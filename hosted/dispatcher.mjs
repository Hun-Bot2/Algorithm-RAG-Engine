import {listManualContests} from './core/store.mjs';
import {prepareMission,seoulDate} from './core/policy.mjs';
import {claimDelivery,ensureDelivery,expireSending,listUnresolved,missReady,transitionDelivery} from './notifications.mjs';

const KST_OFFSET='+09:00';
export const deliveryTimes = localDate => ({
  cutoff:new Date(`${localDate}T05:40:00${KST_OFFSET}`),
  target:new Date(`${localDate}T06:00:00${KST_OFFSET}`),
  deadline:new Date(`${localDate}T06:10:00${KST_OFFSET}`),
});
const iso = date => date.toISOString().replace('.000Z','.000000Z');
const retrySeconds = response => {
  const value=Number(response.headers.get('Retry-After'));
  return Number.isFinite(value) && value>=1 ? Math.min(Math.ceil(value),3600) : 60;
};
async function slackCall(method,{env,fetcher,payload={},query=null}) {
  const url=new URL('https://slack.com/api/'+method);
  if (query) for (const [key,value] of Object.entries(query)) if (value!==undefined && value!==null && value!=='') url.searchParams.set(key,String(value));
  const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),10000);
  try {
    const response=await fetcher(url.href,{method:query?'GET':'POST',headers:{authorization:`Bearer ${env.SLACK_BOT_TOKEN}`,...(query?{}:{'content-type':'application/json; charset=utf-8'})},...(query?{}:{body:JSON.stringify(payload)}),signal:controller.signal});
    if (response.status===429) return {kind:'rate_limited',retrySeconds:retrySeconds(response)};
    if (response.status>=500) return {kind:'unknown'};
    if (!response.ok) return {kind:'rejected',error:'http_'+response.status};
    let body;
    try { body=await response.json(); } catch { return {kind:'unknown'}; }
    if (body?.ok!==true) return {kind:'rejected',error:typeof body?.error==='string'?body.error:'slack_error'};
    return {kind:'ok',body};
  } catch { return {kind:'unknown'}; }
  finally { clearTimeout(timer); }
}

async function sendClaim(db,row,{env,fetcher,now,target,deadline}) {
  const key=row.uniqueness_key, message=JSON.parse(row.message_payload);
  const scheduled=now<target, method=scheduled?'chat.scheduleMessage':'chat.postMessage';
  const payload={channel:row.channel_id,text:message.text,blocks:message.blocks};
  if (scheduled) payload.post_at=Math.floor(target.getTime()/1000);
  // Slack documents a scheduleMessage metadata bug; never add metadata here.
  const result=await slackCall(method,{env,fetcher,payload});
  if (result.kind==='rate_limited') {
    const retryAt=new Date(now.getTime()+result.retrySeconds*1000);
    if (retryAt>deadline) return transitionDelivery(db,{key,from:'sending',to:'missed',now,errorCode:'rate_limit_past_window'});
    return transitionDelivery(db,{key,from:'sending',to:'ready',now,retryAt:iso(retryAt),errorCode:'rate_limited'});
  }
  if (result.kind==='unknown') return transitionDelivery(db,{key,from:'sending',to:'unknown',now,errorCode:'ambiguous_send'});
  if (result.kind==='rejected' && scheduled && result.error==='time_in_past' && now<=deadline)
    return transitionDelivery(db,{key,from:'sending',to:'ready',now,retryAt:iso(target),errorCode:'schedule_time_passed'});
  if (result.kind==='rejected') return transitionDelivery(db,{key,from:'sending',to:'missed',now,errorCode:'slack_rejected'});
  if (scheduled) {
    const id=result.body.scheduled_message_id;
    if (typeof id!=='string' && typeof id!=='number') return transitionDelivery(db,{key,from:'sending',to:'unknown',now,errorCode:'missing_schedule_receipt'});
    return transitionDelivery(db,{key,from:'sending',to:'scheduled',now,scheduledId:String(id),errorCode:null});
  }
  const ts=result.body.ts;
  if (typeof ts!=='string') return transitionDelivery(db,{key,from:'sending',to:'unknown',now,errorCode:'missing_post_receipt'});
  return transitionDelivery(db,{key,from:'sending',to:'posted',now,slackTs:ts,errorCode:null});
}

async function pagedLookup(method,{env,fetcher,row,field,now,maxPages=5}) {
  let cursor='',complete=false;
  for (let page=0;page<maxPages;page++) {
    const params=method==='conversations.history'
      ? {channel:row.channel_id,oldest:Math.floor(deliveryTimes(row.local_date).cutoff.getTime()/1000)-60,limit:100,cursor}
      : {channel:row.channel_id,oldest:Math.floor(deliveryTimes(row.local_date).cutoff.getTime()/1000)-60,latest:Math.floor(deliveryTimes(row.local_date).deadline.getTime()/1000)+3600,limit:100,cursor};
    const result=await slackCall(method,{env,fetcher,...(method==='conversations.history'?{query:params}:{payload:params})});
    if (result.kind!=='ok') return {match:null,complete:false,rateLimit:result.kind==='rate_limited'?result.retrySeconds:null};
    const entries=result.body[field];
    if (!Array.isArray(entries)) return {match:null,complete:false};
    const match=entries.find(item=>typeof item?.text==='string' && item.text.includes(row.uniqueness_key) || method==='chat.scheduledMessages.list' && row.scheduled_message_id && String(item?.id)===row.scheduled_message_id);
    if (match) return {match,complete:true};
    const next=result.body.response_metadata?.next_cursor;
    if (!next) { complete=true; break; }
    if (next===cursor) return {match:null,complete:false};
    cursor=next;
  }
  return {match:null,complete};
}

export async function reconcileDelivery(db,row,{env,fetcher=fetch,now=new Date()}) {
  if (!['unknown','scheduled'].includes(row.state) || row.retry_at && row.retry_at>iso(now)) return row.state;
  const scheduled=await pagedLookup('chat.scheduledMessages.list',{env,fetcher,row,field:'scheduled_messages',now});
  if (scheduled.rateLimit) {
    await transitionDelivery(db,{key:row.uniqueness_key,from:row.state,to:row.state,now,retryAt:iso(new Date(now.getTime()+scheduled.rateLimit*1000)),errorCode:'reconcile_rate_limited'});
    return row.state;
  }
  const history=await pagedLookup('conversations.history',{env,fetcher,row,field:'messages',now});
  if (history.rateLimit) {
    await transitionDelivery(db,{key:row.uniqueness_key,from:row.state,to:row.state,now,retryAt:iso(new Date(now.getTime()+history.rateLimit*1000)),errorCode:'reconcile_rate_limited'});
    return row.state;
  }
  if (history.match) {
    await transitionDelivery(db,{key:row.uniqueness_key,from:row.state,to:'posted',now,slackTs:typeof history.match.ts==='string'?history.match.ts:null,errorCode:null});
    return 'posted';
  }
  if (scheduled.match && row.state==='unknown') {
    await transitionDelivery(db,{key:row.uniqueness_key,from:'unknown',to:'scheduled',now,scheduledId:String(scheduled.match.id ?? row.scheduled_message_id ?? ''),errorCode:null});
    return 'scheduled';
  }
  // Even exhaustive current listings cannot prove Slack did not accept a request.
  return row.state;
}

export async function runDispatcher({db,curriculum,annotations={},now=new Date(),env={},fetcher=fetch,learnerIds,contests=[]}={}) {
  if (env.DELIVERY_ENABLED!=='true') return {disabled:true,processed:[]};
  if (!db || !curriculum || !env.SLACK_BOT_TOKEN || !env.SLACK_CHANNEL_ID || !env.DELIVERY_LEARNER_ID) throw new Error('delivery configuration missing');
  const clock=now instanceof Date ? now : new Date(now);
  if (Number.isNaN(clock.getTime())) throw new Error('invalid dispatcher time');
  const date=seoulDate(clock),times=deliveryTimes(date);
  const owners=learnerIds ?? (await db.prepare('SELECT learner_id FROM learners ORDER BY learner_id').all()).results.map(row=>row.learner_id);
  const processed=[];
  for (const learnerId of owners) {
    if (learnerId!==env.DELIVERY_LEARNER_ID) continue;
    const gate=await db.prepare('SELECT delivery_reconciliation_required FROM learners WHERE learner_id=?').bind(learnerId).first();
    if (gate?.delivery_reconciliation_required===1) {processed.push({learner_id:learnerId,status:'restore_reconciliation_required'});continue;}
    await expireSending(db,learnerId,clock);
    for (const row of await listUnresolved(db,learnerId)) await reconcileDelivery(db,row,{env,fetcher,now:clock});
    if (clock<times.cutoff) { processed.push({learner_id:learnerId,status:'before_cutoff'}); continue; }
    const manual=await listManualContests(db,learnerId);
    const mission=await prepareMission({db,learnerId,curriculum,annotations,now:times.cutoff,preparedAt:clock,contests:[...manual,...contests]});
    const row=await ensureDelivery(db,{learnerId,mission,curriculum,annotations,channelId:env.SLACK_CHANNEL_ID,now:clock});
    if (mission.event_cutoff!==iso(times.cutoff) || mission.curriculum_version!==curriculum.curriculum_version || !mission.activities?.length) {
      await missReady(db,{learnerId,localDate:date,channelId:env.SLACK_CHANNEL_ID,now:clock,reason:'stale_or_empty_mission'});
      processed.push({learner_id:learnerId,status:'stale_or_empty_mission'});
      continue;
    }
    if (clock>times.deadline) {
      await missReady(db,{learnerId,localDate:date,channelId:env.SLACK_CHANNEL_ID,now:clock,reason:'delivery_window_passed'});
      processed.push({learner_id:learnerId,status:'missed'});
      continue;
    }
    const claimed=await claimDelivery(db,{learnerId,localDate:date,channelId:env.SLACK_CHANNEL_ID,now:clock,deadline:times.deadline});
    if (claimed) await sendClaim(db,claimed,{env,fetcher,now:clock,target:times.target,deadline:times.deadline});
    const latest=await db.prepare('SELECT state FROM delivery_outbox WHERE learner_id=? AND local_date=? AND channel_id=?').bind(learnerId,date,env.SLACK_CHANNEL_ID).first();
    processed.push({learner_id:learnerId,status:latest?.state ?? row.state});
  }
  return {disabled:false,processed};
}
