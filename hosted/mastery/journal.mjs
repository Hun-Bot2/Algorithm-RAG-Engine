// Local journal composes the existing planner, ledger, immutable snapshots and catalog.
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {join} from 'node:path';
import {canonicalJson} from '../contest-history.mjs';
import {appendEvents,getPreferences} from '../core/store.mjs';
import {timestamp,timerState} from '../core/events.mjs';
import {seoulDate} from '../core/policy.mjs';
import {readLedger,getSnapshot,saveSnapshot,solvedView,SOLVED_OUTCOMES} from './ledger.mjs';
import {attachCatalog,loadCatalog} from './catalog.mjs';
import {planDay,problemStates,weekInfo} from './planner.mjs';
import {reviewQueue} from './reviews.mjs';
import {buildWeeklyReport} from './weekly.mjs';
import {canonicalKey,trustedConcepts} from './related.mjs';
import interview from './interview-curriculum.v1.json' with {type:'json'};
import patternConfig from './progression-pattern.v1.json' with {type:'json'};
import {planProgression} from './progression.mjs';
import {mockExposurePlans} from './mock-contest.mjs';

const digest=value=>createHash('sha256').update(canonicalJson(value)).digest('hex');
const POLICY_CODE_HASH=digest(Object.fromEntries(['./journal.mjs','./progression.mjs','./planner.mjs','./reviews.mjs','./calibration.mjs','../core/policy.mjs'].map(path=>[path,readFileSync(new URL(path,import.meta.url),'utf8')])));
export const JOURNAL_VERSION='journal-v1';
export function activeLocalCatalog(db,ledger,root) {
  const switches=ledger.catalog_switches ?? [],current=switches.at(-1);
  if (!current) throw new Error('no active catalog: run node hosted/cli.mjs catalog use --version catalog-v4');
  const info=attachCatalog(db.sqlite,join(root,current.payload.to_version,`${current.payload.to_version}.sqlite3`));
  if(info.sha256!==current.payload.catalog_sha256)throw new Error('active catalog file differs from the logged switch');
  const catalog=loadCatalog(db.sqlite);catalog.base_version=switches[0].payload.to_version;return catalog;
}
const level=p=>p?.platform==='leetcode'?({1:'Easy',2:'Medium',3:'Hard'}[p.difficulty_value] ?? p.difficulty_value):p?.difficulty_value ?? null;
const stageRows=p=>p.concepts?Object.entries(p.concepts).map(([pattern,s])=>({...s,pattern,stage:s.next_stage ?? s.stage ?? s.stages?.[s.stage_index]?.id ?? null})):p.patterns ?? p.stages ?? [];
const nextCandidate=p=>p.next ?? p.candidate ?? stageRows(p).find(x=>x.candidate)?.candidate ?? null;

export function journalPlan({date,now,ledger,catalog,preferences,priorPlans=[]}) {
  const rhythm=[...preferences.weekly_rhythm],day=weekInfo(date).weekday;
  const contestDay=['virtual','upsolve'].includes(rhythm[day]);
  if(contestDay)rhythm[day]='practice';
  const base=planDay({date,now,ledger,catalog,preferences:{...preferences,weekly_rhythm:rhythm},priorPlans});
  if(contestDay)base.notes.push('Journal keeps review/curriculum available; contest rhythm remains in Today/Contests and the legacy CLI plan.');
  const progression=planProgression({date,ledger,catalog,priorPlans,config:{...patternConfig,...interview,version:patternConfig.version,target_concepts:preferences.mastery?.target_concepts ?? []}});
  const budget=preferences.weekday_minutes,slots=base.slots.filter(s=>s.kind==='review');
  let used=slots.reduce((s,x)=>s+x.estimated_minutes,0);
  let next=nextCandidate(progression);
  const latest=new Map(ledger.attempts.map(a=>[canonicalKey(catalog,a.problem_key),a]));
  const advancedFailure=[...latest].some(([key,a])=>catalog.problems.get(key)?.platform==='leetcode' && catalog.problems.get(key)?.difficulty_value===3 && ['failed','assisted'].includes(a.outcome));
  const advancedRepair=base.slots.find(s=>s.kind==='remediation');
  if(advancedFailure && advancedRepair && next?.status!=='remediation'){
    next={...advancedRepair,status:'remediation',stage:'advanced_repair',reason:'Advanced challenge needs targeted repair; Easy/Medium evidence is preserved. '+advancedRepair.reason};
    base.focus_override={pattern:advancedRepair.components?.concept,stage:'advanced_repair',status:'remediation',reason:next.reason};
  }
  if(next && used+30<=budget) {
    const key=next.problem_key ?? next.key,p=catalog.problems.get(key);
    slots.push({...next,kind:next.status==='remediation'?'remediation':next.stage==='cross_platform_transfer'?'transfer':next.kind ?? 'practice',problem_key:key,title:p?.title ?? key,platform:p?.platform ?? next.platform,url:p?.url ?? null,
      difficulty:level(p),concepts:[...trustedConcepts(p)],reason:next.reason ?? progression.reason ?? 'Next curriculum evidence needed.',estimated_minutes:30});
  }
  if(next && used+30>budget)base.notes.push(`Next curriculum activity needs 30 minutes; ${Math.max(0,budget-used)} minutes remain after reviews.`);
  // If a pattern has no eligible next candidate, do not silently replace its stage with arbitrary practice.
  for(const slot of slots)slot.difficulty=level(catalog.problems.get(slot.problem_key));
  return {...base,policy_bundle:JOURNAL_VERSION,slots,progression,
    notes:[...base.notes,...(!next?['No supported next curriculum candidate; review or adjust prerequisites/variation coverage.']:[])],
    budget_minutes:budget,curriculum_version:interview.version,progression_config_sha256:digest({interview,patternConfig}),implementation_sha256:POLICY_CODE_HASH};
}

export function createJournalService({db,catalogRoot,catalogProvider=null}) {
  const load=ledger=>catalogProvider?catalogProvider(ledger):activeLocalCatalog(db,ledger,catalogRoot);
  async function view(learnerId,clock=new Date()) {
    const at=timestamp(clock.toISOString()),date=seoulDate(clock),ledger=await readLedger(db,learnerId,{cutoff:at});
    let catalog;
    try{catalog=await load(ledger);}catch(error){if(error.message.startsWith('no active catalog'))return {available:false,reason:error.message};throw error;}
    const preferences=await getPreferences(db,learnerId);
    const prior=(await db.prepare('SELECT kind,local_date,payload FROM recommendation_snapshots WHERE learner_id=? ORDER BY local_date,kind').bind(learnerId).all()).results;
    // Mock contests are not plans (no slots); they only add exposures for new daily plans (specs/mock-contest R6.4).
    const rows=prior.map(r=>({...r,raw:JSON.parse(r.payload)})),isMock=r=>r.kind==='mock_contest' || r.raw.original_kind==='mock_contest';
    // Restored backups wrap them as inert archives: shown as history, never used as exposures (like archived plans).
    const mockContests=rows.filter(isMock).map(r=>r.kind==='mock_contest'?{...r.raw,archived:false}:{...r.raw.original_payload,local_date:r.local_date,archived:true});
    const historyPlans=rows.filter(r=>!isMock(r)).map(r=>{const raw=r.raw,archived=raw.archive_type==='recommendation_snapshot',p=archived?raw.original_payload:raw;return {...p,snapshot_id:p.snapshot_id ?? `${archived?raw.original_kind:r.kind}-${r.local_date}`,local_date:r.local_date,archived};});
    const plans=historyPlans.filter(p=>!p.archived);
    const key='journal-'+digest({date,version:JOURNAL_VERSION,catalog:catalog.version,attempts:ledger.attempts,ratings:ledger.rating_observations,preferences,interview,patternConfig,implementation:POLICY_CODE_HASH}).slice(0,40);
    let plan=await getSnapshot(db,learnerId,key,date);
    if(!plan){plan=journalPlan({date,now:at,ledger,catalog,preferences,priorPlans:[...plans,...mockExposurePlans(mockContests.filter(m=>!m.archived))]});plan.snapshot_id=key;plan=await saveSnapshot(db,learnerId,key,date,plan);}
    const all=[...historyPlans.filter(p=>p.archived || p.snapshot_id!==key),plan];
    const queue=reviewQueue(problemStates(ledger.attempts,catalog),catalog,date);
    const entries=[];
    for(const a of ledger.attempts){
      const pk=canonicalKey(catalog,a.problem_key),p=catalog.problems.get(pk);
      const q=queue.find(r=>r.problem_key===pk);
      entries.push({id:a.id,type:'attempt',occurred_at:a.occurred_at,recorded_at:a.recorded_at,problem_key:pk,title:p?.title ?? pk,platform:p?.platform ?? pk.split(':')[0],difficulty:level(p),pattern:[...trustedConcepts(p)].join(', '),
        activity_type:a.activity_type ?? null,transfer_from:a.transfer_from ?? null,evidence_kind:a.activity_type==='review'?'known_problem_review':a.activity_type==='transfer'?'transfer_claim':'practice',outcome:a.outcome,display_outcome:a.outcome==='independent'&&a.hints===false?'O':a.outcome==='reported_completion'?'?':'X',hints:a.hints,self_minutes:a.self_minutes,...(a.self_seconds!=null?{self_seconds:a.self_seconds}:{}),time_basis:a.self_minutes===null&&a.self_seconds==null?'not_recorded':'self_reported',priority:['failed','assisted'].includes(a.outcome)?'priority':'normal',evidence_source:a.evidence_source});
    }
    // Corrections remove evidence from active replay, not from the chronological audit trail.
    const activeIds=new Set(ledger.attempts.map(a=>a.id));
    for(const [table,idField] of [['events','event_id'],['git_observations','entry_id']]){
      const raw=(await db.prepare(`SELECT canonical FROM ${table} WHERE learner_id=? AND sequence<=? ORDER BY sequence`).bind(learnerId,ledger.high_water[table] ?? 0).all()).results.map(r=>JSON.parse(r.canonical));
      for(const r of raw){
        if((table==='events'&&r.kind!=='attempt')||activeIds.has(r[idField])||r.occurred_at>at||r.recorded_at>at)continue;
        const p=catalog.problems.get(canonicalKey(catalog,r.payload.problem_key));
        entries.push({id:r[idField],type:'attempt',occurred_at:r.occurred_at,recorded_at:r.recorded_at,problem_key:r.payload.problem_key,title:p?.title ?? r.payload.problem_key,platform:p?.platform ?? r.payload.problem_key.split(':')[0],difficulty:level(p),pattern:[...trustedConcepts(p)].join(', '),activity_type:r.payload.activity_type ?? null,transfer_from:r.payload.transfer_from ?? null,...solvedView(r.payload.outcome,r.payload.hints),display_outcome:SOLVED_OUTCOMES.has(r.payload.outcome)?'O':'X',self_minutes:r.payload.self_minutes ?? null,...(r.payload.self_seconds!=null?{self_seconds:r.payload.self_seconds}:{}),priority:['failed','assisted'].includes(r.payload.outcome)?'priority':'normal',status:'superseded',reason:'Historical record corrected by an append-only successor; excluded from active mastery evidence.',superseded_by:raw.find(x=>x.supersedes===r[idField])?.[idField] ?? null});
      }
    }
    for(const p of all.filter(x=>x.snapshot_id))for(const [i,s] of p.slots.entries())entries.push({...s,id:(p.archived?'archive-':'')+p.snapshot_id+'-'+i,type:'recommendation',archived:Boolean(p.archived),occurred_at:p.prepared_at ?? p.event_cutoff ?? null,local_date:p.date ?? p.local_date,snapshot_id:p.snapshot_id,priority:s.review_priority ?? 'normal',pattern:(s.concepts ?? []).join(', ')});
    // Reproducible mastery changes are derived at evidence prefixes, not an editable second ledger.
    const initial=planProgression({date,catalog,ledger:{...ledger,attempts:[]},config:{...patternConfig,...interview,version:patternConfig.version},evidenceOnly:true});
    let previous=new Map(stageRows(initial).map(s=>[s.pattern,canonicalJson({stage:s.stage,status:s.status})]));
    for(let i=0;i<ledger.attempts.length;i++){
      const a=ledger.attempts[i],result=planProgression({date:seoulDate(new Date(a.occurred_at)),catalog,ledger:{...ledger,attempts:ledger.attempts.slice(0,i+1)},config:{...patternConfig,...interview,version:patternConfig.version},evidenceOnly:true});
      for(const state of stageRows(result)){
        const pattern=state.pattern ?? state.concept ?? state.id,signature=canonicalJson({stage:state.stage,status:state.status});
        if(previous.get(pattern)!==signature){entries.push({id:`mastery-${a.id}-${pattern}`,type:'mastery_change',occurred_at:a.occurred_at,pattern,stage:state.stage,status:state.status,reason:state.reason ?? 'Evidence-derived curriculum state changed.'});previous.set(pattern,signature);}
      }
    }
    const intervals=timerState(ledger.events,at);
    for(const e of entries.filter(e=>e.type==='attempt')){
      const spans=intervals.filter(t=>t.problem_key===e.problem_key&&t.status==='confirmed'&&seoulDate(new Date(t.confirmed_end))===seoulDate(new Date(e.occurred_at)));
      e.confirmed_seconds=spans.length?spans.reduce((n,t)=>n+t.confirmed_seconds,0):null;
      if(e.confirmed_seconds!==null)e.time_basis='confirmed_intervals_same_day_not_attributed_to_attempt';
    }
    const byDate=new Map();
    for(const e of entries){const d=e.occurred_at?seoulDate(new Date(e.occurred_at)):e.local_date;if(!byDate.has(d))byDate.set(d,[]);byDate.get(d).push(e);}
    const patterns=stageRows(plan.progression);
    return {available:true,date,mock_contests:mockContests,focus:plan.focus_override ?? {pattern:plan.progression.next?.concept ?? plan.progression.focus?.pattern ?? plan.progression.focus ?? patterns.find(p=>p.candidate)?.pattern ?? null,stage:plan.progression.next?.stage ?? plan.progression.stage ?? patterns.find(p=>p.candidate)?.stage ?? null,status:plan.progression.next?.status ?? plan.progression.status ?? null,reason:plan.slots.find(s=>s.kind!=='review')?.reason ?? plan.notes.at(-1)},
      plan,days:[...byDate].sort((a,b)=>b[0].localeCompare(a[0])).map(([date,entries])=>({date,entries:entries.sort((a,b)=>(b.occurred_at ?? '').localeCompare(a.occurred_at ?? '')||a.id.localeCompare(b.id))})),
      review_selections:(preferences.review_targets ?? []).map(t=>{const p=catalog.problems.get(t.problem_key);return {...t,title:p?.title ?? t.problem_key,url:p?.url ?? null,platform:p?.platform ?? t.problem_key.split(':')[0],difficulty:level(p)};}),
      reviews:{priority:queue.filter(r=>r.review_priority==='priority'),normal:queue.filter(r=>r.review_priority!=='priority')},patterns,
      weekly:buildWeeklyReport({ledger,catalog,plans:all.filter(p=>!p.archived),weekStart:weekInfo(date).weekStart,preferences}),upcoming:queue.filter(r=>!r.due).map(r=>({...r,date:r.due_date,kind:'review',reason:`Scheduled ${r.review_priority} review; stage ${r.review_stage}.`})),
      limitations:['Outcomes are user reported, not judge-verified.','AI-reviewed concepts are not verified algorithm equivalence.','Policy tests do not establish educational effectiveness.']};
  }
  async function attempt(learnerId,input,clock=new Date()){
    const allowed=['event_id','problem_key','outcome','hints','activity_type','transfer_from','self_minutes','self_seconds','occurred_at','supersedes'];
    if(!input||Object.keys(input).some(k=>!allowed.includes(k)))throw new Error('invalid journal attempt');
    const ledger=await readLedger(db,learnerId),catalog=await load(ledger);
    const key=canonicalKey(catalog,input.problem_key);if(!catalog.problems.has(key))throw new Error('invalid journal problem');
    const existing=await db.prepare('SELECT canonical FROM events WHERE learner_id=? AND event_id=?').bind(learnerId,input.event_id).first();
    const old=existing?JSON.parse(existing.canonical):null;
    const now=old?.recorded_at ?? timestamp(clock.toISOString()),payload={problem_key:key,outcome:input.outcome,hints:input.hints,activity_type:input.activity_type ?? 'practice'};
    if(payload.activity_type==='transfer')payload.transfer_from=canonicalKey(catalog,input.transfer_from);
    if('self_minutes'in input)payload.self_minutes=input.self_minutes;
    if('self_seconds'in input)payload.self_seconds=input.self_seconds;
    const added=await appendEvents(db,learnerId,[{schema_version:1,event_id:input.event_id,learner_id:learnerId,kind:'attempt',source:'manual',occurred_at:input.occurred_at ?? old?.occurred_at ?? now,recorded_at:now,payload,supersedes:input.supersedes ?? null}]);
    return {added,journal:await view(learnerId,clock)};
  }
  return {view,attempt};
}
