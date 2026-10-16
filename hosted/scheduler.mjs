/** Mission freezing is independent of Slack transport and of all AI providers. */
import {prepareMission,seoulDate} from './core/policy.mjs';
import {listManualContests} from './core/store.mjs';
import {cachedProviders} from './provider-service.mjs';
import {deliveryTimes,runDispatcher} from './dispatcher.mjs';
export async function runScheduler({db,curriculum,annotations={},env={},now=new Date(),fetcher}={}) {
  if(env.SCHEDULER_ENABLED!=='true')return {disabled:true,frozen:0};
  const times=deliveryTimes(seoulDate(now));
  if(now<times.cutoff)return {disabled:false,frozen:0};
  const learners=(await db.prepare('SELECT learner_id FROM learners ORDER BY learner_id').all()).results;
  let frozen=0;
  for(const {learner_id:learnerId} of learners){
    const cached=await cachedProviders(db,learnerId,now);
    const manual=await listManualContests(db,learnerId);
    await prepareMission({db,learnerId,curriculum,annotations,now:times.cutoff,preparedAt:now,contests:[...manual,...cached.items]});
    frozen++;
  }
  const cached=env.DELIVERY_LEARNER_ID?await cachedProviders(db,env.DELIVERY_LEARNER_ID,now):{items:[]};
  return {disabled:false,frozen,delivery:await runDispatcher({db,curriculum,annotations,env,now,fetcher,contests:cached.items})};
}
