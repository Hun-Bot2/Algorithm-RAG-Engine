/** Optional opening-problem practice goals; never changes problem ranking or liveProposals. */
import {fitAll,predict,practiceObservations,MIN_EVIDENCE} from './calibration.mjs';
import {contestKind} from './contests.mjs';
import {canonicalKey} from './related.mjs';
export const DISCOVERY_FIT_VERSION='contest-opening-goal-v1';
const kindOf=c=>c.platform==='leetcode'?(/biweekly/i.test(c.title)?'biweekly':/weekly/i.test(c.title)?'weekly':'other'):contestKind(c.platform,c.contest_id?.replace(/^(atcoder|codeforces):/,'')??c.id?.replace(/^(atcoder|codeforces):/,'')??'',c.title);
export function contestDiscoveryFit(contest,{ledger,catalog,now=new Date()}={}){
 const attempts=ledger?.attempts??[],platform=contest.platform;
 const independent=attempts.filter(a=>a.outcome==='independent'&&a.hints===false&&a.activity_type==='practice'&&catalog?.problems.get(canonicalKey(catalog,a.problem_key))?.platform===platform);
 const distinct=new Set(independent.map(a=>canonicalKey(catalog,a.problem_key)));
 const nativeDistinct=new Set([...distinct].filter(key=>{const p=catalog.problems.get(key);return Number.isFinite(p.difficulty_value)&&((platform==='codeforces'&&p.difficulty_kind==='cf_rating'&&p.difficulty_value>=0)||(platform==='atcoder'&&p.difficulty_kind==='atc_irt'&&p.difficulty_value>-10000)||(platform==='leetcode'&&p.difficulty_kind==='lc_level'&&[1,2,3].includes(p.difficulty_value)));}));
 const evidence={version:DISCOVERY_FIT_VERSION,catalog_version:catalog?.version??null,platform_attempts:attempts.filter(a=>(catalog?.problems.get(canonicalKey(catalog,a.problem_key))?.platform??a.problem_key?.split(':')[0])===platform).length,independent:independent.length,distinct_independent:distinct.size,distinct_native_independent:nativeDistinct.size,attempt_ids:independent.map(a=>a.id),independent_difficulties:[...distinct].map(key=>{const p=catalog.problems.get(key);return {problem_key:key,difficulty_kind:p.difficulty_kind,value:p.difficulty_value??null};}),participation_ids:[],historical_contests:[],native_difficulties:[],calibration:null,eligibility:contest.eligibility??{status:'unknown'}};
 const result=(status,reason)=>({status,label:{recommended:'수준에 맞는 추천',stretch:'도전 과제',insufficient:'판단 근거 부족'}[status],reason,evidence});
 if(!catalog)return result('insufficient','활성 카탈로그가 없어 플랫폼별 난이도 근거를 비교할 수 없어요.');
 const abilities=fitAll(attempts,catalog,ledger.rating_observations),ability=abilities[platform];
 evidence.calibration=ability?{status:ability.status,n:ability.n,...(platform==='leetcode'?{levels:ability.levels}:{mean:ability.mean,sd:ability.sd,prior:ability.prior})}:null;
 evidence.calibration_attempt_ids=practiceObservations(attempts,catalog,platform).map(o=>o.attempt_id);
 evidence.rating_ids=(ledger.rating_observations??[]).filter(r=>r.payload?.platform===platform).map(r=>r.entry_id);
 if(nativeDistinct.size<MIN_EVIDENCE||ability?.status!=='estimated')return result('insufficient',`같은 플랫폼의 서로 다른 연습 풀이가 ${distinct.size}개예요. 이 중 난이도가 확인된 문제는 ${nativeDistinct.size}개예요. 최소 ${MIN_EVIDENCE}개의 서로 다른 해결 기록과 플랫폼별 난이도 관측이 필요해 아직 수준을 추정하기 어려워요. 다른 플랫폼의 풀이로 대신 추정하지 않아요.`);
 if(contest.stale)return result('insufficient','일정 출처가 오래되어 공식 일정과 참가 조건을 먼저 확인해야 해요.');
 const kind=kindOf(contest);
 if(['other','ahc'].includes(kind))return result('insufficient','이 대회 형식을 비교할 검증된 난이도 자료가 없어요. 휴리스틱 대회와 일반 알고리즘 실력을 환산하지 않아요.');
 const history=(catalog.contests??[]).filter(c=>c.platform===platform&&kindOf(c)===kind&&Date.parse(c.start_at??c.start_time??c.contest_start)<Math.min(now.getTime(),Date.parse(contest.start_at)))
 .sort((a,b)=>Date.parse(b.start_at??b.start_time??b.contest_start)-Date.parse(a.start_at??a.start_time??a.contest_start)).slice(0,5);
 const estimates=[];
 for(const old of history){
  const candidates=(catalog.contestProblems.get(old.contest_id)??[]).map(r=>catalog.problems.get(r.problem_key)).filter(p=>p?.platform===platform&&p.eligible!==false&&Number.isFinite(p.difficulty_value)&&((platform==='atcoder'&&p.difficulty_kind==='atc_irt'&&p.difficulty_value>-10000)||(platform==='codeforces'&&p.difficulty_kind==='cf_rating')||(platform==='leetcode'&&p.difficulty_kind==='lc_level'))).sort((a,b)=>a.difficulty_value-b.difficulty_value||a.key.localeCompare(b.key));
  const p=candidates[0];if(!p)continue;const prediction=predict(p,abilities);if(!Number.isFinite(prediction.p_conservative))continue;
  evidence.historical_contests.push(old.contest_id);evidence.native_difficulties.push({problem_key:p.key,difficulty_kind:p.difficulty_kind,value:p.difficulty_value,p_conservative:prediction.p_conservative});estimates.push(prediction.p_conservative);
 }
 if(estimates.length<3)return result('insufficient','같은 형식의 과거 대회에서 비교할 수 있는 플랫폼 고유 난이도 자료가 3회분 미만이에요. 공개되지 않은 이번 대회의 난이도를 만들지 않아요.');
 const score=estimates.reduce((a,b)=>a+b,0)/estimates.length;evidence.opening_proxy_probability=score;
 const caveat=`같은 형식의 과거 ${estimates.length}회에서 난이도가 알려진 가장 쉬운 문제를 비교한 연습용 추정이에요. 이번 문제 난이도·전체 완주·공식 레이팅을 예측한 값은 아니며 참가 자격은 공식 공지에서 확인해야 해요.`;
 if(score>=.65)return result('recommended','첫 문제 해결을 목표로 참가해 볼 만해요. '+caveat);
 if(score>=.30)return result('stretch','첫 문제부터 시간이 더 필요할 수 있는 도전 과제예요. '+caveat);
 return result('insufficient','현재 보수적 추정으로는 첫 문제 목표도 자신 있게 추천하기 어려워요. '+caveat);
}
