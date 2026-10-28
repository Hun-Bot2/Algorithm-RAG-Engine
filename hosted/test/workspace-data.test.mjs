import test from 'node:test';
import assert from 'node:assert/strict';
import {localDate,calendarDates,navigateDate,workspaceEvents,activeAttempts,filterAttempts,conversion,outcomeText,outcomeSymbol,eventCategory,filterCalendarEvents,daySummary,contestSuitability,contestList} from '../public/workspace-data.mjs';

test('calendar covers leap February and navigates months without rollover',()=>{
 const dates=calendarDates('2024-02-29');assert.equal(dates[0],'2024-01-29');assert.equal(dates.at(-1),'2024-03-03');assert.ok(dates.includes('2024-02-29'));
 assert.equal(navigateDate('2024-01-31',1),'2024-02-01');assert.equal(navigateDate('2024-12-31',1),'2025-01-01');
 assert.deepEqual(calendarDates('2026-10-18','week'),['2026-10-12','2026-10-13','2026-10-14','2026-10-15','2026-10-16','2026-10-17','2026-10-18']);
});
test('local dates respect midnight and explicit timezone including DST',()=>{
 assert.equal(localDate('2026-10-19T15:00:00Z'),'2026-10-20');assert.equal(localDate('2026-10-19T14:59:59Z'),'2026-10-19');assert.equal(localDate('bad'),null);
 assert.equal(localDate('2026-03-08T07:00:00Z','America/New_York'),'2026-03-08');
});
const journal={available:true,date:'2026-10-20',plan:{date:'2026-10-20',slots:[{problem_key:'leetcode:a',title:'Current',explanation:{results:[]}}]},days:[{date:'2026-10-19',entries:[
 {id:'actual',type:'attempt',problem_key:'leetcode:a',occurred_at:'2026-10-19T15:01:00Z',outcome:'assisted',platform:'leetcode',difficulty:'Easy',pattern:'arrays',self_minutes:null},
 {id:'old',type:'attempt',problem_key:'leetcode:a',occurred_at:'2026-10-19T10:00:00Z',outcome:'failed',status:'superseded'},
 {id:'plan',type:'recommendation',problem_key:'leetcode:a',local_date:'2026-10-18',occurred_at:'2026-10-19T15:00:00Z'},
 {id:'concept',type:'mastery_change',occurred_at:'2026-10-19T10:00:00Z'}
 ]},{date:'2026-10-20',entries:[{id:'current-old',type:'recommendation',local_date:'2026-10-20',problem_key:'leetcode:a'}]}]};
test('actual timestamps never follow plan dates, plans never count as activity',()=>{
 const rows=workspaceEvents(journal,{contests:[{id:'c',start_at:'2026-10-23T15:30:00Z',duration_minutes:120}]});
 assert.equal(rows.find(x=>x.id==='actual').date,'2026-10-20');assert.equal(rows.find(x=>x.id==='plan').date,'2026-10-18');
 assert.equal(rows.find(x=>x.layer==='contest').date,'2026-10-24');assert.equal(activeAttempts(rows).length,1);assert.equal(rows.filter(x=>x.date==='2026-10-20'&&x.layer==='plan').length,1);assert.equal(rows.some(x=>x.type==='mastery_change'),false);
 assert.equal(rows.find(x=>x.id==='actual').self_minutes,null);assert.equal(rows.filter(x=>x.status==='superseded').length,1);
});
test('history search and platform difficulty filters preserve real outcomes and missingness',()=>{
 const rows=workspaceEvents(journal,{});assert.equal(filterAttempts(rows,{query:'arrays',platform:'leetcode',difficulty:'Easy',outcome:'assisted',from:'2026-10-20',to:'2026-10-20'}).length,1);
 assert.equal(filterAttempts(rows,{to:'2026-10-19'}).length,0);assert.equal(outcomeText({outcome:'reported_completion'}),'완료 보고 · 도움 미상');assert.notEqual(outcomeText({outcome:'independent',hints:null}),'독립 해결');
});
test('conversion deduplicates snapshots and requires a subsequent same-problem attempt within seven days',()=>{
 const rows=[{id:'p1',date:'2026-10-01',problem_key:'leetcode:a',layer:'plan'},{id:'p2',date:'2026-10-01',problem_key:'leetcode:a',layer:'plan'},
 {id:'past',date:'2026-09-30',problem_key:'leetcode:a',layer:'actual'},{id:'late',date:'2026-10-09',problem_key:'leetcode:a',layer:'actual'}];
 assert.deepEqual([conversion(rows).matched,conversion(rows).total],[0,1]);rows.push({date:'2026-10-08',problem_key:'leetcode:a',layer:'actual'});assert.equal(conversion(rows).matched,1);
});
test('core-only runtime keeps imported completions uncertain and preserves corrected records',()=>{
 const rows=workspaceEvents({available:false},{events:[{event_id:'a',kind:'attempt',occurred_at:'2026-10-18T10:00:00Z',payload:{problem_key:'codeforces:4A',outcome:'reported_completion',hints:null}},{event_id:'b',kind:'attempt',supersedes:'a',occurred_at:'2026-10-19T10:00:00Z',payload:{problem_key:'codeforces:4A',outcome:'failed',hints:null}}]});assert.equal(rows.length,2);assert.equal(activeAttempts(rows).length,1);assert.equal(rows.find(x=>x.id==='a').status,'superseded');
});

test('calendar uses distinct recorded independence symbols and prioritizes actual activity',()=>{
 assert.equal(outcomeSymbol({outcome:'independent',hints:false}),'✓');assert.equal(outcomeSymbol({outcome:'assisted'}),'↗');assert.equal(outcomeSymbol({outcome:'reported_completion'}),'?');
 const rows=workspaceEvents(journal,{}).filter(e=>e.date==='2026-10-20');assert.equal(rows[0].layer,'actual');
});


test('October 20 and 21 remain distinct at KST midnight; civil dates never get reinterpreted',()=>{
 assert.equal(localDate('2026-10-20T14:59:59.999Z'),'2026-10-20');
 assert.equal(localDate('2026-10-20T15:00:00.000Z'),'2026-10-21');
 assert.equal(localDate('2026-10-20T23:59:59+09:00'),'2026-10-20');
 assert.equal(localDate('2026-10-21T00:00:00+09:00'),'2026-10-21');
 assert.equal(localDate('2026-10-20','America/Los_Angeles'),'2026-10-20');
 for(const date of [null,undefined,'','2026-99-99','2026-02-30','2026-10-20T23:30:00'])assert.equal(localDate(date),null);
});
test('calendar categories form a disjoint partition; filter choices never count plans as solved',()=>{
 const rows=[{id:'i',layer:'actual',outcome:'independent',hints:false},{id:'a',layer:'actual',outcome:'assisted'},
 {id:'r',layer:'actual',outcome:'reported_completion'},{id:'f',layer:'actual',outcome:'failed'},
 {id:'u',layer:'actual',outcome:null},{id:'p',layer:'plan',outcome:'independent'},
 {id:'c',layer:'contest'},{id:'x',layer:'actual',outcome:'independent',status:'superseded'}];
 assert.deepEqual(rows.slice(0,7).map(eventCategory),['solved','solved','solved','attempted','attempted','planned','contest']);
 const groups=['solved','attempted','planned','contest'].map(key=>filterCalendarEvents(rows,{[key]:true}));
 assert.equal(groups.flat().length,7);assert.equal(new Set(groups.flat().map(e=>e.id)).size,7);
 assert.deepEqual(filterCalendarEvents(rows,{solved:true,contest:true}).map(e=>e.id),['i','a','r','c']);
 assert.deepEqual(daySummary(rows),{attempted:5,solved:3,unsolved:1,unknown:1,independent:1,assisted:1});
});
test('contest suitability never derives skill from titles, activity volume or suitable=false',()=>{
 const contest={id:'contest:codeforces:123',contest_id:'codeforces:123',platform:'codeforces',title:'Div. 4 beginner ABC 999',source:'https://codeforces.com/api/contest.list'};
 const events=Array.from({length:100},(_,i)=>({id:String(i),layer:'actual',platform:'codeforces',outcome:'independent',hints:false}));
 const fit=contestSuitability(contest,events,[]);assert.equal(fit.status,'insufficient');assert.equal(fit.evidence.independent,100);
 assert.equal(contestSuitability({...contest,source:'manual',suitable:false},events).status,'insufficient');
 assert.equal(contestSuitability({...contest,source:'manual',suitable:true},[]).label,'추천 · 직접 선택');
 const participation=[{entry_id:'old',contest_id:'codeforces:123',outcome:'completed',occurred_at:'2026-10-18T01:00:00Z'},
 {entry_id:'corrected',contest_id:'codeforces:123',outcome:'partial',occurred_at:'2026-10-18T01:00:00Z',supersedes:'old'}];
 assert.equal(contestSuitability(contest,events,participation).status,'stretch');
 assert.deepEqual(contestSuitability(contest,events,participation).evidence.participation_ids,['corrected']);
 assert.equal(contestSuitability(contest,events,participation.slice(0,1)).status,'recommended');
 assert.equal(contestSuitability({...contest,contest_id:'codeforces:999'},events,participation).status,'insufficient');
});
test('contest list sorts real timestamps, includes known ongoing events and shares calendar dates',()=>{
 const events=workspaceEvents({available:true},{contests:[
 {id:'future',platform:'leetcode',start_at:'2026-10-20T15:00:00Z',duration_minutes:90},
 {id:'past',platform:'atcoder',start_at:'2026-10-19T01:00:00Z',duration_minutes:100},
 {id:'ongoing',platform:'codeforces',start_at:'2026-10-20T14:00:00Z',duration_minutes:120}]});
 const rows=contestList(events,{now:'2026-10-20T14:30:00Z'});assert.deepEqual(rows.map(x=>x.contest_id),['ongoing','future']);
 assert.equal(rows[0].date,'2026-10-20');assert.equal(rows[1].date,'2026-10-21');
 assert.equal(contestList(events,{platform:'leetcode',now:'2026-10-20T14:30:00Z'}).length,1);
 assert.equal(contestList(events,{scope:'all'}).length,3);
});

test('same-day recommendation revisions are grouped without deleting their evidence or actual attempts',()=>{
 const view=structuredClone(journal);view.days[0].entries.push({id:'plan-revision',type:'recommendation',problem_key:'leetcode:a',local_date:'2026-10-18',occurred_at:'2026-10-20T01:00:00Z',reason:'Revised source evidence'});
 const rows=workspaceEvents(view,{}),plan=rows.find(e=>e.layer==='plan'&&e.date==='2026-10-18');
 assert.equal(rows.filter(e=>e.layer==='plan'&&e.date==='2026-10-18').length,1);
 assert.equal(plan.reason,'Revised source evidence');assert.deepEqual(new Set(plan.versions.map(e=>e.id)),new Set(['plan','plan-revision']));
 assert.equal(rows.filter(e=>e.layer==='actual').length,2);
 const current=rows.find(e=>e.layer==='plan'&&e.date==='2026-10-20');assert.equal(current.current,true);assert.equal(current.versions.length,2);
});

test('supplied server contest evidence is preserved instead of replaced by UI heuristics',()=>{
 const recommendation={status:'stretch',label:'도전 과제',reason:'Native-platform historical opening-problem evidence',evidence:{platform_attempts:12,independent:8,participation_ids:[],historical_contests:['native:123']}};
 assert.strictEqual(contestSuitability({id:'contest:leetcode:weekly-1',recommendation,source:'manual',suitable:true},[],[]),recommendation);
});
test('the current snapshot duplicated in journal days is one revision, with its explanation preserved',()=>{
 const view={available:true,date:'2026-10-20',plan:{date:'2026-10-20',snapshot_id:'snapshot',slots:[{problem_key:'leetcode:a',explanation:{results:[{provider:'gemini',text:'cached'}]}}]},days:[{date:'2026-10-20',entries:[{id:'snapshot-0',snapshot_id:'snapshot',type:'recommendation',local_date:'2026-10-20',problem_key:'leetcode:a'}]}]};
 const rows=workspaceEvents(view,{});assert.equal(rows.length,1);assert.equal(rows[0].explanation.results[0].text,'cached');assert.equal(rows[0].versions,undefined);
});


test('automatic review schedules are omitted while actual review history and ordinary plans remain',()=>{
 const view={available:true,date:'2026-10-23',plan:{date:'2026-10-23',slots:[{problem_key:'leetcode:a',kind:'review'},{problem_key:'leetcode:b',kind:'practice'}]},
 days:[{date:'2026-10-22',entries:[{id:'actual-review',type:'attempt',activity_type:'review',problem_key:'leetcode:a',occurred_at:'2026-10-22T03:00:00Z',outcome:'independent',hints:false},
 {id:'old-review-plan',type:'recommendation',kind:'review',problem_key:'leetcode:a',local_date:'2026-10-22'}]}],
 upcoming:[{problem_key:'leetcode:a',date:'2026-10-24',kind:'review'}]};
 const rows=workspaceEvents(view,{});
 assert.equal(rows.filter(e=>e.layer==='actual').length,1);
 assert.deepEqual(rows.filter(e=>e.layer==='plan').map(e=>e.problem_key),['leetcode:b']);
 assert.equal(rows.some(e=>e.date==='2026-10-24'),false);
 const core=workspaceEvents({available:false},{mission:{date:'2026-10-23',activities:[{type:'review',problem_key:'leetcode:a'},{type:'practice',problem_key:'leetcode:b'}]}});
 assert.deepEqual(core.filter(e=>e.layer==='plan').map(e=>e.problem_key),['leetcode:b']);
});

test('recommendation badges follow system provenance without marking manual reviews or earlier attempts',()=>{
 const j={available:true,plan:{date:'2026-10-23',slots:[{problem_key:'leetcode:a',kind:'practice'}]},review_selections:[{date:'2026-10-23',problem_key:'leetcode:b'}],days:[{date:'2026-10-23',entries:[
 {id:'before',type:'attempt',problem_key:'leetcode:a',occurred_at:'2026-10-22T02:00:00Z'},
 {id:'after',type:'attempt',problem_key:'leetcode:a',occurred_at:'2026-10-23T02:00:00Z'},
 {id:'manual-only',type:'attempt',problem_key:'leetcode:b',occurred_at:'2026-10-23T02:00:00Z'},
 {id:'timed-later',type:'recommendation',problem_key:'leetcode:c',local_date:'2026-10-23',occurred_at:'2026-10-23T04:00:00Z'},
 {id:'same-day-before',type:'attempt',problem_key:'leetcode:c',occurred_at:'2026-10-23T02:00:00Z'}]}]};
 const events=workspaceEvents(j,{}),byId=id=>events.find(e=>e.id===id);
 assert.equal(events.find(e=>e.layer==='plan'&&e.problem_key==='leetcode:a').recommended,true);
 assert.equal(byId('before').recommended,false);assert.equal(byId('after').recommended,true);
 assert.deepEqual(byId('after').recommended_dates,['2026-10-23']);
 assert.equal(byId('manual-only').recommended,false);assert.equal(byId('same-day-before').recommended,false);
 assert.equal(events.find(e=>e.explicit_review).recommended,false);
});
test('recommendation basis distinguishes policy, rank, semantic fallback and missing scores',async()=>{
 const {recommendationBasis}=await import('../public/workspace-data.mjs');
 assert.match(recommendationBasis({stage:'leetcode_easy'}),/벡터 유사도 점수를 기준으로 선택한 추천은 아니/);
 assert.match(recommendationBasis({components:{channel_rank:1.9}}),/순위 가중 점수와 코사인/);
 assert.match(recommendationBasis({semantic_only:true}),/알고리즘 동등성을 뜻하지/);
 assert.match(recommendationBasis({explicit_review:true}),/직접 지정/);
 assert.match(recommendationBasis({}),/제공되지/);
});

test('mock contests project as their own hidden-by-default layer with display-only result matching',()=>{
 const a=(id,key,occurred_at,outcome,extra={})=>({id,type:'attempt',problem_key:key,occurred_at,outcome,hints:outcome==='independent'?false:true,title:key,platform:'leetcode',...extra});
 const journal={available:true,days:[{date:'2026-11-02',entries:[
   a('before','leetcode:p1','2026-10-30T12:00:00Z','independent'),
   a('p1-sat','leetcode:p1','2026-10-31T13:00:00Z','independent'),
   a('p2-old','leetcode:p2','2026-11-01T01:00:00Z','independent',{status:'superseded'}),
   a('p3-late','leetcode:p3','2026-11-02T03:00:00Z','failed'),
   a('p3-first','leetcode:p3','2026-11-01T03:00:00Z','assisted')]}],
  mock_contests:[{policy:'mock-contest-v1',local_date:'2026-10-31',start_at:'2026-10-31T11:00:00.000Z',duration_minutes:60,composition:{easy:1,medium:1,hard:1},
   expected_solved:1.8,target_expected:1.8,evidence_label:null,placement_note:null,notes:[],
   problems:['p1','p2','p3'].map((s,i)=>({slot:i+1,problem_key:`leetcode:${s}`,title:s,url:`https://leetcode.com/problems/${s}/`,level:i+1}))}]};
 const contest={id:'atcoder:abc430',platform:'atcoder',title:'ABC 430',start_at:'2026-10-31T12:00:00Z',duration_minutes:100};
 const events=workspaceEvents(journal,{contests:[contest]});
 const mock=events.find(e=>e.layer==='mock');
 assert.equal(mock.id,'mock:2026-10-31');assert.equal(mock.duration_minutes,60);assert.ok(!('start_at' in mock)&&!('placement_note' in mock),'no clock time, even from a v1 record');assert.equal(mock.date,'2026-10-31');assert.equal(eventCategory(mock),'mock');
 assert.deepEqual(mock.problems.map(p=>p.result?.attempt_id ?? null),['p1-sat',null,'p3-first'],'earliest active attempt on or after the mock date');
 assert.equal(mock.problems[0].result.symbol,'✓');assert.equal(mock.problems[2].result.text,'도움 해결');
 assert.equal(mock.attempted,2);
 const saturday=events.filter(e=>e.date==='2026-10-31').map(e=>e.layer);
 assert.deepEqual(saturday,['actual','mock','contest']);
 assert.ok(!filterCalendarEvents(events).some(e=>e.layer==='mock'),'hidden until the UI enables the mock layer');
 assert.ok(filterCalendarEvents(events,{solved:true,attempted:true,planned:true,contest:true,mock:true}).some(e=>e.layer==='mock'));
 assert.equal(events.find(e=>e.id==='p1-sat').recommended,false,'a mock contest is not a recommendation');
 assert.equal(conversion(events).total,0);
 assert.equal(activeAttempts(events).length,4,'no attempts are created');
 assert.deepEqual(journal.mock_contests[0].problems.map(p=>'result' in p),[false,false,false],'input records are not mutated');
});

test('archived mock contests keep a distinct id beside an active one on the same date',()=>{
 const m=archived=>({local_date:'2026-10-31',duration_minutes:120,archived,problems:[]});
 const events=workspaceEvents({available:true,days:[],mock_contests:[m(true),m(false)]},{});
 assert.deepEqual(events.filter(e=>e.layer==='mock').map(e=>[e.id,e.archived]).sort(),[['mock-archive:2026-10-31',true],['mock:2026-10-31',false]]);
});
