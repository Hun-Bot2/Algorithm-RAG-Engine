import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,rm,readFile,stat,writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {DatabaseSync} from 'node:sqlite';
import {createExplanationService,explanationEvidence,explanationProviders,withExplanations,aggregateExplanationUsage,EXPLANATION_VERSION} from '../mastery/explanations.mjs';
import {createExplanationStore} from '../mastery/explanation-store.mjs';
import {explanationOtlp} from '../mastery/explanation-observability.mjs';
import {createDatabase} from './helpers/d1.mjs';
import {createJournalService} from '../mastery/journal.mjs';
import {catalogFromData} from '../mastery/catalog.mjs';
import {run} from '../cli.mjs';

const slot={problem_key:'leetcode:two-sum',title:'Two Sum',kind:'practice',concepts:['arrays-hashing'],reason:'Next unseen Easy candidate; 0/3 independent solves.',difficulty:'Easy'};
const plan={date:'2026-10-20',catalog_version:'fixture-v1',policy_bundle:'fixture',snapshot_id:'journal-fixture',slots:[slot]};
const view=()=>({available:true,plan:structuredClone(plan),days:[{date:plan.date,entries:[{...slot,type:'recommendation',snapshot_id:plan.snapshot_id}]}]});
const env={GEMINI_API_KEY:'secret-gemini',NVIDIA_API_KEY:'secret-nvidia',NVIDIA_EXPLANATION_MODEL:'fixture/model'};
const text='현재 학습 단계에 맞춰 아직 풀지 않은 연습 문제를 골랐어요. 풀이 결과를 기록해 주세요.';
const turn=()=>new Promise(resolve=>setImmediate(resolve));
async function fixture(t){const root=await mkdtemp(join(tmpdir(),'explanations-dual-'));t.after(()=>rm(root,{recursive:true,force:true}));return root;}
function request(options){const body=JSON.parse(options.body),gemini=Boolean(body.contents);return {body,provider:gemini?'gemini':'nvidia',instructions:gemini?body.systemInstruction.parts[0].text:body.messages[0].content,input:gemini?body.contents[0].parts[0].text:body.messages[1].content};}
function responseFor(options,{override={},withUsage=true,finish=null}={}){
 const {provider,input}=request(options),result=JSON.stringify({text,evidence_id:JSON.parse(input).evidence_id,...override});
 return Response.json(provider==='gemini'?{responseId:'gemini-response',modelVersion:'gemini-resolved',candidates:[{finishReason:finish ?? 'STOP',content:{parts:[{text:result}]}}],...(withUsage?{usageMetadata:{promptTokenCount:100,candidatesTokenCount:25,totalTokenCount:130,thoughtsTokenCount:5,cachedContentTokenCount:10}}:{})}
 :{id:'nvidia-response',model:'nvidia-resolved',choices:[{finish_reason:finish ?? 'stop',message:{content:result}}],...(withUsage?{usage:{prompt_tokens:80,completion_tokens:20,total_tokens:100,completion_tokens_details:{reasoning_tokens:2}}}:{})});
}
const successFetch=async(u,o)=>responseFor(o);

test('two production provider roles always exist; read-only and missing configuration cause no dispatch or DB creation',async t=>{
 const root=await fixture(t);let calls=0;const service=createExplanationService({cacheRoot:root,fetcher:()=>{calls++;throw Error('should not call');}});
 const before=view(),output=await service.decorate('owner',before,{generate:true});
 assert.equal(calls,0);assert.equal(output.plan.slots[0].explanation.results.length,2);
 assert.deepEqual(output.plan.slots[0].explanation.results.map(r=>r.status),['not_configured','not_configured']);
 assert.equal(output.plan.slots[0].explanation.fallback.source,'template');assert.equal(before.plan.slots[0].explanation,undefined);
 assert.deepEqual(service.usage('owner').groups,[]);assert.deepEqual(service.records('owner'),[]);
 await assert.rejects(stat(join(root,'research.sqlite3')),/ENOENT/);
 assert.equal(explanationProviders({}).length,2);
});

test('both providers start concurrently with byte-identical frozen evidence and instructions; each result persists independently and leases deduplicate',async t=>{
 const root=await fixture(t),calls=[];let release;
 const gate=new Promise(resolve=>{release=resolve;});
 const fetcher=async(url,options)=>{const r=request(options);calls.push(r);if(r.provider==='nvidia')await gate;return responseFor(options);};
 const service=createExplanationService({cacheRoot:root,env,fetcher}),input=view();
 const pending=service.decorate('owner',input,{generate:true});await turn();
 assert.equal(calls.length,2);assert.equal(calls[0].input,calls[1].input);assert.equal(calls[0].instructions,calls[1].instructions);
 assert.equal(calls[0].body.generationConfig.temperature,calls[1].body.temperature);
 const partial=service.records('owner');assert.equal(partial.length,2);assert.equal(partial.filter(r=>r.result?.status==='success').length,1);
 assert.ok(partial.every(r=>r.input.input_json===calls[0].input && r.input.prompt_version===EXPLANATION_VERSION));
 const second=createExplanationService({cacheRoot:root,env,fetcher});
 const during=await second.decorate('owner',input,{generate:true});assert.deepEqual(during.plan.slots[0].explanation.results.map(r=>r.status),['success','in_progress']);assert.equal(calls.length,2);
 input.plan.slots[0].concepts.push('mutated-after-dispatch');release();
 const done=await pending;assert.equal(done.plan.slots[0].explanation.results.filter(r=>r.status==='success').length,2);
 assert.ok(service.records('owner').every(r=>!r.input.input_json.includes('mutated-after-dispatch')));
 const counts=service.usage('owner');await second.decorate('owner',done,{generate:false});assert.deepEqual(service.usage('owner'),counts);
 const restored=await second.explain('owner',plan,slot);assert.ok(restored.results.every(r=>r.cache_hit));
 assert.equal((await stat(join(root,'research.sqlite3'))).mode&0o777,0o600);
});

test('partial failure retains Gemini; explicit retry records a new NVIDIA attempt only; cache identity is per model/input/learner',async t=>{
 const root=await fixture(t);let fail=true;const calls=[];
 const service=createExplanationService({cacheRoot:root,env,fetcher:async(u,o)=>{const r=request(o);calls.push(r.provider);return fail&&r.provider==='nvidia'?new Response('secret-provider-error',{status:429}):responseFor(o);}});
 const first=await service.explain('owner',plan,slot,{generate:true});assert.deepEqual(first.results.map(r=>r.status),['success','failed']);assert.equal(first.fallback,null);
 await service.explain('owner',plan,slot,{generate:true});assert.equal(calls.length,2);
 fail=false;const retry=await service.explain('owner',plan,slot,{generate:true,retryFailed:true});assert.ok(retry.results.every(r=>r.status==='success'));assert.deepEqual(calls,['gemini','nvidia','nvidia']);
 const records=service.records('owner');assert.equal(records.length,3);assert.equal(records[2].attempt.retry_of,records[1].attempt.attempt_id);
 assert.equal(records[2].input.input_id,records[1].input.input_id);assert.equal(records[1].result.usage.input_tokens,null);
 const changed=createExplanationService({cacheRoot:root,env:{...env,GEMINI_EXPLANATION_MODEL:'new-gemini'},fetcher:async(u,o)=>{calls.push(request(o).provider);return responseFor(o);}});
 await changed.explain('owner',plan,slot,{generate:true});assert.equal(calls.length,4);assert.equal(calls.at(-1),'gemini');
 assert.ok((await service.explain('other',plan,slot)).results.every(r=>r.status==='not_generated'));
 assert.ok((await service.explain('owner',{...plan,snapshot_id:'new-snapshot'},slot)).results.every(r=>r.status==='not_generated'));
 assert.ok((await service.explain('owner',plan,{...slot,reason:'new evidence'})).results.every(r=>r.status==='not_generated'));
 assert.ok(!JSON.stringify(service.records('owner')).includes('secret-provider-error'));
});

test('both failures show Korean template, keep validation failures and usage, and omit upstream errors/credentials',async t=>{
 const root=await fixture(t);
 const service=createExplanationService({cacheRoot:root,env,fetcher:async(u,o)=>request(o).provider==='gemini'?responseFor(o,{override:{evidence_id:'invented'}}):Promise.reject(Error('secret-key /private/path'))});
 const result=await service.explain('owner',plan,{...slot,semantic_only:true,source_code:'private-code',path:'/private/path'},{generate:true});
 assert.equal(result.fallback.source,'template');assert.match(result.fallback.text,/제목/);assert.deepEqual(result.results.map(r=>r.status),['failed','failed']);
 const records=service.records('owner');assert.equal(records[0].result.usage.input_tokens,100);assert.equal(records[0].result.validation.valid,false);
 assert.ok(!JSON.stringify(records).includes('private-code'));assert.ok(!JSON.stringify(records).includes('/private/path'));assert.ok(!JSON.stringify(records).includes('secret-key'));
 assert.equal(records[0].result.evidence_ids[0],'invented');assert.ok(records[0].result.raw_response_text);
});

test('invalid JSON, truncated output, markup and missing usage remain auditable instead of erasing failures',async t=>{
 const root=await fixture(t);
 const variants=[async()=>Response.json({}),async(u,o)=>responseFor(o,{finish:'length'}),async(u,o)=>responseFor(o,{override:{text:'한국어 설명이지만 <script>금지된 내용</script>입니다.'}}),async(u,o)=>responseFor(o,{withUsage:false})];
 for(let i=0;i<variants.length;i++){
  const service=createExplanationService({cacheRoot:root,env,fetcher:variants[i]});const output=await service.explain(`owner-${i}`,plan,slot,{generate:true});
  assert.equal(service.records(`owner-${i}`).length,2);
  if(i<3)assert.ok(output.results.every(r=>r.status==='failed'));else{assert.ok(output.results.every(r=>r.status==='success'));assert.equal(service.usage(`owner-${i}`).groups[0].input_tokens,null);}
 }
});

test('aggregate usage/cost has coverage, counts retries once, preserves native reasoning, and supports append-only actual receipts',async t=>{
 const root=await fixture(t),prices={...env,EXPLANATION_PRICES_JSON:JSON.stringify({'gemini/gemini-3.5-flash-lite':{input_per_million:1,output_per_million:2,source:'test-rate-v1'}})};
 const service=createExplanationService({cacheRoot:root,env:prices,fetcher:successFetch});await service.explain('owner',plan,slot,{generate:true});
 const [g,n]=service.usage('owner').groups;assert.equal(g.requests,1);assert.equal(g.input_tokens,100);assert.equal(g.output_tokens,25);assert.equal(g.reasoning_tokens,5);
 assert.ok(Math.abs(g.estimated_cost_usd-0.00016)<1e-12);assert.equal(g.actual_cost_usd,null);assert.equal(n.estimated_cost_usd,null);
 assert.equal(n.input_usage_known,1);assert.equal(n.actual_cost_known,0);
 const receipt={receipt_id:'receipt-1',attempt_id:service.records('owner')[0].attempt.attempt_id,actual_usd:0,source:'user invoice 1'};
 assert.deepEqual(service.addCostReceipts('owner',[receipt]),{added:1});assert.deepEqual(service.addCostReceipts('owner',[receipt]),{added:0});
 assert.equal(service.usage('owner').groups[0].actual_cost_usd,0);assert.equal(service.usage('owner').groups[0].actual_cost_known,1);
 assert.throws(()=>service.addCostReceipts('other',[receipt]),/unknown/);
 assert.throws(()=>service.addCostReceipts('owner',[{...receipt,actual_usd:1}]),/conflicting/);
 const db=new DatabaseSync(join(root,'research.sqlite3'));
 try{assert.throws(()=>db.exec('DELETE FROM explanation_results'),/append-only/);assert.throws(()=>db.exec('UPDATE explanation_attempts SET model=\'changed\''),/append-only/);}finally{db.close();}
 const before=service.usage('owner');await service.explain('owner',plan,slot);assert.deepEqual(service.usage('owner'),before);
});

test('unresolved attempts remain unknown, cannot be retried during lease, and retain lineage after expiry',async t=>{
 const root=await fixture(t),store=createExplanationStore(join(root,'research.sqlite3'));
 const input={input_id:'input',learner_hash:'learner',evidence_hash:'hash',prompt_version:'v1'};
 const attempt={attempt_id:'attempt-1',input_id:'input',cache_key:'key',provider:'gemini',model:'model',started_ms:0,lease_until_ms:100};
 assert.equal(store.claim(input,attempt,{nowMs:0}).claimed,true);
 assert.equal(store.claim(input,{...attempt,attempt_id:'attempt-2'},{nowMs:50,retryFailed:true}).claimed,false);
 assert.equal(store.claim(input,{...attempt,attempt_id:'attempt-2'},{nowMs:200,retryFailed:true}).attempt.retry_of,'attempt-1');
 const report=aggregateExplanationUsage(store.records('learner'));assert.equal(report.groups[0].requests,2);assert.equal(report.groups[0].unresolved,2);assert.equal(report.groups[0].input_tokens,null);
});

test('redacted OTLP projection is offline, deterministic and contains no evidence, prose, learner or problem IDs',async t=>{
 const root=await fixture(t),service=createExplanationService({cacheRoot:root,env,fetcher:successFetch});await service.explain('private-owner',plan,slot,{generate:true});
 const records=service.records('private-owner'),payload=explanationOtlp(records),raw=JSON.stringify(payload);
 assert.deepEqual(payload,explanationOtlp(records));assert.equal(payload.resourceSpans[0].scopeSpans[0].spans.length,2);
 for(const secret of ['private-owner',slot.problem_key,slot.title,slot.reason,text,'secret-gemini'])assert.ok(!raw.includes(secret),secret);
});

test('journal wrapping leaves snapshot IDs, ranking and frozen storage unchanged; CLI reports use durable records without API calls',async t=>{
 const root=await fixture(t),db=createDatabase();t.after(()=>db.close());let calls=0;
 const explanations=createExplanationService({cacheRoot:join(root,'explanations'),env,fetcher:async(u,o)=>{calls++;return responseFor(o);}});
 const catalog=catalogFromData({version:'fixture',problems:Array.from({length:3},(_,i)=>({key:`leetcode:easy-${i}`,platform:'leetcode',title:`Easy ${i}`,url:`https://leetcode.com/problems/easy-${i}/`,difficulty_value:1,difficulty_kind:'lc_level',variation_group:`v${i}`,concepts:[{id:'arrays-hashing',status:'approved'}]})),concepts:[{id:'arrays-hashing',label:'Arrays'}]});
 const rawJournal=createJournalService({db,catalogProvider:()=>catalog}),journal=withExplanations(rawJournal,explanations),clock=new Date('2026-10-20T03:00:00Z');
 const raw=await rawJournal.view('owner',clock),before=db.sqlite.prepare('SELECT payload FROM recommendation_snapshots').all();
 const generated=await explanations.decorate('owner',raw,{generate:true}),next=await journal.view('owner',clock);
 assert.equal(calls,raw.plan.slots.length*2);assert.equal(next.plan.snapshot_id,raw.plan.snapshot_id);
 assert.equal(next.plan.slots[0].problem_key,raw.plan.slots[0].problem_key);assert.deepEqual(db.sqlite.prepare('SELECT payload FROM recommendation_snapshots').all(),before);
 assert.equal(next.days.flatMap(d=>d.entries).find(e=>e.snapshot_id===next.plan.snapshot_id).explanation.results.length,2);
 assert.equal(explanationEvidence(generated.plan,generated.plan.slots[0]).evidence_id,explanationEvidence(raw.plan,raw.plan.slots[0]).evidence_id);
 const output=[];const options={print:value=>output.push(JSON.parse(value)),env:{},fetcher:()=>{throw Error('must not call');}};
 assert.equal(await run(['explanation-usage','--db',join(root,'learner.sqlite3'),'--learner','owner'],options),0);assert.equal(output[0].groups.length,2);
 assert.equal(output[0].groups[0].requests,raw.plan.slots.length);
 assert.equal(await run(['explanation-records','--db',join(root,'learner.sqlite3'),'--learner','owner'],options),0);assert.equal(output[1].length,calls);
 const out=join(root,'redacted.json');assert.equal(await run(['explanation-otel','--db',join(root,'learner.sqlite3'),'--learner','owner','--out',out],{...options,print:()=>{}}),0);
 assert.ok(!(await readFile(out,'utf8')).includes(text));
});

test('missing NVIDIA configuration still retains the configured Gemini response without a synthetic attempt',async t=>{
 const root=await fixture(t);let calls=0;
 const service=createExplanationService({cacheRoot:root,env:{GEMINI_API_KEY:'fixture'},fetcher:async(u,o)=>{calls++;return responseFor(o);}});
 const result=await service.explain('owner',plan,slot,{generate:true});assert.equal(calls,1);
 assert.deepEqual(result.results.map(r=>r.status),['success','not_configured']);assert.equal(service.records('owner').length,1);assert.equal(result.fallback,null);
});

test('timeout is preserved as a failed independent attempt and does not discard the successful peer',async t=>{
 const root=await fixture(t);
 const service=createExplanationService({cacheRoot:root,env,timeoutMs:5,fetcher:async(u,o)=>{
  if(request(o).provider==='gemini')return responseFor(o);
  return new Promise((resolve,reject)=>{
   const guard=setTimeout(()=>reject(Error('test timeout guard')),200);
   o.signal.addEventListener('abort',()=>{clearTimeout(guard);reject(o.signal.reason);},{once:true});
  });
 }});
 const output=await service.explain('owner',plan,slot,{generate:true});assert.equal(output.results[0].status,'success');assert.equal(output.results[1].error.code,'timeout');
 assert.equal(service.records('owner').length,2);assert.equal(service.usage('owner').groups[1].failures,1);assert.equal(service.usage('owner').groups[1].input_tokens,null);
});

test('current default and cached HTTP diagnostics guide access fixes without rewriting prior attempts',async t=>{
 const {explanationErrorHint}=await import('../mastery/explanations.mjs');
 assert.equal(explanationProviders({})[0].model,'gemini-3.5-flash-lite');
 assert.match(explanationErrorHint('gemini',{http_status:404}),/모델/);
 assert.match(explanationErrorHint('nvidia',{http_status:403}),/정확한 원인을 확정/);
 const root=await fixture(t),service=createExplanationService({cacheRoot:root,env,fetcher:async(u,o)=>new Response('',{status:request(o).provider==='gemini'?404:403})});
 await service.explain('owner',plan,slot,{generate:true});const before=service.records('owner');
 const cached=await service.explain('owner',plan,slot);
 assert.ok(cached.results.every(r=>r.error.hint));assert.deepEqual(service.records('owner'),before);
});

test('HTTP error classification preserves safe provider categories without raw messages or credentials',async t=>{
 const {classifyProviderError}=await import('../mastery/explanations.mjs');
 assert.equal(classifyProviderError('403 Forbidden - Authorization failed'),'authorization_failed');
 assert.equal(classifyProviderError('{"error":{"status":"PERMISSION_DENIED"}}'),'permission_denied');
 assert.equal(classifyProviderError('Cloudflare request blocked'),'gateway_denied');
 const root=await fixture(t),service=createExplanationService({cacheRoot:root,env,fetcher:async()=>new Response('Authorization failed; secret-credential /private/log',{status:403})});
 await service.explain('owner',plan,slot,{generate:true});const records=service.records('owner');
 assert.ok(records.every(r=>r.result.error.provider_category==='authorization_failed'));
 assert.ok(!JSON.stringify(records).includes('secret-credential'));assert.ok(!JSON.stringify(records).includes('/private/log'));
});

test('Nemotron short prose disables thinking and persists its actual parameters without changing Gemini options or cache',async t=>{
 const {explanationGenerationParameters}=await import('../mastery/explanations.mjs');
 assert.deepEqual(explanationGenerationParameters({name:'gemini',model:'gemini-3.5-flash-lite'}),{temperature:0.2,max_output_tokens:512});
 const root=await fixture(t),configured={...env,NVIDIA_EXPLANATION_MODEL:'nvidia/nemotron-3.5-lightning-30b-a3b'};const calls=[];
 const fetcher=async(u,o)=>{const r=request(o);calls.push(r.provider);if(r.provider==='nvidia'){assert.deepEqual(r.body.chat_template_kwargs,{enable_thinking:false});assert.equal(r.body.max_tokens,512);}return responseFor(o);};
 const service=createExplanationService({cacheRoot:root,env:configured,fetcher});await service.explain('owner',plan,slot,{generate:true});
 assert.deepEqual(service.records('owner')[1].attempt.generation_parameters.chat_template_kwargs,{enable_thinking:false});
 await createExplanationService({cacheRoot:root,env:configured,fetcher}).explain('owner',plan,slot,{generate:true,retryFailed:true});
 assert.deepEqual(calls,['gemini','nvidia']);
});
