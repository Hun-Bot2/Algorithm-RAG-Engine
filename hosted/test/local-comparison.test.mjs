import test from 'node:test';
import assert from 'node:assert/strict';
import {comparisonResponse} from '../local-comparison.mjs';
const request=(path,options={})=>new Request('http://127.0.0.1:8787'+path,options);
test('local comparison proxies only allowed viewer routes, preserving scope and same-origin frame policy',async()=>{
 let called;
 const result=await comparisonResponse(request('/comparison/api/compare?scope=leetcode&key=leetcode%3Abuddy-strings'),{fetcher:async(url,options)=>{called={url,options};return Response.json({ok:true});}});
 assert.equal(called.url.origin,'http://127.0.0.1:8789');assert.equal(called.url.pathname,'/api/compare');assert.equal(called.url.searchParams.get('scope'),'leetcode');assert.equal(called.options.redirect,'error');
 assert.match(result.headers.get('content-security-policy'),/frame-ancestors 'self'/);assert.deepEqual(await result.json(),{ok:true});
});
test('comparison health exposes availability and scope names without forwarding private content',async()=>{
 const result=await comparisonResponse(request('/api/local/comparison'),{fetcher:async url=>{assert.equal(url.searchParams.get('scope'),'leetcode');return Response.json({methods:{title:{}},scopes:{leetcode:'LeetCode 내부'},private:'not forwarded'});}});
 assert.deepEqual(await result.json(),{available:true,scopes:{leetcode:'LeetCode 내부'}});
});
test('read-only route, origin and path restrictions prevent arbitrary forwarding',async()=>{
 const options={fetcher:async()=>{throw Error('must not fetch');}};
 assert.equal((await comparisonResponse(request('/comparison/api/meta',{method:'POST'}),options)).status,405);
 assert.equal((await comparisonResponse(request('/comparison/api/meta',{headers:{origin:'https://untrusted.test'}}),options)).status,403);
 assert.equal((await comparisonResponse(request('/comparison/private-learning/database'),options)).status,404);
 assert.equal(await comparisonResponse(request('/api/other'),options),null);
 await assert.rejects(comparisonResponse(request('/comparison/'),{upstream:'https://example.test',...options}),/loopback/);
});
test('offline or invalid comparison service produces an unavailable state',async()=>{
 for(const fetcher of [async()=>{throw Error('offline');},async()=>new Response('missing',{status:404}),async()=>Response.json({unexpected:true})]){
  const result=await comparisonResponse(request('/api/local/comparison'),{fetcher});assert.equal(result.status,503);assert.equal((await result.json()).available,false);
 }
});
test('notebook and article exports are allowlisted, with inline figures and notebook download headers',async()=>{
 for(const path of ['/comparison/notebook.html','/comparison/blog.html','/comparison/report.html','/comparison/notebook.ipynb']){
  const result=await comparisonResponse(request(path),{fetcher:async url=>{assert.ok(['/notebook.html','/blog.html','/report.html','/notebook.ipynb'].includes(url.pathname));return new Response('artifact',{headers:{'content-type':path.endsWith('ipynb')?'application/x-ipynb+json':'text/html'}});}});
  assert.equal(result.status,200);assert.match(result.headers.get('content-security-policy'),/img-src 'self' data:/);
  if(path.endsWith('ipynb'))assert.match(result.headers.get('content-disposition'),/attachment.*recommendation_experiments.ipynb/);
 }
 const denied=await comparisonResponse(request('/comparison/notebook-last-run.json'),{fetcher:async()=>{throw Error('must not fetch');}});assert.equal(denied.status,404);
});
