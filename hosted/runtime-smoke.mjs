/** Cloudflare workerd+D1 local smoke. No real Cloudflare account or Slack calls. */
import {Miniflare,convertV4MiniflareOptions} from 'miniflare';
import assert from 'node:assert/strict';
import {readFile,readdir,mkdir,writeFile} from 'node:fs/promises';
const issuer='https://runtime-smoke.cloudflareaccess.com',audience='runtime-smoke';
const pair=await crypto.subtle.generateKey({name:'RSASSA-PKCS1-v1_5',modulusLength:2048,publicExponent:new Uint8Array([1,0,1]),hash:'SHA-256'},true,['sign','verify']);
const jwk={...await crypto.subtle.exportKey('jwk',pair.publicKey),kid:'runtime-key',alg:'RS256',use:'sig'};
const b64=object=>Buffer.from(JSON.stringify(object)).toString('base64url');
const now=Math.floor(Date.now()/1000);
const header=b64({alg:'RS256',kid:'runtime-key'}),claims=b64({iss:issuer,aud:[audience],sub:'test-owner',email:'smoke@example.test',nbf:now-30,exp:now+300});
const signature=await crypto.subtle.sign('RSASSA-PKCS1-v1_5',pair.privateKey,new TextEncoder().encode(header+'.'+claims));
const token=header+'.'+claims+'.'+Buffer.from(signature).toString('base64url');
const external=[];
const mf=new Miniflare(convertV4MiniflareOptions({name:'runtime-smoke',modules:true,scriptPath:new URL('./dist/worker.js',import.meta.url).pathname,compatibilityDate:'2026-10-16',
  d1Databases:{DB:'runtime-test'},bindings:{ACCESS_ISSUER:issuer,ACCESS_AUDIENCE:audience,ACCESS_EMAIL_ALLOWLIST:'smoke@example.test',DELIVERY_ENABLED:'false',INSTRUMENTATION_ENABLED:'true'},
  serviceBindings:{ASSETS:async()=>new Response('<!doctype html><title>asset fixture</title>',{headers:{'content-type':'text/html'}})},
  outboundService:async request=>{external.push(request.url);if(request.url===issuer+'/cdn-cgi/access/certs')return new Response(JSON.stringify({keys:[jwk]}));throw Error('Unexpected remote call: '+request.url);}}));
const start=performance.now();
try{
  const db=await mf.getD1Database('DB');
  for(const name of (await readdir(new URL('./migrations/',import.meta.url))).filter(n=>n.endsWith('.sql')).sort()){
    let statement='';
    for(const line of (await readFile(new URL('./migrations/'+name,import.meta.url),'utf8')).split('\n')){
      statement+=line+'\n';
      if(!statement.trim().endsWith(';'))continue;
      if(/^\s*CREATE TRIGGER/i.test(statement) && !/END;\s*$/i.test(statement))continue;
      await db.prepare(statement.trim()).run();statement='';
    }
    assert.equal(statement.trim(),'','unterminated migration');
  }
  assert.equal((await mf.dispatchFetch('https://app.example.test/')).status,401);
  const headers={'Cf-Access-Jwt-Assertion':token};
  assert.equal((await mf.dispatchFetch('https://app.example.test/',{headers})).status,200);
  const response=await mf.dispatchFetch('https://app.example.test/api/state',{headers});
  assert.equal(response.status,200);const state=await response.json();
  assert.match(state.learner_id,/^access-/);assert.equal(state.capabilities.delivery_enabled,false);
  const event={schema_version:1,event_id:'runtime-1',learner_id:state.learner_id,occurred_at:new Date().toISOString(),recorded_at:new Date().toISOString(),kind:'attempt',source:'manual',payload:{problem_key:state.curriculum.problems[0].problem_key,outcome:'assisted',hints:true},supersedes:null};
  const post=(path,body)=>mf.dispatchFetch('https://app.example.test'+path,{method:'POST',headers:{...headers,origin:'https://app.example.test','content-type':'application/json'},body:JSON.stringify(body)});
  assert.equal((await post('/api/events',event)).status,201);
  assert.equal((await (await post('/api/events',event)).json()).added,0);
  assert.equal((await post('/api/events',{...event,event_id:'bad',learner_id:'another-owner'})).status,400);
  const exported=await(await mf.dispatchFetch('https://app.example.test/api/export',{headers})).json();
  assert.equal(exported.events.length,1);
  const conflicting={...exported,events:[{...event,event_id:'new'}, {...event,payload:{...event.payload,outcome:'failed'}}]};
  assert.ok((await post('/api/import',conflicting)).status>=400);
  assert.equal((await(await mf.dispatchFetch('https://app.example.test/api/export',{headers})).json()).events.length,1);
  const backupResponse=await mf.dispatchFetch('https://app.example.test/api/backup/export',{headers});
  assert.equal(backupResponse.status,200);
  const backup=await backupResponse.json();assert.equal(backup.event_document.events.length,1);
  assert.equal((await post('/api/backup/import',backup)).status,400);
  const operations=await(await mf.dispatchFetch('https://app.example.test/api/operations',{headers})).json();
  assert.equal(operations.limitations.wall_time_is_cpu,false);
  assert.ok(external.every(url=>url===issuer+'/cdn-cgi/access/certs'));
  const receipt={at:new Date().toISOString(),runtime:'local Cloudflare workerd via pinned Wrangler/Miniflare',elapsed_wall_ms:performance.now()-start,
    checks:['D1 migrations','Access JWT+JWKS signature','authenticated assets','state API','owner-bound append','idempotency','atomic conflicting restore','full D1 learning backup','nonempty restore refusal','privacy-preserving operation counters','no inference/Slack calls'],
    limitations:'Local workerd/D1 emulation; not deployed Access, free-tier CPU measurement, delivery or phone alert proof.'};
  await mkdir('private-learning',{recursive:true,mode:0o700});await writeFile('private-learning/runtime-receipt.json',JSON.stringify(receipt,null,2),{mode:0o600});
  console.log(JSON.stringify(receipt,null,2));
}finally{await mf.dispose();}
