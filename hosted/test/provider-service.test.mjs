import test from 'node:test';
import assert from 'node:assert/strict';
import {createDatabase} from './helpers/d1.mjs';
import {cachedProviders,syncProvider} from '../provider-service.mjs';
import {listEvents} from '../core/store.mjs';
const now=new Date('2026-10-16T00:00:00Z');
test('provider cache stays owner scoped and disabled refresh never fetches',async()=>{
  const db=createDatabase();try{
    assert.equal((await cachedProviders(db,'me',now)).sources[0].status,'unavailable');
    await assert.rejects(syncProvider({db,learnerId:'me',platform:'atcoder',now,fetcher:()=>{throw Error('network');}}),/disabled/);
    await syncProvider({db,learnerId:'me',platform:'atcoder',env:{PROVIDERS_ENABLED:'true'},now,
      fetcher:async()=>new Response(JSON.stringify([{id:'abc1',title:'ABC1',start_epoch_second:now.getTime()/1000+86400,duration_second:6000}]))});
    assert.equal((await cachedProviders(db,'me',now)).items.length,1);
    assert.equal((await cachedProviders(db,'other',now)).items.length,0);
    assert.equal((await cachedProviders(db,'me',new Date('2026-10-18'))).sources[1].status,'stale');
  }finally{db.close();}
});
test('imported evidence is idempotent and handle change resets watermark',async()=>{
  const db=createDatabase();try{
    const options={db,learnerId:'me',platform:'atcoder',handle:'user1',env:{PROVIDERS_ENABLED:'true'},now,
      fetcher:async()=>new Response(JSON.stringify([{id:17,result:'AC',epoch_second:1,problem_id:'abc001_a'}]))};
    assert.equal((await syncProvider(options)).imported,1);
    assert.equal((await syncProvider(options)).imported,0);
    assert.equal((await listEvents(db,'me'))[0].payload.outcome,'reported_completion');
    const changed=await syncProvider({...options,handle:'user2',fetcher:async url=>{
      assert.match(url,/from_second=0/);return new Response('[]');}});
    assert.equal(changed.imported,0);
  }finally{db.close();}
});
