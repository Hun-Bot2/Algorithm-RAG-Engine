import test from 'node:test';
import assert from 'node:assert/strict';
import {createDatabase} from './helpers/d1.mjs';
import {recordObservation,operationsSnapshot} from '../operations.mjs';
test('operational counters omit private paths, learner content and CPU claims',async()=>{
  const db=createDatabase();try{
    const now=new Date('2026-10-16T00:00:00Z');
    await recordObservation(db,{path:'/api/private-secret?email=secret',status:404,elapsedMs:2,now});
    await recordObservation(db,{path:'/api/private-secret?email=secret',status:404,elapsedMs:4,now});
    const snapshot=await operationsSnapshot(db,'me');
    assert.equal(snapshot.observations[0].route,'unknown_api');
    assert.equal(snapshot.observations[0].requests,2);assert.equal(snapshot.observations[0].elapsed_sum_ms,6);
    assert.equal(JSON.stringify(snapshot).includes('email=secret'),false);
    assert.equal(snapshot.limitations.d1_billing_usage,null);
    assert.equal(snapshot.limitations.wall_time_is_cpu,false);
  }finally{db.close();}
});
