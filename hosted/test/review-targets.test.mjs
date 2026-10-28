import test from 'node:test';
import assert from 'node:assert/strict';
import {normalizePreferences} from '../core/preferences.mjs';
import {createDatabase} from './helpers/d1.mjs';
import {putPreferences,getPreferences} from '../core/store.mjs';
import {createJournalService} from '../mastery/journal.mjs';
import {catalogFromData} from '../mastery/catalog.mjs';
import {workspaceEvents,activeAttempts} from '../public/workspace-data.mjs';
import {exportLearningBackup,importLearningBackup} from '../backup.mjs';
const target={problem_key:'leetcode:buddy-strings',date:'2026-10-23'};

test('review selections validate exact keys, civil dates and uniqueness without changing old preferences',()=>{
 assert.equal(normalizePreferences({}).review_targets,undefined);
 assert.deepEqual(normalizePreferences({review_targets:[target]}).review_targets,[target]);
 for(const review_targets of [null,[{...target,date:'2026-02-30'}],[{...target,date:'2026-99-99'}],[target,target],[{...target,outcome:'independent'}],[{...target,problem_key:'Buddy Strings'}]])assert.throws(()=>normalizePreferences({review_targets}),/invalid|platform-qualified/);
});
test('manual selections persist, enrich from catalog and survive backup without creating an attempt',async t=>{
 const db=createDatabase(),restored=createDatabase();t.after(()=>{db.close();restored.close();});
 const catalog=catalogFromData({version:'review-fixture',problems:[{key:target.problem_key,platform:'leetcode',title:'Buddy Strings',url:'https://leetcode.com/problems/buddy-strings/',difficulty_kind:'lc_level',difficulty_value:1,concepts:[]}],concepts:[]});
 await putPreferences(db,'owner',{weekday_minutes:45,review_targets:[target]});
 const journal=await createJournalService({db,catalogProvider:()=>catalog}).view('owner',new Date('2026-10-23T01:00:00Z'));
 assert.equal(journal.review_selections[0].title,'Buddy Strings');
 const projected=workspaceEvents(journal,{preferences:await getPreferences(db,'owner')});
 assert.equal(projected.filter(e=>e.explicit_review).length,1);
 assert.equal(projected.find(e=>e.explicit_review).date,'2026-10-23');
 assert.equal(activeAttempts(projected).length,0);
 const backup=await exportLearningBackup(db,'owner');await importLearningBackup(restored,'owner',backup);
 assert.deepEqual((await getPreferences(restored,'owner')).review_targets,[target]);
 assert.equal((await getPreferences(restored,'owner')).weekday_minutes,45);
 assert.equal(restored.sqlite.prepare('SELECT COUNT(*) AS n FROM events').get().n,0);
});
