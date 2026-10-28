import test from 'node:test';
import assert from 'node:assert/strict';
import {catalogFromData} from '../mastery/catalog.mjs';
import {buildWeeklyReport,renderMarkdown} from '../mastery/weekly.mjs';
import {weeklySlackPayload} from '../mastery/slack-weekly.mjs';

test('weekly report labels policy v3 completion while keeping the isr schema',()=>{
  const ledger={attempts:[],events:[],participation:[],rating_observations:[],practice_feedback:[],git_watermarks:[],cutoff:null,high_water:{}};
  const report=buildWeeklyReport({ledger,catalog:catalogFromData({}),weekStart:'2026-10-19'});
  const markdown=renderMarkdown(report);
  assert.match(markdown,/Completion rate \(policy v3\) \(this week\): no data/);
  assert.match(markdown,/Completion rate \(policy v3\) \(4 weeks\): no data/);
  assert.doesNotMatch(markdown,/- Solve rate \(/);
  assert.ok(Object.hasOwn(report.outcomes,'isr'));
  assert.deepEqual(Object.keys(report.outcomes.isr).sort(),['rolling','week']);
  const payload=JSON.stringify(weeklySlackPayload(report));
  assert.match(payload,/Completion rate \(policy v3\)/);
  assert.doesNotMatch(payload,/Independent solve rate/);
  assert.ok(Object.hasOwn(report.outcomes,'isr'));
});
