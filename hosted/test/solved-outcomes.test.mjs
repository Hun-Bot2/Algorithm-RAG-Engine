import test from 'node:test';
import assert from 'node:assert/strict';
import {attemptStream} from '../mastery/ledger.mjs';
import {fitAll} from '../mastery/calibration.mjs';
import {catalogFromData} from '../mastery/catalog.mjs';

// ADR 005 / specs/mastery-system/solved-only-outcomes.md: help is not tracked; any completion is solved.
const at='2026-10-28T10:00:00.000000Z';
const event=(id,outcome,hints)=>({event_id:id,kind:'attempt',occurred_at:at,recorded_at:at,source:'manual',payload:{problem_key:`leetcode:${id}`,outcome,hints}});
const git=(id,outcome,hints)=>({entry_id:id,occurred_at:at,recorded_at:at,source:'git',payload:{problem_key:`leetcode:${id}`,outcome,hints,activity_type:null,transfer_from:null,self_minutes:5}});

test('attemptStream projects every completion to solved and keeps the stored outcome', () => {
  const stream=attemptStream({events:[event('a','independent',false),event('b','assisted',true),event('c','failed',true)],
    git_observations:[git('d','reported_completion',null),git('e','failed',null)]});
  assert.deepEqual(stream.map(a=>[a.id,a.outcome,a.hints,a.recorded_outcome,a.recorded_hints]),[
    ['a','independent',false,'independent',false],
    ['b','independent',false,'assisted',true],
    ['c','failed',null,'failed',true],
    ['d','independent',false,'reported_completion',null],
    ['e','failed',null,'failed',null]]);
});

test('daily-log completions (reported_completion) now count as solved calibration evidence', () => {
  const catalog=catalogFromData({problems:['d1','d2','d3'].map(slug=>({key:`leetcode:${slug}`,platform:'leetcode',title:slug,difficulty_kind:'lc_level',difficulty_value:1,tags:['Array']}))});
  const attempts=attemptStream({events:[],git_observations:['d1','d2','d3'].map(slug=>git(slug,'reported_completion',null))});
  const easy=fitAll(attempts,catalog,[]).leetcode.levels[1];
  assert.equal(easy.n,3,'all three completions are observations');
  assert.ok(easy.mean>0.8,'and all three are successes');
});
