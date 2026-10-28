import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {canonicalJson} from '../contest-history.mjs';
import {defaultConfig} from '../mastery/progression-v2-config.mjs';
import {sha256,canonicalConfigHash} from '../mastery/progression-hash.mjs';
const reference=s=>createHash('sha256').update(s,'utf8').digest('hex');

test('SHA-256 matches Node for UTF-8 and block boundaries',()=>{
  for(const text of ['', 'abc', '알고리즘 학습', 'a'.repeat(55), 'a'.repeat(56), 'a'.repeat(64), '가나다'.repeat(1000)])
    assert.equal(sha256(text),reference(text));
});
test('config hash uses the existing recursive canonical JSON',()=>{
  for(const value of [defaultConfig,{z:[{b:1,a:2}],a:{한글:'값',x:3}}, {b:2,a:1}])
    assert.equal(canonicalConfigHash(value),reference(canonicalJson(value)));
  assert.equal(canonicalConfigHash({b:2,a:1}),canonicalConfigHash({a:1,b:2}));
});
