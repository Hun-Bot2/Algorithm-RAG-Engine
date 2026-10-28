import authoredConfig from './progression-v2.config.json' with {type:'json'};

const keys=(value, expected, name)=>{
  if (!value || typeof value!=='object' || Array.isArray(value) ||
      Object.keys(value).length!==expected.length || expected.some(k=>!Object.hasOwn(value,k))) {
    throw new TypeError(`Invalid ${name}`);
  }
};
const number=(value, name, min, max, exclusiveMin=false, exclusiveMax=false)=>{
  if (typeof value!=='number' || !Number.isFinite(value) ||
      (exclusiveMin ? value<=min : value<min) || (exclusiveMax ? value>=max : value>max)) {
    throw new TypeError(`Invalid ${name}`);
  }
};
const days=(value,name)=>{
  if (!Number.isSafeInteger(value) || value<=0) throw new TypeError(`Invalid ${name}`);
};

// Validate and copy each field so freezing the result never changes caller-owned objects.
export function loadProgressionConfig(input=authoredConfig) {
  keys(input,['version','tau','credible_mass','readiness_probability','level_priors','kappa','lambda','weights',
    'follow_up_days','review_days','transfer_platforms','transfer_native_difficulty_floor','exclude_experimental_atcoder'],'config');
  if (typeof input.version!=='string' || !input.version.trim()) throw new TypeError('Invalid version');
  number(input.tau,'tau',0,1,true,true);
  number(input.credible_mass,'credible_mass',0,1,true,true);
  number(input.readiness_probability,'readiness_probability',0,1);
  number(input.kappa,'kappa',0,Infinity,true);
  number(input.lambda,'lambda',0,1);
  keys(input.level_priors,['1','2','3'],'level_priors');
  const priors={};
  for (const level of ['1','2','3']) {
    const pair=input.level_priors[level];
    if (!Array.isArray(pair) || pair.length!==2) throw new TypeError(`Invalid level_priors.${level}`);
    pair.forEach(x=>number(x,`level_priors.${level}`,0,Infinity,true));
    priors[level]=Object.freeze([...pair]);
  }
  const weightKeys=['challenge_fit','information','coverage','follow_up','continuity'];
  keys(input.weights,weightKeys,'weights');
  const weights={};
  for (const key of weightKeys) { number(input.weights[key],`weights.${key}`,0,Infinity); weights[key]=input.weights[key]; }
  days(input.follow_up_days,'follow_up_days');
  if (!Array.isArray(input.review_days) || input.review_days.length===0) throw new TypeError('Invalid review_days');
  input.review_days.forEach((value,i)=>{
    days(value,'review_days');
    if (i && value<=input.review_days[i-1]) throw new TypeError('Invalid review_days');
  });
  if (!Array.isArray(input.transfer_platforms) || !input.transfer_platforms.length ||
      input.transfer_platforms.some(x=>!['codeforces','atcoder'].includes(x)) ||
      new Set(input.transfer_platforms).size!==input.transfer_platforms.length) throw new TypeError('Invalid transfer_platforms');
  if (input.transfer_native_difficulty_floor!==null) number(input.transfer_native_difficulty_floor,'transfer_native_difficulty_floor',0,Infinity);
  if (typeof input.exclude_experimental_atcoder!=='boolean') throw new TypeError('Invalid exclude_experimental_atcoder');
  return Object.freeze({...input,level_priors:Object.freeze(priors),weights:Object.freeze(weights),
    review_days:Object.freeze([...input.review_days]),transfer_platforms:Object.freeze([...input.transfer_platforms])});
}

export const defaultConfig=loadProgressionConfig();
