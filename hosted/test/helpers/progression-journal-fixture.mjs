import {catalogFromData} from '../../mastery/catalog.mjs';
import {normalizePreferences} from '../../core/preferences.mjs';
export function journalInput(){
 const problems=[];for(const l of [1,2])for(let i=0;i<5;i++)problems.push({key:`leetcode:p${l}-${i}`,platform:'leetcode',difficulty_kind:'lc_level',difficulty_value:l,title:`Fixture ${l}-${i}`,tags:[`v${i}`],concepts:[{id:'arrays-hashing',status:'approved'}]});
 return {date:'2026-10-29',now:'2026-10-29T00:00:00Z',ledger:{attempts:[],events:[],rating_observations:[],cutoff:'2026-10-29T00:00:00Z'},catalog:catalogFromData({version:'v1-preservation-fixture',problems,concepts:[{id:'arrays-hashing',label:'Arrays'}]}),preferences:normalizePreferences({})};
}
