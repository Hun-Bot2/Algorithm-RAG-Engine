// Synthetic child process for actual restart regression; never receives real API keys.
import {dirname,join} from 'node:path';
import {startLocal} from '../../local.mjs';
import {createJournalService} from '../../mastery/journal.mjs';
import {createExplanationService} from '../../mastery/explanations.mjs';
import {catalogFromData} from '../../mastery/catalog.mjs';

const [database,mode]=process.argv.slice(2);
let externalCalls=0,mockCalls=0;
globalThis.fetch=async()=>{externalCalls++;throw new Error('unexpected external fetch in restart fixture');};
const env={GEMINI_API_KEY:'synthetic',NVIDIA_API_KEY:'synthetic',NVIDIA_EXPLANATION_MODEL:'fixture/model'};
const catalog=catalogFromData({version:'restart-fixture',problems:Array.from({length:3},(_,i)=>({key:`leetcode:easy-${i}`,platform:'leetcode',title:`Easy ${i}`,url:`https://leetcode.com/problems/easy-${i}/`,difficulty_value:1,difficulty_kind:'lc_level',variation_group:`v${i}`,concepts:[{id:'arrays-hashing',status:'approved'}]})),concepts:[{id:'arrays-hashing',label:'Arrays'}]});
const service=await startLocal({port:0,database,learnerId:'restart-fixture',curriculum:{schema_version:1,curriculum_version:'restart-fixture',problems:[],relationships:[]},catalogProvider:()=>catalog,autoGitSync:false,explanationEnv:env});
const explanations=createExplanationService({cacheRoot:join(dirname(database),'explanations'),env,fetcher:async(url,options)=>{
 mockCalls++;const body=JSON.parse(options.body),gemini=Boolean(body.contents);
 if(mode==='seed-partial'&&!gemini)return new Response('',{status:403});
 const input=JSON.parse(gemini?body.contents[0].parts[0].text:body.messages[1].content);
 const text=JSON.stringify({text:`${gemini?'Gemini':'NVIDIA'} 합성 응답이에요. 서버를 다시 시작해도 같은 답변을 표시해야 해요.`,evidence_id:input.evidence_id});
 return Response.json(gemini?{candidates:[{content:{parts:[{text}]}}],usageMetadata:{promptTokenCount:10,candidatesTokenCount:20,thoughtsTokenCount:0,totalTokenCount:30}}:{choices:[{message:{content:text}}],usage:{prompt_tokens:10,completion_tokens:20,total_tokens:30}});
}});
if(mode.startsWith('seed-')){
 const journal=createJournalService({db:service.db,catalogProvider:()=>catalog});
 await explanations.decorate('restart-fixture',await journal.view('restart-fixture'),{generate:true});
}
process.on('message',async message=>{
 if(message.type==='status')process.send({type:'status',externalCalls,mockCalls,usage:explanations.usage('restart-fixture'),records:explanations.records('restart-fixture')});
 if(message.type==='close'){await service.close();process.exit(0);}
});
process.send({type:'ready',url:service.url,pid:process.pid});
