// Dependency-free, OFFLINE OTLP/JSON projection. No prompts, prose or learner/problem IDs.
import {createHash} from 'node:crypto';
const digest=value=>createHash('sha256').update(value).digest('hex');
const value=v=>typeof v==='number'?{doubleValue:v}:{stringValue:String(v)};
export function explanationOtlp(records){
  const spans=[];
  for(const {attempt,result} of records){
    if(!result)continue;
    const attrs={'langfuse.observation.type':'generation','gen_ai.system':attempt.provider,'gen_ai.request.model':attempt.model,
      'langfuse.observation.model.name':attempt.model,'explanation.prompt_version':attempt.prompt_version,
      'explanation.status':result.status,'explanation.validation_valid':String(result.validation.valid)};
    for(const [key,v] of Object.entries({input:result.usage.input_tokens,output:result.usage.output_tokens,total:result.usage.total_tokens}))if(v!==null)attrs[`gen_ai.usage.${key}_tokens`]=v;
    if(result.cost.estimated!==null)attrs['explanation.estimated_cost_usd']=result.cost.estimated;
    if(result.cost.actual!==null)attrs['explanation.actual_cost_usd']=result.cost.actual;
    if(result.error)attrs['error.type']=result.error.code;
    spans.push({traceId:digest(attempt.input_id).slice(0,32),spanId:digest(attempt.attempt_id).slice(0,16),name:'recommendation-explanation',kind:3,
      startTimeUnixNano:(BigInt(attempt.started_ms)*1000000n).toString(),
      endTimeUnixNano:(BigInt(Date.parse(result.completed_at))*1000000n).toString(),
      attributes:Object.entries(attrs).map(([key,v])=>({key,value:value(v)})),status:{code:result.status==='success'?1:2}});
  }
  return {resourceSpans:[{resource:{attributes:[{key:'service.name',value:{stringValue:'algorithm-rag-explanations'}}]},scopeSpans:[{scope:{name:'local-explanation-ledger',version:'1'},spans}]}]};
}
