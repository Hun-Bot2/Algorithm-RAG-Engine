// progression-v2: pure derived estimates. v1 modules remain unchanged.
import {defaultConfig} from './progression-v2-config.mjs';
import {betaQuantile} from './beta-math.mjs';
const logit=p=>Math.log(p/(1-p)),sigmoid=x=>1/(1+Math.exp(-x));
export function leetCodeCells(observations,concept,config=defaultConfig){
  const ability=observations.filter(o=>o.kind==='ability'&&[1,2,3].includes(o.level));
  const cells=[];let delta=0;
  for(const level of [1,2,3]){
    const outside=ability.filter(o=>o.level===level&&!o.concepts.includes(concept));
    const own=ability.filter(o=>o.level===level&&o.concepts.includes(concept));
    const [a,b]=config.level_priors[level];
    const mean=(a+outside.reduce((s,o)=>s+o.y,0))/(a+b+outside.length);
    const centre=sigmoid(logit(mean)+config.lambda*delta);
    const alpha=config.kappa*centre+own.reduce((s,o)=>s+o.y/o.concepts.length,0);
    const beta=config.kappa*(1-centre)+own.reduce((s,o)=>s+(1-o.y)/o.concepts.length,0);
    delta=own.length?logit(alpha/(alpha+beta))-logit(centre):0;
    cells.push({level,alpha,beta,n_own:own.length,n_level_outside:outside.length,prior:outside.length?'learner':'authored'});
  }
  const blocks=cells.map(c=>({alpha:c.alpha,beta:c.beta,members:[c.level]}));
  for(let i=0;i<blocks.length-1;){
    const a=blocks[i],b=blocks[i+1];
    if(a.alpha/(a.alpha+a.beta)<b.alpha/(b.alpha+b.beta)){
      blocks.splice(i,2,{alpha:a.alpha+b.alpha,beta:a.beta+b.beta,members:[...a.members,...b.members]});i=Math.max(0,i-1);
    }else i++;
  }
  const tail=(1-config.credible_mass)/2;
  return cells.map(cell=>{
    const block=blocks.find(b=>b.members.includes(cell.level));
    return {...cell,alpha:block.alpha,beta:block.beta,p:block.alpha/(block.alpha+block.beta),
      interval:[betaQuantile(tail,block.alpha,block.beta),betaQuantile(1-tail,block.alpha,block.beta)]};
  });
}
