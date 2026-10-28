// Deterministic beta posterior helpers; no runtime or third-party dependencies.
const coefficients=[676.5203681218851,-1259.1392167224028,771.32342877765313,-176.61502916214059,12.507343278686905,-0.13857109526572012,9.984369578019572e-6,1.5056327351493116e-7];
function logGamma(z){
  if(z<0.5)return Math.log(Math.PI)-Math.log(Math.sin(Math.PI*z))-logGamma(1-z);
  z-=1;let x=0.99999999999980993;
  for(let i=0;i<coefficients.length;i++)x+=coefficients[i]/(z+i+1);
  const t=z+7.5;return 0.5*Math.log(2*Math.PI)+(z+0.5)*Math.log(t)-t+Math.log(x);
}
function validate(a,b,x){
  if(!Number.isFinite(a)||a<=0||!Number.isFinite(b)||b<=0)throw new RangeError('positive finite beta parameters required');
  if(x!==undefined&&(!Number.isFinite(x)||x<0||x>1))throw new RangeError('probability must be in [0,1]');
}
function fraction(a,b,x){
  const tiny=1e-300,protect=v=>Math.abs(v)<tiny?tiny:v;
  let c=1,d=1/protect(1-(a+b)*x/(a+1)),h=d;
  for(let m=1;m<=300;m++){
    const twice=2*m;
    let aa=m*(b-m)*x/((a+twice-1)*(a+twice));
    d=1/protect(1+aa*d);c=protect(1+aa/c);h*=d*c;
    aa=-(a+m)*(a+b+m)*x/((a+twice)*(a+twice+1));
    d=1/protect(1+aa*d);c=protect(1+aa/c);const delta=d*c;h*=delta;
    if(Math.abs(delta-1)<3e-14)return h;
  }
  throw new Error('beta continued fraction did not converge');
}
export function betaCdf(x,a,b){
  validate(a,b,x);if(x===0||x===1)return x;
  const factor=Math.exp(logGamma(a+b)-logGamma(a)-logGamma(b)+a*Math.log(x)+b*Math.log1p(-x));
  const result=x<(a+1)/(a+b+2)?factor*fraction(a,b,x)/a:1-factor*fraction(b,a,1-x)/b;
  return Math.max(0,Math.min(1,result));
}
export function betaSurvival(x,a,b){validate(a,b,x);return betaCdf(1-x,b,a);}
export function betaQuantile(p,a,b){
  validate(a,b,p);if(p===0||p===1)return p;
  let lo=0,hi=1;
  for(let i=0;i<100;i++){const mid=(lo+hi)/2;if(betaCdf(mid,a,b)<p)lo=mid;else hi=mid;}
  return (lo+hi)/2;
}
export function betaVariance(a,b){validate(a,b);return a*b/((a+b)**2*(a+b+1));}
// Expected relative variance reduction from one unit-weight Bernoulli observation.
export function betaInformation(a,b){validate(a,b);return 1/(a+b+1);}
