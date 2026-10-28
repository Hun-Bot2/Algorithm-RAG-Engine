/** Local-only, read-only bridge. Never import into the Worker/core bundle. */
const ROUTES=new Set(['/comparison/','/comparison/api/meta','/comparison/api/search','/comparison/api/problem','/comparison/api/compare','/comparison/api/map','/comparison/notebook.html','/comparison/notebook.ipynb','/comparison/blog.html','/comparison/report.html']);
const json=(value,status=200)=>new Response(JSON.stringify(value),{status,headers:{'content-type':'application/json; charset=utf-8','cache-control':'no-store'}});
export async function comparisonResponse(request,{upstream='http://127.0.0.1:8789',fetcher=fetch}={}){
  const url=new URL(request.url),health=url.pathname==='/api/local/comparison';
  if(!health&&!url.pathname.startsWith('/comparison/'))return null;
  if(request.method!=='GET')return json({error:'read_only'},405);
  if(request.headers.get('origin')&&request.headers.get('origin')!==url.origin)return json({error:'same_origin_required'},403);
  if(!health&&!ROUTES.has(url.pathname))return json({error:'not_found'},404);
  const target=new URL(upstream);
  if(target.protocol!=='http:'||target.hostname!=='127.0.0.1'||target.username||target.password||target.pathname!=='/'||target.search||target.hash)throw Error('comparison upstream must be a loopback origin');
  const path=health?'/api/meta':url.pathname.slice('/comparison'.length);
  const destination=new URL(path+url.search,target);
  if(health)destination.searchParams.set('scope','leetcode');
  try{
    const response=await fetcher(destination,{method:'GET',redirect:'error',signal:AbortSignal.timeout(2500)});
    if(health){
      if(!response.ok)return json({available:false,reason:'comparison_unavailable'},503);
      const meta=await response.json();
      if(!meta.scopes||!meta.methods)return json({available:false,reason:'comparison_unavailable'},503);
      return json({available:true,scopes:meta.scopes});
    }
    const body=await response.arrayBuffer();
    return new Response(body,{status:response.status,headers:{
      'content-type':response.headers.get('content-type')||'text/plain; charset=utf-8','cache-control':'no-store','x-content-type-options':'nosniff',
      'content-security-policy':"default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'self'",
      'cross-origin-resource-policy':'same-origin',
      ...(url.pathname==='/comparison/notebook.ipynb'?{'content-disposition':'attachment; filename="recommendation_experiments.ipynb"'}:{}),
    }});
  }catch{return json({available:false,reason:'comparison_unavailable'},503);}
}
