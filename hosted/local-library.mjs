/** Local-only, read-only research library. Files are served by catalog id, never by a requested path. Never import into the Worker/core bundle. */
import {readFile,realpath,stat} from 'node:fs/promises';
import {extname,isAbsolute,join,normalize,sep} from 'node:path';

const TYPES={'.png':'image/png','.svg':'image/svg+xml','.html':'text/html; charset=utf-8'};
const KINDS={'.png':'image','.svg':'image','.html':'page'};
const ROOTS=['research/','private-learning/'];
// Pages run scripts in an opaque sandbox: no app cookies, no same-origin API, no network.
const PAGE_CSP="sandbox allow-scripts; default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:; media-src data: blob:";
const SVG_CSP="sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data:";
const json=(value,status=200)=>new Response(JSON.stringify(value),{status,headers:{'content-type':'application/json; charset=utf-8','cache-control':'no-store'}});

export function validateLibrary(catalog){
  const ids=new Set();
  if(!Array.isArray(catalog?.folders))throw Error('library: folders required');
  for(const folder of catalog.folders){
    if(!/^[a-z0-9-]+$/.test(folder.id ?? '')||!Array.isArray(folder.items))throw Error('library: invalid folder '+folder.id);
    for(const item of folder.items){
      const path=item.path ?? '';
      if(!/^[a-z0-9-]+$/.test(item.id ?? '')||ids.has(item.id))throw Error('library: invalid or duplicate id '+item.id);
      if(isAbsolute(path)||normalize(path)!==path||path.split('/').includes('..')||!ROOTS.some(r=>path.startsWith(r))||!TYPES[extname(path)])throw Error('library: invalid path '+path);
      ids.add(item.id);
    }
  }
  return catalog;
}

export async function loadLibrary(root,file='research/library.v1.json'){
  return validateLibrary(JSON.parse(await readFile(join(root,file),'utf8')));
}

async function resolveFile(root,path){
  // Reject symlinks that leave the repository.
  try{const real=await realpath(join(root,path)),base=await realpath(root);if(!real.startsWith(base+sep))return null;const info=await stat(real);return info.isFile()?{real,size:info.size}:null;}
  catch{return null;}
}

export async function libraryResponse(request,{root,catalogFile='research/library.v1.json'}={}){
  const url=new URL(request.url),index=url.pathname==='/api/local/library',file=url.pathname.match(/^\/library\/file\/([a-z0-9-]+)$/);
  if(!index&&!file&&!url.pathname.startsWith('/library/'))return null;
  if(request.method!=='GET')return json({error:'read_only'},405);
  if(request.headers.get('origin')&&request.headers.get('origin')!==url.origin)return json({error:'same_origin_required'},403);
  let catalog;
  try{catalog=await loadLibrary(root,catalogFile);}catch{return json({available:false,reason:'library_unavailable'},503);}
  const items=catalog.folders.flatMap(f=>f.items);
  if(index){
    const folders=[];
    for(const folder of catalog.folders){
      const rows=[];
      for(const item of folder.items){
        const found=await resolveFile(root,item.path);
        rows.push({id:item.id,title:item.title,note:item.note ?? null,source:item.source ?? null,path:item.path,private:Boolean(item.private),kind:KINDS[extname(item.path)],
          format:extname(item.path).slice(1),available:Boolean(found),size:found?.size ?? null,url:found?`/library/file/${item.id}`:null});
      }
      folders.push({id:folder.id,title:folder.title,description:folder.description ?? null,items:rows});
    }
    return json({available:true,version:catalog.version,notice:catalog.notice ?? null,folders});
  }
  const item=file&&items.find(i=>i.id===file[1]);
  const found=item&&await resolveFile(root,item.path);
  if(!found)return json({error:'not_found'},404);
  const ext=extname(item.path);
  return new Response(await readFile(found.real),{headers:{'content-type':TYPES[ext],'cache-control':'no-store','x-content-type-options':'nosniff','cross-origin-resource-policy':'same-origin',
    ...(ext==='.html'?{'content-security-policy':PAGE_CSP}:ext==='.svg'?{'content-security-policy':SVG_CSP}:{})}});
}
