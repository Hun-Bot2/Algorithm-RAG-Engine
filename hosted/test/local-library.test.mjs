import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,rm,symlink,writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {libraryResponse,loadLibrary,validateLibrary} from '../local-library.mjs';

const ROOT=new URL('../..',import.meta.url).pathname;
const request=(path,init={})=>new Request('http://127.0.0.1:8787'+path,init);
async function fixture(t,items){
 const root=await mkdtemp(join(tmpdir(),'library-'));t.after(()=>rm(root,{recursive:true,force:true}));
 await mkdir(join(root,'research/figs'),{recursive:true});await mkdir(join(root,'private-learning/maps'),{recursive:true});
 await writeFile(join(root,'research/figs/a.png'),'png-bytes');await writeFile(join(root,'private-learning/maps/m.html'),'<p>map</p>');
 await writeFile(join(root,'outside.svg'),'<svg/>');
 await writeFile(join(root,'research/library.v1.json'),JSON.stringify({version:'t',folders:[{id:'f',title:'F',items}]}));
 return root;
}

test('the committed catalog is valid and every listed file exists locally or is marked private',async()=>{
 const catalog=await loadLibrary(ROOT);
 const response=await libraryResponse(request('/api/local/library'),{root:ROOT}),body=await response.json();
 assert.equal(body.available,true);assert.equal(body.folders.length,catalog.folders.length);
 for(const item of body.folders.flatMap(f=>f.items))assert.ok(item.available||item.private,`${item.path} missing`);
});

test('files are served by catalog id with sandbox headers; paths, unknown ids and writes are refused',async t=>{
 const root=await fixture(t,[{id:'fig',title:'Fig',path:'research/figs/a.png'},{id:'map',title:'Map',path:'private-learning/maps/m.html',private:true},{id:'gone',title:'Gone',path:'research/figs/missing.svg'}]);
 const index=await (await libraryResponse(request('/api/local/library'),{root})).json();
 assert.deepEqual(index.folders[0].items.map(i=>[i.id,i.available,i.kind,i.url]),[['fig',true,'image','/library/file/fig'],['map',true,'page','/library/file/map'],['gone',false,'image',null]]);
 const image=await libraryResponse(request('/library/file/fig'),{root});assert.equal(image.headers.get('content-type'),'image/png');assert.equal(await image.text(),'png-bytes');
 const page=await libraryResponse(request('/library/file/map'),{root});assert.match(page.headers.get('content-security-policy'),/^sandbox allow-scripts;/);assert.match(page.headers.get('content-security-policy'),/default-src 'none'/);
 for(const path of ['/library/file/gone','/library/file/nope','/library/file/..%2Foutside.svg','/library/research/figs/a.png'])assert.equal((await libraryResponse(request(path),{root})).status,404,path);
 assert.equal((await libraryResponse(request('/library/file/fig',{method:'POST'}),{root})).status,405);
 assert.equal((await libraryResponse(request('/api/local/library',{headers:{origin:'https://evil.test'}}),{root})).status,403);
 assert.equal(await libraryResponse(request('/api/state'),{root}),null);
});

test('catalog paths must stay in research/ or private-learning/ and symlinks may not escape the root',async t=>{
 for(const path of ['/etc/passwd','../x.png','research/../outside.svg','src/a.png','research/a.txt','research/./a.png'])
  assert.throws(()=>validateLibrary({folders:[{id:'f',items:[{id:'x',path}]}]}),/invalid path/,path);
 assert.throws(()=>validateLibrary({folders:[{id:'f',items:[{id:'x',path:'research/a.png'},{id:'x',path:'research/b.png'}]}]}),/duplicate/);
 const outside=await mkdtemp(join(tmpdir(),'library-outside-'));t.after(()=>rm(outside,{recursive:true,force:true}));await writeFile(join(outside,'secret.png'),'secret');
 const root=await fixture(t,[{id:'link',title:'Link',path:'research/figs/link.png'}]);await symlink(join(outside,'secret.png'),join(root,'research/figs/link.png'));
 assert.equal((await libraryResponse(request('/library/file/link'),{root})).status,404);
});
