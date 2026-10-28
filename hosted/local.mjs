/** Loopback-only development service. Never bundled into the production Worker. */
import {comparisonResponse} from './local-comparison.mjs';
import {libraryResponse} from './local-library.mjs';
import {createServer} from 'node:http';
import {readFile, mkdir, chmod} from 'node:fs/promises';
import {resolve, dirname, extname} from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createGitSync} from './mastery/git-sync.mjs';
import {cachedProviders,syncProvider} from './provider-service.mjs';
import {contestDiscoveryFit} from './mastery/contest-discovery-fit.mjs';
import {readLedger,highWaterMarks} from './mastery/ledger.mjs';
import {activeLocalCatalog,createJournalService} from './mastery/journal.mjs';
import {createExplanationService,withExplanations} from './mastery/explanations.mjs';
import {createApp} from './core/app.mjs';
import {createDatabase} from './runtime/database.mjs';

const ROOT=fileURLToPath(new URL('..',import.meta.url));
const PUBLIC = new URL('./public/',import.meta.url);
const TYPES = {'.html':'text/html; charset=utf-8','.css':'text/css; charset=utf-8','.mjs':'text/javascript; charset=utf-8'};
export async function startLocal({port=8787,database=resolve(ROOT,'private-learning/hosted.sqlite3'),learnerId='local-owner',
  curriculum,annotations={},catalogRoot=resolve(ROOT,'private-learning/mastery'),catalogProvider=null,gitSyncIntervalMs=5000,autoGitSync=true,explanationEnv=process.env,contestFetcher}={}) {
  const databasePath = database === ':memory:' ? database : resolve(database);
  if (databasePath !== ':memory:') await mkdir(dirname(databasePath),{recursive:true,mode:0o700});
  const db = createDatabase(databasePath);
  if (databasePath !== ':memory:') await chmod(databasePath,0o600);
  const assets = {async fetch(request) {
    const pathname = new URL(request.url).pathname;
    const filename = pathname === '/' ? 'index.html' : pathname.slice(1);
    if (!['index.html','app.mjs','workspace.mjs','workspace-data.mjs','styles.css'].includes(filename)) return new Response('Not found',{status:404});
    try { return new Response(await readFile(new URL(filename,PUBLIC)),{headers:{'content-type':TYPES[extname(filename)]}}); }
    catch { return new Response('Not found',{status:404}); }
  }};
  const gitSync=createGitSync({db,learnerId,intervalMs:gitSyncIntervalMs,catalogRoot,catalogProvider});
  const explanations=createExplanationService({cacheRoot:resolve(dirname(databasePath),'explanations'),env:explanationEnv});
  const journal=withExplanations(createJournalService({db,catalogRoot,catalogProvider}),explanations,db);
  const contests=async({learnerId,now})=>{
    const cache=await cachedProviders(db,learnerId,now),ledger=await readLedger(db,learnerId,{cutoff:now.toISOString()});
    let catalog;try{catalog=catalogProvider?await catalogProvider(ledger):activeLocalCatalog(db,ledger,catalogRoot);}catch{/* No catalog: explicit insufficient evidence. */}
    return {...cache,items:cache.items.map(c=>({...c,recommendation:contestDiscoveryFit(c,{ledger,catalog,now})}))};
  };
  const contestSync=async({learnerId,now})=>{
    const results=await Promise.allSettled(['codeforces','atcoder','leetcode'].map(platform=>syncProvider({db,learnerId,platform,discovery:true,env:{PROVIDERS_ENABLED:'true'},now,fetcher:contestFetcher})));
    return {results:results.map((r,i)=>({platform:['codeforces','atcoder','leetcode'][i],...(r.status==='fulfilled'?r.value:{status:'unavailable',error:'sync_in_progress_or_failed'})}))};
  };
  const app = createApp({db,journal,curriculum,annotations,assets,contests,contestSync,authenticate:async()=>({learner_id:learnerId})});
  const server = createServer(async (incoming,outgoing) => {
    try {
      const host = incoming.headers.host;
      if (!/^127\.0\.0\.1:\d+$/.test(host ?? '')) { outgoing.writeHead(403);outgoing.end();return; }
      const base = `http://${host}`, url = new URL(incoming.url,base);
      if (url.origin !== base) { outgoing.writeHead(403);outgoing.end();return; }
      if(url.pathname==='/api/local/library'||url.pathname.startsWith('/library/')){
        const library=await libraryResponse(new Request(url,{method:incoming.method,headers:incoming.headers}),{root:ROOT});
        outgoing.writeHead(library.status,Object.fromEntries(library.headers));outgoing.end(Buffer.from(await library.arrayBuffer()));return;
      }
      if(url.pathname==='/api/local/comparison'||url.pathname.startsWith('/comparison/')){
        const proxy=await comparisonResponse(new Request(url,{method:incoming.method,headers:incoming.headers}));
        outgoing.writeHead(proxy.status,Object.fromEntries(proxy.headers));outgoing.end(Buffer.from(await proxy.arrayBuffer()));return;
      }
      if(incoming.method==='GET' && url.pathname==='/api/local/sync'){
        const marks=await highWaterMarks(db,learnerId);
        const row=await db.prepare('SELECT revision FROM learners WHERE learner_id=?').bind(learnerId).first();
        outgoing.writeHead(200,{'content-type':'application/json','cache-control':'no-store'});
        outgoing.end(JSON.stringify({revision:JSON.stringify([row?.revision ?? 0,marks]),enabled:autoGitSync,status:gitSync.status()}));return;
      }
      const headers = new Headers();
      for (const [key,value] of Object.entries(incoming.headers)) if (value !== undefined) headers.set(key,Array.isArray(value)?value.join(','):value);
      if (!['GET','HEAD'].includes(incoming.method) && headers.get('origin') !== base) {
        outgoing.writeHead(403);outgoing.end('Same-origin required');return;
      }
      const response = await app.fetch(new Request(url,{method:incoming.method,headers,
        ...(!['GET','HEAD'].includes(incoming.method)?{body:incoming,duplex:'half'}:{})}));
      outgoing.writeHead(response.status,Object.fromEntries(response.headers));
      outgoing.end(Buffer.from(await response.arrayBuffer()));
    } catch { outgoing.writeHead(500);outgoing.end('Local service error'); }
  });
  await new Promise((resolve,reject)=>server.once('error',reject).listen(port,'127.0.0.1',resolve));
  if(autoGitSync)gitSync.start();
  return {url:`http://127.0.0.1:${server.address().port}`,db,server,gitSync,
    async close(){ await gitSync.close();await new Promise(resolve=>server.close(resolve));db.close(); }};
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const {curriculum,annotations} = await import('./generated/curriculum.mjs');
  const service = await startLocal({port:Number(process.env.PORT ?? 8787),database:process.env.LOCAL_DATABASE,learnerId:process.env.LOCAL_LEARNER_ID ?? 'local-owner',curriculum,annotations});
  console.log(`Private development service: ${service.url} (loopback only; not Access authentication)`);
  const stop = async()=>{await service.close();process.exit(0);};
  process.once('SIGINT',stop);process.once('SIGTERM',stop);
}
