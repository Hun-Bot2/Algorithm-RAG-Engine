import {recordObservation} from './operations.mjs';
import {curriculum,annotations} from './generated/curriculum.mjs';
import {runScheduler} from './scheduler.mjs';
import {createApp} from './core/app.mjs';

export default {
  async fetch(request,env,ctx) {
    const started=performance.now();
    const response=await createApp({env,curriculum,annotations}).fetch(request);
    const secured=new Response(response.body,response);
    secured.headers.set('x-content-type-options','nosniff');
    secured.headers.set('referrer-policy','no-referrer');
    secured.headers.set('content-security-policy',"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'");
    if(env.INSTRUMENTATION_ENABLED==='true') {
      const observation=recordObservation(env.DB,{path:new URL(request.url).pathname,status:response.status,elapsedMs:performance.now()-started}).catch(()=>{});
      if(ctx?.waitUntil)ctx.waitUntil(observation);else await observation;
    }
    return secured;
  },
  scheduled(controller,env,ctx) {
    ctx.waitUntil(runScheduler({db:env.DB,curriculum,annotations,env,now:new Date(controller.scheduledTime)}));
  },
};
