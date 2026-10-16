"""Loopback-only private workbench server. No remote content fetching."""
from __future__ import annotations
import argparse,json,secrets
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit,parse_qs
from core import Workbench,prepare,restore,DEFAULT_QUEUE,DEFAULT_CORPUS,DEFAULT_CATALOG,ROOT

STATIC=Path(__file__).with_name('static')
MAX_BODY=65536
class Server(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address,workbench):
        if address[0]!='127.0.0.1':raise ValueError('loopback only')
        self.workbench=workbench;self.token=secrets.token_urlsafe(32)
        super().__init__(address,Handler)
        self.authority=f'127.0.0.1:{self.server_port}';self.origin='http://'+self.authority

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass # No private query/body or reviewer names in logs.
    def respond(self,status,body,ctype='application/json'):
        if ctype=='application/json':body=json.dumps(body,ensure_ascii=False).encode()
        self.send_response(status);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Referrer-Policy','no-referrer');self.send_header('X-Frame-Options','DENY')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        self.end_headers();self.wfile.write(body)
    def trusted(self):
        if self.headers.get('Host')!=self.server.authority:return False
        origin=self.headers.get('Origin')
        if origin is not None and origin!=self.server.origin:return False
        if self.headers.get('Sec-Fetch-Site') in {'cross-site','same-site'}:return False
        return True
    def do_GET(self):
        if not self.trusted():return self.respond(403,{'error':'local origin required'})
        url=urlsplit(self.path);query=parse_qs(url.query);wb=self.server.workbench
        try:
            if url.path=='/api/pilot':
                # Hide retrieval methods and strata before independent annotation.
                pilot={k:v for k,v in wb.pilot.items() if k not in {'sources','strata','items'}}
                pilot['items']=[{k:r[k] for k in ('pair_id','query','target')} for r in wb.pilot['items']]
                return self.respond(200,{'pilot':pilot,'progress':wb.progress(),'write_token':self.server.token})
            if url.path=='/api/pair':return self.respond(200,wb.pair(query.get('id',[''])[0],query.get('reviewer',[''])[0],query.get('mode',['review'])[0]))
            files={'/':'index.html','/index.html':'index.html','/app.js':'app.js','/style.css':'style.css'}
            if url.path not in files:return self.respond(404,{'error':'not found'})
            name=files[url.path];ctype={'html':'text/html; charset=utf-8','js':'text/javascript; charset=utf-8','css':'text/css; charset=utf-8'}[name.rsplit('.',1)[1]]
            return self.respond(200,(STATIC/name).read_bytes(),ctype)
        except KeyError:return self.respond(404,{'error':'pair not in pilot'})
        except (ValueError,TypeError):return self.respond(400,{'error':'invalid request'})
    def do_POST(self):
        if not self.trusted() or self.headers.get('Origin')!=self.server.origin or not secrets.compare_digest(self.headers.get('X-Workbench-Token',''),self.server.token):
            return self.respond(403,{'error':'local origin and write token required'})
        if self.path!='/api/records':return self.respond(404,{'error':'not found'})
        if self.headers.get('Transfer-Encoding') or self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.respond(415,{'error':'JSON required'})
        try:
            length=int(self.headers.get('Content-Length','-1'))
            if not 0<length<=MAX_BODY:return self.respond(413,{'error':'body size outside limit'})
            obj=json.loads(self.rfile.read(length).decode('utf-8'))
            return self.respond(200,self.server.workbench.append(obj))
        except (ValueError,TypeError,KeyError) as e:return self.respond(400,{'error':str(e)})

DEFAULT_DIR=ROOT/'private-learning/verification-workbench/pilot-v1'
def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--output',type=Path,default=DEFAULT_DIR/'pilot.json');a.add_argument('--size',type=int,default=40)
    for name in ('serve','export','restore'):
        a=sub.add_parser(name);a.add_argument('--pilot',type=Path,default=DEFAULT_DIR/'pilot.json');a.add_argument('--db',type=Path,default=DEFAULT_DIR/'reviews.sqlite3')
        if name=='serve':a.add_argument('--port',type=int,default=8765)
        elif name=='export':a.add_argument('--output',type=Path,required=True)
        else:a.add_argument('--input',type=Path,required=True)
    args=p.parse_args()
    if args.command=='prepare':
        result=prepare(DEFAULT_QUEUE,DEFAULT_CORPUS,DEFAULT_CATALOG,args.output,args.size)
        print(json.dumps({'pilot':str(args.output),'pairs':result['actual_pairs'],'strata':result['strata']},indent=2));return
    if args.command=='restore':
        restore(args.pilot,args.input,args.db);print('Private evidence restored and audit verified');return
    wb=Workbench(args.pilot,args.db)
    if args.command=='export':wb.export(args.output);print('Private append-only export created');return
    server=Server(('127.0.0.1',args.port),wb)
    print(f'Private local workbench: {server.origin} (Ctrl-C to stop)',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
if __name__=='__main__':main()
