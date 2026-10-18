"""Real loopback HTTP tests over authored fixtures, never the live annotation store."""
import http.client,json,threading,unittest,re
from server import Server,MAX_BODY
import test_core as fixtures

class HTTPTests(unittest.TestCase):
    def setUp(self):
        fixtures.WorkbenchTest.setUp(self)
        self.server=Server(('127.0.0.1',0),self.w)
        thread=threading.Thread(target=self.server.serve_forever,daemon=True);thread.start()
        self.addCleanup(self.shutdown,thread)
    def shutdown(self,thread):
        self.server.shutdown();self.server.server_close();thread.join(timeout=3)
    def request(self,path,method='GET',body=None,headers=None):
        c=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=5)
        try:
            c.request(method,path,body=body,headers=headers or {});r=c.getresponse();return r.status,r.getheaders(),r.read()
        finally:c.close()
    def test_loopback_pilot_and_static_allowlist(self):
        status,headers,raw=self.request('/api/pilot');obj=json.loads(raw)
        self.assertEqual(status,200);self.assertEqual(len(obj['pilot']['items']),2)
        self.assertNotIn('methods',obj['pilot']['items'][0]);self.assertIn('write_token',obj)
        self.assertEqual(dict(headers)['Cache-Control'],'no-store')
        status,_,html=self.request('/');self.assertEqual(status,200)
        for asset in re.findall(r'(?:src|href)="(/[^"]+)"',html.decode()):
            self.assertEqual(self.request(asset)[0],200,asset)
        self.assertEqual(self.request('/../core.py')[0],404)
        self.assertEqual(self.request('/api/pilot',headers={'Host':'evil.test'})[0],403)
        self.assertEqual(self.request('/api/pilot',headers={'Origin':'https://evil.test'})[0],403)
        self.assertEqual(self.request('/api/pilot',headers={'Sec-Fetch-Site':'cross-site'})[0],403)
    def test_write_token_origin_and_body_validation(self):
        r=fixtures.relationship(self.pid,'fixture-human');body=json.dumps(r)
        headers={'Content-Type':'application/json','Origin':self.server.origin,'X-Workbench-Token':self.server.token}
        self.assertEqual(self.request('/api/records','POST',body,headers)[0],200)
        self.assertEqual(self.request('/api/records','POST',body,headers)[0],200)
        self.assertEqual(self.w.audit()['records'],1)
        for bad in ({**headers,'Origin':'https://evil.test'},{**headers,'X-Workbench-Token':'bad'}):
            self.assertEqual(self.request('/api/records','POST',body,bad)[0],403)
        self.assertEqual(self.request('/api/records','POST','{',headers)[0],400)
        self.assertEqual(self.request('/api/records','POST','[]',headers)[0],400)
        self.assertEqual(self.request('/api/records','POST','x'*(MAX_BODY+1),headers)[0],413)
        self.assertEqual(self.request('/api/records','POST',body,{**headers,'Content-Type':'text/plain'})[0],415)

if __name__=='__main__':unittest.main()
