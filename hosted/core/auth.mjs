import {identifier} from './events.mjs';

const decode = part => {
  if (!/^[A-Za-z0-9_-]+$/.test(part)) throw new Error('invalid token encoding');
  return Uint8Array.from(atob(part.replace(/-/g,'+').replace(/_/g,'/')), c=>c.charCodeAt(0));
};
const jsonPart = part => JSON.parse(new TextDecoder().decode(decode(part)));
const hex = bytes => [...new Uint8Array(bytes)].map(b=>b.toString(16).padStart(2,'0')).join('');
const jwksCache = new Map();
async function accessKeys(issuer,fetcher,now,force=false) {
  const cached=jwksCache.get(issuer);
  if (cached && !force && now()-cached.fetchedAt<300_000) return cached.keys;
  // Unknown KIDs cannot force unbounded fetches from attacker supplied JWTs.
  if (cached && force && now()-cached.fetchedAt<15_000) return cached.keys;
  const controller=new AbortController();
  const timeout=setTimeout(()=>controller.abort(),3000);
  try {
    const response=await fetcher(issuer+'/cdn-cgi/access/certs',{signal:controller.signal});
    if (!response.ok) throw new Error('Access JWKS unavailable');
    const value=await response.json();
    if (!Array.isArray(value.keys)) throw new Error('invalid Access JWKS');
    jwksCache.set(issuer,{keys:value.keys,fetchedAt:now()});
    return value.keys;
  } finally { clearTimeout(timeout); }
}
export async function authenticateAccess(request, env, {fetcher=fetch,now=()=>Date.now()}={}) {
  const issuer = env?.ACCESS_ISSUER, audience = env?.ACCESS_AUDIENCE;
  if (typeof issuer !== 'string' || !/^https:\/\/[A-Za-z0-9-]+\.cloudflareaccess\.com$/.test(issuer) || !audience || !env?.ACCESS_EMAIL_ALLOWLIST) throw new Error('Access configuration missing');
  const token = request.headers.get('Cf-Access-Jwt-Assertion');
  if (!token) throw new Error('Access JWT required');
  const parts = token.split('.');
  if (parts.length !== 3) throw new Error('invalid Access JWT');
  const header = jsonPart(parts[0]), claims = jsonPart(parts[1]);
  if (header.alg !== 'RS256' || typeof header.kid !== 'string' || !header.kid) throw new Error('unsupported Access JWT header');
  const current = Math.floor(now()/1000);
  if (claims.iss !== issuer || !(claims.aud === audience || Array.isArray(claims.aud) && claims.aud.includes(audience)) || typeof claims.exp !== 'number' || !Number.isSafeInteger(claims.exp) || current >= claims.exp || typeof claims.nbf !== 'number' || !Number.isSafeInteger(claims.nbf) || current < claims.nbf) throw new Error('invalid Access JWT claims');
  identifier(claims.sub);
  if (typeof claims.email !== 'string' || !claims.email.trim()) throw new Error('Access email required');
  const allowed = env.ACCESS_EMAIL_ALLOWLIST.split(',').map(s=>s.trim().toLowerCase()).filter(Boolean);
  if (!allowed.includes(claims.email.toLowerCase())) throw new Error('Access email not allowed');
  let keys=await accessKeys(issuer,fetcher,now);
  let jwk=keys.find(k=>k.kid===header.kid && k.kty==='RSA' && (k.alg===undefined || k.alg==='RS256') && (k.use===undefined || k.use==='sig'));
  if (!jwk) { keys=await accessKeys(issuer,fetcher,now,true); jwk=keys.find(k=>k.kid===header.kid && k.kty==='RSA' && (k.alg===undefined || k.alg==='RS256') && (k.use===undefined || k.use==='sig')); }
  if (!jwk) throw new Error('Access signing key unavailable');
  const key = await crypto.subtle.importKey('jwk',jwk,{name:'RSASSA-PKCS1-v1_5',hash:'SHA-256'},false,['verify']);
  const signature = decode(parts[2]);
  const valid = await crypto.subtle.verify('RSASSA-PKCS1-v1_5',key,signature,new TextEncoder().encode(parts[0]+'.'+parts[1]));
  if (!valid) throw new Error('invalid Access JWT signature');
  const digest = await crypto.subtle.digest('SHA-256',new TextEncoder().encode(issuer+'\0'+claims.sub));
  return {learner_id:'access-'+hex(digest),email:claims.email,subject:claims.sub};
}
