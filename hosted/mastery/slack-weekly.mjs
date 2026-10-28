// Optional Slack adapter for the weekly report (MS-902). Builds a Block Kit payload only; delivery
// stays with the existing gated dispatcher and requires explicit approval. Pure.
const MAX_SECTION=3000, MAX_BLOCKS=50;
const escape=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const rate=r=>!r || r.n===0 ? 'no data' : r.status==='insufficient_evidence' ? `${r.successes}/${r.n} (insufficient)` : `${Math.round(r.rate*100)}% (${r.successes}/${r.n})`;
const section=text=>({type:'section',text:{type:'mrkdwn',text:text.length>MAX_SECTION?text.slice(0,MAX_SECTION-1)+'…':text}});
export function weeklySlackPayload(report) {
  const blocks=[{type:'header',text:{type:'plain_text',text:`Weekly report ${report.week_start} – ${report.week_end}`.slice(0,150)}}];
  blocks.push(section(`*Activity* ${report.activity.total} attempts · ${Object.entries(report.activity.by_platform).map(([k,v])=>`${escape(k)} ${v}`).join(', ') || 'none'}`));
  blocks.push(section(`*Completion rate (policy v3)* week ${rate(report.outcomes.isr.week.overall)} · 4 weeks ${rate(report.outcomes.isr.rolling.overall)}\n*Transfer* ${rate(report.outcomes.transfer)} · *On-time reviews* ${rate(report.review.on_time)}`));
  const t=report.outcomes.trajectory;
  blocks.push(section(`*Practice ability* CF ${t.codeforces.current.mean}±${t.codeforces.current.sd} (${t.codeforces.current.status}) · AtCoder ${t.atcoder.current.mean}±${t.atcoder.current.sd} (${t.atcoder.current.status})`));
  if (report.next_week_focus.length) blocks.push(section(`*Next week* ${report.next_week_focus.map(f=>`${escape(f.concept)} — ${escape(f.reason)}`).join('\n')}`));
  blocks.push({type:'context',elements:[{type:'mrkdwn',text:escape(report.disclaimer).slice(0,MAX_SECTION)}]});
  return {text:`Weekly report ${report.week_start}: ${report.activity.total} attempts`,blocks:blocks.slice(0,MAX_BLOCKS)};
}
