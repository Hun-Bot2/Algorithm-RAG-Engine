"""Read-only code/deployment preflight. Configuration presence never proves a live release."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.learning.curriculum import validate_curriculum
from scripts.learning.assets import verify


def check():
    verify()
    seed=json.loads((ROOT/'curriculum/seed.v1.json').read_text())
    validate_curriculum(seed,require_release=True)
    registry=json.loads((ROOT/'assets/curriculum_release.v1.json').read_text())
    for relative,digest in registry['files'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=digest:
            raise ValueError('curated source differs from frozen release registry')
    module=(ROOT/'hosted/generated/curriculum.mjs').read_text()
    line=next(line for line in module.splitlines() if line.startswith('export const manifest = '))
    generated=json.loads(line[len('export const manifest = '):].removesuffix(';'))
    if generated.get('draft') is not False or generated.get('seed_sha256')!=registry['files']['curriculum/seed.v1.json'] or generated.get('annotations_sha256')!=registry['files']['curriculum/annotations.v1.json']:
        raise ValueError('generated hosted curriculum is stale or a draft')
    config=json.loads((ROOT/'hosted/wrangler.jsonc').read_text())
    if config.get('assets',{}).get('run_worker_first') is not True:
        raise ValueError('all static assets must run through Worker authentication')
    configured_ids=[d.get('database_id') for d in config.get('d1_databases',[])]
    bindings=['ACCESS_ISSUER','ACCESS_AUDIENCE','ACCESS_EMAIL_ALLOWLIST']
    auth={name:bool(os.environ.get(name)) for name in bindings}
    delivery={name:bool(os.environ.get(name)) for name in ['SLACK_BOT_TOKEN','SLACK_CHANNEL_ID','DELIVERY_LEARNER_ID']}
    gates=[
      {'gate':'deployment_account_and_D1','passed':bool(configured_ids) and all(v and 'REPLACE' not in v for v in configured_ids)},
      {'gate':'Access_configuration_presence','passed':all(auth.values()),'fields_present':auth},
      {'gate':'Slack_configuration_presence','passed':all(delivery.values()),'fields_present':delivery},
      {'gate':'deployed_authentication_and_quota_measurement','passed':False,'reason':'Requires deployed evidence, not environment presence.'},
      {'gate':'authorized_Slack_smoke_and_phone_alert_measurement','passed':False,'reason':'Live delivery has not been authorized/executed.'},
      {'gate':'30_day_delivery_trial','passed':False,'reason':'Prospective trial required.'},
      {'gate':'optional_encoder_comparison','passed':False,'reason':'Optional local model artifacts not installed; deterministic baselines remain active.'},
      {'gate':'training','passed':False,'reason':'Requires separate reviewed training snapshot, frozen configuration and owned-compute approval.'},
    ]
    return {'code_preflight':'passed','curriculum_problems':len(seed['problems']),
      'schedule_configured':bool(config.get('triggers',{}).get('crons')),
      'scheduler_enabled_in_environment':os.environ.get('SCHEDULER_ENABLED')=='true',
      'delivery_enabled_in_environment':os.environ.get('DELIVERY_ENABLED')=='true',
      'gates':gates,'pending_decisions':['Cloudflare target account/D1 and allowed login email','Slack workspace and private destination channel','Authorization for deployment and one smoke send'],
      'boundary':'No account lookup, secrets printed, deployment, schedule change, send or training performed.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        print(json.dumps(check(),indent=2))
    except (ValueError,KeyError,OSError) as exc:
        sys.exit(str(exc))
