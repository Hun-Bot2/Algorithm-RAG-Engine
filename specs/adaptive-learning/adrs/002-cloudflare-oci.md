# ADR 002: Cloudflare core with optional small OCI batch VM

Status: accepted deployment direction per user, 2026-10-16. Amends ADR 001's
infrastructure scope; no cloud service has been deployed by this decision.

## Research and observed context

User reports an OCI Ubuntu 24.04.5 x86_64 instance: two visible CPUs, AMD EPYC
7551, 954 MiB physical memory, 565 MiB available, no swap, and 43 GiB available
root filesystem. Snapshot uptime is three minutes. These are user-supplied
observations, not our remote inspection or steady-state performance measurements.
Visible CPU count does not establish two dedicated OCPUs, shape identity, free
entitlement or throughput. A second VM can reportedly be provisioned; no work is
assigned to it yet. OCI's idle-reclamation and allocation limits remain relevant
because a small personal batch workload may be inactive for long periods.

## Requirements

- Cloudflare SHALL own private dashboard/API, Access authentication, D1 evidence,
  mission freezing, review scheduling and Slack outbox/reconciliation.
- OCI SHALL be an optional, replaceable offline worker; VM loss MUST NOT block
  morning missions or notifications, or remove authoritative learner history.
- The initial OCI profile SHALL use only Python's standard library for reviewed
  asset checks and metadata/TF-IDF diagnostic evaluation. No inference server,
  model downloads, fitting, public HTTP port or cloud secrets are needed.
- Default service limits SHALL be 256 MiB maximum memory, no service swap,
  25% of one CPU, 180-second total service timeout, one process tree at a time,
  and 14 retained completed/failed runs. These are configured caps, not remotely
  verified cgroup enforcement or a guarantee against whole-VM memory pressure.
- The offline service SHALL run without IP networking. It MUST NOT depend on or
  write the production learner database, Slack or Cloudflare API.
- Deploy/enable actions MUST wait for the user's target/setup selection. No SSH
  target, Cloudflare identity or destination is inferred from machine specs.

## Design and task breakdown

1. Add `scripts/learning/oci_batch.py` with a nonblocking single-run lock,
   immutable run directory, source verification, bounded benchmark subprocess,
   receipt and narrowly scoped retention.
2. Add `deploy/oci/algorithm-batch.service` and `.timer`; weekly Sunday 03:00
   Asia/Seoul, up to 10 minutes jitter, no catch-up run after VM downtime.
3. Add an explicit allowlist tar packager. Include only this stdlib worker,
   reviewed curriculum/evaluation material and unit files; exclude historical
   datasets/indexes, secrets, model weights, hosted dependencies and learner state.
4. Test real runner execution, failure retention, concurrency, immutable inputs,
   deterministic package membership and extraction/run from that package.
5. Document installation, operator verification and disabled-by-default schedule.

## Validation and risks

Local Python tests and a packaged execution can validate the runner and allowlist.
The Mac workspace cannot validate Ubuntu's actual systemd/cgroup state. Run
`systemd-analyze verify`, one manual service start and inspect cgroup peaks on the
actual VM before enabling its timer. CPU/memory quota meanings and timer timezone
syntax were checked against Ubuntu Noble systemd manuals:
- https://manpages.ubuntu.com/manpages/noble/man5/systemd.resource-control.5.html
- https://manpages.ubuntu.com/manpages/noble/man5/systemd.exec.5.html
- https://manpages.ubuntu.com/manpages/noble/man7/systemd.time.7.html
- https://manpages.ubuntu.com/manpages/noble/man5/systemd.timer.5.html

No Docker daemon or transformer runtime is necessary for this profile. The
available-memory snapshot does not justify a local LLM or encoder training on the
1 GiB VM. A separate model experiment needs measured feasibility, labels and
approval. Do not manufacture CPU load to avoid provider idle-reclamation policy.

## Consequences and rollback

This choice adds a low-memory Linux execution option without moving the core
service or introducing a second authoritative database. OCI reports are research
artifacts; they are not automatically deployed candidates or approved edges.
Disable the timer to roll back OCI work; Cloudflare keeps operating independently.
Real deployment, quotas, unattended delivery and the 30-day pilot stay open.
