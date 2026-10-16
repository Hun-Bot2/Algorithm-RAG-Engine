# Cloudflare + OCI small-VM operations

## Roles and resource boundary

Cloudflare remains the authoritative private application: Access, D1, timer/event
history, daily missions, review policy, Slack outbox and five-minute dispatcher.
OCI runs optional offline asset/evaluation diagnostics. It does not host learner
history, authenticate users, deliver Slack, or supply a morning-deadline dependency.
No model server, training, cloud credentials, open web port, Docker or pip install
is needed for the initial worker. Reports never auto-promote a provider or edge.

The user reports Ubuntu 24.04.5, x86_64, 954 MiB RAM with 565 MiB then available,
no swap and 43 GiB disk available. It is a just-booted snapshot, not steady-state
capacity evidence. Two visible CPUs do not establish dedicated cores or Oracle
shape/entitlement. A second VM remains unallocated to this application; add work
only after measuring a need. The architecture decision is in ADR 002.

## Build and inspect the private bundle locally

```bash
mkdir -p private-learning
python3 scripts/learning/package_oci.py --output private-learning/oci-stdlib-v1.tar.gz
```

The builder refuses overwrite and uses an explicit file allowlist. The bundle
contains only standard-library worker code, independently authored curriculum,
evaluation-only judgments, manifests and units. It excludes .env, private history,
keys, OpenAI/Slack clients, raw LeetCode content, indexes, model weights, Node
packages and research output reports. The file manifest checks every bundled
input before any benchmark executes. Transfer only this private bundle over your
chosen SSH connection; no SSH target has been selected or contacted here.

## Install after selecting the target VM

These are operator commands for the Ubuntu VM, not commands already executed.
Use a reviewed archive, verify its reported SHA-256 after transfer, and unpack
only this allowlisted archive. Source files are root-owned and read-only to the
service identity. On an existing installation, preserve the prior source bundle
and /var/lib/algorithm-batch before replacing anything.

```bash
sudo useradd --system --home-dir /var/lib/algorithm-batch --shell /usr/sbin/nologin algorithm-batch
sudo install -d -m 0755 -o root -g root /opt/algorithm-rag-engine
sudo tar -xzf oci-stdlib-v1.tar.gz -C /opt/algorithm-rag-engine --no-same-owner
sudo chown -R root:root /opt/algorithm-rag-engine
sudo install -m 0644 /opt/algorithm-rag-engine/deploy/oci/algorithm-batch.service /etc/systemd/system/algorithm-batch.service
sudo install -m 0644 /opt/algorithm-rag-engine/deploy/oci/algorithm-batch.timer /etc/systemd/system/algorithm-batch.timer
sudo systemd-analyze verify /etc/systemd/system/algorithm-batch.service /etc/systemd/system/algorithm-batch.timer
sudo systemctl daemon-reload
sudo systemctl start algorithm-batch.service
```

Create the dedicated user only if absent (`getent passwd algorithm-batch` first).
No environment file is needed; the example contains only an optional Python
runtime setting, no credentials. systemd creates the private state directory.
A single manual run is required before timer activation.

## Verify on the actual VM

```bash
sudo systemctl show algorithm-batch.service -p Result -p ExecMainStatus -p MemoryPeak -p MemoryMax -p MemorySwapMax -p CPUQuotaPerSecUSec -p TasksMax
sudo journalctl -u algorithm-batch.service -n 30 --no-pager
sudo find /var/lib/algorithm-batch/runs -maxdepth 2 -name receipt.json -print
sudo systemd-analyze calendar 'Sun *-*-* 03:00:00 Asia/Seoul'
```

Inspect the newest private receipt and benchmark.json. Confirm status completed,
source hashes, cgroup memory/cpu controls, and actual memory peak. The default
MemoryHigh=192M, MemoryMax=256M, MemorySwapMax=0 and CPUQuota=25% constrain the
whole service tree. CPU quota is relative to one CPU, not the reported two CPUs.
TimeoutStartSec=180 limits total service wall time and the benchmark subprocess
also has a 120-second timeout. No socket networking is available to the service.
A direct CLI run does not enforce systemd controls. cgroup observations are null
when unavailable, never assumed passed from configuration alone.

Only after manual verification, explicitly activate the optional weekly timer:

```bash
sudo systemctl enable --now algorithm-batch.timer
sudo systemctl list-timers algorithm-batch.timer
```

Runs occur Sunday 03:00 Asia/Seoul with up to ten minutes jitter. Persistent=false
avoids catch-up work after a stopped VM. Neither package creation nor installation
automatically enables the timer. CPU time is not artificially burned to influence
Oracle idle classification. If OCI reclaims or stops the VM, optional reports
stop; Cloudflare missions and notifications remain independent.

## Outputs, retention and recovery

Every attempt uses a fresh run directory with private benchmark/stdout/stderr and
receipt files. A nonblocking flock prevents concurrent worker runs. Only directories
carrying this worker's matching receipt identity are eligible for retention;
unknown directories and symlinks are preserved. The latest 14 recognized runs,
including failures, are retained. No learner history or upstream source is pruned.
A crash before a receipt can leave an orphan for operator inspection.

If a resource limit, invalid hash or benchmark error occurs, inspect the receipt
and service journal. Keep the prior good report for comparison. Do not raise
memory or add swap automatically from one failed run. Disable the timer to pause:

```bash
sudo systemctl disable --now algorithm-batch.timer
```

The diagnostic currently has sparse ranking labels: only six multi-candidate query
groups after graph-overlap exclusion. Perfect conditional scores are not model
quality evidence. No fitting, encoder download or automatic publication is part
of this service. Reports can be copied back for review; Cloudflare does not fetch
or depend on this VM, and its D1 backup stays a separate private operation.

## Verification boundary

Local tests execute the worker and an extracted package, check deterministic
membership/hashes, retention, lock handling and failed inputs. This Mac environment
has no systemd-analyze and has not SSHed into the OCI VM. Ubuntu service parsing,
real cgroup enforcement, resource peaks, timer activation and cloud availability
remain operator verification steps. The official Noble systemd references used
for the settings are linked in ADR 002.

## Local verification receipt

On 2026-10-16, five stdlib tests verified deterministic bundle bytes and membership,
per-file hashes, execution from an extracted bundle with an isolated environment,
tampering/path rejection, conservative retention and nonblocking locking. These
are development-machine checks. Ubuntu systemd parsing, cgroup enforcement, VM
memory peak, SSH access and timer operation have not been verified.
