# Phase A Storage Incident and NVMe Offload Audit

**Date:** 2026-09-19
**Branch:** `agent/Watson-dev`
**Scope:** host storage pressure observed after the Phase A PBT/autoresearch
loop, preservation of experiment evidence, and preparation of safe NVMe-backed
Ray output paths. This audit does not change solver, reward, task, or recurrent
architecture.

## Decision

The Phase A PBT runs contributed to storage pressure but did not directly fill
the system drive. The immediate failure was a single WSL crash dump for
`/usr/bin/node` measuring 77,937,594,360 bytes, written under
`%LOCALAPPDATA%\Temp\wsl-crashes` at approximately 16:47. The repository's
experiment outputs occupied 3,391,042,441 bytes and the user-level
`ray_results` tree occupied 2,428,525,076 bytes.

The crash dump explains the abrupt transition from low headroom to a blocked
drive. A recent Microsoft WSL issue reports the same failure class: individual
Linux-process crash dumps of roughly 58--65 GiB exhausting the Windows host
temporary volume. The installed WSL version here is 2.7.14.0, so the behavior
is not safely assumed fixed by the local version. See
[microsoft/WSL#41314](https://github.com/microsoft/WSL/issues/41314) and the
[official WSL advanced-settings reference](https://learn.microsoft.com/windows/wsl/wsl-config).

## Reproducible signal

The incident check is deliberately small and red-capable:

```powershell
$free = (Get-PSDrive -Name C).Free
if ($free -lt 5GB) { exit 2 }
```

Before remediation it reported `0.214 GB` free and exited 2. After remediation
it reported `72.792 GB` free and exited 0; the final verified state after the
remaining offloads is `78.250 GB` free.

## Attribution

| Storage owner | Bytes observed | Interpretation |
|---|---:|---|
| WSL `/usr/bin/node` crash dump | 77,937,594,360 | Immediate exhaustion event; diagnostic only |
| Two older WSL init dumps | 1,007,616 | Diagnostic only |
| Repository experiment outputs | 3,391,042,441 | Durable Phase A/B evidence and Ray checkpoints |
| User-level `ray_results` | 2,428,525,076 | Historical/default Ray results |
| Ray sessions under `%TEMP%\ray` | 141,794,546 | Disposable but not material to the incident |
| Repository `.venv` | about 2.37 GB | Required Windows-native runtime; retained |
| Repository `.git` | about 676 MB | Required source history; retained |
| Ubuntu `ext4.vhdx` | 76,946,604,032 | Live WSL distro, not disposable; retained on C for now |

The PBT-related result trees therefore consumed about 5.82 GB of file payload.
They reduced safety margin but were an order of magnitude smaller than the
single crash dump.

## Completed remediation

1. Moved all three WSL crash dumps to
   `E:\WSL-diagnostics\wsl-crashes-20260919`. The destination contains exactly
   three files and 77,938,601,976 bytes; the source archive no longer exists on
   C or D.
2. Moved `C:\Users\Admin\ray_results` to
   `D:\iTRADER-offload\active\ray_results` and restored the original path as
   an NTFS junction. Verification found 9,494 files and 2,428,525,076 bytes.
3. Moved 37 readable `phase*` experiment directories, totaling 11,790 files
   and 3,374,436,366 bytes, to
   `D:\iTRADER-offload\active\itrader-azr\experiment-outputs`. Each original
   path is now an independently verified NTFS junction. Existing report plots
   still resolve through their original paths.
4. Left one inaccessible, zero-byte pytest scratch directory untouched. It
   prevented replacing the entire `outputs` root with one junction but did not
   prevent per-run offload.
5. Added the following user-level `%USERPROFILE%\.wslconfig`:

   ```ini
   [wsl2]
   crashDumpFolder=E:\\WSL-diagnostics\\wsl-crashes
   maxCrashDumpCount=1
   ```

   The setting is documented by Microsoft and will take effect on the next
   normal WSL restart. WSL was not forcibly restarted because Ubuntu currently
   hosts Hermes, OpenClaw, ZeroTier, and the browser harness.

## Verified final layout

| Volume | Role | Free space after remediation |
|---|---|---:|
| C: Kingston SATA SSD | Windows, repository, Python environment, live WSL VHDX | 78.250 GB |
| D: Fanxiang NVMe SSD | Active Ray results and per-run experiment directories | 163.846 GB |
| E: WDC HDD | Retained crash diagnostics and future WSL crash dumps | 259.786 GB |

All 37 experiment junctions point beneath the intended D-drive root, the
default `ray_results` junction points to D, and the main Phase A comparison plot
is readable through its original C-drive path.

## Deferred coordinated WSL move

Ubuntu's live `ext4.vhdx` is 76.95 GB on C. WSL 2.7.14 supports
`wsl --manage Ubuntu --move <Location>`, and D is healthy NTFS storage. Moving
the distro could free another roughly 71.7 GiB on C and improve Linux-side
GitNexus/Node I/O, but it requires terminating the active distro and its
services. Do this only in a coordinated maintenance window:

```powershell
wsl --shutdown
wsl --manage Ubuntu --move D:\WSL\Ubuntu
wsl --list --verbose
wsl -d Ubuntu -- uname -a
```

Afterward, verify Hermes, OpenClaw, ZeroTier, the browser harness, repository
mounts, and the GitNexus workflow before declaring the move complete. Do not
unregister/import the distro as a shortcut.

## Operational guardrails

- Check C-drive free space before every long PBT run and require at least 20 GB.
- Keep active Ray/checkpoint outputs on D and cold diagnostics on E.
- Treat `wsl-crashes` as diagnostics, never as experiment evidence.
- Do not move an active Ray run or WSL distro.
- Preserve Markdown/HTML report links through the verified junctions.
- Run the storage A/B on fixed compute/economic settings; compare wall time,
  checkpoint latency, and sampled steps per second, not policy return.

## 2026-09-20 live-run verification

Two memory-safe four-member PBT probes ran with active output on D while an
independent watchdog sampled C free space and searched for new Ubuntu/WSL crash
files every 30 seconds. C remained above 76.08 GB free, D ended above 157.5 GB
free, and the watchdog recorded zero crash files. The final post-fix probe
terminated all four Ray trials successfully.

The prior training interruption is now attributed to Windows RAM pressure, not
storage or WSL: four concurrent trials with two rollout workers each exceeded
the practical capacity of the 15.92 GB host. The validated setting is one
rollout worker per trial with a four-CPU Ray cap, which schedules two trials at
a time and retained at least about 6 GB free RAM during the proof. Keep that
topology for Phase A overnight work unless a separately measured capacity test
supports higher concurrency.

## 2026-09-20 GitNexus crash follow-up

After the training and repository checks were complete, the required stale-index
refresh exposed the same WSL failure class independently of Ray. GitNexus's WSL
Node process (`pid 12713`) crashed during `gitnexus analyze`. The watchdog caught
C free space falling from 76.078 GB and the appearance of
`wsl-crash-1789857188-12713-_usr_bin_node-11.dmp`. The analyzer was interrupted
immediately, but the dump continued expanding to 82,457,206,776 bytes and drove
C to effectively 0 GB before its writer released the file. Ubuntu remained
running, the completed PBT evidence was already on D, and no Ray process was
active, so this event did not invalidate an experiment.

The configured `crashDumpFolder=E:\\WSL-diagnostics\\wsl-crashes` had not yet
taken effect because the live WSL instance has deliberately not been restarted.
Once the lock released, the dump was preserved at
`E:\WSL-diagnostics\wsl-crashes-20260920` through a verified cross-volume move.
C recovered to 76.84 GB free and E retained 182.99 GB free; WSL was not shut
down. Do not retry GitNexus analysis in the current WSL lifetime. At the next
approved restart, first verify that a controlled WSL process crash routes to E,
then refresh GitNexus. The repository's existing index remains stale and must
not be treated as current architecture evidence.
