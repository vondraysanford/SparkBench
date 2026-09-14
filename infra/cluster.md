# Cluster `Lucia` — two DGX Sparks over ConnectX-7

Bridged **2026-09-06** with NVIDIA Sync's [Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html), which detected the QSFP cabling, created the CX-7 network, validated link speed, and distributed SSH keys in one pass. Evidence: `experiments/01-finetune-vs-frontier/evidence/phase-1/2026-09-06-cluster-assistant-*.png` and `2026-09-07-nvidia-sync-cluster-lucia-2-nodes-online.png`.

## Nodes

| | `spark-927a.local` | `spark-11ed.local` |
|---|---|---|
| **Proposed role** | **Spark 1 — train** (NGC PyTorch container) | **Spark 2 — serve + eval** (NGC vLLM container) |
| Cluster IP 1 (CX-7 link A) | `10.100.240.1/24` | `10.100.240.2/24` |
| Cluster IP 2 (CX-7 link B) | `10.100.241.1/24` | `10.100.241.2/24` |
| Reported link speed | 200 Gbps | 200 Gbps |
| Username / UID / GID | `admin` / 1000 / 1000 | `admin` / 1000 / 1000 |
| Memory (NVIDIA Sync, GB) | 131 GB | 131 GB |
| Logical cores | 20 | 20 |
| Idle temp / power at capture | 41 °C / 5 W | 37 °C / 3 W |

Cluster total as reported by NVIDIA Sync: **261 GB**, 2 nodes online.

Role assignment is a proposal: `spark-927a` holds the `.1` addresses so it is "node 1" by the assistant's numbering. Swap if you prefer; whichever node trains is the one whose page cache never competes with vLLM's KV cache.

## Validation

| Check | Result |
|---|---|
| SSH access, GB10 device, minimal OS, sudo | verified on both |
| User details consistent across nodes | verified (`admin`, 1000/1000) |
| Existing CX-7 network | none found; new network created |
| Cluster Assistant speed test (1/1) | **185.96 Gbps** (expected > 180 Gbps) |
| SSH: directory, keypair, public key share, config, known hosts, test connection | complete on both |

## Not done yet (non-gating)

- [x] **Done 2026-09-13.** Port mapping, same on both nodes (see [Ports](#ports) below): link A `10.100.240.x` is RoCE device `rocep1s0f1` on interface `enp1s0f1np1`; link B `10.100.241.x` is `roceP2p1s0f1`. RoCE v2 GID index **3** on both. Each node exposes four RoCE devices; the two with all-zero GID tables (`rocep1s0f0`, `roceP2p1s0f0`) are not cabled. Still owed: paste the raw `ibdev2netdev` output from each node here.
- [ ] Management (LAN / Wi-Fi) IPs of both nodes — the addresses you SSH to from the Mac. Needed by the NCCL playbook's `mpirun -H` line. Head `spark-927a` is `192.168.1.165` (DHCP — reserve it in the router). Worker `spark-11ed`: _record it_.
- [ ] NCCL `all_gather_perf` across both nodes per the [NCCL playbook](https://build.nvidia.com/spark/nccl) → `bench/results/dual-spark/`.
- [ ] `ib_write_bw` and `ib_write_lat` per the [performance guide](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/performance_benchmarking_guide.md) → `bench/results/dual-spark/`. NVIDIA's own run shows ~92 + ~97 Gbps across the two RDMA links.
- [ ] Confirm the role assignment above and reference the hostnames from `infra/serve/` scripts.

## Ports

Verified 2026-09-13 by a passing NCCL preflight and a running tensor-parallel job.

| | Link A (`10.100.240.x`) | Link B (`10.100.241.x`) |
|---|---|---|
| RoCE device | `rocep1s0f1` | `roceP2p1s0f1` |
| Interface | `enp1s0f1np1` | `enP2p1s0f1np1` (by DGX naming; confirm with `ibdev2netdev`) |
| RoCE v2 GID index | 3 (`::ffff:10.100.240.x`) | 3 (`::ffff:10.100.241.x`) |

How to re-derive: `ibdev2netdev` maps device to interface; `ip -br -4 addr` shows which interface holds which address; `cat /sys/class/infiniband/<dev>/ports/1/gids/{0..7}` with `gid_attrs/types/` shows the index whose entry is the node's own IPv4-mapped address with type RoCE v2. The IPv6 `fe80::` rows are not usable by NCCL.

Where the names go: NVIDIA's two-Spark vLLM recipe wants the interface for `MN_IF_NAME`, `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, and the device for `UCX_NET_DEVICES` / `NCCL_IB_HCA`, plus `NCCL_IB_GID_INDEX=3`.

## First workload over the link (2026-09-13)

GLM-5.3-Flash (320B MoE, 4-bit EXL3) served tensor-parallel 2 across both nodes with the community [MiaAI-Lab recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks): head `spark-927a` rank 0 + OpenAI API on port 8888, worker `spark-11ed` rank 1 headless, NCCL over RoCE v2 on link A only. The 164 GiB weight cache rsynced head → worker over the link at ~391 MB/s (rsync- and NVMe-bound, not link-bound). Health returned 200, `17 × 23` returned `391` at temperature 0, and VS Code Copilot Chat drives it as a custom OpenAI-compatible model. Session notes are in `BUILDLOG.md`.

**This is not a SparkBench condition.** It runs a community overlay image, not the pinned NGC vLLM container, and the checkpoint is outside NVIDIA's vLLM-on-Spark support matrix. It stays out of the master table. It is useful as proof the link carries a real NCCL workload and, later, as a hand-run reviewer for sanity-checking planted bugs.

## Gotchas

- **The GLM cluster owns both Sparks while it runs.** Its two containers hold ~85 % of memory on each node. Before any Phase 1 training run: `cd ~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks && ./start.sh stop` on the head. Do not delete either node's `~/.cache/huggingface`; a relaunch then skips the download and the copy.
- **From the Mac, use `192.168.1.165`, not `spark-927a.local`.** The `.local` name advertises the cluster addresses too, and Chromium-based apps (VS Code) can pick one the Mac cannot route to (`ERR_ADDRESS_UNREACHABLE`). VS Code also needs macOS **Privacy & Security → Local Network** switched on before it can reach any LAN address; Terminal already has it, which is why `curl` works when VS Code does not.
- **Do not apply the Connect Two Sparks playbook's netplan example on top of this.** The playbook uses `192.168.100.x` / `192.168.101.x`; the Cluster Assistant chose `10.100.240.x` / `10.100.241.x`. Where a playbook says `192.168.100.10`, substitute the addresses above.
- NVIDIA Sync reports memory in GB by default (131 GB = 128 GiB). Use one unit in `bench/` and say which.
- Both nodes need the same username for every NVIDIA multi-node recipe. That is already true (`admin`). Do not create per-node users.
- If a container fails to allocate memory well below 128 GB, flush the page cache first: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` (NVIDIA's documented unified-memory workaround).
