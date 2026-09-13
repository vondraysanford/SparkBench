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

- [ ] `ibdev2netdev` on each node → record which logical interface (`enp1s0f1np1` / `enP2p1s0f1np1` / `…f0np0`) carries `10.100.240.x` and `10.100.241.x`. NVIDIA's two-Spark vLLM recipe needs that name for `MN_IF_NAME`, `NCCL_SOCKET_IFNAME`, `UCX_NET_DEVICES`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`.
- [ ] Management (LAN / Wi-Fi) IPs of both nodes — the addresses you SSH to from the Mac. Needed by the NCCL playbook's `mpirun -H` line.
- [ ] NCCL `all_gather_perf` across both nodes per the [NCCL playbook](https://build.nvidia.com/spark/nccl) → `bench/results/dual-spark/`.
- [ ] `ib_write_bw` and `ib_write_lat` per the [performance guide](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/performance_benchmarking_guide.md) → `bench/results/dual-spark/`. NVIDIA's own run shows ~92 + ~97 Gbps across the two RDMA links.
- [ ] Confirm the role assignment above and reference the hostnames from `infra/serve/` scripts.

## Gotchas

- **Do not apply the Connect Two Sparks playbook's netplan example on top of this.** The playbook uses `192.168.100.x` / `192.168.101.x`; the Cluster Assistant chose `10.100.240.x` / `10.100.241.x`. Where a playbook says `192.168.100.10`, substitute the addresses above.
- NVIDIA Sync reports memory in GB by default (131 GB = 128 GiB). Use one unit in `bench/` and say which.
- Both nodes need the same username for every NVIDIA multi-node recipe. That is already true (`admin`). Do not create per-node users.
- If a container fails to allocate memory well below 128 GB, flush the page cache first: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` (NVIDIA's documented unified-memory workaround).
