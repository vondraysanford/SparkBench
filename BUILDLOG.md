# BUILDLOG

Session notes, newest first. Three bullets per session — what worked, what broke, what surprised you — plus one line naming the day's capture that is post-worthy. Numbers land here as text the moment they exist; screenshots rot, text greps.

---

## 2026-09-06 · Two Sparks bridged (non-gating Phase 1 task, done ahead of order)

- **Worked:** NVIDIA Sync's Cluster Assistant took `spark-927a.local` and `spark-11ed.local` from two boxes with a cable between them to a validated cluster named `Lucia` in one pass. It verified SSH, GB10, OS, and sudo on both; confirmed `admin` (UID/GID 1000) matches on both; found no existing CX-7 network and created one on `10.100.240.0/24` and `10.100.241.0/24` at a reported 200 Gbps per link; measured **185.96 Gbps** on its speed test (threshold > 180 Gbps); and distributed SSH keys, config, and known hosts with a passing test connection. The cluster view the next morning: 2 nodes online, 261 GB total, idle at 37–41 °C and 3–5 W.
- **Broke:** _fill in — anything that failed or had to be retried before this run?_
- **Surprised:** _fill in — the assistant's subnet choice, how long the speed test took, anything about the cabling?_
- **Capture:** eight screenshots in `experiments/01-finetune-vs-frontier/evidence/phase-1/` (`2026-09-06-cluster-assistant-01…07-*.png`, `2026-09-07-nvidia-sync-cluster-lucia-2-nodes-online.png`). Post-worthy: the 185.96 Gbps test and the two-node cluster view. Still owed for this milestone: a photo of the two Sparks with the QSFP cable, and the NCCL / `ib_write_bw` numbers (see `infra/cluster.md`).
