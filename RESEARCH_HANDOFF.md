# Autonomous Research Handoff — hoplas KGE (WN18RR)

**Audience:** an autonomous agent given free rein to modify code and launch runs to
improve knowledge-graph-embedding (KGE) scores on WN18RR. Read this fully before
changing anything. Author: handed off from an interactive session with Scott Hawley.

---

## 0. TL;DR mission

Two parallel tracks, both on WN18RR, both maximizing ranking metrics:

1. **Learnable-PHM track (primary).** We embed entities in a learnable table and
   represent each relation as a *learnable hypercomplex (PHM) operator*. Trained with
   an MSE "attraction" loss + **SIGReg** (an isotropic-Gaussian regularizer) that
   **replaces negative sampling**. Current best ≈ **MRR 0.44 / H@10 0.58**, already
   competitive with published QuatE on recall but **weak on H@1 (~0.37 vs 0.44)**.
   Goal: lift **all** metrics — especially **H@1 / MRR** — without sacrificing H@10.

2. **Raw-quaternion track (secondary).** The same framework with a *frozen* Hamilton
   quaternion operator (`--op quat`) is our QuatE reproduction. It currently scores
   **MRR 0.244** — only the pykeen-repro tier, far below published QuatE (0.481).
   Goal: push the **frozen-quaternion** numbers up toward published QuatE.

Improving each is partly independent; do both. **One variable at a time** for any
claim; log everything to wandb; save checkpoints.

---

## 1. The task & framing

WN18RR: 40,559 entities, 11 relations (→ 22 with inverse relations), ~87k train
triples. Standard transductive link prediction; report **filtered** MRR, MR, Hits@{1,3,10}.

We reframe KGE as the hoplas "ring task": a relation is a **group action / operator**
that transforms a head entity into its tail. `t ≈ op_r(h)`. This mirrors `train_ops.py`
(the synthetic ring task) but with (a) a learnable entity *embedding table* instead of
given coordinates, and (b) **one operator per relation**.

**Why SIGReg instead of negatives:** standard KGE uses negative sampling to prevent the
trivial collapse (all entities identical). We instead use **SIGReg** (Sliced Isotropic
Gaussian Regularizer, Epps–Pulley statistic; `hoplas/losses.py`) on the entity cloud to
hold it spread/Gaussian. This is cleaner (no negative sampling) and is the scientifically
interesting contribution. It works: the cloud stays spread and the model learns.

---

## 2. Code map (repo root: `~/github/hoplas`)

- **`train_kge.py`** — the KGE trainer (modeled on `train_ops.py`). Contains:
  - `KGEModel`: `nn.Embedding` entity table + `nn.ModuleList` of one `OpWrapper` per
    relation. `apply_relation(h_emb, r)` loops over the (≤22) relations in a batch.
  - `evaluate()`: **filtered** MRR/Hits, fully self-contained (no pykeen evaluator).
    Scores all entities via `-||op_r(h) - E||²` using the matmul expansion
    `-( |pred|² - 2·pred·Eᵀ + |E|² )` (no `(B,N,d)` blow-up). With inverse triples in
    the eval split, tail-ranking covers **both** directions (the "both" metric).
  - `freeze_quaternion` / `init_quaternion`: set a relation's PHM algebra `a` to the
    exact Hamilton table — frozen (raw quaternion) or trainable warm-start.
  - `algebra_metrics()`: logs `algebra_dist_quat = ‖a − a_quat‖_F` (Frobenius distance
    to the exact Hamilton table; **NOT** basis-invariant — see §7).
  - `embedding_viz()`: wandb 3D-PCA scatter of tail vs op(head), colored by relation.
  - Checkpointing: saves `checkpoints/<run>_best.pt` and `_final.pt` with the per-relation
    learned algebra tensors, entity table, args, metrics.
- **`hoplas/data.py::KGTripleDataset`** — WN18RR triples as `(h,r,t)` int IDs. Sources
  canonical IDs/splits from **pykeen** (data download only — *not* pykeen training).
  Adds inverse triples `(t, r+R, h)`. `true_tails()` builds the filter set.
- **`hoplas/ops.py::OpWrapper`** — wraps the relation operator. `--op` choices include
  `ph` (learnable PHM, random `a` init), `quat` (frozen Hamilton), `kquat`
  (Kingdon quaternion), `filmr`/`filmr_expm`/`matop*`. `op_resid=True` → `x + op(x)`
  (skip connection). `unit_norm` L2-normalizes output (we keep it **off** for KGE; SIGReg
  wants Gaussian, not a sphere).
- **`hoplas/ph_layers.py::PHMLinear`** — the PHM layer. Weight = Σᵢ kron(aᵢ, sᵢ); `a` is
  the (n,n,n) learnable algebra (the "structure constants"), `s` the per-block weights.
- **`hoplas/losses.py`** — `SIGReg(x, global_step, num_slices=256, chunk_size=32)` and
  `MomMatchLoss(...)` (class-conditional moment matching; currently off, `--lambda-mom 0`).
- **`configs/kge_*.cfg`** — configargparse config files (dash-keys). Existing:
  `kge_quat_4`, `kge_ph_4`, `kge_ph_quatinit`, `kge_ph_quatinit_1k`,
  `kge_ph_lambd0.01/0.2/0.3`.
- **`scripts/launch.sh`** — `./scripts/launch.sh <host> <config> [gpu]`. rsyncs source +
  configs to `<host>` and nohups the run there. **Task-aware:** `kge_*.cfg` → `train_kge.py`,
  else `train_ops.py`.
- **`scripts/launch_queue.sh`** — `./scripts/launch_queue.sh <host> [--par N] [--gpu ID]
  <configs...>`. Same rsync+SSH dispatch, but keeps `--par` runs busy and refills the queue
  as they finish. Also task-aware (per-config). Good for sweeps:
  `./scripts/launch_queue.sh lecun --par 3 configs/kge_*.cfg`.

---

## 3. The model & loss (precise)

- Entity embeddings `E ∈ R^{Ne×nd}`, init `N(0,1)`. Default `nd=200` (must be divisible by
  `--order`, default 4).
- Per relation `r`: `pred = op_r(E[h])`, where `op_r` is `x + PHMLinear_r(x)` (skip),
  no unit-norm.
- **Loss per batch of true triples:**
  `loss = (1−lambd)·(lambda_sim·MSE(pred, E[t]) + lambda_mom·MomMatch) + lambd·SIGReg(E[sample])`
  - `MSE(pred, E[t])` = "attraction": pull relation-transformed head onto its tail.
  - `SIGReg` is computed on a **fresh random sample of `--sigreg-n` entities (default 4096),
    decoupled from `--batch-size`** (so the anti-collapse pressure doesn't wobble with batch
    size; the Epps–Pulley estimate is sample-size sensitive).
  - `lambda_mom` defaults 0 (MomMatch off).
- Optimizer: AdamW, two param groups (entity table lr `--lr`, relation ops `--op-lr` or `--lr`).
- LR schedule: `--scheduler {none,warmup,onecycle}`. `onecycle` steps per-batch to/from
  `--max-lr`. **Lesson learned: `max-lr=0.1` diverged** (loss spiked, algebra blew up
  ~epoch 130); **`max-lr=0.02` is stable.** `warmup` is a per-epoch linear ramp.

---

## 4. How to run (env, GPU, wandb, checkpoints)

- **Env:** `~/envs/hoplas` (uv-managed). `pykeen` is installed there *for data only*.
  In the dev container, ensure: torch (CUDA build matching the container's driver — we hit
  a cu130-vs-driver mismatch before; pick a wheel ≤ the driver's CUDA version), `pykeen`,
  `wandb`, `configargparse`, `plotly`. SIGReg uses complex `exp` → **needs CUDA or CPU,
  not MPS**.
- **Launching runs:** the `scripts/launch*.sh` scripts run *locally* (in the container) and
  dispatch jobs to a GPU **host via SSH/rsync** — the container is the control plane, the
  host (e.g. `lecun`) is the compute. Just ensure the container has SSH access to the host
  (ssh config + keys). Use `./scripts/launch.sh <host> <config> [gpu]` for one run or
  `./scripts/launch_queue.sh <host> --par N configs/kge_*.cfg` for a sweep.
- **Or run directly** (e.g. if the GPU is local to the container, or for a quick test):
  `python train_kge.py --config configs/kge_ph_4.cfg` (CLI overrides config keys).
  One-off: `python train_kge.py --op ph --nd 256 --epochs 500 ...`.
- **wandb:** project `drscotthawley/hoplas-kge`. `wandb login` must be done in the
  container. Use `--no-wandb` for quick smoke tests.
- **Checkpoints** land in `checkpoints/<run_name>.pt` variants. **Always** keep them; they
  hold the learned per-relation algebra for offline analysis.
- **Smoke test pattern** before any real launch: `--epochs 2 --eval-every 0 --nd 8
  --batch-size 8192 --cpu --no-wandb` — catches wiring bugs in seconds.
- **GPU:** historically lecun (RTX 4090, 24 GB). Runs are tiny (~1.6 GB, <10% util), so
  **stacking several concurrent runs is fine**. In the container use whatever GPU is present.

---

## 5. Current results (WN18RR test, filtered; nd=200 unless noted)

| Model | MRR | MR | H@10 | H@3 | H@1 |
|---|---|---|---|---|---|
| **Published QuatE¹** (paper) | 0.481 | 3472 | 0.564 | 0.500 | 0.436 |
| **Published QuatE³** (paper, best) | 0.488 | 2314 | 0.582 | 0.508 | 0.438 |
| pykeen QuatE repro (40k ep) | 0.208 | 7440 | 0.390 | 0.256 | 0.120 |
| our **frozen quat** (`--op quat`) | 0.244 | **1669** | 0.521 | 0.354 | 0.082 |
| our **PHM random-init** | 0.440 | 1684 | 0.573 | 0.478 | 0.367 |
| our **PHM quat-init** | 0.444 | 1735 | 0.573 | 0.476 | 0.374 |
| our **PHM quat-init, 1k ep, onecycle .02** | 0.444 | 1933 | 0.567 | 0.477 | 0.376 |
| our **PHM lambd=0.2** (300 ep) | 0.442 | 1802 | **0.587** | 0.488 | 0.360 |
| our **PHM lambd=0.3** | (in progress) | | **>0.58** | | (other metrics dipped) |

Key reads:
- **Our MR (~1670–1930) is far better than published QuatE (2314–3472)** — SIGReg's spread
  gives excellent tail behavior (few catastrophic mis-ranks).
- **H@10 is QuatE-competitive or better** (0.57–0.59).
- **H@1 is the weakness** (~0.37 vs published 0.44). It caps MRR at ~0.44 vs 0.48.
- Learnable algebra ≈ **1.8× the frozen quaternion on MRR, ~4.5× on H@1** (0.082→0.37) —
  the controlled in-framework win. The frozen quaternion's H@1 of 0.082 is its worst feature.
- **quat-init ≈ random-init** at convergence (init washes out; the landscape here is benign).
  More epochs (300→1000) did **not** help (already plateaued).
- `lambd` trades fit vs spread: higher `lambd` → better H@10, slightly worse H@1; MRR ~flat.

---

## 6. Concrete experiment ideas

### Track 1 — learnable PHM, lift H@1/MRR without losing H@10
The diagnosis: recall (H@10) is great, **precision-at-1 is the gap**. Ideas, roughly in
order of expected value / cheapness:
1. **Scoring/readout function.** Eval scores by `-||pred − t||²` and trains by MSE. A
   distance metric is blunt at rank-1. Try: cosine similarity, a **learned temperature**,
   a **margin** term, or a small learned bilinear/diagonal readout on top of `pred`. This
   directly targets H@1. (Make sure train objective and eval scoring stay consistent.)
2. **Capacity:** larger `nd` (256, 400, 512 — keep divisible by `order`). Published QuatE
   uses 400-real-dim; we use 200. More neurons may especially help H@1.
3. **`lambd` schedule:** the static `lambd` trades H@10↔H@1. Try **annealing lambd**
   (start high for spread, decay so MSE sharpens rank-1 late), or a per-term schedule.
4. **MomMatch** (`--lambda-mom > 0`, `hoplas/losses.py`): class-conditional covariance
   matching forbids within-class variance collapse — may sharpen per-relation structure.
5. **`order` (n) of the hypercomplex algebra:** try n=2, 8 (nd must stay divisible). Does a
   richer algebra help, or is 4 (quaternion-sized) best?
6. **Operator family:** compare `--op ph` vs `kquat` (Kingdon), and `filmr_expm` (rotation
   via matrix exp). `op_resid` on/off, `unit_norm` on/off (but unit_norm conflicts with
   SIGReg's Gaussian target — test carefully).
7. **Optimizer/schedule:** onecycle `max-lr` in [0.01, 0.03] (0.1 diverges), longer cosine,
   separate `--op-lr`. Bigger/smaller batch.
8. **`sigreg-n`:** the SIGReg sample size (default 4096). Larger may stabilize the estimate.

### Track 2 — raw frozen-quaternion (`--op quat`), toward published QuatE (0.48)
Our frozen quaternion (MRR 0.244, H@1 0.082) badly underperforms published QuatE (0.481).
This is a *different* training regime than QuatE's (we use MSE+SIGReg, no negatives;
QuatE uses sLCWA/BCE). The frozen-quaternion's terrible H@1 suggests the MSE+SIGReg recipe
is the wrong objective for a fixed quaternion. Ideas:
1. **Scoring function** (as Track-1 #1) likely matters even more here.
2. **Tune the same knobs** for the frozen op: `nd`, `lambd`, `lr/schedule`, `sigreg-n`,
   `op-resid`, longer runs — the frozen op was never tuned (we mostly tuned the learnable one).
3. Consider whether the **relation should be a unit quaternion** (QuatE normalizes the
   relation). Our frozen op is the Hamilton *multiplication-by-a-fixed-table*; QuatE multiplies
   by a learned **unit** quaternion. A per-relation **unit-quaternion** parameterization (learn
   a 4-vector per relation, normalize, Hamilton-multiply) may be the faithful "raw QuatE" and
   could close much of the gap. This is a small new op worth adding.
4. As a sanity ceiling, it's fine to also reproduce QuatE in its native regime to confirm the
   ~0.48 target is reachable in *this* codebase.

---

## 7. Methodology & guardrails (please follow)

- **One variable per claim.** Name configs/tags so the diff is obvious (e.g.
  `kge_ph_nd400.cfg`). Don't change scheduler *and* lambd *and* nd in one run and attribute
  a delta.
- **Always report TEST, not val**, for headline numbers; val is fine for tracking. Our
  `evaluate()` does filtered ranking — don't "fix" it without re-deriving filtered MRR
  carefully (filtered eval is exactly where KGE bugs hide).
- **Watch for two failure modes we already hit:**
  (a) *collapse* — if SIGReg weight is too low or scoring degenerate, embeddings/algebra
  collapse and MRR → ~0 (loss flattens near a trivial value). The earlier pykeen-bilinear
  approach died this way; SIGReg is what prevents it here.
  (b) *LR divergence* — onecycle `max-lr ≥ ~0.1` blows up (~epoch 130). Stay ≤ 0.03.
- **`algebra_dist_quat` is NOT basis-invariant.** A large value doesn't mean "not a
  quaternion algebra" (could be an isomorphic one). For real "did it learn a quaternion?"
  analysis, load the saved per-relation `a` tensors and compare up to a change of basis.
  (Open question we never answered: does random-init converge to a quaternion-*equivalent*
  algebra, or a genuinely different equally-good one? The checkpoints enable this.)
- **Smoke-test on CPU (nd=8, 2 epochs, --no-wandb) before every real launch.**
- **Keep checkpoints and wandb runs.** Name wandb runs descriptively (the trainer already
  builds `run_name` from op/order/nd/lambd/tag).
- **Stacking runs is fine** (tiny GPU footprint), but label them clearly.

## 8. Known gotchas (paid for already)
- torch CUDA wheel must match the driver's CUDA version (we hit cu130-on-12.4-driver →
  silent CPU fallback). Verify `torch.cuda.is_available()` at startup.
- SIGReg's complex `exp` does **not** run on Apple MPS — CUDA/CPU only.
- pykeen is for **data sourcing only**; do not reintroduce pykeen training loops (that path
  was a time-sink and the bilinear-PHM-in-pykeen kept collapsing).
- `launch.sh`/`launch_queue.sh` dispatch to a GPU host via SSH/rsync from wherever they're
  run; the container needs SSH access to that host (or run `train_kge.py` directly if the
  GPU is local).

## 9. Definition of success
Primary: raise **MRR and H@1** of the learnable-PHM model toward/above published QuatE
(MRR 0.48, H@1 0.44) **while keeping H@10 ≥ 0.57 and MR ≤ ~1700**. Secondary: raise the
**frozen-quaternion** MRR well above 0.244 (ideally toward the QuatE tier). Document each
result in a short table (config → test MRR/MR/H@{1,3,10}) and keep the winning configs.
