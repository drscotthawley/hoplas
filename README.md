# hoplas

**Hypercomplex Operational Latent Spaces**

`hoplas` studies how the *choice of operator* shapes the geometry of a learned
latent space. We train a small projector to map data into a new space in which a
single learned operator advances each sample to "the next one in a sequence"
(a ring/cycle), and we compare operator families — ordinary matrices, rotations,
and **hypercomplex** (parameterized Kronecker-product) layers — to see which
inductive biases produce clean, low-degree-of-freedom cyclic structure.

This is a follow-up to earlier work on
**[Operational Latent Spaces (OpLaS)](https://drscotthawley.github.io/oplas/)**
(Hawley & Tackett, [arXiv:2406.02699](https://arxiv.org/abs/2406.02699)),
which showed that self-supervised latent spaces can support semantically
meaningful *operations* (analogous to op-amps in electronics) — e.g. learning
the Circle of Fifths as a rotation in a 64-d space via the **FiLMR** rotation
layer. `hoplas` pushes on the central question raised there: how does the
operator's algebra (its degrees of freedom vs. its built-in structure) trade off
against the geometry the latent space is forced to adopt?

All training runs log to [Weights & Biases](https://wandb.ai/).

---

## Operators

The operators under comparison (in [`hoplas/filmr.py`](hoplas/filmr.py) and
[`hoplas/ph_layers.py`](hoplas/ph_layers.py)):

| name         | what it is                                                                 |
|--------------|----------------------------------------------------------------------------|
| `matop`      | square matrix applied twice (`x·Mᵀ` then `·M`) — baseline                   |
| `matop2`     | single square matrix multiply (`x·M`) — cleanest unconstrained baseline     |
| `filmr`      | FiLM + rotation in nd via the Aguilera–Perez algorithm (uses `arctan2`)     |
| `filmr_expm` | FiLM + rotation via a low-rank skew-symmetric generator and `matrix_exp`    |
| `ph`         | `PHMLinear` at hypercomplex order *n* — a parameterized Kronecker-sum layer |

`filmr_expm` is the workhorse rotation: its generator is built from `rank//2`
vector pairs, so `rank=2` is a single rotation plane (the right prior for a ring)
and `rank=nd` recovers a general `SO(nd)` rotation. It exposes
`rotation_angle_deg()` to read off the learned rotation angle. Unlike `filmr` it
has well-conditioned gradients everywhere (no `arctan2` plateau near identity).

The `ph` operator (from
[Grassucci et al.'s HyperNets](https://github.com/eleGAN23/HyperNets)) is genuinely
*constrained* at low order: order *n* saturates the parameterization and behaves
like a full unconstrained matrix only once `n ≳ nd^(2/3)`. Below that it carries
a real algebraic prior — which is the whole point of studying it here.

---

## Installation

There is no PyPI release yet; install from GitHub. Python ≥ 3.10.

**Option A — clone and install editable** (recommended for development):

```bash
git clone https://github.com/drscotthawley/hoplas.git
cd hoplas
pip install -e .          # or: uv pip install -e .
```

**Option B — install directly from GitHub**:

```bash
pip install git+https://github.com/drscotthawley/hoplas.git
# or: uv pip install git+https://github.com/drscotthawley/hoplas.git
```

Dependencies (torch, torchvision, wandb, plotly, gdown, safetensors, …) are
declared in [`pyproject.toml`](pyproject.toml) and installed automatically. Dev
extras (pytest): `pip install -e ".[dev]"`.

Log in to W&B once before training (or set `WANDB_MODE=offline`):

```bash
wandb login
```

---

## Experiments

### 1. Simple rotation benchmark — `train_simple_rot.py`

Supervised sanity check: learn a single fixed nd-dimensional rotation with each
operator and compare final MSE. Useful for confirming an operator *can* represent
a rotation, and for probing how inter-channel correlation in the data interacts
with constrained (low-order `ph`) operators.

```bash
# single run
./train_simple_rot.py --method filmr_expm --nd 64

# using a config file (CLI args override)
./train_simple_rot.py --config my_rot.yaml --nd 128

# full sweep over methods × correlation breadth, prints a final-loss table
./run_simple_rot.sh 0.9      # 0.9 = correlation strength among correlated channels
```

Both scripts accept `--config <file.yaml>` (via
[ConfigArgParse](https://github.com/bw2/ConfigArgParse)). Config keys use
underscore `dest` names (e.g. `corr_nd: 4`, `n_samples: 50000`). CLI args
always override the config file.

Key args: `--method {filmr,filmr_expm,matop,matop2,ph}`, `--nd`, `--order`
(hypercomplex order for `ph`), `--corr` (correlation strength in [0,1]),
`--corr-nd` (how many channels share that correlation), `--lr`, `--epochs`.
Results: <https://wandb.ai/drscotthawley/simple%20rot>

Final MSE after training, by method, sweeping over `corr_nd` — the number of
input channels sharing an inter-channel correlation (`corr=0.9`):

| run        |     nd=1 |     nd=2 |     nd=4 |     nd=8 |    nd=16 |
|------------|---------:|---------:|---------:|---------:|---------:|
| filmr      | 0.001216 | 0.000698 | 0.000496 | 0.002542 | 0.001237 |
| filmr_expm | 0.000000 | 0.000001 | 0.000001 | 0.000008 | 0.000028 |
| matop      | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| matop2     | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| ph_2       | 0.054844 | 0.055158 | 0.055196 | 0.047945 | 0.038888 |
| ph_4       | 0.068751 | 0.067965 | 0.067233 | 0.061337 | 0.054137 |
| ph_8       | 0.061031 | 0.060617 | 0.059060 | 0.054227 | 0.046926 |
| ph_16      | 0.000000 | 0.000000 | 0.000001 | 0.000006 | 0.000010 |

The `ph_N` rows are `PHMLinear` at hypercomplex order N. Orders below
nd^(2/3)=16 are genuinely constrained algebras (non-zero structural floor),
while order 16 saturates the Kronecker-sum parameterization and can represent
any matrix, so it solves the rotation exactly. The constrained ph orders (2/4/8)
improve as more input channels are correlated (loss drops left-to-right).

### 2. Operator-learning task — `train_ops.py`

The main experiment. A `Projector` maps data into a new space, a `trans_op`
(one of the operators above) transforms each point — advancing it to the next in
a cycle (`--target ring`) or reflecting it (`--target reflect`, dihedral
inversion) — and an inverse projector maps back. Training is **self-supervised**: a similarity loss
pulls `trans_op(proj(xᵢ))` toward `proj(xᵢ₊₁)`, **SIGReg** (an Epps–Pulley
normality-test regularizer, [`hoplas/losses.py`](hoplas/losses.py)) spreads the
distribution, and a reconstruction loss keeps the projector invertible. There are
no negative pairs — only attraction + regularization.

Two datasets:

- **`line`** — a synthetic quantized line in nd-space with wraparound
  (a clean ring; good for fast iteration).
- **`mnist`** — VAE encodings of MNIST, cycling through digit classes
  `0→1→…→9→0`. Requires a one-time encoding step (below).

```bash
# synthetic ring
./train_ops.py --dataset line --op filmr_expm --nd 3

# MNIST digit cycle (nd is forced to the VAE latent dim, 16)
./train_ops.py --dataset mnist --op filmr_expm --epochs 500

# using a config file (CLI args override)
./train_ops.py --config mnist_filmr.yaml --epochs 200
```

Both scripts accept `--config <file.yaml>` (via
[ConfigArgParse](https://github.com/bw2/ConfigArgParse)). Config keys use
underscore `dest` names (e.g. `op: filmr_expm`, `op_lr: 0.0001`,
`lambda_recon: 1.0`). CLI args always override the config file.

Notable args: `--op`, `--rank` (rotation planes for `filmr_expm`), `--order`
(hypercomplex order for `ph`), `--op-lr` (separate, usually *lower* LR for the
operator so its rotation angle climbs gently and locks onto the first closure
sheet), `--lambd` (SIGReg weight), `--lambda-recon`, `--op-resid`/`--no-op-resid`,
`--unit-norm`/`--no-unit-norm`, `--val-every`. Runs log a live 3-D PCA scatter of
the embedding (colored by digit) to W&B. The best-validation checkpoint is saved
to `checkpoints/<run_name>.pt` (projector, inverse projector, operator, and all
CLI args).

Projects: `ring` (line) and `ring-mnist` (mnist) on W&B.

### MNIST encoding (one-time prep for the mnist ring task)

```bash
./scripts/encode_mnist.py
```

This downloads Marco Cassar's pretrained MNIST VAE (code + weights fetched on
first use into `third_party/` and `~/datasets/hoplas_vae/`, both gitignored),
encodes every MNIST image to its 16-d latent `mu`, and saves
`~/datasets/mnist_latents.pt`. The ring task reads from there.

### 3. Inference demo — `hoplas/inference.py`

Visualizes what the learned operator *does* in pixel space. Loads a checkpoint,
builds a 10×10 grid of MNIST test images (each column one digit class), runs
`encode → project → trans_op → inverse-project → decode`, and saves PNGs:

```bash
python hoplas/inference.py checkpoints/mnist_filmr_expm.pt
```

Outputs three grids: `mnist_input.png` (raw images), `mnist_recon.png` (VAE
round-trip with no operator — a decoder-quality baseline), and
`mnist_transformed.png` (the full pipeline). If training succeeded, column *c*
(all digit *c*) should decode to digit *(c+1) mod 10* — i.e. the latent operator
advances each digit to the next one. The core logic is exposed as
`run_demo(...)` / `apply_operation(...)` for reuse in notebooks or a Gradio app.

### 4. Knowledge-graph embedding — `train_kge.py`

Applies the same operator families to **link prediction** on standard
knowledge-graph benchmarks (`WN18RR` by default; also `WN18`, `FB15k`,
`FB15k237` via `--dataset`). Each entity gets a learned `nd`-d embedding and each
relation *r* gets one of the operators above as a transform `op_r`; the model
scores a triple `(h, r, t)` by how close `op_r(E[h])` lands to `E[t]`. The
dataset adds an inverse relation for every relation (so `r`'s inverse is `r+R`),
which the multi-hop and inverse evals below rely on.

Training reuses the ring task's negative-free recipe and adds an optional light
contrastive term:

- a **similarity** loss pulls `op_r(E[h])` toward `E[t]` (`--sim {mse,cos}`),
- **SIGReg** (`--lambd`) spreads the embedding distribution (anti-collapse),
- an optional **in-batch cosine-InfoNCE** contrastive term (`--lambda-neg`,
  `--neg-temp`) — the lever that sharpens Hits@1,
- optional **MomMatch** (`--lambda-mom`) and explicit **inverse-consistency**
  (`--lambda-inv`, an MSE that drives `op_{r⁻¹}(op_r(E[h])) → E[h]`).

```bash
# the WN18RR champion recipe (beats published QuatE on every metric)
./train_kge.py --dataset WN18RR --op ph --order 2 --nd 512 \
    --lambd 0.10 --lambda-neg 0.20 --neg-temp 0.05 \
    --batch-size 8192 --score cos --epochs 300 --lr 0.01 --tag champ

# via a config file (configs/kge_*.cfg; CLI overrides)
./train_kge.py --config configs/kge_ph_neg20_bs8192.cfg
```

Evaluation is **filtered ranking** (all known-true tails removed before ranking);
the final line reports `TEST[l2|dot|cos]` MRR / MR / Hits@{1,3,10} so you can
compare score functions without retraining. Notable args: `--op` (any operator,
plus `trans` for a TransE-style translation baseline), `--order` (hypercomplex
order, `nd` must be divisible by it), `--score {l2,dot,cos}`, `--apply
{loop,vec,check}` (`vec` is the vectorized relation-apply, ~8× faster and needed
for many-relation graphs; `check` asserts it matches the loop), `--lambd`,
`--lambda-neg`, `--neg-temp`, `--lambda-inv`, `--scheduler {none,warmup,onecycle}`,
`--seed`, `--tag`. The best-validation checkpoint is saved to
`checkpoints/<dataset>_<op>_<order>_nd<nd>_lambd<lambd>_<tag>_best.pt` (entity
embeddings, all relation ops, and the CLI args). W&B project defaults to
`hoplas-kge-<dataset>`.

For unattended sweeps on a remote GPU box, `scripts/launch_queue.sh <host> --par
N --gpu ID configs/kge_*.cfg` keeps *N* runs in flight; `scripts/results.sh` and
`scripts/scores.sh` harvest the metric tables.

### 5. Composability evaluation — `eval_kge.py`

Offline (no training) probe of whether the *learned relation operators compose
algebraically*. Loads a `train_kge.py` checkpoint, rebuilds the model, and runs
four tests:

- **`hop`** — multi-hop path queries (`op_{r_k}∘…∘op_{r_1}(E[h])` ranked against
  true path endpoints) for `k = 1 … --max-k`; measures how composition degrades
  with chain length.
- **`inv`** — inverse round-trip: does `op_{r⁻¹}(op_r(E[h]))` recover `h`?
- **`sym`** — involution on auto-detected symmetric relations: does `op_r²(E[h])`
  return to `h`?
- **`alg`** — operator-algebra structure: commutator norms and identity distance
  on probe vectors.

```bash
# all four tests on a checkpoint
python eval_kge.py checkpoints/WN18RR_ph_2_nd512_lambd0.1_champ_best.pt --score cos

# just the path-query test, more hops
python eval_kge.py checkpoints/<ckpt>.pt --tests hop --max-k 4 --n-queries 2000
```

Args: `--tests {all,hop,inv,sym,alg}` (comma-separated subset), `--max-k`,
`--n-queries`, `--score {l2,dot,cos}` (defaults to the checkpoint's training
score), `--device`. Test 0 is a 1-hop sanity check that must reproduce the
training `TEST` MRR — if it doesn't, the path-query answer sets are leaking the
training split. To run it on a remote host against a checkpoint there:
`scripts/launch_eval_kge.sh <host> <checkpoint_filename> [extra eval args]`.

---

## Package layout

```
hoplas/
  filmr.py        FiLMR, FiLMR_expm, MatOp, MatOp2, rotation utilities
  ph_layers.py    PHMLinear (hypercomplex / parameterized Kronecker layer)
  ops.py          OpWrapper — builds an operator (optional residual + unit-norm)
  models.py       Projector (nd→nd MLP with optional residual + sphere norm)
  data.py         LineDataset, MNISTEncodingsDataset, CIFAR-10 loaders, KGTripleDataset
  losses.py       SIGReg (Epps–Pulley normality regularizer)
  vae.py          loader for the pretrained MNIST VAE (third-party, auto-fetched)
  viz.py          3-D embedding scatter for W&B
  inference.py    checkpoint → MNIST grid transform demo
train_simple_rot.py   supervised rotation benchmark
train_ops.py         self-supervised ring task (main experiment)
train_kge.py         knowledge-graph link prediction (WN18RR / WN18 / FB15k / FB15k-237)
eval_kge.py          offline composability eval (multi-hop / inverse / involution / algebra)
run_simple_rot.sh     sweep + results table for the benchmark
configs/              config files for train runs (kge_*.cfg for the KGE experiments)
scripts/encode_mnist.py   one-time MNIST→latent encoding
scripts/launch_queue.sh   keep N training runs in flight on a remote host
scripts/launch_eval_kge.sh  run eval_kge.py against a remote checkpoint
scripts/results.sh, scores.sh   harvest metric tables from remote run logs
tests/                pytest sanity checks
```

## Testing

```bash
pytest          # after: pip install -e ".[dev]"
```

## Acknowledgements

- **OpLaS** prior work: <https://drscotthawley.github.io/oplas/>
- `PHMLinear` from Grassucci et al.,
  [HyperNets](https://github.com/eleGAN23/HyperNets).
- MNIST VAE by Marco Cassar (Ocramaru/dl_experimentation), fetched at runtime.
- CIFAR-10 embedding loaders contributed by Emanuele "Manu" Rucci.

## License

See [LICENSE](LICENSE).
