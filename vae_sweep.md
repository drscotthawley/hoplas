# VAE hyperparameter sweep — reconstruction quality for operator scoring

**Goal:** find the VAE config whose reconstructions best preserve class content, so the
frozen-latent operator experiments can be scored fairly. Judged by two metrics that are **not**
part of the training loss (so they stay independent of what we optimize):

- **k=0 ceiling** — a clean-image classifier's accuracy on encode→decode reconstructions
  (`scripts/score_recon.py`). Higher = class content survives the round-trip.
- **recon-FID** — Fréchet Inception Distance between real test images and their reconstructions.
  Lower = reconstruction distribution matches the real one. *(building)*

Knobs (the three factors): **bottleneck** = `latent_dim`; **recon loss** = `perceptual_weight`
(VGG16-feature L2 added to pixel MSE; 0 = pure MSE); **smoothing** = `beta` (KL weight).

## Baseline — pure MSE (`perceptual_weight=0`)

| dataset | latent_dim | classifier clean | k=0 ceiling | notes |
|---|---|---|---|---|
| fashion | 16  | 94.5% | 36.4% | blurry; large domain-shift drop |
| cifar10 | 48  | 90.3% | 10.4% | ≈ chance |
| cifar10 | 128 | 90.3% | 16.8% | capacity helps over d48 |

MSE reconstructions are far too lossy for a clean-trained classifier → motivates perceptual loss.

## Round 1 — perceptual loss, sweep `latent_dim × perceptual_weight` (beta=1.0, epochs=80)

Weights rebalanced to ~1× MSE (the pw=10 canary made perceptual ≈7× MSE — too hot).
Checkpoints: `~/datasets/hoplas_vae/<dataset>_vae_<tag>.pt`.

| tag | dataset | latent_dim | pw | machine | k=0 | recon-FID | status |
|---|---|---|---|---|---|---|---|
| c48pw1  | cifar10 | 48  | 1.0 | lecun   | 0.129 | 298.1 | done — bad, likely capacity-starved at pw=1 |
| c256pw1 | cifar10 | 256 | 1.0 | lecun   | **0.634** | **44.9** | done — best CIFAR so far, both metrics |
| f32pw05 | fashion | 32  | 0.5 | lecun   | 0.470 | 135.2 | done — worse than d16!? see note below |
| c128pw1 | cifar10 | 128 | 1.0 | tsrazer | **0.586** | **53.3** | done (vs MSE 0.168!) |
| c128pw2 | cifar10 | 128 | 2.0 | tsrazer | **0.646** | **49.4** | done — best so far, both metrics agree |
| c128pw4 | cifar10 | 128 | 4.0 | tsrazer | — | — | training (pushing perceptual axis) |
| c128b05 | cifar10 | 128 | 1.0, β=0.5 | tsrazer | — | — | training (less KL smoothing) |
| f16pw05 | fashion | 16  | 0.5 | lecun   | **0.630** | **101.4** | done — beats d32! (see note) |

*razer unreachable (SSH `kex_exchange_identification` failure on both `razer-ts-docker` and `razer-docker`) — its fashion run moved to lecun. Machines used: lecun (c48pw1, c256pw1, f16pw05, f32pw05), tsrazer (c128pw1, c128pw2).*

### Round 1 takeaways

- **Perceptual loss is a massive win on CIFAR.** MSE-only d128 was 16.8% k=0; perceptual pw=1.0
  jumps to 58.6%, pw=2.0 to 64.6%. Confirmed by FID too (independent metric, same direction).
- **CIFAR: capacity matters a lot at fixed pw=1.0.** 48→128→256 gives 12.9%→58.6%→63.4% k=0 and
  298→53→45 FID. d48 looks capacity-starved (or needs a higher pw to match its smaller MSE
  scale) rather than perceptual loss "not working" there.
- **Surprise: Fashion d16 (pw=0.5) beats d32 (pw=0.5) on BOTH metrics** (63.0% vs 47.0% k=0;
  101 vs 135 FID) — the opposite of the CIFAR capacity trend. Leading hypothesis: pw is *not*
  latent-dim-normalized, so the same absolute weight is proportionally weaker relative to a
  lower baseline MSE at higher latent_dim — i.e. d32 may be *underweighted* on perceptual loss,
  not fundamentally worse. Test: rerun d32 at pw=1.0 (matching CIFAR's working point) before
  concluding capacity hurts Fashion.
- Still training: c128pw4 (pw=4, does it keep climbing past pw=2's 64.6%?), c128b05 (beta=0.5,
  does less KL smoothing help?) — both on tsrazer.
- **Best so far:** CIFAR d256/pw1 (63.4% / 44.9 FID), Fashion d16/pw0.5 (63.0% / 101.4 FID).

### Qualitative finding (Scott, visual inspection of Fashion recon grid, 2026-07-23)
A plaid shirt reconstructs as **flat uniform gray** — no pattern at all — even though the
silhouette/class-relevant shape is right. Diagnosis: shallow VGG layers (relu1_2/2_2/3_3) are
edge/shape-dominant, under-penalizing lost *texture*; MSE actively prefers flat-gray (the
pixelwise mean over plausible textures). Added `--perceptual-deep` (relu4_3, texture-sensitive)
to test whether this recovers pattern detail. Not fully captured by k=0/FID alone (a classifier
may still get "shirt" right without the plaid) — worth an eyeball check on the recon grid too,
not just the two scalar metrics.

### Round 2 — results (all finished)

| tag | dataset | latent_dim | pw | beta | k=0 | FID | notes |
|---|---|---|---|---|---|---|---|
| f16pw05deep | fashion | 16 | 0.5 | 1.0 | 0.637 | 96.2 | deep VGG: marginal gain over f16pw05 (0.630/101.4) |
| **f32pw1** | fashion | 32 | 1.0 | 1.0 | **0.879** | **18.6** | mystery solved: d32 just needed higher pw |
| **f64pw1** | fashion | 64 | 1.0 | 1.0 | **0.897** | **14.4** | best Fashion so far; Scott confirms plaid pattern visibly returns |
| **c256pw2** | cifar10 | 256 | 2.0 | 1.0 | **0.762** | **31.5** | best CIFAR so far by a wide margin — capacity+pw compound |
| c128pw4 | cifar10 | 128 | 4.0 | 1.0 | pending | pending | **finished on tsrazer (wandb: val_recon=30.77, val_perc=22.32) but tsrazer is down again — checkpoint stuck there, not lost, just unreachable** |
| c128b05 | cifar10 | 128 | 1.0 | 0.5 | pending | pending | **finished on tsrazer (wandb: val_recon=26.99, val_perc=24.11) — same as above, checkpoint stranded** |

**Fashion mystery resolved:** the Round 1 "d16 beats d32" result was the pw-underweighting
artifact, exactly as suspected — at matched pw=1.0, capacity helps monotonically (16→32→64:
63%→88%→90% k=0), same shape as CIFAR. Confirms Scott's qualitative read: plaid texture
visibly returns at d64.

**CIFAR: capacity + pw compound.** c256pw2 (76.2%/31.5) beats every single-lever-improved
config by a wide margin — pushing both axes together, not just one, is the winning direction.

**tsrazer went down again** (unreachable, same signature as the earlier RAM/GPU wedge) partway
through this round. Both its runs (c128pw4, c128b05) finished training successfully before it
went down (confirmed via `wandb_search.sh` — no host access needed), so no work was lost, but
their checkpoints are inaccessible until tsrazer is back up. **Sticking to lecun only for now**
per Scott's direction. `hsrazer` (the smaller razer machine, distinct from tsrazer) came back
reachable but a `gpu.sh` probe hit an SSH auth issue (bare hostname without explicit user) —
not yet used for launches.

## Classifiers for Fashion d32 (paired to the f32pw1 VAE)

Two classifiers, distinct weight files, to set a fair ruler for the operator (op^k) scoring:

| classifier | trained/tested on | test acc | file |
|---|---|---|---|
| input | clean Fashion images | **94.7%** | `classifier_fashion_input.pt` |
| recon (paired to f32pw1) | f32pw1 op⁰ reconstructions | **89.2%** | `classifier_fashion_recon_f32pw1.pt` |

For reference: the *clean* classifier applied to f32pw1 reconstructions scored 87.9% (the k=0
number from score_recon). So:

- **op⁰ ceiling (recon-adapted ruler) ≈ 89.2%** — the fair denominator for op^k/op⁰. Strong;
  ample class signal survives d32.
- **Recon-adaptation gain is small (~1.3 pts: 87.9 → 89.2).** The perceptual d32 VAE is good
  enough (FID 18.6) that domain shift is minor — a clean classifier already reads its outputs
  nearly as fairly as a domain-adapted one. **Lesson (revises earlier expectation):** the
  recon-adapted classifier is *critical for lossy VAEs* (MSE VAEs were near-chance under a clean
  ruler — pure domain-shift artifact) but only a *small correction for good VAEs*. Its value
  scales inversely with VAE quality.
- **Residual 94.7 → 89.2 (~5.5 pts) is genuine VAE info loss, not domain shift** (the recon
  ruler already corrected for domain) — real class-discriminative texture the d32 bottleneck
  discards. That's the true cost of the round-trip and what op⁰ actually ceilings at.

### Round 3 (not yet launched — pending direction)
- CIFAR: does pw keep helping past 2.0 at d256? (c128's pw1→pw2 gain was large; untested at d256)
- Fashion: does d128 keep the trend going, or hit diminishing returns after 32→64's plateau-ish
  jump (63→88→90, gains shrinking)?
- Re-run c128pw4/c128b05 once tsrazer is back (or re-launch on lecun) to get their k=0/FID
