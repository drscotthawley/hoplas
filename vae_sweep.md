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
| c48pw1  | cifar10 | 48  | 1.0 | lecun   | — | — | launched |
| c256pw1 | cifar10 | 256 | 1.0 | lecun   | — | — | launched |
| f32pw05 | fashion | 32  | 0.5 | lecun   | — | — | launched |
| c128pw1 | cifar10 | 128 | 1.0 | tsrazer | — | — | trained → on lecun, ready to eval |
| c128pw2 | cifar10 | 128 | 2.0 | tsrazer | — | — | trained → on lecun, ready to eval |
| c128pw4 | cifar10 | 128 | 4.0 | tsrazer | — | — | training (pushing perceptual axis) |
| c128b05 | cifar10 | 128 | 1.0, β=0.5 | tsrazer | — | — | training (less KL smoothing) |
| f16pw05 | fashion | 16  | 0.5 | lecun   | — | — | training |

*razer unreachable (SSH `kex_exchange_identification` failure on both `razer-ts-docker` and `razer-docker`) — its fashion run moved to lecun. Machines used: lecun (c48pw1, c256pw1, f16pw05, f32pw05), tsrazer (c128pw1, c128pw2).*

*(Results filled in as runs finish; then Round 2 refines toward the winning region + varies beta.)*
