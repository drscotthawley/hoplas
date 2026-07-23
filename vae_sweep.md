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

### Round 2 (planned, pending Round 1 stragglers)
- Fashion: d32 @ pw=1.0 and pw=2.0 (test the underweighting hypothesis)
- CIFAR: d256 @ pw=2.0 (does the pw1→pw2 gain at d128 transfer to d256?)
- Once c128pw4/c128b05 land: extend whichever axis (pw or beta) is still improving
