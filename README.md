# hoplas
hypercomplex operational latent spaces 

## Simple rotation benchmark

Supervised learning of a single-plane rotation in nd=64 (see `train_simple_rot.py`).
Final MSE loss after training, by method, sweeping over `corr_nd` — the number
of input channels sharing an inter-channel correlation (`corr=0.9`):

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

The `ph_N` rows are PHMLinear at hypercomplex order N. Orders below
nd^(2/3)=16 are genuinely constrained algebras (non-zero structural floor),
while order 16 saturates the Kronecker-sum parameterization and can represent
any matrix, so it solves the rotation exactly. The constrained ph orders (2/4/8)
improve as more input channels are correlated (loss drops left-to-right).

See graphs: https://wandb.ai/drscotthawley/simple%20rot

