# hoplas
hypercomplex operational latent spaces 

## Simple rotation benchmark

Supervised learning of a single-plane rotation in nd=64 (see `train_simple_rot.py`).
Final MSE loss after training, by method:

| run        | final loss |
|------------|-----------:|
| filmr      |   0.001190 |
| filmr_expm |   0.000001 |
| matop      |   0.000000 |
| matop2     |   0.000000 |
| ph_2       |   0.054875 |
| ph_4       |   0.068795 |
| ph_8       |   0.061082 |
| ph_16      |   0.000000 |

The `ph_N` rows are PHMLinear at hypercomplex order N. Orders below
nd^(2/3)=16 are genuinely constrained algebras (non-zero structural floor),
while order 16 saturates the Kronecker-sum parameterization and can represent
any matrix, so it solves the rotation exactly.

See graphs: https://wandb.ai/drscotthawley/simple%20rot

