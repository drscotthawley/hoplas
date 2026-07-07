"""KGE relation model + filtered-ranking evaluation, shared by train_kge.py (training)
and eval_kge.py (offline composability eval).

Single source of truth for both the model and the ranking metrics, so the training-time
validation MRR and the offline evaluator's numbers are identical by construction (they run
the exact same score_matrix / filtered_ranks / mrr_hits code, not two look-alike copies).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from hoplas.ops import OpWrapper, freeze_quaternion, init_quaternion
from hoplas.models import Projector


class KGEModel(nn.Module):
    """Learnable entity embeddings + one relation operator per relation id."""

    def __init__(self, num_entities, num_relations, nd, op, order,
                 op_resid=True, rank=2, unit_norm=False, quat_init=False,
                 use_proj=False, pnd=None, proj_n_hid=None, proj_layers=3, proj_resid=False):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, nd)
        nn.init.normal_(self.entity_emb.weight, std=1.0)  # ~N(0,1); SIGReg keeps Cov ~ I

        # The op acts in the projected space (pd) iff --projector, else directly in entity
        # space (nd). pd must be known here, BEFORE the ops are built with it as their dim.
        self.use_proj = use_proj
        pd = (pnd or nd) if use_proj else nd

        self.ops = nn.ModuleList([
            OpWrapper(op, pd, order, op_resid, rank, unit_norm) for _ in range(num_relations)
        ])
        if op == "quat":
            for o in self.ops:
                freeze_quaternion(o.op)          # frozen exact quaternion (raw baseline)
        elif op == "ph" and quat_init:
            for o in self.ops:
                init_quaternion(o.op)            # learnable, warm-started at exact quaternion

        # (Note: projector doesn't help, so you can mostly ignore this next section)
        # Optional projector + inverse projector around the relation op (a nonlinear "lift", as in
        # train_ops): entity -> proj -> op -> inv_proj. The op then acts in the projected (pnd)
        # space, but SIGReg + eval scoring stay in the ORIGINAL entity space (predict() maps back
        # through the inverse projector), so ranking is unchanged.
        if use_proj:
            nh = proj_n_hid or nd
            self.proj = Projector(nd, pd, n_hid=nh, n_layers=proj_layers, proj_resid=proj_resid, unit_norm=unit_norm)
            self.inv_proj = Projector(pd, nd, n_hid=nh, n_layers=proj_layers, proj_resid=proj_resid, unit_norm=False)
            # Near-identity init: shrink each projector's output layer so that, with the residual
            # skip (valid when pnd == nd), proj ~= inv_proj ~= identity at the start. Paired with
            # --proj-freeze-epochs this lets the plain KGE train first, then the projectors unfreeze
            # and depart from identity (instead of a random projector wrecking early training).
            for pr in (self.proj, self.inv_proj):
                nn.init.normal_(pr.out_proj.weight, std=1e-4)
                nn.init.zeros_(pr.out_proj.bias)

    def _apply_relation_loop(self, h_emb, r):
        """Reference: loop over the (<=Nr) relations present, apply each op to its rows.
        Correct for any op type, but O(#distinct relations) Python iterations per batch
        — the bottleneck on many-relation datasets (FB15k-237: 474, FB15k: 2690)."""
        out = h_emb.new_empty(h_emb.shape)
        for rid in r.unique().tolist():
            m = r == rid
            out[m] = self.ops[rid](h_emb[m])
        return out

    def _phm_stack_ok(self):
        """True iff every relation op is a PHMLinear (ph/quat): has a, s, bias, n."""
        return all(hasattr(o.op, "a") and hasattr(o.op, "s") and hasattr(o.op, "bias")
                   and hasattr(o.op, "n") for o in self.ops)

    def _apply_relation_vec(self, h_emb, r, chunk=2048):
        """Vectorized equivalent of the loop for PHMLinear ops, via the implicit-einsum
        math (PHMLinear_Implicit: no per-relation weight materialization). Stacks the
        per-relation algebra a (Nr,n,n,n), block weights s (Nr,n,do,di) and bias (Nr,nd),
        then for each BATCH CHUNK gathers per-sample by relation id and contracts in a
        memory-frugal order. Chunking + the explicit two-step contraction keep memory
        bounded (the naive single fused einsum blew up to ~18GB at nd512/bs8192).
        Handles op_resid / unit_norm uniformly from the (identical) OpWrappers."""
        op0 = self.ops[0]
        n = op0.op.n
        A = torch.stack([o.op.a for o in self.ops])      # (Nr, n, n, n)  [i, a, b]
        S = torch.stack([o.op.s for o in self.ops])      # (Nr, n, do, di) [i, j, k]
        Bk = torch.stack([o.op.bias for o in self.ops])  # (Nr, nd)
        B = h_emb.shape[0]
        out = h_emb.new_empty(B, h_emb.shape[1])
        for lo in range(0, B, chunk):
            sl = slice(lo, lo + chunk)
            hc, rc = h_emb[sl], r[sl]
            a_r, s_r, b_r = A[rc], S[rc], Bk[rc]          # gather per sample (this chunk)
            X = hc.reshape(hc.shape[0], n, -1)            # (p, b, k) = (p, n, di)
            T = torch.einsum("pijk,pbk->pijb", s_r, X)    # contract k -> (p, i, j, b) small
            Y = torch.einsum("piab,pijb->paj", a_r, T)    # contract i,b -> (p, a, j)=(p,n,do)
            opx = Y.reshape(hc.shape[0], -1) + b_r        # == PHMLinear(hc), per row
            oc = hc + opx if op0.op_resid else opx
            out[sl] = F.normalize(oc, dim=-1) if op0.unit_norm else oc
        return out

    def apply_relation(self, h_emb, r):
        """Apply each sample's relation operator. r: (B,) relation ids.
        apply_mode (set by --apply): 'loop' (default), 'vec' (fast, PHM only), or
        'check' (run both and assert they match — for verifying the vectorization)."""
        mode = getattr(self, "apply_mode", "loop")
        if mode in ("vec", "check") and self._phm_stack_ok():
            vec = self._apply_relation_vec(h_emb, r)
            if mode == "vec":
                return vec
            ref = self._apply_relation_loop(h_emb, r)
            d = (ref - vec).abs().max().item()
            print(f"[apply check] max|loop-vec|={d:.3e}", flush=True)
            assert d < 1e-3, f"vectorized apply_relation mismatch: {d}"
            return ref
        return self._apply_relation_loop(h_emb, r)

    def project(self, e):
        """Entity (nd) -> projected space: proj(e) (pnd) iff self.use_proj, else e unchanged (no-op)."""
        return self.proj(e) if self.use_proj else e

    def inv_project(self, z):
        """Projected space -> original entity space (nd): inv_proj iff self.use_proj, else unchanged (no-op)."""
        return self.inv_proj(z) if self.use_proj else z

    def predict(self, h, r):
        """Original-space relation prediction for scoring/eval: project E[h], transform, then map
        back through the inverse projector, so ranking stays in the entity space either way."""
        return self.inv_project(self.apply_relation(self.project(self.entity_emb(h)), r))

    def forward(self, h, r, t):
        return self.apply_relation(self.entity_emb(h), r), self.entity_emb(t)


# ---------------------------------------------------------------------------
# Filtered-ranking evaluation primitives (shared by evaluate() and eval_kge tests)
# ---------------------------------------------------------------------------

def score_matrix(model, pred, score="l2"):
    """(B, Ne) similarity of each `pred` row against every entity embedding.
      l2  -> -||pred - E||^2 (the original distance score)
      dot -> pred . E         (bilinear; drops the spurious -||E||^2 per-entity term)
      cos -> cosine(pred, E)  (dot, with pred and E row-normalized)."""
    E = model.entity_emb.weight
    if score == "dot":
        return pred @ E.t()
    elif score == "cos":
        return F.normalize(pred, dim=-1) @ F.normalize(E, dim=-1).t()
    else:  # "l2": -||pred - E||^2 via the matmul expansion (no (B, Ne, nd) tensor)
        E_sq = (E ** 2).sum(-1)
        return -((pred ** 2).sum(-1, keepdim=True) - 2 * pred @ E.t() + E_sq.unsqueeze(0))


def filtered_ranks(scores, targets, filter_sets, device):
    """1-based filtered ranks (optimistic-free, strict-greater).
    scores: (B, Ne); targets: (B,) true entity per row; filter_sets: B iterables of
    known-true entities to mask to -inf before ranking (the target itself is kept)."""
    ranks = []
    for i in range(scores.size(0)):
        tgt = int(targets[i])
        s = scores[i]
        others = [x for x in filter_sets[i] if x != tgt]
        if others:
            s = s.clone()
            s[torch.tensor(others, device=device)] = float("-inf")
        ranks.append(1 + int((s > s[tgt]).sum().item()))
    return ranks


def mrr_hits(ranks):
    """Filtered MRR / MR / Hits@{1,3,10} from a list of 1-based integer ranks."""
    t = torch.tensor(ranks, dtype=torch.float)
    return dict(mrr=(1.0 / t).mean().item(), mr=t.mean().item(),
                h1=(t <= 1).float().mean().item(),
                h3=(t <= 3).float().mean().item(),
                h10=(t <= 10).float().mean().item())


@torch.no_grad()
def evaluate(model, eval_ds, hr2t, device, batch=512, score="l2"):
    """Filtered MRR / Hits@k over eval_ds (which includes inverse triples), built on the
    shared score_matrix / filtered_ranks / mrr_hits primitives so the numbers match the
    offline evaluator exactly. `score` selects the ranking function (l2 / dot / cos)."""
    model.eval()
    triples = eval_ds.triples
    ranks = []
    for i in range(0, len(triples), batch):
        chunk = triples[i:i + batch].to(device)
        h, r, t = chunk[:, 0], chunk[:, 1], chunk[:, 2]
        pred = model.predict(h, r)                                   # (B, nd)
        scores = score_matrix(model, pred, score)                   # (B, Ne)
        filt = [hr2t[(int(h[b]), int(r[b]))] for b in range(chunk.size(0))]
        ranks.extend(filtered_ranks(scores, t, filt, device))
    model.train()
    return mrr_hits(ranks)
