#!/usr/bin/env python3
"""Offline composability evaluator for KGE checkpoints.

Tests whether learned relation operators compose gracefully across hops
(Guu et al. 2015 "Traversing KGs in Vector Space" path-query framing),
inverse round-trips, involution of symmetric relations, and raw
operator-algebra structure (commutator norms).

The k=1 path-query result MUST reproduce the run's TEST MRR (sanity check).

Usage:
    python eval_kge.py <checkpoint.pt> \\
        [--tests all|hop|inv|sym|alg] \\
        [--max-k 3] [--n-queries 2000] [--score cos|l2|dot] [--device cuda]
"""
import argparse
import random
from collections import defaultdict

import torch
import torch.nn.functional as F

from hoplas.data import KGTripleDataset
from hoplas.ops import OpWrapper
from train_kge import KGEModel, evaluate


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device)
    saved_args = ck["args"]  # dict stored by train_kge.save_ckpt
    a = argparse.Namespace(**saved_args)
    # Rebuild datasets using the checkpoint's dataset name
    train_ds = KGTripleDataset(a.dataset, "train", create_inverse=True)
    valid_ds = KGTripleDataset(a.dataset, "valid", create_inverse=True)
    test_ds  = KGTripleDataset(a.dataset, "test",  create_inverse=True)
    # Build model skeleton and load weights
    model = KGEModel(
        train_ds.num_entities, train_ds.num_relations, a.nd, a.op,
        a.order, a.op_resid, getattr(a, "rank", 2),
        getattr(a, "unit_norm", False), quat_init=False,
    ).to(device)
    model.entity_emb.weight.data.copy_(ck["entity_emb"].to(device))
    model.ops.load_state_dict({k: v.to(device) for k, v in ck["ops"].items()})
    model.eval()
    print(f"Loaded checkpoint '{path}'  dataset={a.dataset}  op={a.op}  nd={a.nd}  "
          f"order={getattr(a,'order','-')}  epoch={ck.get('epoch','?')}")
    return model, a, train_ds, valid_ds, test_ds


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def build_adj(datasets):
    """(h, r) -> set(t) over all given datasets (for path traversal)."""
    adj = defaultdict(set)
    for ds in datasets:
        for h, r, t in ds.triples.tolist():
            adj[(h, r)].add(t)
    return adj


def build_hr2t(datasets):
    """(h, r) -> set(t) for filtered ranking."""
    return build_adj(datasets)  # same structure


def score_matrix(model, pred, score_mode, device):
    """Return (B, Ne) score matrix for `pred` against all entity embeddings."""
    E = model.entity_emb.weight
    if score_mode == "dot":
        return pred @ E.t()
    elif score_mode == "cos":
        pred_n = F.normalize(pred, dim=-1)
        E_n = F.normalize(E, dim=-1)
        return pred_n @ E_n.t()
    else:  # l2: -||pred - E||^2
        E_sq = (E ** 2).sum(-1)
        return -((pred ** 2).sum(-1, keepdim=True) - 2 * pred @ E.t() + E_sq.unsqueeze(0))


def filtered_ranks(scores, targets, filter_sets, device):
    """
    scores: (B, Ne) float
    targets: (B,) int — the true target entity for each row
    filter_sets: list of B sets (each = known-true entities for that query)
    Returns (B,) integer ranks (1-based, optimistic-free strict-greater).
    """
    ranks = []
    for i in range(scores.size(0)):
        tgt = targets[i].item()
        s = scores[i].clone()
        others = [x for x in filter_sets[i] if x != tgt]
        if others:
            s[torch.tensor(others, device=device)] = float("-inf")
        ranks.append(1 + int((s > s[tgt]).sum().item()))
    return ranks


def mrr_hits(ranks):
    t = torch.tensor(ranks, dtype=torch.float)
    return dict(mrr=(1.0 / t).mean().item(), mr=t.mean().item(),
                h1=(t <= 1).float().mean().item(),
                h3=(t <= 3).float().mean().item(),
                h10=(t <= 10).float().mean().item())


# ---------------------------------------------------------------------------
# Test 1: Multi-hop path queries
# ---------------------------------------------------------------------------

def _sample_path_queries_fast(adj, all_rel_ids, k, n_queries=2000, seed=42,
                              test_pairs=None, test_hr2t=None):
    """Sample k-hop path queries. Returns (h, rels, targets, filt) tuples.

    Starting (h, r1) pairs are drawn from distinct test (h, r) pairs so the eval
    is not contaminated by training examples. Path extensions for k≥2 use adj
    (all splits) to traverse the full known graph (Guu et al. path-query task).

    targets = the entities whose rank we measure:
      k=1   -> the held-out TEST tails of (h, r1) only (so k=1 reproduces the
               filtered TEST MRR; train/valid tails are NOT scored).
      k>=2  -> the full set of graph-reachable path endpoints.
    filt = all known positives to mask out when ranking a target (the full
      reachable set across splits, the standard filtered setting)."""
    rng = random.Random(seed)
    if test_pairs is None or len(test_pairs) == 0:
        return []
    # Distinct test (h, r) starting pairs that have at least one neighbour in adj.
    valid_pairs = sorted({(h, r) for h, r in test_pairs if adj.get((h, r))})
    if not valid_pairs:
        return []
    rng.shuffle(valid_pairs)
    queries = []
    attempts = 0
    max_attempts = max(n_queries, len(valid_pairs)) * 30
    pi = 0
    while len(queries) < n_queries and attempts < max_attempts:
        attempts += 1
        h, r1 = valid_pairs[pi % len(valid_pairs)]
        pi += 1
        cur = set(adj[(h, r1)])
        rels = [r1]
        valid = True
        for _ in range(k - 1):
            r_next = rng.choice(all_rel_ids)
            rels.append(r_next)
            nxt = set()
            for m in cur:
                nxt.update(adj.get((m, r_next), set()))
            if not nxt:
                valid = False
                break
            cur = nxt
        if not (valid and cur):
            continue
        if k == 1:
            targets = test_hr2t.get((h, r1), set()) if test_hr2t else set()
            if not targets:
                continue
            queries.append((h, rels, targets, cur))   # filter = all known tails
        else:
            queries.append((h, rels, cur, cur))        # endpoints = filter set
    return queries


@torch.no_grad()
def test_hop(model, adj, hr2t, all_rel_ids, test_ds, device, score_mode, max_k=3, n_queries=2000):
    """Path-query MRR/Hits@k for k=1..max_k.
    Starting (h,r) pairs are sampled from test triples only; path extensions
    use all-split adj. k=1 should reproduce TEST MRR as a sanity check."""
    # Build test (h, r) pairs and the test-split (h,r)->tails map (base relations only, r < R).
    R = test_ds.num_base_relations
    test_pairs = []
    test_hr2t = defaultdict(set)
    for h, r, t in test_ds.triples.tolist():
        if r < R:
            test_pairs.append((h, r))
            test_hr2t[(h, r)].add(t)
    results = {}
    for k in range(1, max_k + 1):
        queries = _sample_path_queries_fast(adj, all_rel_ids, k, n_queries=n_queries,
                                            test_pairs=test_pairs, test_hr2t=test_hr2t)
        if not queries:
            print(f"  hop k={k}: no queries found, skipping")
            continue
        all_ranks = []
        batch = 256
        qs = queries
        for i in range(0, len(qs), batch):
            chunk = qs[i:i + batch]
            h_ids = torch.tensor([q[0] for q in chunk], dtype=torch.long, device=device)
            pred = model.entity_emb(h_ids)  # start: E[h]
            # apply each hop's relation operator
            rel_seq_len = len(chunk[0][1])
            for hop_idx in range(rel_seq_len):
                r_ids = torch.tensor([q[1][hop_idx] for q in chunk], dtype=torch.long, device=device)
                pred = model.apply_relation(pred, r_ids)
            scores = score_matrix(model, pred, score_mode, device)
            # Rank each target (k=1: held-out test tails; k>=2: path endpoints),
            # filtering the other known positives out (standard filtered setting).
            for b_idx, q in enumerate(chunk):
                targets, filt = q[2], q[3]
                for tgt in targets:
                    s = scores[b_idx].clone()
                    others = [x for x in filt if x != tgt]
                    if others:
                        s[torch.tensor(others, device=device)] = float("-inf")
                    all_ranks.append(1 + int((s > s[tgt]).sum().item()))
        m = mrr_hits(all_ranks)
        results[k] = m
        print(f"  hop k={k}  n_queries={len(queries)}  MRR={m['mrr']:.4f}  "
              f"H@1={m['h1']:.4f}  H@3={m['h3']:.4f}  H@10={m['h10']:.4f}  MR={m['mr']:.1f}")
    return results


# ---------------------------------------------------------------------------
# Test 2: Inverse round-trip
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_inverse(model, test_ds, hr2t, device, score_mode, n_queries=2000):
    """op_{r+R}(op_r(E[h])) ranked against E[h]. Tests op⁻¹∘op ≈ id."""
    R = test_ds.num_base_relations  # inverse rels start at index R
    triples = test_ds.triples  # (T, 3): h, r, t  (only base triples; r < R)
    base = triples[triples[:, 1] < R]  # filter to base relations
    n = min(n_queries, len(base))
    idx = torch.randperm(len(base))[:n]
    base = base[idx]

    all_ranks = []
    batch = 256
    for i in range(0, len(base), batch):
        chunk = base[i:i + batch].to(device)
        h, r, t = chunk[:, 0], chunk[:, 1], chunk[:, 2]
        # forward then inverse
        pred = model.apply_relation(model.entity_emb(h), r)
        r_inv = r + R
        pred2 = model.apply_relation(pred, r_inv)
        scores = score_matrix(model, pred2, score_mode, device)
        # rank h (the starting entity — should be recovered)
        for b in range(chunk.size(0)):
            h_tgt = h[b].item()
            s = scores[b]
            all_ranks.append(1 + int((s > s[h_tgt]).sum().item()))

    m = mrr_hits(all_ranks)
    # Also report round-trip embedding error: ||op_inv(op_r(E[h])) - E[h]|| / ||E[h]||
    with torch.no_grad():
        sample = base[:min(500, len(base))].to(device)
        h_s, r_s = sample[:, 0], sample[:, 1]
        e_h = model.entity_emb(h_s)
        p1 = model.apply_relation(e_h, r_s)
        p2 = model.apply_relation(p1, r_s + R)
        err = (p2 - e_h).norm(dim=-1) / (e_h.norm(dim=-1) + 1e-9)
        round_trip_err = err.mean().item()
    print(f"  inverse  n={len(base)}  MRR={m['mrr']:.4f}  H@1={m['h1']:.4f}  "
          f"H@10={m['h10']:.4f}  MR={m['mr']:.1f}  round_trip_err={round_trip_err:.4f}")
    return m, round_trip_err


# ---------------------------------------------------------------------------
# Test 3: Symmetric-relation involution
# ---------------------------------------------------------------------------

def find_symmetric_relations(datasets):
    """Relations r where (h,r,t) and (t,r,h) both exist (base relations only)."""
    R = datasets[0].num_base_relations
    pair_set = set()
    for ds in datasets:
        for h, r, t in ds.triples.tolist():
            if r < R:
                pair_set.add((h, r, t))
    sym_rels = set()
    for h, r, t in pair_set:
        if (t, r, h) in pair_set:
            sym_rels.add(r)
    return sym_rels


@torch.no_grad()
def test_involution(model, datasets, hr2t, device, score_mode, n_queries=2000):
    """For symmetric relation r: op_r²(E[h]) should rank h highly."""
    sym_rels = find_symmetric_relations(datasets)
    if not sym_rels:
        print("  involution: no symmetric relations found")
        return {}
    print(f"  involution: {len(sym_rels)} symmetric relations")
    # Collect test triples with symmetric relations (test-only, not train/valid)
    R = datasets[0].num_base_relations
    test_ds = datasets[2]
    sym_triples = []
    for h, r, t in test_ds.triples.tolist():
        if r < R and r in sym_rels:
            sym_triples.append((h, r, t))
    if not sym_triples:
        print("  involution: no triples with symmetric relations")
        return {}
    rng = random.Random(42)
    rng.shuffle(sym_triples)
    sym_triples = sym_triples[:n_queries]

    all_ranks = []
    invol_errors = []
    batch = 256
    for i in range(0, len(sym_triples), batch):
        chunk = sym_triples[i:i + batch]
        h_ids = torch.tensor([x[0] for x in chunk], dtype=torch.long, device=device)
        r_ids = torch.tensor([x[1] for x in chunk], dtype=torch.long, device=device)
        e_h = model.entity_emb(h_ids)
        p1 = model.apply_relation(e_h, r_ids)
        p2 = model.apply_relation(p1, r_ids)
        err = (p2 - e_h).norm(dim=-1) / (e_h.norm(dim=-1) + 1e-9)
        invol_errors.extend(err.tolist())
        scores = score_matrix(model, p2, score_mode, device)
        for b in range(len(chunk)):
            h_tgt = h_ids[b].item()
            s = scores[b]
            all_ranks.append(1 + int((s > s[h_tgt]).sum().item()))

    m = mrr_hits(all_ranks)
    invol_err = sum(invol_errors) / len(invol_errors) if invol_errors else float("nan")
    print(f"  involution  n={len(sym_triples)}  MRR={m['mrr']:.4f}  H@1={m['h1']:.4f}  "
          f"H@10={m['h10']:.4f}  MR={m['mr']:.1f}  invol_err={invol_err:.4f}")
    return m, invol_err


# ---------------------------------------------------------------------------
# Test 4: Operator-algebra structure (entity-free)
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_algebra_structure(model, device, n_probe=200, max_rel_pairs=50):
    """Commutator norms and identity-distance on probe vectors (sampled entity embeddings).

    commutator: ||op_a(op_b(z)) - op_b(op_a(z))||_2
    identity_dist: ||op_r(z) - z||_2 / ||z||_2  (how far from identity each op is)
    """
    E = model.entity_emb.weight.detach()
    n_ent = E.size(0)
    probe_idx = torch.randperm(n_ent, device=device)[:n_probe]
    Z = E[probe_idx]  # (n_probe, nd)

    Nr = len(model.ops)
    rel_ids = list(range(Nr))
    rng = random.Random(42)

    # Identity distance: sample all relations on all probes
    id_dists = []
    for r in rel_ids:
        r_t = torch.full((n_probe,), r, dtype=torch.long, device=device)
        opz = model.apply_relation(Z, r_t)
        d = (opz - Z).norm(dim=-1) / (Z.norm(dim=-1) + 1e-9)
        id_dists.append(d.mean().item())
    mean_id_dist = sum(id_dists) / len(id_dists)

    # Commutator: sample pairs of relations
    if Nr * Nr > max_rel_pairs:
        pairs = [(rng.randrange(Nr), rng.randrange(Nr)) for _ in range(max_rel_pairs)]
    else:
        pairs = [(a, b) for a in rel_ids for b in rel_ids]
    comm_norms = []
    for a, b in pairs:
        ra = torch.full((n_probe,), a, dtype=torch.long, device=device)
        rb = torch.full((n_probe,), b, dtype=torch.long, device=device)
        ab = model.apply_relation(model.apply_relation(Z, ra), rb)
        ba = model.apply_relation(model.apply_relation(Z, rb), ra)
        cn = (ab - ba).norm(dim=-1).mean().item()
        comm_norms.append(cn)
    mean_comm = sum(comm_norms) / len(comm_norms) if comm_norms else float("nan")

    print(f"  algebra  n_probe={n_probe}  mean_id_dist={mean_id_dist:.4f}  "
          f"mean_commutator={mean_comm:.4f}  (n_pairs={len(pairs)})")
    return dict(mean_id_dist=mean_id_dist, mean_commutator=mean_comm)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="KGE composability evaluation (eval-only, no training).")
    p.add_argument("checkpoint", help="path to a _best.pt or _final.pt checkpoint")
    p.add_argument("--tests", default="all",
                   help="comma-separated subset of {hop,inv,sym,alg} or 'all'")
    p.add_argument("--max-k", type=int, default=3, help="max hops for path-query test")
    p.add_argument("--n-queries", type=int, default=2000, help="queries per test/hop")
    p.add_argument("--score", choices=["l2", "dot", "cos"], default=None,
                   help="scoring function (default: from checkpoint args, else 'cos')")
    p.add_argument("--device", default=None,
                   help="compute device (default: cuda if available, else cpu)")
    args = p.parse_args()

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device={device}")

    model, ckpt_args, train_ds, valid_ds, test_ds = load_checkpoint(args.checkpoint, device)
    score_mode = args.score or getattr(ckpt_args, "score", "cos")
    print(f"score_mode={score_mode}")

    tests = {t.strip() for t in args.tests.split(",")} if args.tests != "all" else {"hop", "inv", "sym", "alg"}
    all_ds = [train_ds, valid_ds, test_ds]
    hr2t = build_hr2t(all_ds)
    adj = build_adj(all_ds)
    all_rel_ids = list(range(train_ds.num_relations))

    print("\n=== Test 0: 1-hop sanity check (must match TEST MRR from training log) ===")
    m1 = evaluate(model, test_ds, hr2t, device, score=score_mode)
    print(f"  1-hop eval  MRR={m1['mrr']:.4f}  H@1={m1['h1']:.4f}  "
          f"H@3={m1['h3']:.4f}  H@10={m1['h10']:.4f}  MR={m1['mr']:.1f}")

    if "hop" in tests:
        print(f"\n=== Test 1: Multi-hop path queries (k=1..{args.max_k}) ===")
        test_hop(model, adj, hr2t, all_rel_ids, test_ds, device, score_mode,
                 max_k=args.max_k, n_queries=args.n_queries)

    if "inv" in tests:
        print("\n=== Test 2: Inverse round-trip ===")
        test_inverse(model, test_ds, hr2t, device, score_mode, n_queries=args.n_queries)

    if "sym" in tests:
        print("\n=== Test 3: Symmetric-relation involution ===")
        test_involution(model, all_ds, hr2t, device, score_mode, n_queries=args.n_queries)

    if "alg" in tests:
        print("\n=== Test 4: Operator-algebra structure ===")
        test_algebra_structure(model, device, n_probe=200, max_rel_pairs=100)

    print("\nDone.")


if __name__ == "__main__":
    main()
