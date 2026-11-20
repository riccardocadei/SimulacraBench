"""SimulacraBench submission v102.

Every held-out cell is a prediction of P(answer to PREDICT question j | the person's GIVEN facts), learned at run
time from the survey's visible people. The schema's routing is not assumed: "not asked" is an ordinary answer.
Pure by construction: no processes, no file writes, no network. TabPFN and TabICL weights ship in weights/; TabFM
comes from the grader's Hugging Face cache (models.txt).

One predict() call handles one survey:

1. Design. GIVEN facts one-hot (plus frequent pairwise interactions, and a scaled position for ordered facts:
   numeric bands, and education by EDUCATION_ORDER), and as typed integer codes for the in-context models.
   Folds over the visible people: 10 in the final phase where they fit (FOLDS10_MAX_*), else 5.
2. Time plan. One clock for the whole run (three calls). Each call gets a share of what is left in proportion to
   its instrument's measured member cost (INSTRUMENT_COST); in the final phase the TabPFN fine-tune gets what the
   members leave (at least FT_SHARE of the share).
3. Fine-tuned TabPFN (final phase). TabPFN-3.5 (the fast checkpoint; on World Bank and UNHCR the 220M default
   checkpoint, fp16) is fine-tuned in memory on the visible people, every question one task, for at most FT_EPOCHS
   passes. World Bank and UNHCR questions with up to 160 answers are tasks, UNHCR's near-constant ones too
   (FT_*_BY); the learning rate is 1e-5, 3e-5 on World Bank. Inference takes the fine-tune's class permutations
   (align_class_orders). Any failure keeps the original weights.
4. Members. Each gives an out-of-fold (OOF) distribution for every visible person and a full-data one for every
   held-out person, strongest first so that the time guard drops the cheapest:
   TabPFN; shared-trunk networks (plain, ordered-fact inputs, hurdle heads for routed questions), trained as one
   GPU batch of K + 1 models whose row masks hide each fold and distilled from TabPFN's OOF and held-out
   distributions (final phase: 600 steps on World Bank and UNHCR, else 400; UNHCR's also see answer-share profiles of
   high-cardinality facts, and after its fine-tune are 4 seeds each of two networks); TabFM (UNICEF, final phase);
   softmax regressions at several L2 strengths; a latent-class regression; a family-stacked TabPFN (questions
   with the same options and gate as one long table); and the crowd marginal.
5. Blend. The fine-tuned member's OOF likelihood is discounted for the rows its weights saw (c x passes x
   questions / visible rows nats per cell; c = FT_DOSE_C, World Bank's in FT_DOSE_BY). Per question: a convex
   linear mixture by EM, shrunk toward survey-wide weights by a strength chosen on cross-fitted likelihood; a
   geometric (log-linear) pool fitted out of fold, with a ridge toward survey-wide weights; then EM between the
   two. On UNHCR both shrink through the question's routing family (question -> family -> survey). A pool that is
   not finite is dropped (the linear mixture stays).
6. Post-processing. Routing rules whose parent is a GIVEN fact are applied where the visible rows confirm them,
   and the grader's probability floor is pre-compensated.

The development phase runs the same pipeline without fine-tuning and TabFM (one TabPFN estimator). Strict
development builds (package.py --strict) fail on a member error and smoke-test the final-only models. In the
final phase a run that would finish degraded raises a distinct exception instead (LOUD_FINAL), so it can be retried.

Environment: SIMBENCH_FT_EPOCHS caps the fine-tuning passes (pin it for reproducible runs; otherwise the number
of passes follows the clock); SIMBENCH_DEBUG prints progress and raises instead of falling back; SIMBENCH_STRICT=1
runs the strict development path locally.

Method and evidence: the paper (paper/). Version history and every experiment: docs/log.md.
"""

import os
import re
import time

import numpy as np
import pandas as pd
import torch

K_BASE = 5                # folds: development, and final calls where 10 do not fit
K_FINAL = 10              # folds in the final phase where they fit
K_FOLDS = K_BASE          # this call's fold count (set by _predict)
FOLDS10_MAX_ROWS = 15000  # above this many visible people the final keeps 5 folds (UNICEF)
FOLDS10_MAX_ITEMS = 120   # ... and above this many questions (UNHCR)
TABFM_REPO, TABFM_MAX_CLASSES = "google/tabfm-1.0.0-pytorch", 10
TABFM_REV = "77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83"   # v57: the snapshot every test used (Oct 9-16)
TABFM_ESTIMATORS = 1
TABFM_SKIP = ("unhcr", "world_bank")   # U15: off on UNHCR (seconds to the 220M fine-tune); v98: off on World Bank (U21c +0.000009; pays for 600 network steps)
SEED = 0
SMOOTH = 0.5              # pseudo-count per option in the crowd marginal
PAIR_MIN_COUNT = 30       # interaction levels rarer than this are dropped
WEIGHT_PRIOR = 20.0       # pseudo-rows pulling question weights toward survey-wide weights (second blend stage)
EM_ITERS = 300
WPRIOR_GRID = [5.0, 20.0, 80.0, 320.0]   # H2: per-question shrinkage, chosen by cross-fitted likelihood
LINEAR = [                # (features, λ): L2 on weights, loss is a per-row mean
    ("oh", 3e-4), ("oh", 1e-3), ("oh", 3e-3), ("oh", 1e-2),
    ("int", 3e-3), ("oh_ord", 1e-3),
]
MLP_GRID = [              # (hidden width, weight decay, dropout, features)
    (256, 1e-2, 0.3, "oh"),
    (512, 1e-2, 0.5, "oh"),
    (256, 1e-2, 0.3, "oh_ord"),
    (256, 1e-2, 0.3, "oh+hurdle"),   # M6: routed questions as P(asked) x P(answer | asked)
]
MLP_SEEDS = 2
MLP_GRID_BY = {"unhcr": [(512, 1e-2, 0.5, "oh", 4), (256, 1e-2, 0.3, "oh_ord", 4)]}   # U24g: with a fine-tuned teacher,
                                                                                     # the same 8 fits as 4 seeds each
MLP_STEPS, MLP_LR, MLP_EVAL_EVERY = 400, 3e-3, 20
MLP_STEPS_BY = {"world_bank": 600, "unhcr": 600}   # U22b (WB), U23a (UNHCR): their networks stopped at the 400-step cap
LBFGS_ITERS = 150
TABICL_REPO, TABICL_CKPT = "jingang/TabICL", "tabicl-classifier-v2-20260212.ckpt"   # smoke test of strict builds only
TABICL_ESTIMATORS = 1
TABPFN_REPO = "Prior-Labs/tabpfn_3_5"
TABPFN_FAST = "tabpfn-v3.5-fast-20260909.safetensors"
TABPFN_FULL16 = "tabpfn-v3.5-20260909-fp16.safetensors"   # U2: the 220M default checkpoint, fp16 values (bundled)
TABPFN_ESTIMATORS = {1: 1, 2: 3}                       # ensemble size per phase
FT_FULL_FOR = ("world_bank", "unhcr")   # U2 (World Bank), U15 (UNHCR): the TabPFN member fine-tunes the default checkpoint
FT_PLUS = {"unhcr": 479.0}   # U15: TabFM 628 s - slower 220M inference 149 s (H100), given to UNHCR's fine-tune
FT_SHARE = 0.35           # H6: minimum share of a final call's time for fine-tuning
FT_EPOCHS = int(os.environ.get("SIMBENCH_FT_EPOCHS", 10))     # H6: at most this many passes over the questions
FT_DEV = False            # rehearsal builds only (package.py --ft-dev): fine-tune in the development phase too
FT_DEV_SKIP = ()          # rehearsal builds only (package.py --ft-dev-skip): surveys that still skip it in development
FT_TASKS = 16             # H8g: questions per fine-tuning "epoch"
FT_STATS = {}             # last fine-tuning: passes over the questions completed (diagnostics and ftdose)
FT_MAX_CLASSES, FT_MIN_MINORITY = 10, 0.005   # tasks: at most this many answers; skip near-constant questions
FT_MAX_CLASSES_BY = {"world_bank": 160, "unhcr": 160}   # U3, U24e: the 3.5 checkpoint's max_num_classes; wide questions join
FT_MIN_MINORITY_BY = {"unhcr": 0.0}         # U24e: UNHCR fine-tunes on its near-constant questions too
FT_LR = 1e-5              # fine-tuning learning rate
FT_NO_CKPT = ("world_bank",)   # U27 ckpt: no activation checkpointing in the fine-tune (same math, ~1.3x faster)
TABPFN_ABS = ("world_bank",)  # U30 ABS: TabPFN member averages its estimators in logit space (0 s)
TABPFN_SNAP = {"world_bank": (0.7, 0.25)}   # U35: (when, weight) of a snapshot of the same fine-tune as a 4th estimator
SNAP_SECONDS = {"world_bank": 80.0}         # U35: H100 seconds of that estimator, taken from the fine-tune
FT_SNAP = {}              # U35: the snapshot weights of the last fine-tune
FT_LR_BY = {"world_bank": 3e-5}   # U23g (results/u23_wblr): World Bank, with its dose constant in FT_DOSE_BY
FAMILY_MIN = 3            # a family needs this many questions with the same options (and gate)
FAMILY_CONTEXT = 20000    # rows of the long (person x question) table as TabPFN context
FLOOR = 1e-3              # the grader's floor (starter-kit config.yml, scoring.floor), pre-compensated at the end
MIN_ROWS = 50             # a routing rule needs this many visible people per direction ...
MAX_VIOLATION = 0.01      # ... and at most this share breaking it, to count as verified
TOTAL_SECONDS = {1: 780, 2: 3350}   # hard stop for the whole run (grader: 900 / 3600)
N_CALLS = 3                         # predict() is called once per survey
SAFETY_SECONDS = 25                 # reserved per call for blending and output
POOL_RESERVE = 60                   # U32: the log pool runs unless the whole run is this close to TOTAL_SECONDS
LOGPOOL_RIDGE = 80.0      # M10: pull of each question's pool weights toward the survey-wide ones
LOGPOOL_ITERS = 100       # L-BFGS iterations per stage
FT_DOSE_C = 0.01           # H47: optimism of the fine-tuned member per (pass x question / visible row)
FT_DOSE_BY = {"world_bank": 0.03}    # T1 (results/t1_ftdose), U23g: World Bank's c at lr 3e-5
FAMILY_SHRINK = ("unhcr",)  # U23e (results/u23_ens): stacking weights pulled question -> routing family -> survey
FAMILY_PULL, FAMILY_RIDGE = 1.0, 320.0   # family weights' pull to the survey's (x visible rows); log-pool family ridge
LOG_FLOOR = 1e-6          # member probabilities are clipped here before taking logs
_CLOCK = {"start": time.time(), "calls": 0}
LAST_DIAGNOSTICS = {}     # last call's in-run CV skill, member weights, seconds (local inspection only)
DEBUG = bool(os.environ.get("SIMBENCH_DEBUG"))
# Strict development uploads (package.py --strict sets this to True in the zip): in the development
# phase only, a member that crashes makes the run fail visibly instead of falling back,
# and the final-only models (TabICL, TabPFN ensemble, TabFM) are loaded and run on a few rows. Never active
# in the final phase.
STRICT_DEFAULT = False
STRICT_DEFAULT = STRICT_DEFAULT or os.environ.get("SIMBENCH_STRICT") == "1"   # local test of the strict path

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------- contract --

def item_options(schema, item):
    """Option list in the order the grader reads it: values, then the sentinel if gated."""
    record = schema["items"][item]
    return list(record["values"]) + ([schema["gated_value"]] if record.get("gate") else [])


def shipped_items(schema, frame):
    """GIVEN and PREDICT items in schema key order (the canonical within-row order)."""
    return [name for name, record in schema["items"].items()
            if record["class"] in ("GIVEN", "PREDICT") and name in frame.columns]


def assemble(frame, schema, probs):
    """One vector per blank cell: rows top to bottom, items in schema order.

    probs[item] is an (n_rows, K) array aligned with frame rows.
    """
    items = [i for i in shipped_items(schema, frame) if schema["items"][i]["class"] == "PREDICT"]   # H56: scored cells only
    blank = frame[items].isna().to_numpy()
    return [probs[items[col]][row].tolist() for row, col in zip(*np.nonzero(blank))]


# ------------------------------------------------------------------ design --

def one_hot(columns, min_count=1):
    """Dense one-hot of string columns, keeping levels seen at least min_count times."""
    blocks = []
    for col in columns.T:
        levels, codes = np.unique(col, return_inverse=True)
        keep = np.bincount(codes) >= min_count
        if keep.sum() < 2:
            continue
        block = np.zeros((len(col), keep.sum()), np.float32)
        remap = np.cumsum(keep) - 1
        rows = np.flatnonzero(keep[codes])
        block[rows, remap[codes[rows]]] = 1.0
        blocks.append(block)
    return np.hstack(blocks) if blocks else np.zeros((len(columns), 0), np.float32)


_NUM = re.compile(r"\d+(?:[.,]\d+)?")


EDUCATION_ORDER = {     # H14c: education facts in their natural order (proxies afro Q94, arab Q1003 for the lab)
    'Q94': ['No formal schooling', 'Informal schooling only (including Koranic schooling)', 'Some primary schooling', 'Primary school completed', 'Intermediate school or Some secondary school / high school', 'Secondary school / high school completed', 'Post-secondary qualifications, other than university e.g. a diploma or degree from a polytechnic or college', 'Some university', 'University completed', 'Post-graduate'],
    'Q1003': ['No formal education', 'Elementary', 'Preparatory/Basic', 'Secondary', 'Technical degree', 'BA', 'MA and above'],
    's2_1c_education_level_lbl': ['1. No formal education', '2. Early Childhood Education or Pre-primary', '3. Primary', '4. Secondary', '5. Secondary - Technical and Vocational Education and Training (TVET)', '6. Post-secondary - Technical and Vocational Education and Training (TVET)', '7. Tertiary and above'],
    'user.educationLevel': ['No schooling completed', 'Primary school', 'Some high school, no matric certificate', 'Secondary school', 'Matric', 'Trade / technical / vocational training (at a TVET, FET or private college)', 'University bachelor’s degree', 'University honours degree', 'Master’s degree', 'Professional degree (Law, Medicine, PhD)'],
}


def numeric_order(values):
    """Positions of the options that form an increasing numeric chain, or None.

    An option counts if it reads as a number or a band ('25-34', '<2.5', '11+', '0',
    '10000-14999'); enumerated labels ('1. Head of household', '98. Don't know') and
    words ('Not answered', the sentinel) stay outside the chain. A chain needs at least
    three options in strictly increasing order.
    """
    keyed = []
    for i, v in enumerate(values):
        text = str(v).strip()
        if not re.match(r"^[<>]?\d", text) or re.match(r"^\d+\.\s*\D", text):
            continue
        nums = [float(x.replace(",", ".")) for x in _NUM.findall(text)]
        keyed.append((i, sum(nums) / len(nums) - (1e-6 if text.startswith("<") else 0)))
    if len(keyed) < 3 or any(b[1] <= a[1] for a, b in zip(keyed, keyed[1:])):
        return None
    return np.array([i for i, _ in keyed])


class Design:
    """Encodings of the GIVEN block, targets, folds and option-count buckets."""

    def __init__(self, frame, schema):
        items = shipped_items(schema, frame)
        self.given = [i for i in items if schema["items"][i]["class"] == "GIVEN"]
        self.targets = [i for i in items if schema["items"][i]["class"] == "PREDICT"]
        self.visible = frame[self.targets].notna().all(axis=1).to_numpy()
        self.vis_idx = np.flatnonzero(self.visible)
        self.hid_idx = np.flatnonzero(~self.visible)

        G = frame[self.given].astype(str).to_numpy()
        pairs = np.column_stack([np.char.add(np.char.add(G[:, a], "|"), G[:, b])
                                 for a in range(G.shape[1]) for b in range(a + 1, G.shape[1])])
        self.X = {"oh": one_hot(G)}
        self.X["int"] = np.hstack([self.X["oh"], one_hot(pairs, PAIR_MIN_COUNT)])
        self.nominal_cols = []
        self.gated = np.array([bool(schema["items"][i].get("gate")) for i in self.targets])
        # Typed codes (M1): ordered facts by numeric order, others by schema order.
        ord_cols, typed = [], []
        for c, name in enumerate(self.given):
            values = list(schema["items"][name]["values"])
            chain = numeric_order(values)
            labels = [str(v) for v in values]
            if (chain is None and name in EDUCATION_ORDER          # H14c: schema prior on education
                    and all(v in labels for v in EDUCATION_ORDER[name])):          # v57: fail closed on a label change
                chain = np.array([labels.index(v) for v in EDUCATION_ORDER[name]])
            pos = {v: i for i, v in enumerate(values)}
            col = np.array([pos.get(v, len(values)) for v in G[:, c]], float)
            if chain is None:
                self.nominal_cols.append(c)
            if chain is not None:
                rank = {values[i]: r for r, i in enumerate(chain)}
                mid = (len(chain) - 1) / 2                       # unordered levels -> middle
                col = np.array([rank.get(v, mid) for v in G[:, c]], float)
                ord_cols.append((col - col.mean()) / (col.std() + 1e-9))
            typed.append(col)
        self.codes = np.column_stack(typed)
        if ord_cols:
            self.X["oh_ord"] = np.hstack([self.X["oh"], np.column_stack(ord_cols).astype(np.float32)])
        self.phase2 = len(self.vis_idx) > schema.get("split", {}).get("n_train", 10 ** 9)

        self.k = np.array([len(item_options(schema, i)) for i in self.targets])
        vis = frame.loc[self.visible]
        self.Y = np.column_stack([
            vis[item].map({o: j for j, o in enumerate(item_options(schema, item))})
            .fillna(0).to_numpy(int) for item in self.targets])

        # Buckets: items with the same option count share one (N, J_k, K) logit tensor.
        self.buckets = {int(k): np.flatnonzero(self.k == k) for k in np.unique(self.k)}

        rng = np.random.default_rng(SEED)
        self.folds = rng.integers(0, K_FOLDS, len(self.vis_idx))
        # Row weights per batch member: fold models hide their fold, the last sees everything.
        mask = np.ones((K_FOLDS + 1, len(self.vis_idx)), np.float32)
        for f in range(K_FOLDS):
            mask[f, self.folds == f] = 0.0
        self.mask = torch.tensor(mask, device=DEVICE)
        self.Y_t = torch.tensor(self.Y, device=DEVICE)


def marginal(y, k):
    counts = np.bincount(y, minlength=k).astype(float) + SMOOTH
    return counts / counts.sum()


# ------------------------------------------------------ batched multi-task --

def bucket_nll(logits, d):
    """Per-(batch, row) negative log-likelihood summed over items, from bucketed logits."""
    total = 0.0
    for k, idx in d.buckets.items():
        logp = torch.log_softmax(logits[k], dim=-1)             # (B, N, J_k, k)
        y = d.Y_t[:, idx].unsqueeze(0).unsqueeze(-1)             # (1, N, J_k, 1)
        total = total + logp.gather(-1, y.expand(logp.shape[0], -1, -1, -1)).squeeze(-1).sum(-1)
    return -total                                                # (B, N)


def masked_loss(logits, d, mask=None):
    mask = d.mask if mask is None else mask
    nll = bucket_nll(logits, d)
    return ((nll * mask).sum(1) / mask.sum(1)).sum()


def collect(logits_vis, logits_hid, d, out_oof, out_pred, out_bucket):
    """Write OOF true-class probabilities and full held-out distributions (and, for the
    geometric pool, every question's full OOF distributions per bucket into out_bucket)."""
    for k, idx in d.buckets.items():
        p_vis = torch.softmax(logits_vis[k], -1)                 # (B, N_vis, J_k, k)
        y = d.Y_t[:, idx]
        folds = torch.tensor(d.folds, device=DEVICE)
        p_own = p_vis[folds, torch.arange(len(folds), device=DEVICE)]   # each row from its fold model
        true = p_own.gather(-1, y.unsqueeze(-1)).squeeze(-1)     # (N_vis, J_k)
        out_oof[:, idx] = true.float().cpu().numpy()
        out_bucket[k] = p_own.float()                           # (N_vis, J_k, k), on the device
        p_hid = torch.softmax(logits_hid[k][-1], -1).float().cpu().numpy()   # full model
        for j, item_j in enumerate(idx):
            out_pred[item_j] = p_hid[:, j]


class LinearHeads(torch.nn.Module):
    """B independent softmax regressions for every bucket: (B, D) x (D, J_k * k)."""

    def __init__(self, d, dim, B):
        super().__init__()
        self.shape = {k: (len(idx), k) for k, idx in d.buckets.items()}
        self.W = torch.nn.ParameterDict({str(k): torch.nn.Parameter(
            torch.zeros(B, dim, len(idx) * k, device=DEVICE)) for k, idx in d.buckets.items()})
        self.b = torch.nn.ParameterDict({str(k): torch.nn.Parameter(
            torch.zeros(B, 1, len(idx) * k, device=DEVICE)) for k, idx in d.buckets.items()})

    def forward(self, H):                       # H: (N, D) or (B, N, D)
        out = {}
        for k, (J, K) in self.shape.items():
            z = (H @ self.W[str(k)] if H.dim() == 3 else torch.einsum("nd,bdo->bno", H, self.W[str(k)]))
            out[k] = (z + self.b[str(k)]).reshape(z.shape[0], z.shape[1], J, K)
        return out

    def penalty(self):
        return sum((w ** 2).sum() for w in self.W.values())


# T5 / P-UNHCR (results/t5_profiles): UNHCR's networks and linear heads also see each high-cardinality GIVEN fact's
# level as its cross-fitted, shrunk answer-share profile (SVD components); rs44 arab_real +0.0001, afro / wgm lose.
PROF_INSTRUMENTS = ("unhcr",)
PROF_MIN_LEVELS = 15        # a GIVEN fact with at least this many schema levels gets answer profiles
PROF_PSEUDO = 20.0          # pseudo-people pulling a level's shares toward the overall shares
PROF_RANK = 6               # profile components per fact


def prof_inputs(d, features):
    """The plain inputs (N, D), or with profiles (B, N, D + R): fold model b sees its own cross-fitted profiles."""
    X = torch.tensor(d.X[features], device=DEVICE)
    X_vis, X_hid = X[d.vis_idx], X[d.hid_idx]
    prof = getattr(d, "prof", None)
    if prof is None:
        return X_vis, X_hid
    pv, ph = prof
    B = pv.shape[0]
    return (torch.cat([X_vis.unsqueeze(0).expand(B, -1, -1), pv], -1),
            torch.cat([X_hid.unsqueeze(0).expand(B, -1, -1), ph], -1))


def level_profiles(frame, schema, d):
    """Cross-fitted, shrunk answer-share profiles of each high-cardinality GIVEN fact's level, reduced by SVD.
    Returns (B, N_vis, R) and (B, N_hid, R) tensors, or None if no fact qualifies."""
    K, B = K_FOLDS, K_FOLDS + 1
    nv = len(d.vis_idx)
    offs = np.concatenate([[0], np.cumsum(d.k)])
    Yoh = np.zeros((nv, offs[-1]), np.float64)
    for j in range(len(d.targets)):
        Yoh[np.arange(nv), offs[j] + d.Y[:, j]] = 1.0
    fv_all, fh_all, used = [], [], []
    for name in d.given:
        L_schema = len(schema["items"][name].get("values") or [])
        if L_schema < PROF_MIN_LEVELS:
            continue
        lev = pd.factorize(frame[name].astype(str).to_numpy())[0]
        L = int(lev.max()) + 1
        lv, lh = lev[d.vis_idx], lev[d.hid_idx]
        Cf, Nf = np.zeros((K, L, offs[-1])), np.zeros((K, L))
        for g in range(K):
            rows = d.folds == g
            oh = np.zeros((rows.sum(), L))
            oh[np.arange(rows.sum()), lv[rows]] = 1.0
            Cf[g], Nf[g] = oh.T @ Yoh[rows], oh.sum(0)
        Ct, Nt = Cf.sum(0), Nf.sum(0)

        def prof(excl):
            C = Ct - Cf[excl].sum(0) if excl else Ct
            n = Nt - Nf[excl].sum(0) if excl else Nt
            glob = C.sum(0) / max(n.sum(), 1.0)
            return (C + PROF_PSEUDO * glob) / (n[:, None] + PROF_PSEUDO) - glob, n

        R = min(PROF_RANK, L - 1)
        fv, fh = np.zeros((B, nv, R)), np.zeros((B, len(d.hid_idx), R))
        for b in range(B):
            ex = [b] if b < K else []
            Pb, nb = prof(ex)
            _, _, Vt = np.linalg.svd(np.sqrt(nb)[:, None] * Pb, full_matrices=False)
            V = Vt[:R].T
            proj_b = Pb @ V
            for g in range(K):
                rows = d.folds == g
                if g == b:
                    fv[b, rows] = proj_b[lv[rows]]
                else:
                    fv[b, rows] = (prof(ex + [g])[0] @ V)[lv[rows]]
            fh[b] = proj_b[lh]
            tr = d.folds != b
            mu, sd = fv[b, tr].mean(0), fv[b, tr].std(0) + 1e-6
            fv[b], fh[b] = (fv[b] - mu) / sd, (fh[b] - mu) / sd
        fv_all.append(fv), fh_all.append(fh), used.append(f"{name}:{L}:{R}")
    LAST_DIAGNOSTICS["profile_facts"] = used
    if not used:
        return None
    t = lambda a: torch.tensor(np.concatenate(a, -1), dtype=torch.float32, device=DEVICE)
    return t(fv_all), t(fh_all)


# U5 (results/u5_distill): distil the fine-tuned TabPFN into the networks and linear heads. Fold model b also fits
# TabPFN's out-of-fold distributions on fold b's rows (context without fold b), the full model TabPFN's held-out
# distributions on the held-out people's facts, with weight DISTILL_LAMBDA per row (the full model's rescaled to
# the fold models' soft/hard mass ratio). No held-out labels; no fold sees a soft label made with its own fold.
DISTILL_LAMBDA = 3.0
DISTILL_LAMBDA_BY = {"world_bank": 10.0}   # U21d/e: WB students under the 220M fine-tuned teacher; UNHCR (U17), UNICEF (U21e) peak at 3


def distill_targets(d, names, preds, buckets, lam):
    d.soft_W = d.soft_T = None
    if lam <= 0 or "tabpfn" not in names or len(buckets) != len(names):
        return
    m = names.index("tabpfn")
    K, nv, nh = K_FOLDS, len(d.vis_idx), len(d.hid_idx)
    T = {}
    for k, idx in d.buckets.items():
        tv = torch.as_tensor(buckets[m][k], dtype=torch.float32, device=DEVICE)
        th = torch.as_tensor(np.stack([np.asarray(preds[m][j]) for j in idx], 1), dtype=torch.float32, device=DEVICE)
        t = torch.cat([tv, th], 0).clamp_min(1e-8)
        T[k] = t / t.sum(-1, keepdim=True)
    W = torch.zeros(K + 1, nv + nh, device=DEVICE)
    nb = np.bincount(d.folds, minlength=K).astype(float)
    for b in range(K):
        W[b, :nv][torch.as_tensor(d.folds == b, device=DEVICE)] = lam
    W[K, nv:] = lam * float(np.mean(nb / (nv - nb))) * nv / max(nh, 1)
    d.soft_W, d.soft_T = W, T


def train_loss(model, X_vis, X_hid, d, mask=None):
    """masked_loss on the visible rows + the distillation term when d.soft_W is set."""
    mask = d.mask if mask is None else mask
    W = getattr(d, "soft_W", None)
    if W is None:
        return masked_loss(model(X_vis), d, mask)
    nv = X_vis.shape[-2]
    out = model(torch.cat([X_vis, X_hid], -2))
    hard = bucket_nll({k: z[:, :nv] for k, z in out.items()}, d)
    soft = 0.0
    for k in d.buckets:
        soft = soft - (d.soft_T[k][None] * torch.log_softmax(out[k], -1)).sum((-1, -2))
    return (((hard * mask).sum(1) + (soft * W).sum(1)) / mask.sum(1)).sum()


def fit_linear(d, features, lam):
    X_vis, X_hid = prof_inputs(d, features)
    B = K_FOLDS + 1
    model = LinearHeads(d, X_vis.shape[-1], B)
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=LBFGS_ITERS,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = train_loss(model, X_vis, X_hid, d) + lam * model.penalty()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return model(X_vis), model(X_hid)


class HurdleHeads(torch.nn.Module):
    """Per-question heads that factor a routed question as P(asked) x P(answer | asked).

    Each routed question has its *own* asked-logit (no sharing between siblings, unlike
    the routing heads that failed in v3), so the factorisation imposes nothing: it only
    separates the frequent "not asked" decision from the rarer substantive answers.
    Returns log-probabilities (valid logits for the softmax downstream).
    """

    def __init__(self, d, width, B):
        super().__init__()
        self.d = d
        self.answers = LinearHeads(d, width, B)
        self.group = torch.as_tensor(np.arange(len(d.targets)), device=DEVICE)
        G = int(self.group.max()) + 1
        self.Wa = torch.nn.Parameter(torch.zeros(B, width, G, device=DEVICE))
        self.ba = torch.nn.Parameter(torch.zeros(B, 1, G, device=DEVICE))
        self.gated = torch.as_tensor(d.gated, device=DEVICE)

    def forward(self, h):
        ans = self.answers(h)
        asked = (h @ self.Wa + self.ba)[..., self.group]             # (B, N, J)
        out = {}
        for k, idx in self.d.buckets.items():
            plain = torch.log_softmax(ans[k], -1)
            if k < 2 or not self.d.gated[idx].any():
                out[k] = plain
                continue
            idx_t = torch.as_tensor(idx, device=DEVICE)
            a = asked[..., idx_t].unsqueeze(-1)
            routed = torch.cat([torch.nn.functional.logsigmoid(a)
                                + torch.log_softmax(ans[k][..., :-1], -1),
                                torch.nn.functional.logsigmoid(-a)], -1)
            out[k] = torch.where(self.gated[idx_t][None, None, :, None], routed, plain)
        return out


class MLP(torch.nn.Module):
    """B independent shared-trunk networks: one-hot -> h -> h -> per-item heads."""

    def __init__(self, d, dim, width, dropout, B, seed=SEED, hurdle=False):
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(seed)
        init = lambda *s: torch.nn.Parameter(
            (torch.randn(*s, generator=g) / np.sqrt(s[-2])).to(DEVICE))
        self.W1, self.b1 = init(B, dim, width), torch.nn.Parameter(torch.zeros(B, 1, width, device=DEVICE))
        self.W2, self.b2 = init(B, width, width), torch.nn.Parameter(torch.zeros(B, 1, width, device=DEVICE))
        self.heads = HurdleHeads(d, width, B) if hurdle else LinearHeads(d, width, B)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, X):
        W1 = self.W1 * self.w1_mask if getattr(self, "w1_mask", None) is not None else self.W1   # H54: unseen levels
        h = torch.nn.functional.gelu(torch.einsum("bnd,bdh->bnh" if X.dim() == 3 else "nd,bdh->bnh", X, W1) + self.b1)
        h = self.dropout(h)
        h = torch.nn.functional.gelu(h @ self.W2 + self.b2) + h
        return self.heads(self.dropout(h))


def fit_mlp(d, width, wd, dropout, features="oh", seeds=MLP_SEEDS):
    """Average of `seeds` runs, in probability space; returned as log-probabilities.
    A features name ending in '+hurdle' uses the hurdle heads."""
    hurdle = features.endswith("+hurdle")
    features = features.split("+")[0]
    runs = [fit_mlp_once(d, width, wd, dropout, features, SEED + s, hurdle)
            for s in range(seeds)]
    avg = lambda parts: {k: torch.log(sum(torch.softmax(p[k], -1) for p in parts) / len(parts))
                         for k in d.buckets}
    return avg([r[0] for r in runs]), avg([r[1] for r in runs])


def fit_mlp_once(d, width, wd, dropout, features, seed, hurdle=False):
    """Adam, full batch. Each fold picks its best checkpoint on its own fold; the
    full model uses the median of those steps (its checkpoints are kept)."""
    torch.manual_seed(seed)
    X_vis, X_hid = prof_inputs(d, features)
    B = K_FOLDS + 1
    model = MLP(d, X_vis.shape[-1], width, dropout, B, seed, hurdle)
    nz = (X_vis != 0).float()                                    # U4: count rows, not values (centred columns sum to ~0)
    seen = torch.einsum("bn,bnd->bd", d.mask, nz) if nz.dim() == 3 else d.mask @ nz
    model.w1_mask = (seen > 0).float().unsqueeze(-1)   # H54: a column no training row has keeps 0 weight, not its random init
    opt = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=wd)
    holdout = 1.0 - d.mask[:K_FOLDS]                             # (K, N_vis)

    best = [np.inf] * K_FOLDS
    best_vis = [None] * K_FOLDS
    best_step = [0] * K_FOLDS
    hid_ckpt = {}
    for step in range(1, getattr(d, "mlp_steps", MLP_STEPS) + 1):
        model.train()
        opt.zero_grad()
        train_loss(model, X_vis, X_hid, d).backward()
        opt.step()
        if step % MLP_EVAL_EVERY:
            continue
        model.eval()
        with torch.no_grad():
            lv = model(X_vis)
            nll = bucket_nll(lv, d)[:K_FOLDS]
            val = ((nll * holdout).sum(1) / holdout.sum(1)).cpu().numpy()
            for f in range(K_FOLDS):
                if val[f] < best[f]:
                    best[f], best_step[f] = val[f], step
                    best_vis[f] = {k: v[f].clone() for k, v in lv.items()}
            hid_ckpt[step] = {k: v[-1].clone() for k, v in model(X_hid).items()}
            del lv
        if time.time() > d.deadline:                            # out of time: keep what we have
            break

    step = int(np.median(best_step))
    step = max(MLP_EVAL_EVERY, step - step % MLP_EVAL_EVERY)
    logits_vis = {k: torch.stack([best_vis[f][k] for f in range(K_FOLDS)]) for k in d.buckets}
    logits_hid = {k: hid_ckpt[step][k].unsqueeze(0) for k in d.buckets}
    return logits_vis, logits_hid


# ----------------------------------------------------------- latent classes --

LATENT_CLASSES = [64]     # latent-class members (number of classes)
LATENT_STEPS = 1500
LATENT_LR = 0.02
LATENT_HIDDEN = 128


class LatentClasses(torch.nn.Module):
    """Latent-class regression (LCA with covariates): p(y_1..y_J | x) = sum_c pi_c(x) prod_j p_jc(y_j).
    Fitted on the *joint* likelihood of each visible person's answers, so how answers co-occur
    across questions (shared traits) informs every question's estimate; predicts the marginal
    p(y_j | x) = sum_c pi_c(x) p_jc. B independent copies (folds + full), like the networks."""

    def __init__(self, d, dim, C, B, seed):
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.d, self.C = d, C
        H = LATENT_HIDDEN
        self.W1 = torch.nn.Parameter((torch.randn(B, dim, H, generator=g) / np.sqrt(dim)).to(DEVICE))
        self.b1 = torch.nn.Parameter(torch.zeros(B, 1, H, device=DEVICE))
        self.W = torch.nn.Parameter((torch.randn(B, H, C, generator=g) * 0.01).to(DEVICE))
        self.b = torch.nn.Parameter(torch.zeros(B, 1, C, device=DEVICE))
        # class profiles start at the observed answer shares, perturbed, so classes split from the crowd
        self.theta = torch.nn.ParameterDict()
        for k, idx in d.buckets.items():
            base = np.log(np.stack([marginal(d.Y[:, j], k) for j in idx]))            # (J_k, k)
            noise = torch.randn(B, C, len(idx), k, generator=g) * 0.7
            self.theta[str(k)] = torch.nn.Parameter((torch.tensor(base, dtype=torch.float32) + noise).to(DEVICE))

    def logits_pi(self, X):
        h = torch.nn.functional.gelu(torch.einsum("nd,bdh->bnh", X, self.W1) + self.b1)
        return h @ self.W + self.b

    def log_pi(self, X):
        return torch.log_softmax(self.logits_pi(X), -1)                       # (B, N, C)

    def joint_ll(self, X, Y1h):
        """log p(y_i | x_i) per (batch, row), from one-hot answers per bucket."""
        acc = self.log_pi(X)                                                  # (B, N, C)
        for k in self.d.buckets:
            lp = torch.log_softmax(self.theta[str(k)], -1).flatten(2)         # (B, C, J_k * k)
            acc = acc + torch.einsum("nf,bcf->bnc", Y1h[k], lp)
        return torch.logsumexp(acc, -1)                                       # (B, N)

    def marginal_logits(self, X):
        pi = torch.softmax(self.logits_pi(X), -1)
        return {k: torch.log(torch.einsum("bnc,bcjk->bnjk", pi, torch.softmax(self.theta[str(k)], -1)) + 1e-9)
                for k in self.d.buckets}


def fit_latent(d, C, seed=SEED, steps=None, lr=None, wd=1e-3):
    """Each fold copy picks its step by the *marginal* log-likelihood on its own held-out fold (the
    score we are judged on); the full copy uses the median of those steps."""
    steps, lr = steps or LATENT_STEPS, lr or LATENT_LR
    torch.manual_seed(seed)
    X = torch.tensor(d.X["oh"], device=DEVICE)
    X_vis, X_hid = X[d.vis_idx], X[d.hid_idx]
    Y1h = {k: torch.nn.functional.one_hot(d.Y_t[:, idx], k).float().flatten(1) for k, idx in d.buckets.items()}
    B = K_FOLDS + 1
    model = LatentClasses(d, X.shape[1], C, B, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    holdout = 1.0 - d.mask[:K_FOLDS]
    best, best_vis, best_step, hid_ckpt = [np.inf] * K_FOLDS, [None] * K_FOLDS, [0] * K_FOLDS, {}
    for step in range(1, steps + 1):
        opt.zero_grad()
        ll = model.joint_ll(X_vis, Y1h)                                       # (B, N)
        loss = -(ll * d.mask).sum(1) / d.mask.sum(1)
        reg = wd * ((model.W ** 2).sum((1, 2)) + (model.W1 ** 2).sum((1, 2)))
        (loss + reg).sum().backward()
        opt.step()
        if step % MLP_EVAL_EVERY:
            continue
        with torch.no_grad():
            lv = model.marginal_logits(X_vis)
            nll = bucket_nll(lv, d)[:K_FOLDS]
            val = ((nll * holdout).sum(1) / holdout.sum(1)).cpu().numpy()
            for f in range(K_FOLDS):
                if val[f] < best[f]:
                    best[f], best_step[f] = val[f], step
                    best_vis[f] = {k: v[f].clone() for k, v in lv.items()}
            hid_ckpt[step] = {k: v[-1].clone() for k, v in model.marginal_logits(X_hid).items()}
            del lv
        if time.time() > d.deadline:
            break
    step = int(np.median(best_step))
    step = max(MLP_EVAL_EVERY, step - step % MLP_EVAL_EVERY)
    logits_vis = {k: torch.stack([best_vis[f][k] for f in range(K_FOLDS)]) for k in d.buckets}
    logits_hid = {k: hid_ckpt[step][k].unsqueeze(0) for k in d.buckets}
    return logits_vis, logits_hid


# ------------------------------------------- in-context members (TabPFN, TabFM) --

def as_member_output(p_vis, p_hid):
    """Wrap per-row OOF probabilities (N_vis, J_k, k) and held-out ones (N_hid, J_k, k)
    in the (batch, rows, items, options) log-prob layout that collect() reads."""
    logs = lambda p: torch.log(torch.clamp(p, min=1e-12))
    return ({k: logs(v).unsqueeze(0).expand(K_FOLDS, *v.shape) for k, v in p_vis.items()},
            {k: logs(v).unsqueeze(0) for k, v in p_hid.items()})


def fit_incontext(d, clf, coarse=False):
    """A pretrained in-context classifier, refit per question and per fold on the typed
    known-fact codes. Questions not reached before the deadline fall back to crowd
    shares, so the blend gives the member no weight there."""
    X_vis, X_hid = d.codes[d.vis_idx], d.codes[d.hid_idx]
    # coarse: at most 5 folds (pairs of the 10 merged) for an expensive member; still out of fold
    KF = min(K_FOLDS, 5) if coarse else K_FOLDS
    folds = d.folds % KF if coarse else d.folds
    p_vis = {k: np.zeros((len(d.vis_idx), len(idx), k)) for k, idx in d.buckets.items()}
    p_hid = {k: np.zeros((len(d.hid_idx), len(idx), k)) for k, idx in d.buckets.items()}

    def fit_predict(train, X_out, y, k):
        if np.unique(y[train]).size < 2:
            return np.tile(marginal(y[train], k), (len(X_out), 1))
        rows = np.flatnonzero(train)
        clf.fit(X_vis[rows], y[rows])
        out = np.zeros((len(X_out), k))
        out[:, clf.classes_] = clf.predict_proba(X_out)
        return out

    def crowd_only(k, jj, y):
        for f in range(KF):
            te = folds == f
            p_vis[k][te, jj] = marginal(y[~te], k)
        p_hid[k][:, jj] = marginal(y, k)

    for k, idx in d.buckets.items():
        for jj, j in enumerate(idx):
            y = d.Y[:, j]
            if time.time() > d.deadline:                         # out of time: crowd shares
                crowd_only(k, jj, y)
                continue
            for f in range(KF + 1):                              # folds, then the full fit
                if time.time() > d.deadline:                     # checked per fit, not per question
                    crowd_only(k, jj, y)
                    break
                if f < KF:
                    te = folds == f
                    p_vis[k][te, jj] = fit_predict(~te, X_vis[te], y, k)
                else:
                    p_hid[k][:, jj] = fit_predict(np.ones(len(y), bool), X_hid, y, k)
    to_t = lambda p: {k: torch.as_tensor(v, dtype=torch.float32, device=DEVICE) for k, v in p.items()}
    return as_member_output(to_t(p_vis), to_t(p_hid))


class _TabFMAdapter:
    """TabFM on typed codes, nominal columns passed as categories. Questions with more classes
    than TabFM supports get their training shares instead."""

    def __init__(self, model, nominal):
        from tabfm import TabFMClassifier
        self.clf = TabFMClassifier(model=model, n_estimators=TABFM_ESTIMATORS, batch_size=TABFM_ESTIMATORS,
                                   random_state=SEED)
        self.nominal = set(nominal)

    def _frame(self, X):
        return pd.DataFrame({f"c{c}": (X[:, c].astype(int).astype(str) if c in self.nominal else X[:, c])
                             for c in range(X.shape[1])})

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.shares = np.bincount(np.searchsorted(self.classes_, y)) / len(y)
        self.too_many = len(self.classes_) > TABFM_MAX_CLASSES
        if not self.too_many:
            self.clf.fit(self._frame(X), y)
            self.classes_ = self.clf.classes_
        return self

    def predict_proba(self, X):
        if self.too_many:
            return np.tile(self.shares, (len(X), 1))
        return self.clf.predict_proba(self._frame(X))


def tabfm_path():
    """The pinned TabFM snapshot if the grader's cache holds it, else whatever main it fetched (v57)."""
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(TABFM_REPO, revision=TABFM_REV, local_files_only=True)
    except Exception:
        return snapshot_download(TABFM_REPO, local_files_only=True)


def fit_tabfm(d):
    """TabFM v1.0.0 (code Apache-2.0, weights TabFM Non-Commercial Licence; fetched by the grader
    from models.txt, which the v34 probe proved works). Final phase only, at most 5 folds."""
    from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1

    model = tabfm_v1.load(model_type="classification", checkpoint_path=tabfm_path(), device=str(DEVICE))
    return fit_incontext(d, _TabFMAdapter(model, d.nominal_cols), coarse=True)


HERE = os.path.dirname(os.path.abspath(__file__))


def model_file(repo, name):
    """A shipped copy first (weights/ next to main.py), else the Hugging Face cache."""
    local = os.path.join(HERE, "weights", name)
    if os.path.exists(local):
        return local
    from huggingface_hub import snapshot_download
    return os.path.join(snapshot_download(repo, local_files_only=True), name)


def align_class_orders(on):
    """H50c: TabPFN shuffles the class order per estimator config. The fine-tune trains under the configs of
    n_estimators=2; at inference with n >= 3 estimators, configs 0 and 1 already share its column order and category
    codes but not its class order. With `on`, they take the fine-tune's class permutations (same seed and class
    count), so the fine-tuned weights predict under the views they were trained on; estimator 2 is unchanged.
    `on=False` restores TabPFN's own function. Any failure falls back to the unaligned configs."""
    import tabpfn.classifier as tc
    original = getattr(tc, "_simbench_original_configs", None) or tc.generate_classification_ensemble_configs
    tc._simbench_original_configs = original
    if not on:
        tc.generate_classification_ensemble_configs = original
        return

    def aligned(**kw):
        configs = original(**kw)
        try:
            if kw.get("num_estimators", 0) >= 3:
                tuned = original(**{**kw, "num_estimators": 2})
                for i in range(2):
                    configs[i].class_permutation = tuned[i].class_permutation
        except Exception:
            if DEBUG:
                raise
        return configs
    tc.generate_classification_ensemble_configs = aligned


def fit_tabpfn(d):
    """TabPFN-3.5 (non-commercial research licence, allowed by the rules), weights shipped in weights/.
    Nominal facts are flagged as categorical. With d.ft_seconds > 0 it is first fine-tuned on the survey (H6)."""
    import tabpfn.checkpoint
    from tabpfn import TabPFNClassifier

    # HF cache files are symlinks to suffix-less blobs; TabPFN picks the loader by suffix.
    tabpfn.checkpoint.Checkpoint.is_safetensors = property(lambda self: True)
    os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", "1")      # build the network once
    # The fast checkpoint by default; World Bank and UNHCR fine-tune the 220M default one (fp16) when there is
    # time to fine-tune (U2, U15). Without fine-tuning the default checkpoint is too slow for its gain.
    ckpt = model_file(TABPFN_REPO, TABPFN_FAST)
    if getattr(d, "ft_full", False) and getattr(d, "ft_seconds", 0) > 0:   # U2: fine-tune the default checkpoint
        try:
            full = model_file(TABPFN_REPO, TABPFN_FULL16)
            if not os.path.exists(full):          # a missing file would make TabPFN try a gated download
                raise FileNotFoundError(full)
            ckpt = full
        except Exception:
            if DEBUG or _CLOCK.get("strict"):
                raise
    state = None
    if getattr(d, "ft_seconds", 0) > 0:
        t = time.time()
        try:
            state = finetune_tabpfn(d.codes[d.vis_idx], d.Y, d.nominal_cols, ckpt, d.ft_seconds, FT_EPOCHS,
                                    d.ft_max_classes, d.ft_min_minority, d.ft_lr, d.ft_act_ckpt,
                                    snap_frac=(getattr(d, "snap", None) or (None,))[0])
        except Exception:
            if DEBUG or _CLOCK.get("strict"):
                raise
        if DEBUG:
            print(f"    tabpfn fine-tuning: {time.time() - t:.1f}s of {d.ft_seconds:.0f}s allowed, "
                  f"{FT_STATS.get('passes')} passes over {FT_STATS.get('questions')} questions")
    clf = TabPFNClassifier(model_path=ckpt, device=str(DEVICE), n_estimators=TABPFN_ESTIMATORS[2 if d.phase2 else 1],
                           categorical_features_indices=d.nominal_cols or None, random_state=SEED,
                           average_before_softmax=d.tabpfn_abs)
    try:
        if state is not None:
            install_tabpfn_weights(ckpt, state)
            align_class_orders(True)                # H50c
            d.finetuned = True
        out = fit_incontext(d, clf)
        snap = FT_SNAP.pop("state", None)
        if state is not None and snap is not None and getattr(d, "snap", None) \
                and time.time() + SNAP_SECONDS.get(getattr(d, "inst", None), 80.0) < d.deadline:
            try:                                                    # U35: snapshot of the same fine-tune, 1 estimator
                one = TabPFNClassifier(model_path=ckpt, device=str(DEVICE), n_estimators=1,
                                       categorical_features_indices=d.nominal_cols or None, random_state=SEED,
                                       average_before_softmax=d.tabpfn_abs)
                t_snap = time.time()
                install_tabpfn_weights(ckpt, snap)
                extra = fit_incontext(d, one)
                FT_STATS["snap_seconds"] = round(time.time() - t_snap, 1)
                if time.time() <= d.deadline:              # finished in time: every question got the snapshot's fit
                    w = d.snap[1]
                    mix = lambda a, b: torch.log((1 - w) * a.exp() + w * b.exp())
                    out = tuple({k: mix(o[k], e[k]) for k in o} for o, e in zip(out, extra))
                    FT_STATS["snap_loop"] = FT_SNAP.get("loop")
                    if DEBUG:
                        print(f"    tabpfn snapshot: fired, weights after loop {FT_SNAP.get('loop')}, mixed at {w}, "
                              f"{FT_STATS['snap_seconds']}s")
                elif DEBUG:                                # overran: some questions may hold crowd shares; v105's member
                    print(f"    tabpfn snapshot: discarded (overran), {FT_STATS['snap_seconds']}s")
            except Exception:
                if DEBUG or _CLOCK.get("strict"):
                    raise
        elif DEBUG and getattr(d, "snap", None) and state is not None:
            print(f"    tabpfn snapshot: skipped ({'no snapshot taken' if snap is None else 'no time left'})")
        return out
    finally:                                   # the family member and the next survey start from the original
        FT_SNAP.clear()                        # U35: the snapshot weights never outlive this member
        align_class_orders(False)
        if state is not None:
            from tabpfn.model_loading import clear_built_model_cache
            clear_built_model_cache()


def finetune_tabpfn(X, Y, nominal, ckpt, seconds, epochs, max_classes=FT_MAX_CLASSES, min_minority=FT_MIN_MINORITY,
                    lr=FT_LR, act_ckpt=True, snap_frac=None):
    """H6: TabPFN fine-tuned on this survey's visible people, every question one task, in memory. Returns the
    new state dict, or None when no question qualifies. Tasks: 2..max_classes answers present and no answer
    above 1 - min_minority. TabPFN's loop takes a list of datasets and preprocesses each one on its own;
    it is handed this survey's questions instead of the single dataset it asks for, and the class count
    (which the estimator otherwise fixes at the first task's) is set per batch."""
    import tabpfn.finetuning.finetuned_base as fb
    from tabpfn.constants import ModelVersion
    from tabpfn.finetuning import FinetunedTabPFNClassifier

    started = time.time()
    FT_SNAP.clear()
    X = np.asarray(X, np.float32)
    ys = []
    for j in range(Y.shape[1]):
        classes, y = np.unique(Y[:, j], return_inverse=True)
        if 2 <= len(classes) <= max_classes and np.bincount(y).max() <= (1 - min_minority) * len(y):
            ys.append(y.astype(np.int64))
    if not ys:
        return None

    class MultiTask(FinetunedTabPFNClassifier):
        def _setup_batch(self, batch):
            cfg = batch.configs
            while isinstance(cfg, (list, tuple)):
                cfg = cfg[0]
            perm = getattr(cfg, "class_permutation", None)
            labels = max(int(t.max()) for t in [*batch.y_context, batch.y_query]) + 1
            n = max(len(perm), labels) if perm is not None else labels
            self.finetuned_estimator_.n_classes_ = n
            self.finetuned_estimator_.classes_ = torch.arange(n)

    step = min(FT_TASKS, len(ys))                        # H8g: questions per loop epoch, cycling through all
    order = np.random.default_rng(SEED).permutation(len(ys))
    calls = [0]

    def chunk():
        start = calls[0] * step
        calls[0] += 1
        return [ys[order[(start + i) % len(ys)]] for i in range(step)]

    ft = MultiTask(device=str(DEVICE), epochs=epochs * -(-len(ys) // step), time_limit=max(1, int(seconds)),
                   learning_rate=lr,
                   weight_decay=0.01, validation_split_ratio=0, early_stopping=False, n_estimators_finetune=2,
                   finetune_ctx_query_split_ratio=0.2, random_state=SEED, save_checkpoint_interval=None,
                   use_activation_checkpointing=act_ckpt,
                   model_version=ModelVersion.V3_5_FAST if "fast" in os.path.basename(ckpt) else ModelVersion.V3_5,
                   extra_classifier_kwargs={"model_path": ckpt, "categorical_features_indices": nominal or None})
    original, cache = fb.get_preprocessed_dataset_chunks, os.environ.get("TABPFN_MODEL_CACHE_SIZE")
    cap_loops, loop_start = epochs * -(-len(ys) // step), []

    def every_task(calling_instance, X_raw, y_raw, *a, **k):   # called once per loop epoch
        now = time.time()
        done = len(loop_start)                           # loops finished before this one (the first includes the build)
        if snap_frac and "state" not in FT_SNAP and done >= 2:   # U35: the weights at snap_frac of the expected loops
            per = (now - loop_start[1]) / (done - 1)
            expected = min(cap_loops, done + max(0.0, seconds - (now - started)) / per)
            if done >= snap_frac * expected:
                FT_SNAP["state"] = {n: v.detach().clone() for n, v in ft.finetuned_estimator_.model_.state_dict().items()}
                FT_SNAP["loop"] = done
        loop_start.append(now)
        part = chunk()
        return original(calling_instance, [X] * len(part), part, *a, **k)

    fb.get_preprocessed_dataset_chunks = every_task
    os.environ["TABPFN_MODEL_CACHE_SIZE"] = "0"          # train a private copy, never the cached network
    oom = False
    try:
        ft.fit(X, ys[0])
        state = {k: v.detach().clone() for k, v in ft.finetuned_estimator_.model_.state_dict().items()}
        FT_STATS.update(passes=round(calls[0] * step / len(ys), 2), questions=len(ys))
    except torch.cuda.OutOfMemoryError:
        if act_ckpt:
            raise
        oom = True
    finally:
        fb.get_preprocessed_dataset_chunks = original
        if cache is None:
            os.environ.pop("TABPFN_MODEL_CACHE_SIZE", None)
        else:
            os.environ["TABPFN_MODEL_CACHE_SIZE"] = cache
        del ft                     # optimiser state and activations: TabPFN sizes its inference to free memory
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if oom:                                              # U27 ckpt: retry with checkpointing for the time left
        FT_STATS["oom_retry"] = True
        return finetune_tabpfn(X, Y, nominal, ckpt, seconds - (time.time() - started), epochs, max_classes,
                               min_minority, lr, act_ckpt=True, snap_frac=snap_frac)
    return state


def install_tabpfn_weights(ckpt, state):
    """Load `state` into the network TabPFN's built-model cache serves for `ckpt`, so every fit in this call uses it."""
    from tabpfn import TabPFNClassifier
    from tabpfn.model_loading import clear_built_model_cache
    os.environ["TABPFN_MODEL_CACHE_SIZE"] = "1"
    clear_built_model_cache()
    probe = TabPFNClassifier(model_path=ckpt, device=str(DEVICE))
    probe._initialize_model_variables()
    models = probe.models_ if isinstance(probe.models_, (list, tuple)) else [probe.models_]
    for model in models:
        model.load_state_dict(state)


def families(d, schema):
    """Groups of >= FAMILY_MIN questions with identical option lists and identical gates:
    checkbox families (same parent) and Likert batteries. Returns [(k, [j, ...]), ...]."""
    groups = {}
    for j, item in enumerate(d.targets):
        rec = schema["items"][item]
        key = (tuple(rec["values"]), repr(sorted(rec["gate"].items())) if rec.get("gate") else None)
        groups.setdefault(key, []).append(j)
    return [(int(d.k[js[0]]), js) for js in groups.values() if len(js) >= FAMILY_MIN]


def fit_tabpfn_family(d, schema):
    """M12: one TabPFN fit per question family on the long table (person x question) with
    the question id as one more categorical column, so the family's questions share
    their context. Questions outside any family get crowd shares (no weight there)."""
    import tabpfn.checkpoint
    from tabpfn import TabPFNClassifier

    tabpfn.checkpoint.Checkpoint.is_safetensors = property(lambda self: True)
    os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", "1")
    ckpt = model_file(TABPFN_REPO, TABPFN_FAST)
    cat = (d.nominal_cols or []) + [d.codes.shape[1]]
    clf = TabPFNClassifier(model_path=ckpt, device=str(DEVICE), n_estimators=1,
                           categorical_features_indices=cat, random_state=SEED)
    X_vis, X_hid = d.codes[d.vis_idx], d.codes[d.hid_idx]
    p_vis = {k: np.zeros((len(d.vis_idx), len(idx), k)) for k, idx in d.buckets.items()}
    p_hid = {k: np.zeros((len(d.hid_idx), len(idx), k)) for k, idx in d.buckets.items()}
    for k, idx in d.buckets.items():                            # default: crowd shares
        for jj, j in enumerate(idx):
            y = d.Y[:, j]
            for f in range(K_FOLDS):
                te = d.folds == f
                p_vis[k][te, jj] = marginal(y[~te], k)
            p_hid[k][:, jj] = marginal(y, k)
    crowd_vis, crowd_hid = ({k: v.copy() for k, v in p.items()} for p in (p_vis, p_hid))
    rng = np.random.default_rng(SEED)
    pos = {k: {j: jj for jj, j in enumerate(idx)} for k, idx in d.buckets.items()}

    def long(X, js):
        n = len(X)
        return np.vstack([np.column_stack([X, np.full(n, q, X.dtype)]) for q in range(len(js))])

    for k, js in families(d, schema):
        Y = d.Y[:, js]                                           # (N_vis, J_fam)
        for f in range(K_FOLDS + 1):
            if time.time() > d.deadline:
                # A2: an interrupted family keeps crowd shares on its fold rows too, so its out-of-fold quality
                # matches its held-out predictions
                cols = [pos[k][j] for j in js]
                p_vis[k][:, cols], p_hid[k][:, cols] = crowd_vis[k][:, cols], crowd_hid[k][:, cols]
                return as_member_output(*[{kk: torch.as_tensor(v, dtype=torch.float32, device=DEVICE)
                                           for kk, v in p.items()} for p in (p_vis, p_hid)])
            tr = np.ones(len(d.vis_idx), bool) if f == K_FOLDS else d.folds != f
            X_long, y_long = long(X_vis[tr], js), Y[tr].T.reshape(-1)
            if len(y_long) > FAMILY_CONTEXT:
                keep = rng.choice(len(y_long), FAMILY_CONTEXT, replace=False)
                X_long, y_long = X_long[keep], y_long[keep]
            if np.unique(y_long).size < 2:
                continue
            clf.fit(X_long, y_long)
            X_out = X_hid if f == K_FOLDS else X_vis[~tr]
            out = np.zeros((len(X_out) * len(js), k))
            out[:, clf.classes_] = clf.predict_proba(long(X_out, js))
            out = out.reshape(len(js), len(X_out), k)
            for q, j in enumerate(js):
                if f == K_FOLDS:
                    p_hid[k][:, pos[k][j]] = out[q]
                else:
                    p_vis[k][~tr, pos[k][j]] = out[q]
    to_t = lambda p: {kk: torch.as_tensor(v, dtype=torch.float32, device=DEVICE) for kk, v in p.items()}
    return as_member_output(to_t(p_vis), to_t(p_hid))


# ----------------------------------------------------------------- mixture --

def em_weights(P, prior=None, strength=0.0):
    """Convex mixture weights maximising the OOF log-likelihood, for many items at once.

    P: (J, N, M) probability each member gave the true class. Returns (J, M), each row
    maximising sum_i log sum_m w_m P[j, i, m] + strength * sum_m prior_m log w_m.
    """
    P = torch.as_tensor(P, dtype=torch.float32, device=DEVICE)
    J, _, M = P.shape
    prior_t = None if prior is None else torch.as_tensor(prior, dtype=torch.float32, device=DEVICE)
    w = torch.full((J, M), 1.0 / M, device=DEVICE) if prior_t is None else prior_t.expand(J, M).clone()
    for _ in range(EM_ITERS):
        r = P * w[:, None, :]
        r = r / r.sum(-1, keepdim=True)
        new = r.sum(1) + (strength * prior_t if prior_t is not None else 0.0)
        new = new / new.sum(-1, keepdim=True)
        done = (new - w).abs().max() < 1e-7
        w = new
        if done:
            break
    return w.cpu().numpy()


def linear_mixture(P, strength):
    """Survey-wide mixture weights, then per-question weights pulled toward them by `strength` pseudo-rows."""
    w_g = em_weights(P.reshape(1, -1, P.shape[-1]))[0]
    return w_g, em_weights(P, w_g, strength)


def crossfit_mixture(d, P, strength):
    """H2: (J, N_vis) mixture probability of the true answer, each row from weights fitted without its fold."""
    out = np.empty(P.shape[:2])
    for f in range(K_FOLDS):
        te = d.folds == f
        _, W = linear_mixture(P[:, ~te], strength)
        out[:, te] = (P[:, te] * W[:, None, :]).sum(-1)
    return out


def routing_families(schema, d):
    """U23e: family id per question: a gate parent and every question it gates (transitively) share an id;
    questions outside any family (or in a one-question family) share id 0."""
    items = schema["items"]

    def root(i):
        seen = set()
        while items.get(i, {}).get("gate") and i not in seen:
            seen.add(i)
            i = items[i]["gate"]["parent"]
        return i
    parents = {items[t]["gate"]["parent"] for t in d.targets if items[t].get("gate")}
    keys = [root(t) if (items[t].get("gate") or t in parents) else None for t in d.targets]
    ids, out = {}, np.zeros(len(d.targets), int)
    for j, k in enumerate(keys):
        if k is not None and keys.count(k) >= 2:
            out[j] = ids.setdefault(k, len(ids) + 1)
    return out


def family_mixture(P, fam, strength, n_vis):
    """U23e: survey EM weights -> family weights (pulled to the survey's by FAMILY_PULL x visible rows) -> question
    weights (pulled to their family's by `strength` pseudo-rows). P (J, N, M). Returns (J, M)."""
    J, N, M = P.shape
    w_s = em_weights(P.reshape(1, -1, M))[0]
    W = np.zeros((J, M))
    for f in np.unique(fam):
        jj = np.flatnonzero(fam == f)
        W[jj] = em_weights(P[jj], em_weights(P[jj].reshape(1, -1, M), w_s, FAMILY_PULL * n_vis)[0], strength)
    return W


def fit_log_pool(d, oof_buckets, hid_buckets, w_init):
    """M10: geometric pool p(a) ∝ exp(Σ_m w_m log p_m(a)), weights unconstrained.

    oof_buckets / hid_buckets: per member, {k: (N, J_k, k) probabilities} (own-fold OOF for
    visible rows, full-data model for held-out rows). Weights are fitted first survey-wide,
    then per question with a ridge toward the survey-wide ones (with d.family: toward their family's, which are
    pulled toward the survey-wide ones by FAMILY_RIDGE), as a batch of K_FOLDS + 1
    problems: fold problems hide their fold, so each visible row's pooled OOF probability
    comes from weights that never saw it. Returns (OOF true-class probabilities (N_vis, J),
    {question index: held-out distribution (N_hid, k)}, survey-wide weights of the full problem).
    """
    folds = torch.as_tensor(d.folds, device=DEVICE)
    rows = torch.arange(len(d.folds), device=DEVICE)
    to_log = lambda ps: torch.log(torch.clamp(torch.stack(ps, 0), min=LOG_FLOOR))   # (M, N, J_k, k)
    L = {k: to_log([torch.as_tensor(b[k], device=DEVICE, dtype=torch.float32) for b in oof_buckets])
         for k in d.buckets}
    B, M = K_FOLDS + 1, len(oof_buckets)

    def nll(weights_of):                                         # summed over problems, rows, items
        total = 0.0
        for k, idx in d.buckets.items():
            z = torch.log_softmax(torch.einsum("mnjk,bjm->bnjk", L[k], weights_of(k)), -1)
            y = d.Y_t[:, idx].unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1)
            total = total - (z.gather(-1, y).squeeze(-1).sum(-1) * d.mask).sum()
        return total

    def run(params, loss_fn):
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=LOGPOOL_ITERS, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            loss = loss_fn()
            loss.backward()
            return loss
        opt.step(closure)

    g = torch.as_tensor(w_init, dtype=torch.float32, device=DEVICE).expand(B, M).clone().requires_grad_()
    run([g], lambda: nll(lambda k: g[:, None, :].expand(B, len(d.buckets[k]), M)))
    g = g.detach()
    W = {k: g[:, None, :].expand(B, len(idx), M).clone().requires_grad_() for k, idx in d.buckets.items()}
    if d.family is None:
        run(list(W.values()), lambda: nll(lambda k: W[k])
            + LOGPOOL_RIDGE * sum(((W[k] - g[:, None, :]) ** 2).sum() for k in d.buckets))
    else:                                                        # U23e: question -> family -> survey ridges
        fid = torch.as_tensor(d.family, device=DEVICE)
        G = g[:, None, :].expand(B, int(d.family.max()) + 1, M).clone().requires_grad_()
        run(list(W.values()) + [G], lambda: nll(lambda k: W[k])
            + LOGPOOL_RIDGE * sum(((W[k] - G[:, fid[torch.as_tensor(idx, device=DEVICE)], :]) ** 2).sum()
                                  for k, idx in d.buckets.items())
            + FAMILY_RIDGE * ((G - g[:, None, :]) ** 2).sum())

    oof, hid = np.empty((len(d.vis_idx), len(d.targets))), {}
    with torch.no_grad():
        for k, idx in d.buckets.items():
            w = W[k].detach()
            p_own = torch.softmax(torch.einsum("mnjk,njm->njk", L[k], w[folds]), -1)
            oof[:, idx] = p_own[rows[:, None], torch.arange(len(idx), device=DEVICE), d.Y_t[:, idx]].cpu().numpy()
            Lh = to_log([torch.as_tensor(np.stack([b[j] for j in idx], 1), device=DEVICE, dtype=torch.float32)
                         for b in hid_buckets])
            p_hid = torch.softmax(torch.einsum("mnjk,jm->njk", Lh, w[-1]), -1).cpu().numpy()
            for jj, j in enumerate(idx):
                hid[j] = p_hid[:, jj]
    return oof, hid, g[-1].cpu().numpy()


# Measured member seconds per instrument (INSTRUMENT_COST: v51's members on the H100, fine-tuning excluded).
# The grader's call order is unknown, so each call gets a share of the remaining time proportional to its own
# cost against the instruments still to come, instead of an equal third: UNHCR, the slowest, cannot be
# squeezed if it comes first.
INSTRUMENT_ROWS = {10349: "unhcr", 19846: "unicef", 6927: "world_bank"}
INSTRUMENT_ITEMS = {"s2_1c_education_level_lbl": "unhcr", "user.educationLevel": "world_bank", "Q09": "unicef"}   # Q09: only UNICEF has it (Q13 is also afro/arab's)


def instrument(schema):
    """The instrument by its row count, else by a characteristic item (U4: in case a schema is re-issued)."""
    name = INSTRUMENT_ROWS.get(schema.get("dataset", {}).get("n_rows"))
    return name or next((v for k, v in INSTRUMENT_ITEMS.items() if k in schema.get("items", {})), None)
# v51 members on the H100, fine-tuning excluded; UNHCR + 28 s for 600 network steps (U23a), + 5 s family shrinkage (U23e)
INSTRUMENT_COST = {1: {"unhcr": 303, "unicef": 185, "world_bank": 163},
                   2: {"unhcr": 1153 + 28 + 5, "unicef": 174, "world_bank": 342}}


def time_share(schema, phase, left):
    """Seconds this call may use out of `left`; equal shares for unknown instruments."""
    name = instrument(schema)
    seen = _CLOCK.setdefault("seen", set())
    calls_left = max(1, N_CALLS - _CLOCK["calls"])
    _CLOCK["calls"] += 1
    if name is None:
        return left / calls_left
    seen.add(name)
    cost = INSTRUMENT_COST[phase]
    rest = sum(c for n, c in cost.items() if n not in seen)
    proportional = left * cost[name] / (cost[name] + rest)
    # a cheap instrument keeps the old equal share while the ones still to come fit with a 1.5x margin
    # the first instruments may use what is left after reserving 1.15x the measured needs of the rest
    return max(proportional, min(left / calls_left, left - 1.15 * rest))


# U1 (results/u1_kit): failed final submissions do not count and may be retried (starter-kit README:113-116), so in the
# final phase a run that would finish degraded fails instead, with a distinct exception type (the only diagnostic the
# hidden run reports); the retry is the lenient v99_lenient.
LOUD_FINAL = True


class FinalFallback(RuntimeError):
    pass


class FinalNoTabPFN(RuntimeError):
    pass


class FinalNoFinetune(RuntimeError):
    pass


def _predict(frame, schema):
    global K_FOLDS
    started = time.time()
    targets = [i for i in shipped_items(schema, frame) if schema["items"][i]["class"] == "PREDICT"]
    n_visible = int(frame[targets].notna().all(axis=1).sum())
    final = n_visible > schema.get("split", {}).get("n_train", 10 ** 9)
    _CLOCK["final"] = final
    # 10 folds only where they fit: the in-context members' cost grows with the visible rows, and on
    # UNICEF (about 18k) 10 folds overran its time share while adding nothing measurable (+0.00005).
    K_FOLDS = K_FINAL if final and n_visible <= FOLDS10_MAX_ROWS and len(targets) <= FOLDS10_MAX_ITEMS \
        else K_BASE
    d = Design(frame, schema)
    inst = instrument(schema)
    phase = 2 if d.phase2 else 1
    left = TOTAL_SECONDS[phase] - (started - _CLOCK["start"])
    d.deadline = started + max(30.0, time_share(schema, phase, left) - SAFETY_SECONDS)
    share = d.deadline - started
    member_cost = INSTRUMENT_COST[phase].get(inst)
    ft_room = share - 1.15 * member_cost if member_cost else 0.0            # what the members leave (v52)
    dev_ft = FT_DEV and inst not in FT_DEV_SKIP
    d.ft_seconds = max(FT_SHARE * share, ft_room) if d.phase2 or dev_ft else 0.0   # H6: final phase
    J, n_hid = len(d.targets), len(d.hid_idx)
    names, oofs, preds = [], [], []

    # Marginal member (numpy): OOF from fold complements, prediction from all rows.
    oof = np.empty((len(d.vis_idx), J))
    pred = []
    for j in range(J):
        y, k = d.Y[:, j], d.k[j]
        for f in range(K_FOLDS):
            te = d.folds == f
            oof[te, j] = marginal(y[~te], k)[y[te]]
        pred.append(np.tile(marginal(y, k), (n_hid, 1)))
    names.append("marginal"), oofs.append(oof), preds.append(pred)
    buckets = []                                                  # per member: {k: (N_vis, J_k, k)}, for the pool
    bucket = {}
    for k, idx in d.buckets.items():
        b = np.empty((len(d.vis_idx), len(idx), k), np.float32)
        for jj, j in enumerate(idx):
            for f in range(K_FOLDS):
                te = d.folds == f
                b[te, jj] = marginal(d.Y[~te, j], k)
        bucket[k] = b
    buckets.append(bucket)

    d.ft_full = inst in FT_FULL_FOR                                        # U2
    if d.ft_seconds > 0 and d.ft_full:                                     # U15
        d.ft_seconds += FT_PLUS.get(inst, 0.0)
    d.ft_max_classes = int(FT_MAX_CLASSES_BY.get(inst, FT_MAX_CLASSES))
    d.ft_min_minority = FT_MIN_MINORITY_BY.get(inst, FT_MIN_MINORITY)
    d.ft_lr = FT_LR_BY.get(inst, FT_LR)
    d.ft_act_ckpt = inst not in FT_NO_CKPT
    d.tabpfn_abs = inst in TABPFN_ABS
    d.snap = TABPFN_SNAP.get(inst) if d.phase2 else None                  # U35
    d.inst = inst
    if d.snap and d.ft_seconds > 0:
        d.ft_seconds = max(1.0, d.ft_seconds - SNAP_SECONDS.get(inst, 0.0))
    d.prof = None
    if inst in PROF_INSTRUMENTS:
        try:
            d.prof = level_profiles(frame, schema, d)
            if DEBUG:
                print(f"  answer profiles: {LAST_DIAGNOSTICS.get('profile_facts')}")
        except Exception:
            if DEBUG or (STRICT_DEFAULT and not d.phase2):
                raise
    # Most valuable first, so the time guard drops the cheapest members. TabPFN comes first because the
    # networks and linear heads distil from it; a generator, so the network grid can follow its fine-tune (U24g).
    def member_jobs():
        yield "tabpfn", fit_tabpfn, (d,)
        grid = (MLP_GRID_BY.get(inst) if getattr(d, "finetuned", False) else None) \
            or [g + (MLP_SEEDS,) for g in MLP_GRID]
        yield from ((f"mlp_{f}_{w}_{p:g}", fit_mlp, (d, w, wd, p, f, n))
                    for w, wd, p, f, n in grid if f.split("+")[0] in d.X
                    and not (f.endswith("+hurdle") and not d.gated.any()))
        if d.phase2 and inst not in TABFM_SKIP:                  # U15
            yield "tabfm", fit_tabfm, (d,)
        yield from ((f"linear_{f}_{lam:g}", fit_linear, (d, f, lam)) for f, lam in LINEAR if f in d.X)
        yield from ((f"latent_{C}", fit_latent, (d, C)) for C in LATENT_CLASSES)
        if families(d, schema):
            yield "tabpfn_family", fit_tabpfn_family, (d, schema)
    strict = STRICT_DEFAULT and not d.phase2
    _CLOCK["strict"] = strict
    if strict:
        smoke_final_members(d)
    for name, fit, args in member_jobs():
        if time.time() > d.deadline:            # strict mode does not fail on time: the development
            continue                            # budget says nothing about the final's (4x longer)
        try:
            t = time.time()
            if name.startswith(("mlp", "linear")):
                finetuned = getattr(d, "finetuned", False)
                d.mlp_steps = MLP_STEPS_BY.get(inst, MLP_STEPS) if finetuned else MLP_STEPS
                distill_targets(d, names, preds, buckets,
                                DISTILL_LAMBDA_BY.get(inst, DISTILL_LAMBDA) if finetuned else DISTILL_LAMBDA)
            try:
                logits_vis, logits_hid = fit(*args)
            finally:
                d.soft_W = d.soft_T = None
            oof, pred, bucket = np.empty((len(d.vis_idx), J)), [None] * J, {}
            collect(logits_vis, logits_hid, d, oof, pred, bucket)
            del logits_vis, logits_hid
            # A2: one non-finite cell would make every question's blend weights NaN; such a member is dropped
            if not (np.isfinite(oof).all() and all(p is None or np.isfinite(p).all() for p in pred)
                    and all(torch.isfinite(t).all().item() for t in bucket.values())):
                raise FloatingPointError(f"{name}: non-finite predictions")
            names.append(name), oofs.append(oof), preds.append(pred)
            buckets.append(bucket)
            if DEBUG:
                print(f"    {name}: {time.time() - t:.1f}s  oof ll {np.log(oof + 1e-12).mean():.4f}")
        except Exception:
            if DEBUG or strict:
                raise
    if LOUD_FINAL and d.phase2 and _CLOCK.get("final"):
        if "tabpfn" not in names:
            raise FinalNoTabPFN()
        if not getattr(d, "finetuned", False):
            raise FinalNoFinetune()

    P = np.stack(oofs, -1).transpose(1, 0, 2) + 1e-6             # (J, N_vis, M)
    # H47: the fine-tuned TabPFN's weights saw every visible row, so its out-of-fold likelihood is optimistic
    # by about FT_DOSE_C x passes x fine-tuning questions / visible rows nats per cell (measured on 4 splits x
    # 4 proxies). Every fitting stage reads its true-class OOF probability discounted by that much.
    ft_delta = (FT_DOSE_BY.get(inst, FT_DOSE_C) * FT_STATS.get("passes", 0) * FT_STATS.get("questions", 0) / len(d.vis_idx)
                if getattr(d, "finetuned", False) else 0.0)
    ft_idx = [m for m, n in enumerate(names) if n == "tabpfn"]
    if ft_delta > 0 and ft_idx:
        P[:, :, ft_idx] *= np.exp(-ft_delta)
        if len(buckets) == len(names):
            for m in ft_idx:
                for k, idx in d.buckets.items():
                    t = torch.as_tensor(buckets[m][k], device=DEVICE, dtype=torch.float32).clone()
                    y = d.Y_t[:, idx].unsqueeze(-1)
                    t.scatter_(-1, y, t.gather(-1, y) * float(np.exp(-ft_delta)))
                    buckets[m][k] = t
    xfit = {s: crossfit_mixture(d, P, s) for s in WPRIOR_GRID}   # H2: shrinkage by cross-fitted likelihood
    strength = max(WPRIOR_GRID, key=lambda s: np.log(xfit[s]).sum())
    w_global, W = linear_mixture(P, strength)
    mixed_oof = xfit[strength]                                    # (J, N_vis), cross-fitted (H2)
    d.family = None
    if inst in FAMILY_SHRINK:
        fam = routing_families(schema, d)                          # U23e: shrink through routing families
        if fam.max() > 0:
            d.family = fam
            W = family_mixture(P, fam, strength, len(d.vis_idx))
            mixed_oof = np.empty(P.shape[:2])
            for f in range(K_FOLDS):                               # cross-fitted OOF mixture, as H2's
                te = d.folds == f
                Wf = family_mixture(P[:, ~te], fam, strength, len(d.vis_idx))
                mixed_oof[:, te] = (P[:, te] * Wf[:, None, :]).sum(-1)
    uniform = np.mean(np.log(d.k))
    LAST_DIAGNOSTICS.clear()
    LAST_DIAGNOSTICS["wprior"] = strength
    LAST_DIAGNOSTICS["ft_delta"] = ft_delta
    LAST_DIAGNOSTICS["finetuned"] = getattr(d, "finetuned", False)
    LAST_DIAGNOSTICS.update({
        "cv_skill_linear": float(1 + np.log(mixed_oof).mean() / uniform),
        "weights": dict(zip(names, np.round(w_global, 3).tolist())),
    })
    probs = {}
    for j, item in enumerate(d.targets):
        w = W[j]
        full = np.zeros((len(frame), d.k[j]))
        full[d.hid_idx] = sum(wm * p[j] for wm, p in zip(w, preds))
        probs[item] = full

    # M10: geometric pool, then a per-question EM between the linear mixture and the pool. U32: it also runs past this
    # call's deadline (members and the fine-tune may overrun the share on a slower GPU) while the run's clock allows
    pool_until = max(d.deadline + SAFETY_SECONDS / 2, _CLOCK["start"] + TOTAL_SECONDS[phase] - POOL_RESERVE)
    if len(buckets) == len(preds) and time.time() < pool_until:
        try:
            t = time.time()
            pool_oof, pool_hid, g = fit_log_pool(d, buckets, preds, w_global)
            if not (np.isfinite(pool_oof).all() and all(np.isfinite(h).all() for h in pool_hid.values())):
                raise FloatingPointError("log pool: non-finite")   # U24a: keep the linear mixture
            P2 = np.stack([mixed_oof, pool_oof.T], -1) + 1e-6     # (J, N_vis, 2)
            v_global = em_weights(P2.reshape(1, -1, 2))[0]
            V = em_weights(P2, v_global, WEIGHT_PRIOR)
            mixed_oof = (P2 * V[:, None, :]).sum(-1)
            for j, item in enumerate(d.targets):
                probs[item][d.hid_idx] = V[j, 0] * probs[item][d.hid_idx] + V[j, 1] * pool_hid[j]
            LAST_DIAGNOSTICS.update({"pool_share": float(v_global[1]), "pool_weight_sum": float(g.sum()),
                                     "pool_seconds": round(time.time() - t, 1)})
        except Exception:
            if DEBUG:
                raise
    del buckets
    LAST_DIAGNOSTICS["cv_skill"] = float(1 + np.log(mixed_oof).mean() / uniform)
    LAST_DIAGNOSTICS["verified_rules"] = apply_verified_routing(frame, schema, d, probs)
    for item in d.targets:                                        # pre-compensate the grader's floor
        p = np.clip(probs[item][d.hid_idx] - FLOOR, 0.0, None)
        probs[item][d.hid_idx] = p / p.sum(1, keepdims=True)
    if DEBUG:
        top = sorted(zip(w_global, names), reverse=True)[:6]
        print(f"  verified routing rules: {LAST_DIAGNOSTICS['verified_rules']}; members: {len(names)}; "
              f"budget left {d.deadline - time.time():.0f}s, run left "
              f"{_CLOCK['start'] + TOTAL_SECONDS[phase] - time.time():.0f}s")
        if "pool_share" not in LAST_DIAGNOSTICS:
            print("  log pool: skipped")
        else:
            print(f"  log pool: share {LAST_DIAGNOSTICS['pool_share']:.2f}, weight sum "
                  f"{LAST_DIAGNOSTICS['pool_weight_sum']:.2f}, {LAST_DIAGNOSTICS['pool_seconds']}s, in-run CV "
                  f"{LAST_DIAGNOSTICS['cv_skill_linear']:.4f} -> {LAST_DIAGNOSTICS['cv_skill']:.4f}")
        print(f"  {DEVICE} {time.time() - started:.0f}s  weights: "
              + ", ".join(f"{n}={w:.2f}" for w, n in top))
    vectors = assemble(frame, schema, probs)
    if _CLOCK.get("strict") and FT_SHARE > 0 and _CLOCK["calls"] >= N_CALLS:
        smoke_finetune(d)                     # after the last call's output: never displaces a member
    return vectors


def apply_verified_routing(frame, schema, d, probs):
    """Overwrite P(not asked) where a GIVEN-parent routing rule holds in the visible rows.

    Returns the number of (question, direction) rules applied.
    """
    applied = 0
    for j, item in enumerate(d.targets):
        gate = schema["items"][item].get("gate")
        if not gate or schema["items"].get(gate["parent"], {}).get("class") != "GIVEN":
            continue
        k = d.k[j]
        routed_in = frame[gate["parent"]].isin(gate["observed_if"]).to_numpy()
        known = frame[gate["parent"]].notna().to_numpy()      # U4: a NaN parent is neither routed in nor out
        not_asked = d.Y[:, j] == k - 1                        # visible rows, sentinel is last
        vis_in, hid_in = routed_in[d.vis_idx], routed_in[d.hid_idx]
        vis_k, hid_k = known[d.vis_idx], known[d.hid_idx]
        # Routed out: the rule says "not asked". Routed in: the rule says "answered".
        for routed, vis_mask, hid_mask in ((False, ~vis_in & vis_k, ~hid_in & hid_k), (True, vis_in, hid_in)):
            n = int(vis_mask.sum())
            if n < MIN_ROWS or not hid_mask.any():
                continue
            broken = not_asked[vis_mask] if routed else ~not_asked[vis_mask]
            if broken.mean() > MAX_VIOLATION:
                continue
            rate = (broken.sum() + 0.5) / (n + 1)               # smoothed chance the rule is broken
            p_na = rate if routed else 1 - rate
            rows = d.hid_idx[hid_mask]
            p = probs[item][rows]
            body = p[:, :-1] / np.clip(p[:, :-1].sum(1, keepdims=True), 1e-12, None)
            probs[item][rows] = np.column_stack([body * (1 - p_na), np.full(len(rows), p_na)])
            applied += 1
    return applied


def smoke_finetune(d, rows=300):
    """Strict development uploads, once, at the end of the last call (H6): fine-tune TabPFN on a few rows
    and questions, swap the weights in, fit on them and restore the original, so a broken fine-tuning path
    fails the development upload rather than silently falling back in the final."""
    import tabpfn.checkpoint
    from tabpfn import TabPFNClassifier
    from tabpfn.model_loading import clear_built_model_cache
    tabpfn.checkpoint.Checkpoint.is_safetensors = property(lambda self: True)
    j = next(j for j in range(len(d.targets)) if np.unique(d.Y[:rows, j]).size >= 2)
    X, y = d.codes[d.vis_idx[:rows]], d.Y[:rows, j]
    ckpt = model_file(TABPFN_REPO, TABPFN_FAST)
    t = time.time()
    state = finetune_tabpfn(X, d.Y[:rows, :8], d.nominal_cols, ckpt, 60, 1)
    assert state is not None, "no question qualifies for fine-tuning"
    install_tabpfn_weights(ckpt, state)
    TabPFNClassifier(model_path=ckpt, device=str(DEVICE), n_estimators=1, random_state=SEED).fit(X, y).predict_proba(X[:10])
    clear_built_model_cache()
    if DEBUG:
        print(f"  fine-tuning smoke: {time.time() - t:.1f}s")


def smoke_final_members(d, rows=300):
    """Strict development uploads: load the final-only models from the shipped weights and run them
    on a few rows of one question, so a missing file or broken import shows up before the final."""
    started = time.time()
    j = next(j for j in range(len(d.targets)) if np.unique(d.Y[:rows, j]).size >= 2)
    X, y = d.codes[d.vis_idx[:rows]], d.Y[:rows, j]
    from tabicl import TabICLClassifier
    from tabpfn import TabPFNClassifier
    import tabpfn.checkpoint
    tabpfn.checkpoint.Checkpoint.is_safetensors = property(lambda self: True)
    TabICLClassifier(model_path=model_file(TABICL_REPO, TABICL_CKPT), allow_auto_download=False,
                     device=str(DEVICE), n_estimators=TABICL_ESTIMATORS, random_state=SEED).fit(X, y).predict_proba(X[:10])
    ckpt = model_file(TABPFN_REPO, TABPFN_FAST)
    TabPFNClassifier(model_path=ckpt, device=str(DEVICE),
                     n_estimators=TABPFN_ESTIMATORS[2], random_state=SEED).fit(X, y).predict_proba(X[:10])
    from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1   # package from requirements.txt, weights from models.txt
    model = tabfm_v1.load(model_type="classification", device=str(DEVICE),
                          checkpoint_path=tabfm_path())
    _TabFMAdapter(model, d.nominal_cols).fit(X, y).predict_proba(X[:10])
    if DEBUG:
        print(f"  smoke: {time.time() - started:.1f}s in total")


def predict(frame, schema):
    try:
        return _predict(frame, schema)
    except (FinalNoTabPFN, FinalNoFinetune):
        raise
    except Exception as e:
        if DEBUG or _CLOCK.get("strict"):
            raise
        if LOUD_FINAL and _CLOCK.get("final"):
            raise FinalFallback() from e
        return _marginal_fallback(frame, schema)


def _marginal_fallback(frame, schema):
    """Last line of defence: the smoothed crowd marginal, which cannot fail."""
    probs = {}
    targets = [i for i, r in schema["items"].items() if r["class"] == "PREDICT" and i in frame.columns]
    visible = frame[targets].notna().all(axis=1).to_numpy()   # v57: PREDICT columns only, as in Design
    for item in shipped_items(schema, frame):
        options = item_options(schema, item)
        counts = frame.loc[visible, item].value_counts()
        m = np.array([counts.get(o, 0) + SMOOTH for o in options], float)
        probs[item] = np.broadcast_to(m / m.sum(), (len(frame), len(options)))
    return assemble(frame, schema, probs)
