"""Metrics with explicit tie handling and paired study resampling."""
import numpy as np
from scipy.stats import rankdata

LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
          'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
UID = 'StudyInstanceUID'

def percentile_rank(scores):
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError('Expected a nonempty, finite 2D score matrix')
    return rankdata(x, axis=0, method='average') / len(x)

def validate(truth, scores):
    y, p = np.asarray(truth), np.asarray(scores)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != len(LABELS):
        raise ValueError('Expected matching N x 12 truth and scores')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError('Truth must be binary and scores finite')
    if np.any(y.sum(0) == 0) or np.any(y.sum(0) == len(y)):
        raise ValueError('Every label must contain positives and negatives')

def per_class_auc(truth, scores):
    validate(truth, scores)
    y, p = np.asarray(truth), np.asarray(scores)
    npos = y.sum(0)
    ranks = rankdata(p, axis=0, method='average')
    return ((ranks * y).sum(0) - npos * (npos + 1) / 2) / (npos * (len(y) - npos))

def draw_counts(truth, n_resamples=2000, seed=20261021):
    """Ordinary paired study bootstrap; reject complete draws missing any class."""
    y = np.asarray(truth, dtype=int)
    rng = np.random.default_rng(seed)
    counts, attempted = [], 0
    while len(counts) < n_resamples:
        attempted += 1
        if attempted > n_resamples * 100:
            raise ValueError('Too few valid joint resamples; revise uncertainty method')
        ix = rng.integers(0, len(y), len(y))
        c = np.bincount(ix, minlength=len(y))
        positives = c @ y
        if np.all((positives > 0) & (positives < len(y))):
            counts.append(c)
    return np.asarray(counts), attempted

def bootstrap_auc(truth, scores, counts):
    """Exact weighted positive-negative comparisons for each bootstrap draw.

    Multiplicities reproduce indexing scores with sampled study IDs, including
    ties. All labels and all model variants share the same study multiplicities.
    """
    validate(truth, scores)
    y, p = np.asarray(truth), np.asarray(scores)
    values = np.empty((len(counts), y.shape[1]))
    for j in range(y.shape[1]):
        pos, neg = y[:, j] == 1, y[:, j] == 0
        delta = p[pos, j, None] - p[neg, j][None, :]
        wins = (delta > 0).astype(float) + .5 * (delta == 0)
        cp, cn = counts[:, pos], counts[:, neg]
        values[:, j] = np.einsum('bi,ij,bj->b', cp, wins, cn, optimize=True) / (cp.sum(1) * cn.sum(1))
    return values

def operating_point(truth, scores, threshold=.5):
    """Descriptive rank-score cutoff, NOT a calibrated clinical probability."""
    y, pred = np.asarray(truth, dtype=bool), percentile_rank(scores) >= threshold
    tp = (pred & y).sum(0); fn = (~pred & y).sum(0)
    fp = (pred & ~y).sum(0); tn = (~pred & ~y).sum(0)
    return dict(tp=tp, fn=fn, fp=fp, tn=tn,
                recall=tp / (tp + fn), specificity=tn / (tn + fp),
                precision=np.divide(tp, tp + fp, out=np.full(len(tp), np.nan), where=(tp + fp) > 0))
