"""Frozen V13/V15 production fusion, including its float32 CSV boundary."""
from io import StringIO
import numpy as np
import pandas as pd
from .metrics import LABELS, percentile_rank as rank

RAPTOR_WEIGHT = np.array([.5, .5, .5, .5, .5, .2, .6, .5, .8, .35, .5, .5])
READER_WEIGHT = np.array([.025, .02, .02, .008, .025, .008, .008, .004, .003, .035, .03, .008])

def csv_boundary(x):
    # Production serializes float32 scores, then reloads with pandas' default
    # parser before the last rank fusion. Preserve that tie-sensitive boundary.
    return pd.read_csv(StringIO(pd.DataFrame(np.asarray(x, dtype=np.float32)).to_csv(index=False))).to_numpy()

def fusion(branches, *, use_224=True, use_r6=True, use_reader=True,
           equal_class_weights=False, extra=None):
    raptor = rank(branches['384'])
    if use_224:
        raptor = .65 * raptor + .35 * rank(branches['224'])
    if extra is not None:
        raptor = .50 * rank(branches['384']) + .25 * rank(branches['224']) + .25 * rank(branches[extra])
    wr = np.full(12, .5) if equal_class_weights else RAPTOR_WEIGHT
    first = wr * raptor + (1 - wr) * rank(branches['r6']) if use_r6 else raptor
    first = csv_boundary(first)
    if not use_reader:
        return rank(first)
    return rank((1 - READER_WEIGHT) * rank(first) + READER_WEIGHT * rank(branches['reader']))

def variants(branches):
    result = {
        'raptor384_single': rank(branches['384']),
        'raptor224_single': rank(branches['224']),
        'r6_fourfold': rank(branches['r6']),
        'reader_threefold': rank(branches['reader']),
        'raptor384_224': rank(.65 * rank(branches['384']) + .35 * rank(branches['224'])),
        'baseline_v13_v15': fusion(branches),
        'without_224': fusion(branches, use_224=False),
        'without_r6': fusion(branches, use_r6=False),
        'without_reader': fusion(branches, use_reader=False),
        'equal_class_weights': fusion(branches, equal_class_weights=True),
    }
    for name in ['royalacecat', 'syouyatobita_g233', 'kawaharataishi_swa', 'tbongc']:
        if name in branches:
            result[f'extra_{name}'] = fusion(branches, extra=name)
            result[f'single_{name}'] = rank(branches[name])
    return result
