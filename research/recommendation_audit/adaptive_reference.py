#!/usr/bin/env python3
"""Independent SciPy reference for progression-v2 LeetCode cells (design §4.1)."""
import json
import math
from pathlib import Path

import scipy
from scipy.stats import beta as beta_distribution

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'hosted/test/fixtures/adaptive-cells.v1.json'
PRIORS = ((3.2, 0.8), (2.0, 2.0), (0.8, 3.2))
LEVELS = (1, 2, 3)


def reference(observations, concept):
    ability = [o for o in observations if o['kind'] == 'ability']
    cold = not any(concept in o['concepts'] for o in ability)
    cells = []
    previous_delta = 0.0
    for level, (prior_a, prior_b) in zip(LEVELS, PRIORS):
        outside = [o for o in ability if o['level'] == level and concept not in o['concepts']]
        own = [o for o in ability if o['level'] == level and concept in o['concepts']]
        used_outside = [] if cold else outside
        mean = (prior_a + sum(o['y'] for o in used_outside)) / (prior_a + prior_b + len(used_outside))
        centre = 1 / (1 + math.exp(-(math.log(mean / (1 - mean)) + 0.5 * previous_delta)))
        a = 4 * centre + sum(o['y'] / len(o['concepts']) for o in own)
        b = 4 * (1 - centre) + sum((1 - o['y']) / len(o['concepts']) for o in own)
        posterior = a / (a + b)
        previous_delta = math.log(posterior / (1 - posterior)) - math.log(centre / (1 - centre)) if own else 0.0
        cells.append({'level': level, 'alpha': a, 'beta': b, 'n_own': len(own),
                      'n_level_outside': len(outside), 'prior': 'learner' if used_outside else 'authored'})
    # Weighted PAV: each contiguous block has its summed beta pseudo-counts.
    blocks = []
    for i, cell in enumerate(cells):
        blocks.append({'indices': [i], 'alpha': cell['alpha'], 'beta': cell['beta']})
        while len(blocks) >= 2:
            left, right = blocks[-2:]
            if left['alpha'] / (left['alpha'] + left['beta']) >= right['alpha'] / (right['alpha'] + right['beta']):
                break
            blocks[-2:] = [{'indices': left['indices'] + right['indices'],
                            'alpha': left['alpha'] + right['alpha'], 'beta': left['beta'] + right['beta']}]
    for block in blocks:
        a, b = block['alpha'], block['beta']
        for i in block['indices']:
            cells[i].update(alpha=a, beta=b, p=a / (a + b),
                            interval=[float(beta_distribution.ppf(0.1, a, b)), float(beta_distribution.ppf(0.9, a, b))])
    return cells


def observation(level, concepts, y, kind='ability'):
    return {'level': level, 'concepts': concepts, 'y': y, 'kind': kind}


def main():
    o = observation
    cases = [
        ('empty', []),
        ('cold_with_outside_history', [o(1, ['other'], 0)] * 6 + [o(2, ['other'], 1)] * 4),
        ('easy_two_solves', [o(1, ['target'], 1)] * 2),
        ('easy_five_solves', [o(1, ['target'], 1)] * 5),
        ('medium_failure', [o(1, ['target'], 1)] * 2 + [o(2, ['target'], 0)]),
        ('easy_mixed', [o(1, ['target'], 1), o(1, ['target'], 0), o(1, ['target'], 0)]),
        ('outside_evidence', [o(1, ['other'], 0), o(2, ['other'], 1), o(2, ['target'], 1)]),
        ('multi_concept', [o(1, ['target', 'other', 'third'], 1), o(2, ['target', 'other'], 0)]),
        ('pav_violation', [o(1, ['target'], 0)] * 8 + [o(2, ['target'], 1)] * 8 + [o(3, ['target'], 1)] * 8),
        ('retention_ignored', [o(1, ['target'], 1), o(1, ['target'], 0, 'retention')]),
        ('outside_shared_excluded', [o(1, ['target', 'other'], 1), o(1, ['other'], 0)]),
    ]
    rows = [{'name': name, 'observations': observations, 'concept': 'target',
             'cells': reference(observations, 'target')} for name, observations in cases]
    OUTPUT.write_text(json.dumps({'version': 'adaptive-cells-v1', 'scipy_version': scipy.__version__, 'rows': rows},
                                 indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'{OUTPUT.relative_to(ROOT)}: {len(rows)} scenarios, scipy {scipy.__version__}')


if __name__ == '__main__':
    main()
