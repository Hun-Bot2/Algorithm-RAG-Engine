#!/usr/bin/env python3
"""Generate deterministic SciPy reference values for progression-v2 beta math."""
import json
from pathlib import Path

import scipy
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'hosted/test/fixtures/beta-reference.v1.json'
SHAPES = (0.5, 0.8, 2.0, 3.2, 4.0, 20.0, 60.0)
POINTS = (0.01, 0.05, 0.2, 0.5, 0.7, 0.95, 0.99)


def main():
    rows = []
    for i, alpha in enumerate(SHAPES):
        for j, beta_shape in enumerate(SHAPES):
            x = POINTS[(i + j) % len(POINTS)]
            probability = POINTS[(2 * i + j) % len(POINTS)]
            rows.append({
                'alpha': alpha,
                'beta': beta_shape,
                'x': x,
                'cdf': float(beta.cdf(x, alpha, beta_shape)),
                'quantile_probability': probability,
                'quantile': float(beta.ppf(probability, alpha, beta_shape)),
            })
    fixture = {'version': 'beta-reference-v1', 'scipy_version': scipy.__version__, 'rows': rows}
    OUTPUT.write_text(json.dumps(fixture, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'{OUTPUT.relative_to(ROOT)}: {len(rows)} rows, scipy {scipy.__version__}')


if __name__ == '__main__':
    main()
