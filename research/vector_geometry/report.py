"""Render interpretation from the actual executed diagnostic measurements."""
import json
from pathlib import Path


def summary(result):
    search = result['search']
    svd = next(row for row in result['reductions'] if row['method'] == 'TruncatedSVD' and row['dimensions'] == 256)
    pca = next(row for row in result['reductions'] if row['method'] == 'PCA' and row['dimensions'] == 256)
    return (
        f"현재 {result['shape'][0]:,} × {result['shape'][1]:,} float32 인덱스는 "
        f"{result['raw_vector_bytes']/2**20:.2f} MiB입니다. 단일 스레드의 warm-query 중앙값은 "
        f"기존 NumPy 식 {search['numpy_production_expression']['median_ms']:.3f} ms, "
        f"NumPy norm/dot 식 {search['numpy_norm_dot_identity']['median_ms']:.3f} ms, "
        f"FAISS FlatL2 {search['faiss_flat_l2']['median_ms']:.3f} ms였습니다. "
        f"128개 쿼리를 3회 반복한 top-10 순서 일치율은 {search['numpy_faiss_identical_order_fraction']:.1%}입니다.\n\n"
        f"256차원에서 원래 cosine 이웃의 평균 보존율은 PCA {pca['cosine_overlap_at_10']:.1%}, "
        f"SVD {svd['cosine_overlap_at_10']:.1%}입니다. 축소 벡터 저장량은 원본의 1/6이며 변환 행렬 저장량은 별도입니다. "
        "현재 일일 소량 검색에는 NumPy를 유지하고, 압축과 FAISS 도입은 실제 처리량·지연 요구가 생길 때 재평가하는 편이 합리적입니다. "
        "이 실험은 기하 보존과 구현 비용의 진단이며 알고리즘 추천의 정답률을 측정하지 않습니다. "
        "반복된 matmul 경고를 보존했고, 전체 유한성·성분 직교성 및 일부 쿼리의 독립 float64 검사를 통과했습니다. 경고 원인은 미확정입니다."
    )


def write_report(result, output):
    output = Path(output)
    search = result['search']
    lines = ['# 벡터 검색·차원축소 실측 보고서', '', summary(result), '',
        '## 1. 정확 검색 비용', '',
        '![정확 검색 지연](search_latency.png)', '',
        '| 방법 | 중앙값 ms | p95 ms |', '|---|---:|---:|']
    for label, key in [('기존 NumPy 거리식','numpy_production_expression'),('NumPy norm/dot','numpy_norm_dot_identity'),('FAISS FlatL2','faiss_flat_l2')]:
        row = search[key]
        lines.append(f"| {label} | {row['median_ms']:.3f} | {row['p95_ms']:.3f} |")
    lines += ['', f"FAISS 인덱스 생성은 {search['faiss_build_ms']:.3f} ms였습니다. Flat은 압축·근사 검색이 아닌 전수 정확 검색입니다. "
        '운영 코드와 같은 raw squared-L2를 사용하고 자기 자신을 제외했습니다. 거리 계산과 top-10 선택을 함께 측정했으며 운영의 마지막 무작위 3개 선택은 제외했습니다. '
        '운영 식은 큰 임시 차이 배열을 만들고 norm/dot 식은 이를 줄일 수 있지만, 이번에는 운영 코드를 변경하지 않았습니다.', '',
        '측정은 개발 머신의 메모리에 올라온 배열을 대상으로 합니다. 쿼리는 고정 seed 17로 선택한 카탈로그 내부 128개이며, 방법 실행 순서는 고정입니다. '
        '384회는 384개의 독립적인 문제 표본이 아닙니다. p95는 관측 지연 분포이며 신뢰구간이 아닙니다. '
        '파일 로딩, 패키지 import, 쿼리 임베딩, 동시 부하, OCI 성능과 전체 프로세스 peak RSS는 측정하지 않았습니다.', '',
        '## 2. PCA와 SVD의 압축 손실', '', '![차원과 이웃 보존율](overlap.png)', '',
        '| 방법 | 차원 | 이웃 보존@10 | 설명 분산 합 | 벡터 MiB | 적합·변환 초 |', '|---|---:|---:|---:|---:|---:|']
    for row in result['reductions']:
        lines.append(f"| {row['method']} | {row['dimensions']} | {row['cosine_overlap_at_10']:.1%} | {row['explained_variance_ratio_sum']:.1%} | {row['vector_storage_bytes']/2**20:.2f} | {row['fit_transform_seconds']:.3f} |")
    lines += ['', '보존율은 전체 카탈로그에서 자기 자신을 제외한 원본 cosine top-10과 축소 공간 cosine top-10의 교집합 크기/10을 평균한 값입니다. '
        'PCA는 중심화하고 SVD는 중심화하지 않으므로 cosine 원점 자체가 달라집니다. 따라서 이 결과를 두 방법의 보편적인 우열로 해석할 수 없습니다. '
        '설명 분산 비율도 알고리즘 관련성의 정답률이 아닙니다. 모델은 전체 카탈로그에 적합했으며 새 문제 일반화는 평가하지 않았습니다. '
        '모델별 벡터 저장량만 비교했고 변환 행렬·평균 벡터·라이브러리 메모리는 포함하지 않았습니다. 축소 검색 지연 및 신규 쿼리 변환 비용은 별도 후속 측정 대상입니다.', '',
        '## 3. 2차원 시각화', '', '![난이도별 2차원 투영](projections.png)', '',
        '각 점은 한 문제이고 색은 기존 난이도 메타데이터입니다. 축의 방향·스케일은 방법 간 비교 대상이 아닙니다. '
        '난이도 LDA는 이 색 라벨을 학습에 직접 사용한 표본 내 투영입니다. 분리가 보여도 독립 검증의 증거가 아니며, 세 클래스이므로 최대 두 판별축만 가능합니다. '
        'Latent Dirichlet Allocation이라는 토픽 모델 LDA는 비음수 토큰 빈도를 위한 별도 방법이므로 부호가 있는 dense embedding에 적용하지 않았습니다.', '',
        '| 투영 | 원본 cosine 이웃 대비 2D Euclidean 보존@10 | 적합 초 |', '|---|---:|---:|']
    for name, row in result['projections'].items():
        lines.append(f"| {name} | {row['euclidean_overlap_at_10']:.1%} | {row['fit_seconds']:.2f} |")
    lines += ['', f"t-SNE와 UMAP은 동일한 PCA 50차원 전처리를 사용했습니다. 이 전처리 자체의 Euclidean 이웃 보존율은 {result['pca50_euclidean_overlap_at_10']:.1%}입니다. "
        f"seed 17/29 사이의 2D 이웃 일치율은 t-SNE {result['tsne_seed_neighbor_overlap_at_10']:.1%}, UMAP {result['umap_seed_neighbor_overlap_at_10']:.1%}였습니다. "
        '이는 한 번의 seed 변경 진단이며 전체 불확실성 추정이 아닙니다. UMAP 최초 실행시간에는 JIT 준비 비용이 포함될 수 있습니다. '
        '2D 좌표는 임의 원점을 가지므로 Euclidean 이웃을 사용했습니다. 군집 간 간격·밀도·모양만으로 알고리즘군, 선수 관계 또는 전이 효용을 단정하지 않습니다.', '',
        '## 4. 재현과 사용 경계', '',
        f"계산 중 라이브러리 경고 {sum(row['count'] for row in result.get('runtime_warnings', []))}건을 원문·빈도와 함께 results.json에 보존했습니다. matmul 경고를 무시하고 정확성을 가정하지 않았습니다. 이웃 거리 행렬과 축소 결과의 전체 유한성, 축소 성분의 float64 비-BLAS 직교성 검사, 16개 쿼리의 직접 float64 L2 교차 검증을 통과했습니다. 경고의 런타임 원인 자체는 확정하지 않았습니다. 지연 시간에는 해당 환경의 경고 처리 비용도 포함됩니다.", '',
        f"- 입력 SHA-256 전후 일치: `{result['input_sha256_before'] == result['input_sha256_after']}`. 불변 source registry와도 대조했습니다.",
        f"- 난이도 분포/결측: `{json.dumps(result['metadata_missingness'], ensure_ascii=False)}`.",
        f"- 원본 벡터 norm 범위: `{result['vector_norm_range']}`.",
        '- 실행 환경·전체 파라미터·고정 쿼리 행 번호·hash: [results.json](results.json).',
        '- 실행 결과를 포함한 [노트북](vector_geometry.ipynb), [HTML 보기](vector_geometry.html), [연구 의존성](requirements-research.txt).',
        '- 기존 역사적 100-pair benchmark는 이번 실험에 사용하지 않았습니다. 이웃 보존은 의미적 정답률·학습 성과와 구분합니다.',
        '- 역사적 원문·벡터는 재배포하지 않으며 hosted bundle에 포함하지 않습니다. 유료 API, 신규 임베딩 또는 모델 서버는 사용하지 않았습니다.',
        '- 실제 OCI 설치·systemd 자원 한도·스케줄은 별도 미검증 상태입니다. 이번 과학 계산을 1 GB VM에 그대로 배치하지 않습니다.', '',
        '재실행(저장소 루트, Python 3.13 연구 환경):', '', '```bash',
        'python3.13 -m venv .venv', '.venv/bin/python -m pip install -r research/vector_geometry/requirements-research.txt',
        'MPLCONFIGDIR=/tmp/algorithm-vector-mpl JUPYTER_RUNTIME_DIR=/tmp/algorithm-vector-jupyter .venv/bin/python research/vector_geometry/build_notebook.py --execute',
        '.venv/bin/python -m unittest research.vector_geometry.test_experiment -v', '```', '',
        '수치·속도는 패키지/하드웨어에 따라 달라질 수 있습니다. 원본 로컬 자산이 없는 checkout에서는 실행이 실패하며 API로 대체하지 않습니다. '
        '정확한 의존 버전은 직접 사용한 패키지를 기록한 것으로, 플랫폼 간 전이적 전체 lock을 보장하지 않습니다.', '',
        '방법과 공식 문서 링크는 [연구 설계](README.md#method-references)를 참조하세요.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary(result)
