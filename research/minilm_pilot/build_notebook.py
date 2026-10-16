#!/usr/bin/env python3
"""Build and optionally execute the local MiniLM pilot notebook."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from textwrap import dedent

import nbformat

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NOTEBOOK = HERE / "minilm_pilot.ipynb"
HTML = HERE / "minilm_pilot.html"


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbformat.v4.new_code_cell(dedent(source).strip())


def build_notebook():
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Algorithm RAG Engine (.venv)",
        "language": "python",
        "name": "vector-geometry-venv",
    }
    notebook.metadata["language_info"] = {"name": "python"}
    notebook.cells = [
        markdown("""
        # MiniLM 교차 플랫폼 알고리즘 문제 파일럿

        이 실행은 고정된 MiniLM 모델로 90개 영어 문제 요약을 임베딩하고, 같은 입력의 TF-IDF와 비교합니다. 이는 작은 검토 주석 집합에 대한 오프라인 진단이며 전체 문제 카탈로그 검색 성능이나 학습 효과를 측정하지 않습니다.
        """),
        markdown("""
        ## 실행 범위와 비용

        사전 고정된 코호트는 LeetCode 60개, AtCoder 20개, Codeforces 10개입니다. 입력 필드는 제목과 기존 영어 rationale이며, 명시적 concept tag, 플랫폼, 난이도는 입력에 넣지 않습니다. 각 입력을 제목+rationale 및 제목만 사용하는 두 변형으로 계산합니다.

        모델 파일은 고정 revision의 로컬 스냅샷을 사용합니다. 이 분석 단계는 API 호출과 모델 재다운로드 없이 CPU에서 동작하며 추론 비용은 로컬 CPU 시간입니다. 최초 자산 준비는 모델 다운로드와 연구용 Python 의존성 설치 비용이 별도로 듭니다. 개발 머신의 시간·메모리 수치는 OCI 자원 한도나 운영 비용의 증거가 아닙니다.
        """),
        markdown("""
        ## 데이터와 방법

        임베딩 유사도는 검색 제안과 별개로, 기존 검토 관계 라벨이 있는 후보만으로 TF-IDF와 MiniLM을 평가합니다. 후보가 2개 미만인 query group은 ranking 지표에서 제외되며, seed graph와 겹치는 관계 라벨도 제외합니다. 보고서에는 평가 가능한 분모와 제외 수를 함께 표시합니다. nDCG는 이 희소하게 검토된 후보 안에서만 계산하며 미검토 카탈로그 항목은 음성으로 간주하지 않습니다.

        고정된 9개 예제는 세 플랫폼에서 각각 세 문제를 제시합니다. 각 플랫폼을 제외한 상위 3개는 모델 제안이지 승인된 관계가 아닙니다. PCA/t-SNE/UMAP 그림은 90개 작은 코호트의 탐색적 배치입니다. 축이나 군집 간격은 알고리즘 관계를 입증하지 않습니다.
        """),
        code("""
        from pathlib import Path
        import importlib
        import sys

        candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
        ROOT = next((p for p in candidates if (p / "research/minilm_pilot/experiment.py").is_file()
                     and (p / "curriculum/annotations.v1.json").is_file()), None)
        if ROOT is None:
            raise FileNotFoundError("저장소 루트와 파일럿 입력을 찾지 못했습니다")
        OUTPUT = ROOT / "research/minilm_pilot"
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from research.minilm_pilot import experiment
        experiment = importlib.reload(experiment)
        results = experiment.run(ROOT, OUTPUT)
        assert isinstance(results, dict)
        print("실험 완료:", results["scope"])
        """),
        markdown("## 실제 실행 결과"),
        code("""
        import pandas as pd
        from IPython.display import Image, display

        print("모델:", results["model"]["model_id"], "revision:", results["model"]["revision"])
        print("코호트:", results["cohort_counts"], "| 실행 환경:", results["environment"])
        variant_rows = []
        for variant, value in results["variants"].items():
            score = value["scoring"]
            for method, metrics in score["metrics"].items():
                variant_rows.append({"입력": variant, "방법": method,
                    "query groups": metrics.get("query_groups"),
                    "ranking groups": metrics.get("ranking_query_groups"),
                    "nDCG@5": metrics.get("ndcg_at_5"),
                    "Recall@10": metrics.get("recall_at_10"),
                    "그래프 제외 행": score.get("seed_graph_overlap_rows_excluded", 0)})
        print("검토된 후보 범위에서의 ranking 결과 (미검토 후보는 음성이 아님)")
        display(pd.DataFrame(variant_rows).round(4))

        resource_rows = []
        for name, value in results["variants"].items():
            tokens = value["token_lengths"]
            resource_rows.append({"입력": name, "encode 초": value["encode_seconds"],
                "벡터 MiB": value["vector_bytes"] / 2**20,
                "토큰 중앙값": tokens["median"], "토큰 최댓값": tokens["max"],
                "256 초과": tokens["over_256"]})
        print("실측 자원과 토큰 길이")
        display(pd.DataFrame(resource_rows).round(4))

        print("고정 9개 query의 교차 플랫폼 상위 3개 제안")
        example_rows = []
        for example in results["cross_platform_examples"]:
            example_rows.append({"query": example["source"], "제목": example["title"],
                "상위 제안": " | ".join(f"{n['problem_key']} ({n['cosine']:.3f})"
                    for n in example["neighbors"])})
        display(pd.DataFrame(example_rows))
        print(results["example_boundary"])

        print("입력 hash 불변:", results["input_hashes_unchanged"],
              "| peak RSS bytes:", results["process_peak_rss_bytes"],
              "| warnings:", sum(w["count"] for w in results.get("warnings", [])))
        for warning in results.get("warnings", [])[:3]:
            print(f"경고 {warning['category']} × {warning['count']}: {warning['message']}")
        if len(results.get("warnings", [])) > 3:
            print("나머지 경고 종류는 results.json에 보존")
        for name in results.get("figures", []):
            path = OUTPUT / name
            if path.is_file():
                print(name)
                display(Image(filename=str(path)))
        """),
        markdown("""
        ## 해석 경계

        모델은 전체 21,244개 카탈로그 메타데이터나 문제 본문이 아니라 90개 검토된 영어 annotation만 처리합니다. AtCoder 메타데이터의 출처는 공식 API가 아니라 AtCoder Problems 커뮤니티 데이터입니다. 모든 nearest-neighbor 결과는 검토가 필요한 제안입니다.

        판정된 후보가 적고 seed graph 겹침을 제거하므로 nDCG/Recall은 희소한 조건부 지표입니다. 이 값을 전체 카탈로그 recall이나 일반화 성능으로 읽지 않습니다. MiniLM의 어떤 지표가 높더라도 자동 승격 근거가 되지 않으며, 현재 실험은 모델 훈련·승격을 하지 않습니다.

        입력 파일 hash, 모델 manifest, 라이브러리 버전, 경고와 실행 측정값은 `results.json`에 보존합니다. 재현에 필요한 패키지는 [연구 의존성](requirements-research.txt), 모델 파일과 고정 revision은 [manifest](model.manifest.json), 데이터와 사전등록된 범위는 [파일럿 명세](../../specs/adaptive-learning/minilm-pilot.md)에 기록되어 있습니다.
        """),
    ]
    nbformat.validate(notebook)
    return notebook


def execute_notebook(notebook):
    # Reuse the repository's temporary-kernel helper and its .venv Python.
    from research.vector_geometry.build_notebook import execute_notebook as execute
    return execute(notebook)


def export_html(notebook):
    from nbconvert import HTMLExporter
    exporter = HTMLExporter()
    exporter.exclude_input_prompt = True
    exporter.embed_images = True
    body, _ = exporter.from_notebook_node(notebook)
    HTML.write_text(body, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run the offline local pilot")
    args = parser.parse_args()
    notebook = build_notebook()
    if args.execute:
        os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                           "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"})
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        notebook = execute_notebook(notebook)
        from research.minilm_pilot.report import write_report
        result = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
        lead = write_report(result, HERE)
        notebook.cells.insert(1, markdown("## 핵심 결과\n\n" + lead))
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK)
    if args.execute:
        export_html(notebook)
    print(f"Wrote {NOTEBOOK}")
    if args.execute:
        print(f"Wrote {HTML}")


if __name__ == "__main__":
    main()
