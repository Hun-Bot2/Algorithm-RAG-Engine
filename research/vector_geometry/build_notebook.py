#!/usr/bin/env python3
"""Build and optionally execute the local vector geometry research notebook."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from textwrap import dedent

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NOTEBOOK = HERE / "vector_geometry.ipynb"
HTML = HERE / "vector_geometry.html"


def markdown(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(dedent(source).strip())


def code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(dedent(source).strip())


def build_notebook() -> nbformat.NotebookNode:
    """Create the concise Korean explainer and its editable execution cells."""
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Algorithm RAG Engine (.venv)",
        "language": "python",
        "name": "vector-geometry-venv",
    }
    notebook.metadata["language_info"] = {"name": "python"}
    notebook.cells = [
        markdown(
            """
            # 벡터 검색과 저차원 기하 진단

            이 노트북은 로컬 LeetCode 임베딩 카탈로그에서 정확 검색과 차원 축소가 기존 벡터 이웃을 얼마나 보존하는지 살펴봅니다. 결과는 현재 임베딩 공간의 기하를 진단하며 추천 품질이나 학습 효과를 증명하지 않습니다.
            """
        ),
        markdown(
            """
            ## Context

            현재 인덱스는 약 3천 개의 벡터를 담습니다. 이 실험은 기존 로컬 자산만 읽으며 API 호출이나 운영 코드 변경 없이, NumPy의 정확 검색으로 충분한지와 축소 표현이 검색 이웃을 보존하는지 확인합니다.
            """
        ),
        markdown(
            """
            ## Data

            입력은 저장소의 기존 NumPy 벡터 인덱스와 메타데이터입니다. 난이도 결측과 파일 해시는 실행 결과에 기록해 데이터 범위와 입력 불변성을 확인합니다. 전체 쿼리 ID나 좌표는 노트북 결과에 내보내지 않습니다.
            """
        ),
        markdown(
            """
            ## Methods

            NumPy와 FAISS Flat은 같은 거리 척도와 쿼리를 사용해 정확 검색 결과를 비교합니다. PCA는 중심화된 데이터의 분산을, Truncated SVD는 중심화하지 않은 데이터의 성분을 보존하도록 여러 차원에서 적합합니다. 이웃 겹침은 원공간 상위 이웃과 축소 공간 상위 이웃의 중복을 나타냅니다.

            2차원 LDA는 난이도 라벨을 이용한 지도 투영의 표본 내 예시입니다. 여기서 LDA는 문서 주제 모델이 아닙니다. t-SNE와 UMAP은 탐색용 시각화이며 전역 거리나 군집의 의미를 입증하지 않습니다. 비지도 축소는 전체 카탈로그에 적합한 전이적 진단이므로 일반화 성능으로 해석하지 않습니다.
            """
        ),
        code(
            """
            from pathlib import Path
            import importlib
            import json
            import sys

            candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
            ROOT = next(
                (candidate for candidate in candidates
                 if (candidate / "research/vector_geometry/experiment.py").is_file()
                 and (candidate / "assets/legacy_numpy.manifest.v1.json").is_file()),
                None,
            )
            if ROOT is None:
                raise FileNotFoundError("Could not locate the repository and vector asset manifest")
            OUTPUT = ROOT / "research/vector_geometry"
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            from research.vector_geometry import experiment
            experiment = importlib.reload(experiment)
            results = experiment.run(ROOT, OUTPUT)
            assert isinstance(results, dict), "experiment.run must return a dictionary"
            print("실험 완료: 결과 항목 {}개".format(len(results)))
            """
        ),
        markdown(
            """
            ## Results

            아래 표와 그림은 방금 실행한 실험의 반환값과 파일만 표시합니다. 임베딩 자체나 쿼리 ID 전체는 노트북 출력에 복사하지 않습니다.
            """
        ),
        code(
            """
            import pandas as pd
            from IPython.display import Image, display

            print(f"카탈로그: {results['shape'][0]:,}행 × {results['shape'][1]:,}차원, {results['dtype']}")
            search = results["search"]
            exact = search["numpy_faiss_identical_order_fraction"]
            print(f"요약: NumPy와 FAISS의 top-10 순서 일치율 {exact:.3f}")
            search_rows = [
                {"방법": method, "중앙값 ms": timing["median_ms"], "p95 ms": timing["p95_ms"]}
                for method, timing in (
                    ("NumPy production", search["numpy_production_expression"]),
                    ("FAISS FlatL2", search["faiss_flat_l2"]),
                    ("NumPy norm-dot", search["numpy_norm_dot_identity"]),
                )
            ]
            print("정확 검색 지연 시간 (반복 측정)")
            display(pd.DataFrame(search_rows).round(3))
            reductions = pd.DataFrame(results["reductions"])
            print("차원 축소와 원공간 cosine 이웃 겹침")
            display(reductions[["method", "dimensions", "cosine_overlap_at_10",
                                "explained_variance_ratio_sum", "storage_fraction_of_original"]]
                    .rename(columns={"method": "방법", "dimensions": "차원",
                                     "cosine_overlap_at_10": "이웃 겹침@10",
                                     "explained_variance_ratio_sum": "설명 분산 비율 합",
                                     "storage_fraction_of_original": "저장 비율"})
                    .round(4))
            projection_rows = [
                {"투영": name, **metrics}
                for name, metrics in results["projections"].items()
            ]
            print("2차원 투영 진단")
            display(pd.DataFrame(projection_rows).round(4))

            import json
            print("재현 정보:", json.dumps({
                "입력 SHA-256": results["input_sha256_before"],
                "입력 변경 없음": results["input_sha256_before"] == results["input_sha256_after"],
                "난이도 결측/분포": results["metadata_missingness"],
                "계산 환경": results["software"],
            }, ensure_ascii=False, indent=2))

            figures = [OUTPUT / name for name in results.get("figures", [])]
            figures = [figure for figure in figures if figure.is_file()]
            if figures:
                for figure in figures:
                    print(figure.name)
                    display(Image(filename=str(figure)))
            else:
                print("생성된 PNG 그림이 없습니다.")
            """
        ),
        markdown(
            """
            ## Takeaways

            해석은 측정된 실행 결과와 그림에 한정합니다. 정확 검색 간 일치와 축소 후 이웃 보존은 각각 구현 비교와 기하 보존의 증거입니다. 둘 다 알고리즘 간 선후 관계, 추천 적합성 또는 사용자 학습 성과를 직접 측정하지 않습니다. t-SNE와 UMAP의 모양만으로 전역 구조를 주장하지 않습니다.
            """
        ),
    ]
    nbformat.validate(notebook)
    return notebook


def execute_notebook(notebook: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """Execute using this checkout's .venv without installing a global kernel."""
    python = ROOT / ".venv" / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(f"Expected repository virtualenv Python at {python}")
    with tempfile.TemporaryDirectory(prefix="vector-geometry-kernels-") as temporary:
        kernel_root = Path(temporary)
        kernels_dir = kernel_root / "kernels"
        spec_dir = kernels_dir / "vector-geometry-venv"
        spec_dir.mkdir(parents=True)
        (spec_dir / "kernel.json").write_text(
            json_dumps_kernel(python), encoding="utf-8"
        )
        manager = KernelSpecManager(kernel_dirs=[str(kernels_dir)])
        manager.get_kernel_spec("vector-geometry-venv")
        kernel = AsyncKernelManager(
            kernel_name="vector-geometry-venv",
            kernel_spec_manager=manager,
        )
        client = NotebookClient(
            notebook,
            km=kernel,
            resources={"metadata": {"path": str(ROOT)}},
            timeout=3600,
            allow_errors=False,
        )
        return client.execute(cleanup_kc=True)


def json_dumps_kernel(python: Path) -> str:
    import json

    return json.dumps(
        {
            "argv": [str(python), "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Algorithm RAG Engine (.venv)",
            "language": "python",
        },
        indent=2,
    )


def export_html(notebook: nbformat.NotebookNode) -> None:
    from nbconvert import HTMLExporter

    exporter = HTMLExporter()
    exporter.exclude_input_prompt = True
    exporter.embed_images = True
    body, _ = exporter.from_notebook_node(notebook)
    HTML.write_text(body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute the notebook (requires research/vector_geometry/experiment.py)",
    )
    args = parser.parse_args()

    notebook = build_notebook()
    if args.execute:
        notebook = execute_notebook(notebook)
        import json
        from report import write_report
        measured = json.loads((HERE / "results.json").read_text())
        interpretation = write_report(measured, HERE)
        notebook.cells.insert(1, markdown("## tl;dr\n\n" + interpretation))
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK)
    if args.execute:
        export_html(notebook)
    print(f"Wrote {NOTEBOOK}")
    if args.execute:
        print(f"Wrote {HTML}")


if __name__ == "__main__":
    main()
