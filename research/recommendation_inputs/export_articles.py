"""Render local Markdown articles without remote assets or private statement text."""
import base64
from pathlib import Path
import re
from nbconvert.filters.markdown_mistune import markdown2html_mistune

HERE=Path(__file__).resolve().parent
STYLE='''body{font:16px/1.85 system-ui,sans-serif;color:#26382d;background:#f7f7f0;margin:0}main{max-width:920px;margin:auto;padding:38px 25px}h1{font-size:30px;line-height:1.4}h2{font-size:23px;margin-top:42px}a{color:#315d46}img{max-width:100%;height:auto}pre{padding:18px;background:#edf2e9;overflow:auto;font-size:13px;line-height:1.6}code{font-size:14px}table{border-collapse:collapse;display:block;overflow:auto;font-size:14px}td,th{padding:10px 12px;border-bottom:1px solid #d5dfd1;text-align:left}blockquote{border-left:3px solid #52725b;margin-left:0;padding-left:20px}.links{font-size:14px;margin-bottom:25px}hr{border:0;border-top:1px solid #d5dfd1;margin-top:35px}@media(max-width:600px){main{padding:24px 16px}h1{font-size:25px}h2{font-size:20px}}'''


def render(source,destination):
    text=source.read_text()
    def image(match):
        alt,name=match.groups()
        path=(HERE/name).resolve()
        if not path.is_relative_to((HERE/'figures').resolve()) or path.suffix!='.png':
            raise ValueError('only local figure PNGs may be embedded')
        return f'![{alt}](data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()})'
    text=re.sub(r'!\[([^\]]*)\]\((figures/[^)]+)\)',image,text)
    for before,after in [('recommendation_experiments.ipynb','notebook.ipynb'),('recommendation_experiments.html','notebook.html'),('EXPERIMENTS.md','report.html'),('BLOG.md','blog.html')]:
        text=text.replace(']('+before+')',']('+after+')')
    content=markdown2html_mistune(text)
    html=f'<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{"제목이 비슷한 문제와 풀이가 비슷한 문제는 달랐다" if source.name=="BLOG.md" else "입력 표현 실험 기록"}</title><style>{STYLE}</style><main><p class="links"><a href="./">문제 유사도 비교</a> · <a href="notebook.html">실행 노트북</a> · <a href="notebook.ipynb">ipynb 다운로드</a></p>{content}</main></html>'
    destination.write_text(html)


if __name__=='__main__':
    render(HERE/'BLOG.md',HERE/'blog.html')
    render(HERE/'EXPERIMENTS.md',HERE/'experiment_report.html')
