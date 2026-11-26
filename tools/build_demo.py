"""Inline docs/data.json into docs/template.html.

  docs/index.html      a complete page that opens with a double click (and on GitHub Pages)
  docs/fragment.html   the same page without the document wrapper, for hosts that add their own

Run: python -m tools.make_explorer && python -m tools.build_demo
"""
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "docs"
data = (D / "data.json").read_text(encoding="utf-8").replace("</", r"<\/")  # never close the script tag early
page = (D / "template.html").read_text(encoding="utf-8").replace("__DATA__", data, 1)
import base64  # noqa: E402
for key, name in (("__DAY__", "day.jpg"), ("__NIGHT__", "night.jpg"), ("__WATER__", "water.jpg")):
    page = page.replace(key, "data:image/jpeg;base64," + base64.b64encode((D / "tex" / name).read_bytes()).decode(), 1)
(D / "fragment.html").write_text(page, encoding="utf-8")
(D / "index.html").write_text("<!doctype html>\n<html lang=\"en\">\n<body>\n" + page + "\n</body>\n</html>\n", encoding="utf-8")
print("docs/index.html", (D / "index.html").stat().st_size // 1024, "KB")
