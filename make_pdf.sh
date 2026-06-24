#!/usr/bin/env bash
# make_pdf.sh — render a math-heavy markdown file to PDF.
#
# No LaTeX engine on this box, so: pandoc -> self-contained HTML with MathML
# (Chrome renders MathML natively, no JS wait) -> headless Chrome print-to-pdf.
#
# Usage:
#   ./make_pdf.sh docs/theory_landscape.md            # -> docs/theory_landscape.pdf
#   ./make_pdf.sh docs/theory_landscape.md out.pdf    # -> out.pdf
set -euo pipefail

IN="${1:?usage: make_pdf.sh <input.md> [output.pdf]}"
OUT="${2:-${IN%.md}.pdf}"
[ -f "$IN" ] || { echo "no such file: $IN" >&2; exit 1; }

command -v pandoc >/dev/null || { echo "pandoc not found" >&2; exit 1; }
CHROME="$(command -v google-chrome || command -v chromium-browser || command -v chromium || true)"
[ -n "$CHROME" ] || { echo "no chrome/chromium found" >&2; exit 1; }

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

# CSS for readable output
cat > "$TMP/style.css" <<'CSS'
body{max-width:820px;margin:2.5em auto;padding:0 1.5em;font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;line-height:1.5;color:#222}
h1{font-size:1.6em;border-bottom:2px solid #ccc;padding-bottom:.2em}
h2{font-size:1.25em;margin-top:1.6em;border-bottom:1px solid #eee;padding-bottom:.15em}
h3{font-size:1.05em;margin-top:1.1em}
code{background:#f4f4f4;padding:.1em .3em;border-radius:3px;font-size:.92em}
pre{background:#f7f7f7;padding:.8em;border-radius:5px;overflow-x:auto}
blockquote{border-left:3px solid #ccc;margin:1em 0;padding:.2em 1em;color:#555;background:#fafafa}
table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:.3em .6em}
math{font-size:1.05em}
CSS

# Sanitize TeX pandoc's math reader rejects (\rm -> \mathrm, drop \! spacing).
# Source markdown is left untouched (GitHub/MathJax handle those fine).
sed -e 's/{\\rm \([a-zA-Z]*\)}/{\\mathrm{\1}}/g' -e 's/\\!//g' "$IN" > "$TMP/in.md"

TITLE="$(basename "${IN%.md}")"
pandoc "$TMP/in.md" -o "$TMP/out.html" --standalone --mathml \
    --metadata title="$TITLE" --css "$TMP/style.css" --embed-resources

"$CHROME" --headless --no-sandbox --disable-gpu --no-pdf-header-footer \
    --virtual-time-budget=10000 \
    --print-to-pdf="$(realpath "$OUT")" "file://$TMP/out.html" >/dev/null 2>&1 || true

[ -s "$OUT" ] || { echo "PDF not produced" >&2; exit 1; }
echo "wrote $OUT ($(du -h "$OUT" | cut -f1), $(pdfinfo "$OUT" 2>/dev/null | awk '/Pages/{print $2" pages"}'))"
