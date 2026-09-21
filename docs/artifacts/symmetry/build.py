"""Embed the evidence figures as data URIs and write the publishable artifact.
Edit symmetry.src.html (which carries {{FIG:name}} placeholders), then run: python build.py"""
import base64, os, re, sys
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "symmetry.src.html")).read()
def sub(m):
    name = m.group(1); path = os.path.join(here, "fig", name + ".jpg")
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
out = re.sub(r"\{\{FIG:([A-Za-z0-9_]+)\}\}", sub, src)
assert "{{FIG:" not in out, "unresolved figure placeholder"
open(os.path.join(here, "symmetry.html"), "w").write(out)
print(f"built symmetry.html — {len(out)/1024/1024:.2f} MB, {src.count('{{FIG:')} figures embedded")
