"""Download the official Stanford GloVe 6B archive and extract only 100d."""
import argparse
import urllib.request
import zipfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--output", default="assets")
a = p.parse_args()
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
target = out / "glove.6B.100d.txt"
if not target.exists():
    archive = out / "glove.6B.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".zip.part")
        print("Downloading official Stanford glove.6B.zip", flush=True)
        urllib.request.urlretrieve("https://nlp.stanford.edu/data/glove.6B.zip", temporary)
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as z:
        z.extract("glove.6B.100d.txt", out)
print(target.resolve(), flush=True)
