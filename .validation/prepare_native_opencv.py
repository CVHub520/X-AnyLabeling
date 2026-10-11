import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

version = "4.14.0.94"
root = Path("C:/b")
root.mkdir(exist_ok=True)
with urllib.request.urlopen(
    f"https://pypi.org/pypi/opencv-contrib-python-headless/{version}/json",
    timeout=60,
) as response:
    metadata = json.load(response)
source = next(
    item for item in metadata["urls"] if item["packagetype"] == "sdist"
)
archive = root / "opencv.tar.gz"
with (
    urllib.request.urlopen(source["url"], timeout=120) as response,
    archive.open("wb") as output,
):
    shutil.copyfileobj(response, output, 1024 * 1024)
unpacked = root / "unpacked"
unpacked.mkdir()
with tarfile.open(archive) as package:
    package.extractall(unpacked, filter="data")
source_root = next(unpacked.iterdir())
target = root / "opencv"
source_root.rename(target)
setup = target / "setup.py"
text = setup.read_text()
needle = 'if os.name == "nt"\n            else []'
assert text.count(needle) == 1, "The OpenCV packaging layout changed"
text = text.replace(
    needle,
    'if os.name == "nt" and "-DWITH_FFMPEG=OFF" not in os.environ.get("CMAKE_ARGS", "")\n            else []',
)
setup.write_text(text)
print(f"Prepared {target}: native build without an unavailable FFmpeg DLL.")
