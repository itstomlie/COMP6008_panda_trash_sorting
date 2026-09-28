"""Download only the official Menagerie Panda model, meshes and license."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen
import xml.etree.ElementTree as ET

REVISION = "8161bba264d7fa7c99ca301e91e7fb44737676ad"
BASE = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{REVISION}/franka_emika_panda/"
DEST = Path(__file__).resolve().parent / "assets" / "franka_emika_panda"


def download(name):
    path = DEST / name
    if path.exists():
        return
    with urlopen(BASE + name, timeout=60) as response:
        content = response.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


if __name__ == "__main__":
    download("panda.xml")
    root = ET.parse(DEST / "panda.xml").getroot()
    names = ["LICENSE", "README.md"]
    names += ["assets/" + mesh.attrib["file"] for mesh in root.findall("asset/mesh")]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(download, names))
    (DEST / "REVISION").write_text(REVISION + "\n")
    print(f"Panda assets ready: {DEST}")

