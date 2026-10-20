import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
import requests

import config

HF = "https://huggingface.co"
HF_SOURCES = {
    "vstar": {
        "repo": "craigwu/vstar_bench",
        "revision": "d9ae62c903da0c98336e85c5ee89cd863b04b4da",
        "prefixes": ("direct_attributes/", "relative_position/", "test_questions.jsonl"),
        "license": "MIT (code and annotations, https://github.com/penghao-wu/vstar); images from SA-1B under "
                   "the SA-1B research license",
    },
    "hrbench": {
        "repo": "DreamMr/HR-Bench",
        "revision": "83b9013d6293b85dc507e87199ca52517536939c",
        "prefixes": ("hr_bench_8k.parquet", "LICENSE", "README.md"),
        "license": "No license stated; used only for evaluation; images not redistributed.",
    },
    "mme": {
        "repo": "yifanzhang114/MME-RealWorld-Lite",
        "revision": "0e378f64aea87bb2c30dae8bc600f99fae966711",
        "prefixes": ("data.zip", "README.md"),
        "license": "Apache-2.0 for the benchmark (https://github.com/yfzhang114/MME-RealWorld); images keep the "
                   "terms of their source datasets",
    },
}
WORKERS = 8
CHUNK = 1 << 22


def fetch(session, url, path, size=None):
    path = Path(path)
    if path.exists() and (size is None or path.stat().st_size == size):
        return
    part = path.with_name(path.name + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"}
    with session.get(url, headers=headers, stream=True, timeout=120) as reply:
        if reply.status_code == 416 and size == have:
            part.replace(path)
            return
        reply.raise_for_status()
        mode = "ab" if have and reply.status_code == 206 else "wb"
        with open(part, mode) as handle:
            for block in reply.iter_content(CHUNK):
                handle.write(block)
    if size is not None and part.stat().st_size != size:
        raise SystemExit(f"{url}: got {part.stat().st_size} bytes, expected {size}")
    part.replace(path)


def hf_tree(session, repo, revision):
    url = f"{HF}/api/datasets/{repo}/tree/{revision}?recursive=1"
    entries = []
    while url:
        reply = session.get(url, timeout=60)
        reply.raise_for_status()
        entries += reply.json()
        url = reply.links.get("next", {}).get("url")
    return [entry for entry in entries if entry["type"] == "file"]


def hf_source(session, name):
    spec = HF_SOURCES[name]
    folder = config.DOWNLOADS / name
    wanted = [entry for entry in hf_tree(session, spec["repo"], spec["revision"])
              if entry["path"].startswith(spec["prefixes"])]

    def one(entry):
        url = f"{HF}/datasets/{spec['repo']}/resolve/{spec['revision']}/{entry['path']}"
        path = folder / entry["path"]
        fetch(session, url, path, entry["size"])
        digest = config.sha256_file(path)
        lfs = entry.get("lfs")
        if lfs and lfs["oid"] != digest:
            raise SystemExit(f"{path}: SHA-256 {digest} does not match {lfs['oid']}")
        return {"name": entry["path"], "bytes": entry["size"], "sha256": digest, "git_oid": entry["oid"]}

    with ThreadPoolExecutor(WORKERS) as pool:
        files = sorted(pool.map(one, wanted), key=lambda item: item["name"])
    return {"url": f"{HF}/datasets/{spec['repo']}", "revision": spec["revision"], "license": spec["license"],
            "files": files, "downloaded_at": config.now()}


def font():
    path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
    return {"url": "https://matplotlib.org", "revision": f"matplotlib {matplotlib.__version__}",
            "license": "Bitstream Vera and DejaVu font license",
            "files": [{"name": path.name, "bytes": path.stat().st_size, "sha256": config.sha256_file(path)}],
            "downloaded_at": config.now()}


def save(name, record):
    config.write_json(config.DOWNLOADS / f"{name}.json", record)
    merged = {path.stem: json.loads(path.read_text()) for path in sorted(config.DOWNLOADS.glob("*.json"))}
    config.write_json(config.SOURCES_PATH, merged)


def main(names):
    session = requests.Session()
    for name in names:
        if name in HF_SOURCES:
            save(name, hf_source(session, name))
        elif name == "font":
            save(name, font())
        print(name, "done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["font", "vstar", "hrbench", "mme"])
