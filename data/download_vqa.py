"""
Download VQAv2 annotations + COCO val2014 images, optimised for 32 CPUs / 100-150 GB RAM.

Parallelism strategy:
  1. All 3 files download simultaneously (outer ThreadPoolExecutor with 3 workers)
  2. Each file is split into chunks and downloaded in parallel via byte-range requests
     (10 chunk-workers per file → 30 concurrent connections during download)
  3. Only the ~5k needed COCO images are extracted (not all 40k) using 32 threads.
     Each thread opens its own ZipFile handle, so there's no lock contention.

Usage:
    python data/download_vqa.py --config configs/distil_config.yaml [--workers 32]
"""

import argparse
import concurrent.futures
import json
import os
import zipfile
from collections import Counter
from pathlib import Path
from threading import Lock

import requests
import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Parallel chunked download
# ---------------------------------------------------------------------------

def _supports_range(url: str) -> tuple[bool, int]:
    """Return (supports_range, content_length). HEAD request."""
    r = requests.head(url, timeout=30, allow_redirects=True)
    r.raise_for_status()
    accepts = r.headers.get("Accept-Ranges", "none").lower()
    length = int(r.headers.get("Content-Length", 0))
    return (accepts == "bytes" and length > 0), length


def _download_chunk(url: str, dest: Path, byte_start: int, byte_end: int,
                    pbar: tqdm, lock: Lock) -> None:
    """Download [byte_start, byte_end] and write to the correct offset using pwrite."""
    headers = {"Range": f"bytes={byte_start}-{byte_end}"}
    r = requests.get(url, headers=headers, stream=True, timeout=180)
    r.raise_for_status()
    fd = os.open(str(dest), os.O_WRONLY)
    offset = byte_start
    try:
        for data in r.iter_content(chunk_size=1 << 18):  # 256 KB
            os.pwrite(fd, data, offset)  # atomic, no seek races between threads
            offset += len(data)
            with lock:
                pbar.update(len(data))
    finally:
        os.close(fd)


def _download_stream(url: str, dest: Path, pbar: tqdm, lock: Lock) -> None:
    """Fallback single-stream download for servers that don't support range requests."""
    r = requests.get(url, stream=True, timeout=180)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for data in r.iter_content(chunk_size=1 << 18):
            f.write(data)
            with lock:
                pbar.update(len(data))


def download_file_parallel(url: str, dest: Path,
                           chunk_workers: int = 10, label: str = "") -> None:
    """Download a file with parallel byte-range chunks. Idempotent."""
    if dest.exists():
        tqdm.write(f"  skip (exists): {dest.name}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    supports_range, total = _supports_range(url)
    label = label or dest.name

    with tqdm(total=total or None, unit="B", unit_scale=True,
              desc=label, leave=True, position=0) as pbar:
        lock = Lock()

        if not supports_range or total == 0:
            _download_stream(url, dest, pbar, lock)
            return

        # Pre-allocate file so pwrite offsets are valid
        with open(dest, "wb") as f:
            f.seek(total - 1)
            f.write(b"\x00")

        chunk_size = max(total // chunk_workers, 4 << 20)  # min 4 MB per chunk
        ranges: list[tuple[int, int]] = []
        start = 0
        while start < total:
            end = min(start + chunk_size - 1, total - 1)
            ranges.append((start, end))
            start = end + 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(ranges)) as ex:
            futs = [
                ex.submit(_download_chunk, url, dest, s, e, pbar, lock)
                for s, e in ranges
            ]
            for f in concurrent.futures.as_completed(futs):
                f.result()  # re-raise any exception


# ---------------------------------------------------------------------------
# Selective parallel extraction
# ---------------------------------------------------------------------------

def extract_selective(zip_path: Path, dest_dir: Path,
                      zip_names: list[str], n_workers: int) -> None:
    """
    Extract only zip_names from zip_path into dest_dir.
    Each thread opens its own ZipFile handle to avoid contention.
    Files are written flat (directory prefix stripped).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    def _extract_one(name: str) -> None:
        with zipfile.ZipFile(zip_path, "r") as zf:
            data = zf.read(name)
        (dest_dir / Path(name).name).write_bytes(data)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(tqdm(
            ex.map(_extract_one, zip_names),
            total=len(zip_names),
            desc=f"Extracting {len(zip_names)} images",
            unit="img",
        ))


# ---------------------------------------------------------------------------
# Answer vocab
# ---------------------------------------------------------------------------

def build_answer_vocab(samples: list, top_k: int) -> dict:
    counter: Counter = Counter()
    for s in samples:
        for ans in s["answers"]:
            counter[ans["answer"].lower().strip()] += 1
    vocab = {ans: idx for idx, (ans, _) in enumerate(counter.most_common(top_k))}
    vocab["<unk>"] = len(vocab)
    return vocab


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: dict, n_workers: int) -> None:
    data_dir  = Path(cfg["data"]["data_dir"])
    cache_dir = Path(cfg["data"]["cache_dir"])
    num_samples = cfg["data"]["num_samples"]
    num_answers = cfg["data"]["num_answers"]

    ann_dir = data_dir / "annotations"
    img_dir = data_dir / "val2014"
    ann_dir.mkdir(parents=True, exist_ok=True)

    q_zip    = data_dir / "v2_questions_val.zip"
    a_zip    = data_dir / "v2_annotations_val.zip"
    coco_zip = data_dir / "val2014.zip"

    downloads = [
        (cfg["data"]["vqa_questions_url"],   q_zip,    "VQAv2 questions"),
        (cfg["data"]["vqa_annotations_url"], a_zip,    "VQAv2 annotations"),
        (cfg["data"]["coco_images_url"],     coco_zip, "COCO val2014 (~6.3 GB)"),
    ]

    # ---- Step 1: Download all 3 files in parallel -------------------------
    print(f"\n[1/4] Downloading 3 files in parallel (chunk_workers={n_workers // 3} each) ...")
    chunk_workers = max(4, n_workers // 3)
    pending = [(url, dest, lbl) for url, dest, lbl in downloads if not dest.exists()]

    if not pending:
        print("  All files already downloaded.")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(pending)) as ex:
            futs = {
                ex.submit(download_file_parallel, url, dest, chunk_workers, lbl): lbl
                for url, dest, lbl in pending
            }
            for fut in concurrent.futures.as_completed(futs):
                fut.result()

    # ---- Step 2: Extract annotation zips (small, sequential is fine) ------
    print("\n[2/4] Extracting VQAv2 annotations ...")
    q_json = ann_dir / "v2_OpenEnded_mscoco_val2014_questions.json"
    a_json = ann_dir / "v2_mscoco_val2014_annotations.json"
    if not q_json.exists():
        with zipfile.ZipFile(q_zip) as zf:
            zf.extractall(ann_dir)
    if not a_json.exists():
        with zipfile.ZipFile(a_zip) as zf:
            zf.extractall(ann_dir)

    # ---- Step 3: Parse, subset, build vocab --------------------------------
    print("\n[3/4] Parsing annotations and building answer vocab ...")
    with open(q_json) as f:
        questions_data = json.load(f)
    with open(a_json) as f:
        annotations_data = json.load(f)

    ann_by_qid = {a["question_id"]: a for a in annotations_data["annotations"]}

    all_samples = []
    for q in questions_data["questions"]:
        qid = q["question_id"]
        if qid not in ann_by_qid:
            continue
        all_samples.append({
            "question_id": qid,
            "image_id":    q["image_id"],
            "question":    q["question"],
            "answers":     ann_by_qid[qid]["answers"],
        })

    all_samples = all_samples[:num_samples]
    needed_ids  = {s["image_id"] for s in all_samples}
    print(f"  {len(all_samples)} samples | {len(needed_ids)} unique images")

    vocab = build_answer_vocab(all_samples, num_answers)
    print(f"  Answer vocab: {len(vocab)} entries (top-{num_answers} + <unk>)")

    # ---- Step 4: Selective extraction (5k images, not all 40k) -------------
    print(f"\n[4/4] Extracting needed COCO images ({n_workers} threads) ...")
    already = {iid for iid in needed_ids
               if (img_dir / f"COCO_val2014_{iid:012d}.jpg").exists()}
    missing_ids = needed_ids - already
    print(f"  {len(already)} already on disk, {len(missing_ids)} to extract")

    if missing_ids:
        needed_names = {f"COCO_val2014_{iid:012d}.jpg" for iid in missing_ids}
        with zipfile.ZipFile(coco_zip) as zf:
            all_zip_names = zf.namelist()
        to_extract = [n for n in all_zip_names if Path(n).name in needed_names]

        if not to_extract:
            print("  WARNING: no matching entries in zip — check COCO zip structure.")
        else:
            extract_selective(coco_zip, img_dir, to_extract, n_workers)

    # Verify
    still_missing = [iid for iid in needed_ids
                     if not (img_dir / f"COCO_val2014_{iid:012d}.jpg").exists()]
    if still_missing:
        print(f"  WARNING: {len(still_missing)} images still missing after extraction.")

    # ---- Save metadata -----------------------------------------------------
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "vqa_samples.json", "w") as f:
        json.dump(all_samples, f)
    with open(cache_dir / "answer_vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)

    print(f"\nDone.")
    print(f"  Samples → {cache_dir / 'vqa_samples.json'}")
    print(f"  Vocab   → {cache_dir / 'answer_vocab.json'}")
    print(f"  Images  → {img_dir}/ ({len(needed_ids)} files)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/distil_config.yaml")
    parser.add_argument("--workers", type=int, default=32,
                        help="Total thread budget (split across downloads + extraction)")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg, args.workers)
