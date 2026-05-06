"""Bootstrap a small NYU v2 subset as a BenchHub-curated dataset.

Streams `sayakpaul/nyu_depth_v2` from HuggingFace, converts the first N
samples to BenchHub's folder convention, ZIPs them, uploads via the
authenticated API, and flips the result to is_curated=True via the
admin endpoint.

Why a script and not a server-side flow:
- The full dataset is ~24 GB of WebDataset shards, which won't fit on
  the 1 GB Fly volume. We need a controlled subset.
- HF Datasets are parquet/WebDataset by default — not BenchHub's
  folder convention. The conversion happens here, so the production
  importer doesn't have to grow a translator.
- One-shot bootstrap. Once the curated dataset is live, the seed
  user role is "delete and re-seed if you want to update."

Local setup (NOT a production dependency):
    python -m venv .venv-seed && source .venv-seed/bin/activate
    pip install h5py pillow numpy requests

Run:
    BENCHHUB_API_TOKEN=<your-token-from-/settings/api_tokens> \\
    BENCHHUB_BASE_URL=https://benchhub.fly.dev \\
        python scripts/seed_nyu_v2_curated.py --samples 50

Flags:
    --samples N        How many examples to take (default 50).
    --name NAME        BenchHub dataset name (default 'nyu_depth_v2_subset').
    --hf-repo REPO     HF repo to stream (default sayakpaul/nyu_depth_v2).
    --override         Allow overwriting an existing dataset with the same name.
    --no-curate        Upload but skip the curate flip (debugging).

Implementation notes:
- We stream `data/train-000000.tar` (3 GB total) via HTTP and walk it
  with the stdlib `tarfile` module in stream-only mode (`r|`). We stop
  reading the moment we hit our sample budget, so only the first ~30 MB
  of the shard actually transfers — enough for ~50 (rgb, depth) pairs.
- Each tar member is an HDF5 file with two datasets: `rgb` (3,480,640 uint8)
  and `depth` (480,640 float32). We extract the bytes into BytesIO and
  open with h5py.
- The `datasets` HF library was tried first, but `sayakpaul/nyu_depth_v2`
  ships a loader script and HF dropped script support. Going direct to
  the tar avoids the broken dependency.
"""
import argparse
import io
import os
import shutil
import sys
import tarfile
import tempfile
import zipfile

try:
    import h5py
    import numpy as np
    from PIL import Image
    import requests
except ImportError as e:
    sys.stderr.write(
        f"Missing dependency: {e}. This script's requirements are not in "
        "the production requirements.txt — install locally:\n"
        "    pip install h5py pillow numpy requests\n"
    )
    sys.exit(2)


HF_RESOLVE_TEMPLATE = (
    "https://huggingface.co/datasets/{repo}/resolve/main/data/train-000000.tar"
)


def _build_subset_zip(hf_repo: str, n_samples: int, dest_zip_path: str) -> int:
    """Stream the first shard of `hf_repo` from HF, write the first
    n_samples (rgb, depth) pairs into a ZIP at dest_zip_path following
    BenchHub folder convention. Returns the number of samples written."""
    url = HF_RESOLVE_TEMPLATE.format(repo=hf_repo)
    print(f"Streaming {url}\n  (only the first ~30 MB will transfer for {n_samples} samples)")

    work_dir = tempfile.mkdtemp(prefix="benchhub-nyu-")
    try:
        rgb_dir = os.path.join(work_dir, "image_rgb")
        depth_dir = os.path.join(work_dir, "raw_depth")
        os.makedirs(rgb_dir)
        os.makedirs(depth_dir)

        # README at the root keeps process_dataset_zip from treating the
        # one populated subfolder as the dataset root (single-folder
        # unwrap heuristic).
        with open(os.path.join(work_dir, "README.md"), "w") as f:
            f.write(
                f"# NYU v2 subset\n\nFirst {n_samples} samples streamed from "
                f"`{hf_repo}` (`data/train-000000.tar`) and converted to "
                f"BenchHub folder convention.\n"
            )

        written = 0
        with requests.get(url, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            # `r|` = stream-only mode; tarfile reads sequentially without seeking.
            with tarfile.open(fileobj=resp.raw, mode="r|") as tar:
                for member in tar:
                    if written >= n_samples:
                        break
                    if not member.name.endswith(".h5"):
                        continue
                    fobj = tar.extractfile(member)
                    if fobj is None:
                        continue
                    raw = fobj.read()
                    try:
                        with h5py.File(io.BytesIO(raw), "r") as h5:
                            rgb = np.array(h5["rgb"])      # (3, H, W) uint8
                            depth = np.array(h5["depth"])  # (H, W) float32
                    except Exception as e:
                        print(f"  skip {member.name}: {e}")
                        continue

                    sample_id = f"s{written:04d}"
                    # rgb is channels-first; PIL wants (H, W, 3).
                    rgb_hwc = np.transpose(rgb, (1, 2, 0))
                    Image.fromarray(rgb_hwc, mode="RGB").save(
                        os.path.join(rgb_dir, f"{sample_id}.png"), "PNG"
                    )
                    h, w = depth.shape[:2]
                    np.savez(
                        os.path.join(depth_dir, f"{sample_id}_{w}x{h}.npz"),
                        depth=depth,
                    )
                    written += 1
                    if written % 10 == 0:
                        print(f"  ...wrote {written}/{n_samples}")

        if written == 0:
            raise RuntimeError(
                "No samples written — the shard didn't yield any .h5 files."
            )

        # Zip the work_dir contents (not the work_dir itself) — process_dataset_zip
        # has a single-root-folder unwrap heuristic that we already neutralized
        # by including README.md at the top.
        print(f"Zipping {written} samples...")
        with zipfile.ZipFile(dest_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(work_dir):
                for fname in files:
                    abs_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(abs_path, work_dir)
                    zf.write(abs_path, arcname=rel_path)
        return written
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _upload(zip_path: str, dataset_name: str, base_url: str, token: str,
            override: bool) -> int:
    print(f"Uploading {os.path.getsize(zip_path) / 1e6:.1f} MB to {base_url}...")
    with open(zip_path, "rb") as fh:
        resp = requests.post(
            f"{base_url.rstrip('/')}/api/dataset/upload",
            headers={"Authorization": f"Bearer {token}"},
            data={
                "dataset_name": dataset_name,
                "override": "true" if override else "false",
            },
            files={"dataset_zip": (os.path.basename(zip_path), fh)},
            timeout=600,
        )
    if resp.status_code != 201:
        raise RuntimeError(f"Upload failed: {resp.status_code} {resp.text}")
    body = resp.json()
    print(f"  -> dataset_id={body['dataset_id']} ({body.get('message')})")
    return body["dataset_id"]


def _curate(dataset_id: int, base_url: str, token: str) -> None:
    print(f"Marking dataset {dataset_id} as curated...")
    resp = requests.post(
        f"{base_url.rstrip('/')}/api/admin/datasets/{dataset_id}/curate",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Curate flip failed: {resp.status_code} {resp.text}")
    print(f"  -> {resp.json()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--name", default="nyu_depth_v2_subset")
    parser.add_argument("--hf-repo", default="sayakpaul/nyu_depth_v2")
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--no-curate", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("BENCHHUB_API_TOKEN")
    base_url = os.environ.get("BENCHHUB_BASE_URL", "https://benchhub.fly.dev")
    if not token:
        sys.stderr.write(
            "BENCHHUB_API_TOKEN not set. Generate one at "
            f"{base_url}/settings/api_tokens and export it.\n"
        )
        return 2

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = tmp.name
    try:
        n = _build_subset_zip(args.hf_repo, args.samples, zip_path)
        dataset_id = _upload(zip_path, args.name, base_url, token, args.override)
        if not args.no_curate:
            _curate(dataset_id, base_url, token)
        print(
            f"\nDone. {n} samples live as curated dataset "
            f"'{args.name}' (id={dataset_id}). "
            f"Visit {base_url.rstrip('/')}/dataset/{dataset_id}"
        )
        return 0
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)


if __name__ == "__main__":
    sys.exit(main())
