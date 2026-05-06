"""Create a curated leaderboard around the NYU v2 dataset.

Prereq: scripts/seed_nyu_v2_curated.py has already populated the dataset
(default name 'nyu_depth_v2_subset'). The leaderboard goes in the
benchhub-curated project so /explore?curated=1 surfaces it.

Run:
    BENCHHUB_API_TOKEN=<your-admin-token> \\
    BENCHHUB_BASE_URL=https://benchhub.fly.dev \\
        python scripts/seed_nyu_v2_leaderboard.py [--dataset-id N]

If --dataset-id is omitted, the script looks up the dataset by name via
the public list endpoint. If both fail, it errors out.
"""
import argparse
import os
import sys

try:
    import requests
except ImportError:
    sys.stderr.write("pip install requests\n")
    sys.exit(2)


def _resolve_dataset_id(base_url: str, name: str) -> int | None:
    # Public dataset listing is HTML; pull the JSON-friendly metadata via
    # /api/leaderboard/by_name doesn't apply. Fall back to scraping
    # /datasets and grepping. Cleanest: just hit /dataset/<id> until a
    # name match — but that requires knowing the id. Use the unauth
    # listing page and parse for our marker instead.
    resp = requests.get(f"{base_url.rstrip('/')}/datasets", timeout=30)
    resp.raise_for_status()
    # Anchors look like: <a href="/dataset/<id>" ...>name</a>
    import re
    for m in re.finditer(r'/dataset/(\d+)[^>]*>\s*([^<]+)', resp.text):
        if m.group(2).strip() == name:
            return int(m.group(1))
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="monocular_depth_nyu_v2")
    parser.add_argument("--dataset-name", default="nyu_depth_v2_subset")
    parser.add_argument("--dataset-id", type=int, default=None)
    args = parser.parse_args()

    token = os.environ.get("BENCHHUB_API_TOKEN")
    base_url = os.environ.get("BENCHHUB_BASE_URL", "https://benchhub.fly.dev")
    if not token:
        sys.stderr.write("BENCHHUB_API_TOKEN not set.\n")
        return 2

    dataset_id = args.dataset_id
    if dataset_id is None:
        dataset_id = _resolve_dataset_id(base_url, args.dataset_name)
        if dataset_id is None:
            sys.stderr.write(
                f"Couldn't find dataset named '{args.dataset_name}' on "
                f"{base_url}. Run seed_nyu_v2_curated.py first, or pass "
                "--dataset-id.\n"
            )
            return 1
        print(f"Resolved dataset '{args.dataset_name}' -> id={dataset_id}")

    resp = requests.post(
        f"{base_url.rstrip('/')}/api/admin/leaderboards/create",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "name": args.name,
            "dataset_ids": [dataset_id],
        },
        timeout=30,
    )
    if resp.status_code not in (200, 201):
        sys.stderr.write(f"Create failed: {resp.status_code} {resp.text}\n")
        return 1
    body = resp.json()
    status = "created" if body.get("created") else "already existed"
    print(
        f"Leaderboard '{body['name']}' {status} in project "
        f"'{body['project_name']}' (id={body['leaderboard_id']}). "
        f"Visit {base_url}/{body['project_name']}/leaderboard/{body['leaderboard_id']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
