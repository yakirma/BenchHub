"""Release notes for the `benchhub-client` PyPI package.

Server-side only — this module lives at the repo root (like app.py / tasks.py /
metric_engine.py) so it is NOT bundled into the pip package (pyproject only ships
`benchhub*`). The site renders it at /releases and the footer links there.

Keep it in sync with `__version__` in `benchhub/_version.py`: on a release, bump
that string and prepend one entry here (newest first). `date` is an ISO string
(kept static so the module has no import-time side effects).
"""

# Newest first. Each entry: {version, date (YYYY-MM-DD), highlights: [str, ...]}.
RELEASES = [
    {
        "version": "0.1.11",
        "date": "2026-08-20",
        "highlights": [
            "Reference a leaderboard by its `<owner>/<slug>` handle with "
            "`client.leaderboard(...)` (a numeric id still works). This is what "
            "downloaded submission scripts now use.",
            "Downloaded submission scripts pin the client version they were "
            "generated against (`benchhub-client>=0.1.11`), so a script can no "
            "longer call an API that isn't on the installed client.",
            "Raw sample-data downloads now require a signed-in session or an API "
            "token.",
        ],
    },
    {
        "version": "0.1.10",
        "date": "2026-06-05",
        "highlights": [
            "Dev kit: author metrics and visualizations programmatically "
            "(`create_metric`, `create_visualization`) and test them locally "
            "before uploading with `bh.author.test_metric` / `test_visualization`.",
            "User-registered data types: declare a new `kind` with "
            "`create_datatype(...)` — its storage plus a sandboxed `visualize()` "
            "and optional `decode()` hook — and submit predictions for it.",
            "`sequence` kind: `iter_samples` yields sequence containers and "
            "predictions can be packed as containers (comparison view renders "
            "them as video).",
        ],
    },
    {
        "version": "0.1.9",
        "date": "2026-06-05",
        "highlights": [
            "Interim release in the 2026-06-05 dev-kit / data-types batch; "
            "superseded by 0.1.10 the same day.",
        ],
    },
    {
        "version": "0.1.8",
        "date": "2026-06-05",
        "highlights": [
            "Interim release in the 2026-06-05 dev-kit / data-types batch; "
            "superseded by 0.1.10 the same day.",
        ],
    },
    {
        "version": "0.1.3",
        "date": "2026-05-29",
        "highlights": [
            "`iter_samples` yields typed `bh.<Kind>` instances (e.g. `bh.Image` "
            "with `.array`) instead of bare PIL objects.",
            "Bulk-ZIP input download plus an on-disk cache for `iter_samples`, so "
            "re-runs don't re-fetch the dataset.",
            "`SubmissionBuilder.submit()` accepts an optional name.",
        ],
    },
    {
        "version": "0.1.1",
        "date": "2026-05-28",
        "highlights": [
            "Server-driven `iter_samples` — the client no longer needs "
            "torchvision or a local copy of the dataset.",
            "Pass the API token through to the leaderboard-contract fetch.",
        ],
    },
    {
        "version": "0.1.0",
        "date": "2026-05-28",
        "highlights": [
            "First PyPI release: the strict typed contract (`benchhub.types`) plus "
            "`bh.Client` for iterating samples, submitting predictions, and "
            "`BHDatasetCreator` for uploading a typed dataset from Python.",
        ],
    },
]
