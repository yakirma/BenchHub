# templates/ — server-rendered Jinja (vanilla JS, no framework)

Big screens: `leaderboard.html`, `comparison.html`, `dataset_view.html`, `edit_leaderboard.html`. Static assets minimal (`static/css/`, `static/js/`). Theme is hardcoded in `base.html`.

## Palette (warm-cream, Cabinet retheme)
- Body bg `#fcf9f4` (warm off-white), tertiary/card bg `#f5efe2` (warm cream).
- Body/heading text `#3a2614` and `#2a1f10` (warm dark brown). Border/divider `#e8dfc8` (tan).
- Primary accent **kept violet** `#7c3aed` (brand identity). Don't swap it without an explicit ask — badges, focus rings, hover states all lean on it.
- Background radial gradients: amber `rgba(217,119,6,…)` + peach `rgba(244,114,89,…)`. If you re-tint the page bg, keep the gradient stops in the same hue family or it'll clash.

## Theme + layout conventions
- **Light-only by design.** `<html data-bs-theme="light">` is hardcoded in `base.html`. The CSS override block sets identical values for `[data-bs-theme="light"]` and `[data-bs-theme="dark"]`, but Bootstrap's own navbar vars (`--bs-navbar-color`, `--bs-tertiary-bg-rgb`) aren't covered, so any dark-mode rendering leaks white-on-white. `global_settings.theme_mode` still defaults to `'dark'` in SQLite but the template no longer reads it. Don't reintroduce a real dark mode without overriding *every* `--bs-navbar-*` and `-rgb` variant.
- **Navbar text pinned manually** (`.navbar .nav-link { color: #281950 }` etc.) as belt-and-suspenders.
- **`stretched-link` inside a sticky sidebar needs `position: relative` on the parent** — else the first card's anchor covers the whole scroll container and intercepts every later click. Bit us in the `/leaderboards` category tree.
- **Mobile pattern for long lists** (metrics, visualizations): render a `<select>` with `d-md-none`, hide the sidebar with `d-none d-md-block`. Keeps the detail pane on-screen without a Bootstrap collapse dance.
- **Dataset/leaderboard catalog cards use `.bh-card-grid`** (defined in `base.html`), NOT a Bootstrap `row row-cols-* g-3`. It's a fixed-width CSS grid (`repeat(auto-fill, minmax(12rem, 1fr))`, 2-up under 768px) so a `.home-card` is the **same size on every page** (home, landing, `/datasets`, `/leaderboards`) regardless of container width — the column count flexes, the card width doesn't. The old per-page `row-cols` counts made identical cards render at different sizes (e.g. home LB grid was 3-up while datasets were 4-up in the same column). Grid children are bare `<div>`s (no `.col`); keep `home-grid-item`/`ds-search-row` where the page's JS filter needs them. The stale `base.html` comment mentioning a "richer `.ds-card` grid" predates this — catalogs use `.home-card`.

## Comparison view (`/comparison/<lb_id>`) gotchas
- **`samples_only=1` must thread through every navigation link** (pagination, View Options form, filters) or the page collapses back into full submission-comparison mode the moment the URL drops the param.
- **`samples_only_mode` filters out only submission-needing panels** (`per_sample_metrics`, `per_source_stats`, `pred_histogram`, `viz_*`) — NOT scalar/text/json/audio columns.
- **`leaderboard.comparison_display_columns` CSV is legacy.** The renderer uses `available_display_options - hidden_comparison_display_columns`; the form persists only `hidden_*`. Don't rely on the CSV for visibility.
- **Template gates header+cell on `all_field_types.get(col_key) != 'metric'`** (was `not in ['scalar','metric']`). A new field type must not be accidentally excluded.
- Depth column header has a colormap `<select>` that rewrites every `.depth-img[data-col-key]` src in that column on change; `serve_custom_field_image` forwards `?cmap=` through its redirect to `serve_gt_viz` (Flask doesn't carry query args across `redirect()`).
