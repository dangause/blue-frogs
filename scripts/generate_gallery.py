"""Generate HTML gallery of flagged axanthism candidate images.

Reads flagged CSVs from streaming inference and creates a browsable gallery
with lazy-loaded thumbnails, score badges, and links to iNaturalist.

Usage:
    python scripts/generate_gallery.py \
        --flagged-csvs results/streaming_focal_v2/flagged_model_a_v1.csv \
                       results/streaming_focal_v2/flagged_model_c_v1.csv \
        --output results/streaming_focal_v2/gallery.html \
        --intersection
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# iNaturalist URL patterns
INAT_PHOTO_URL = "https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/medium.jpg"
INAT_OBS_URL = "https://www.inaturalist.org/observations/{observation_id}"


def load_flagged_csvs(csv_paths: list[Path]) -> dict[str, pd.DataFrame]:
    """Load flagged CSVs and return dict mapping model name to dataframe."""
    model_dfs = {}
    for path in csv_paths:
        df = pd.read_csv(path)
        # Extract model name from model_version column or filename
        if "model_version" in df.columns:
            model_name = df["model_version"].iloc[0].rsplit("_v", 1)[0]
        else:
            # Fallback: extract from filename like flagged_model_a_v1.csv
            model_name = path.stem.replace("flagged_", "").rsplit("_v", 1)[0]
        model_dfs[model_name] = df
        logger.info(f"Loaded {len(df)} flagged images from {model_name}")
    return model_dfs


def find_intersection(model_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Find photos flagged by ALL models."""
    if len(model_dfs) < 2:
        return list(model_dfs.values())[0]

    # Start with first model's photo_ids
    common_photos = None
    for model_name, df in model_dfs.items():
        photo_set = set(df["photo_id"].unique())
        if common_photos is None:
            common_photos = photo_set
        else:
            common_photos &= photo_set

    logger.info(f"Found {len(common_photos)} photos flagged by all {len(model_dfs)} models")

    # Build merged dataframe with scores from each model
    rows = []
    for photo_id in common_photos:
        row = {"photo_id": photo_id}
        for model_name, df in model_dfs.items():
            match = df[df["photo_id"] == photo_id].iloc[0]
            row["observation_id"] = match["observation_id"]
            row[f"score_{model_name}"] = match["prediction_score"]
        rows.append(row)

    merged = pd.DataFrame(rows)
    # Add average score for sorting
    score_cols = [c for c in merged.columns if c.startswith("score_")]
    merged["avg_score"] = merged[score_cols].mean(axis=1)
    return merged.sort_values("avg_score", ascending=False)


def find_union(model_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine all flagged photos (union)."""
    all_rows = []
    seen_photos = set()

    for model_name, df in model_dfs.items():
        for _, row in df.iterrows():
            photo_id = row["photo_id"]
            if photo_id in seen_photos:
                continue
            seen_photos.add(photo_id)
            all_rows.append({
                "photo_id": photo_id,
                "observation_id": row["observation_id"],
                "prediction_score": row["prediction_score"],
                "source_model": model_name,
            })

    merged = pd.DataFrame(all_rows)
    return merged.sort_values("prediction_score", ascending=False)


def generate_html(
    df: pd.DataFrame,
    model_names: list[str],
    output_path: Path,
    title: str = "Axanthism Candidates Gallery",
) -> None:
    """Generate HTML gallery from merged dataframe."""
    score_cols = [c for c in df.columns if c.startswith("score_")]
    has_multi_scores = len(score_cols) > 0

    # Generate image cards
    cards_html = []
    for _, row in df.iterrows():
        photo_id = int(row["photo_id"])
        obs_id = int(row["observation_id"])
        photo_url = INAT_PHOTO_URL.format(photo_id=photo_id)
        obs_url = INAT_OBS_URL.format(observation_id=obs_id)

        # Build score badges
        if has_multi_scores:
            badges = []
            for col in score_cols:
                model = col.replace("score_", "")
                score = row[col]
                badges.append(f'<span class="badge badge-{model}">{model}: {score:.3f}</span>')
            badges_html = "\n".join(badges)
            avg_score = row.get("avg_score", 0)
            data_score = f'data-avg-score="{avg_score:.4f}"'
        else:
            score = row.get("prediction_score", 0)
            model = row.get("source_model", "unknown")
            badges_html = f'<span class="badge">{model}: {score:.3f}</span>'
            data_score = f'data-avg-score="{score:.4f}"'

        card = f'''
        <div class="card" {data_score} data-photo="{photo_id}" data-obs="{obs_id}">
            <a href="{obs_url}" target="_blank" rel="noopener">
                <img loading="lazy" src="{photo_url}" alt="Photo {photo_id}">
            </a>
            <div class="card-info">
                <div class="badges">{badges_html}</div>
                <div class="links">
                    <a href="{obs_url}" target="_blank">iNat #{obs_id}</a>
                </div>
            </div>
        </div>'''
        cards_html.append(card)

    cards_joined = "\n".join(cards_html)

    # Generate model filter buttons if multiple models
    if has_multi_scores:
        filter_buttons = '<button class="filter-btn active" data-filter="all">All ({n})</button>'.format(
            n=len(df)
        )
        for model in model_names:
            filter_buttons += f'<button class="filter-btn" data-filter="{model}">{model}</button>'
    else:
        filter_buttons = ""

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --bg: #1a1a2e;
            --card-bg: #16213e;
            --text: #eee;
            --accent: #00d4aa;
            --badge-a: #ff6b6b;
            --badge-b: #4ecdc4;
            --badge-c: #45b7d1;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 20px;
        }}
        header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        h1 {{ color: var(--accent); margin-bottom: 10px; }}
        .stats {{ color: #888; font-size: 0.9em; }}
        .controls {{
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .filter-btn {{
            padding: 8px 16px;
            border: 2px solid var(--accent);
            background: transparent;
            color: var(--text);
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .filter-btn:hover, .filter-btn.active {{
            background: var(--accent);
            color: var(--bg);
        }}
        select {{
            padding: 8px 16px;
            background: var(--card-bg);
            color: var(--text);
            border: 2px solid var(--accent);
            border-radius: 8px;
            cursor: pointer;
        }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
        }}
        .card {{
            background: var(--card-bg);
            border-radius: 12px;
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 25px rgba(0, 212, 170, 0.2);
        }}
        .card img {{
            width: 100%;
            aspect-ratio: 4/3;
            object-fit: cover;
            display: block;
        }}
        .card-info {{
            padding: 12px;
        }}
        .badges {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-bottom: 8px;
        }}
        .badge {{
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: 600;
            background: #333;
        }}
        .badge-model_a {{ background: var(--badge-a); color: #000; }}
        .badge-model_b {{ background: var(--badge-b); color: #000; }}
        .badge-model_c {{ background: var(--badge-c); color: #000; }}
        .links a {{
            color: var(--accent);
            text-decoration: none;
            font-size: 0.85em;
        }}
        .links a:hover {{ text-decoration: underline; }}
        .hidden {{ display: none !important; }}
    </style>
</head>
<body>
    <header>
        <h1>{title}</h1>
        <p class="stats">{len(df)} images | Generated from {len(model_names)} model(s)</p>
    </header>

    <div class="controls">
        {filter_buttons}
        <select id="sort-select">
            <option value="score-desc">Score (High to Low)</option>
            <option value="score-asc">Score (Low to High)</option>
            <option value="obs-asc">Observation ID (Asc)</option>
            <option value="obs-desc">Observation ID (Desc)</option>
        </select>
    </div>

    <div class="gallery" id="gallery">
        {cards_joined}
    </div>

    <script>
        const gallery = document.getElementById('gallery');
        const cards = Array.from(gallery.querySelectorAll('.card'));

        // Sorting
        document.getElementById('sort-select').addEventListener('change', (e) => {{
            const [field, dir] = e.target.value.split('-');
            const mult = dir === 'asc' ? 1 : -1;
            cards.sort((a, b) => {{
                let va, vb;
                if (field === 'score') {{
                    va = parseFloat(a.dataset.avgScore);
                    vb = parseFloat(b.dataset.avgScore);
                }} else {{
                    va = parseInt(a.dataset.obs);
                    vb = parseInt(b.dataset.obs);
                }}
                return (va - vb) * mult;
            }});
            cards.forEach(card => gallery.appendChild(card));
        }});

        // Filtering (if multiple models)
        document.querySelectorAll('.filter-btn').forEach(btn => {{
            btn.addEventListener('click', () => {{
                document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                const filter = btn.dataset.filter;
                cards.forEach(card => {{
                    if (filter === 'all') {{
                        card.classList.remove('hidden');
                    }} else {{
                        const hasModel = card.querySelector('.badge-' + filter);
                        card.classList.toggle('hidden', !hasModel);
                    }}
                }});
            }});
        }});
    </script>
</body>
</html>'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    logger.info(f"Gallery saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML gallery of flagged axanthism candidates"
    )
    parser.add_argument(
        "--flagged-csvs", type=Path, nargs="+", required=True,
        help="Paths to flagged CSV files from streaming inference",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output path for HTML gallery",
    )
    parser.add_argument(
        "--intersection", action="store_true",
        help="Only show images flagged by ALL models (default: union)",
    )
    parser.add_argument(
        "--title", type=str, default="Axanthism Candidates Gallery",
        help="Gallery title",
    )
    args = parser.parse_args()

    # Load all flagged CSVs
    model_dfs = load_flagged_csvs(args.flagged_csvs)
    model_names = list(model_dfs.keys())

    # Merge according to mode
    if args.intersection and len(model_dfs) > 1:
        merged = find_intersection(model_dfs)
        logger.info(f"Intersection mode: {len(merged)} images")
    else:
        merged = find_union(model_dfs)
        logger.info(f"Union mode: {len(merged)} images")

    if len(merged) == 0:
        logger.warning("No images to display")
        return

    generate_html(merged, model_names, args.output, args.title)
    logger.info(f"Done! Open {args.output} in a browser to view the gallery.")


if __name__ == "__main__":
    main()
