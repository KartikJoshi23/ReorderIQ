"""
Build the Favorita data pack the dashboard loads (plan §2.3).

    Kaggle CSVs -> curated series + per-store calendar (favorita_data.build_pack) -> compact JSON

Outputs
    app/public/data/favorita-pack.json   deployed with the dashboard (loaded automatically on open)
    data/processed/favorita-pack.json    git-ignored copy read by the notebook and offline reports
    app/src/data/demo-params.json        aggregate statistics for the synthetic demo mode

Usage:
    python scripts/build_data_pack.py
    python scripts/build_data_pack.py --fixture      # synthetic test fixture only (no real data)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import favorita_data as fd  # noqa: E402

APP = fd.ROOT / "app"
PUBLIC_OUT = APP / "public" / "data" / "favorita-pack.json"
PLAIN_OUT = fd.ROOT / "data" / "processed" / "favorita-pack.json"       # git-ignored
DEMO_PARAMS_OUT = APP / "src" / "data" / "demo-params.json"            # aggregate stats only
FIXTURE_DIR = APP / "tests" / "fixtures"


def build_fixture() -> None:
    """Tiny synthetic pack (same schema, made-up numbers) for the TypeScript unit tests."""
    rng = np.random.default_rng(7)
    days = 21
    fixture = {
        "schema": "favorita-pack/1",
        "meta": {"source": "SYNTHETIC TEST FIXTURE", "url": "", "terms": "synthetic"},
        "calendar": {"start": "2016-12-20", "days": days, "trainEnd": "2017-01-05",
                     "modelStart": "2016-12-20", "replayStart": "2017-01-01", "promoKnownFrom": "2016-12-20",
                     "oil": [round(50 + float(v), 2) for v in rng.normal(0, 1, days).cumsum()]},
        "stores": [{"store": 1, "city": "Testville", "state": "Test", "type": "A", "cluster": 1,
                    "holiday": [1 if d in (5, 12) else 0 for d in range(days)], "event": [0] * days,
                    "quake": [0] * days, "workDay": [0] * days,
                    "closed": [1 if d in (5, 12) else 0 for d in range(days)],   # 25 Dec, 1 Jan
                    "names": {"5": "Navidad", "12": "Primer dia del ano"}}],
        "series": [{"id": "1-TEST", "store": 1, "family": "TEST", "unit": "units", "fractional": False,
                    "sales": [None if d > 16 else (0 if d in (5, 12) else int(v))
                              for d, v in enumerate(rng.integers(80, 120, days))],
                    "promo": [int(v) for v in rng.integers(0, 4, days)]}],
        "defaultSeries": "1-TEST",
    }
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURE_DIR / "fixture-pack.json").write_text(json.dumps(fixture, indent=1), encoding="utf-8")
    print(f"Wrote synthetic fixture to {FIXTURE_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=fd.RAW_DIR)
    ap.add_argument("--fixture", action="store_true", help="only build the synthetic test fixture")
    args = ap.parse_args()

    if args.fixture:
        build_fixture()
        return

    stores = {s for s, _ in fd.SHORTLIST}
    print("Loading Kaggle CSVs ...")
    raw = fd.load_raw(args.raw_dir, stores_subset=stores)
    pack = fd.build_pack(raw)
    demo = fd.estimate_demo_params(raw)

    text = json.dumps(pack, separators=(",", ":"), ensure_ascii=False)
    for out in (PUBLIC_OUT, PLAIN_OUT):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    DEMO_PARAMS_OUT.parent.mkdir(parents=True, exist_ok=True)
    DEMO_PARAMS_OUT.write_text(json.dumps(demo, indent=2), encoding="utf-8")

    print(f"Series: {len(pack['series'])} | days: {pack['calendar']['days']} | pack {len(text) / 1024:.1f} KiB -> {PUBLIC_OUT}")
    print(f"Demo parameters (aggregate statistics only) -> {DEMO_PARAMS_OUT}  (fit R^2 = {demo['fit']['r2']:.3f})")


if __name__ == "__main__":
    main()
