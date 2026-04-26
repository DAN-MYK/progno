"""Generate report.md artifact after training (spec §6.6)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from progno_train.config import Paths


def generate_report(paths: "Paths", tour: str = "atp") -> Path:
    """Write artifacts/report.md and return the path."""
    if not paths.model_card.exists():
        raise FileNotFoundError(f"model_card.json not found at {paths.model_card}")

    card = json.loads(paths.model_card.read_text())
    metrics = card.get("metrics", {})
    feature_names: list[str] = card.get("feature_names", [])
    git_sha = card.get("git_sha", "unknown")[:8]
    train_years = card.get("train_years", [])

    lines: list[str] = []
    lines.append(f"# Progno {tour.upper()} — Model Report")
    lines.append(f"\n_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · git {git_sha}_\n")

    # ── Metrics table ─────────────────────────────────────────────────────────
    lines.append("## Metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    if train_years:
        lines.append(f"| Train years | {train_years[0]}–{train_years[1]} |")
    for key, label in [
        ("logloss_catboost", "Log-loss (CatBoost)"),
        ("ece_catboost", "ECE"),
        ("roi_catboost", "ROI 0.25× Kelly"),
        ("n_test", "Test rows"),
    ]:
        if key in metrics:
            val = metrics[key]
            lines.append(f"| {label} | {f'{val:.4f}' if isinstance(val, float) else val} |")
    lines.append("")

    # ── Delta vs previous published version ───────────────────────────────────
    prev = _find_prev_card(paths, tour)
    if prev:
        pm = prev.get("metrics", {})
        lines.append("## Delta vs Previous Version\n")
        lines.append("| Metric | Prev | Now | Δ |")
        lines.append("|--------|------|-----|---|")
        for key, label in [
            ("logloss_catboost", "Log-loss"),
            ("ece_catboost", "ECE"),
            ("roi_catboost", "ROI"),
        ]:
            if key in metrics and key in pm:
                cur_v, prev_v = float(metrics[key]), float(pm[key])
                delta = cur_v - prev_v
                flag = " ⚠" if key != "roi_catboost" and delta > 0.005 else ""
                lines.append(f"| {label} | {prev_v:.4f} | {cur_v:.4f} | {delta:+.4f}{flag} |")
        lines.append("")

    # ── Feature importance top-20 ─────────────────────────────────────────────
    fi = _feature_importance(paths, feature_names)
    if fi:
        lines.append("## Feature Importance (top 20)\n")
        lines.append("| Rank | Feature | Score |")
        lines.append("|------|---------|-------|")
        for i, (name, score) in enumerate(fi[:20], 1):
            lines.append(f"| {i} | `{name}` | {score:.2f} |")
        lines.append("")

    # ── Elo top-20 ────────────────────────────────────────────────────────────
    elo_top = _elo_top20(paths)
    if elo_top:
        lines.append("## Elo Rankings (top 20)\n")
        lines.append("| # | Player | Overall |")
        lines.append("|---|--------|---------|")
        for i, (name, elo) in enumerate(elo_top, 1):
            lines.append(f"| {i} | {name.title()} | {elo:.0f} |")
        lines.append("")

    # ── Monthly accuracy on test set ──────────────────────────────────────────
    monthly = _monthly_stats(paths)
    if monthly is not None and not monthly.empty:
        lines.append("## Monthly Stats (test set)\n")
        has_roi = "roi_pct" in monthly.columns and monthly["roi_pct"].notna().any()
        if has_roi:
            lines.append("| Month | n | Accuracy | Avg Edge | ROI% |")
            lines.append("|-------|---|----------|----------|------|")
            for row in monthly.itertuples():
                lines.append(
                    f"| {row.month} | {row.n} | {row.acc:.1%} "
                    f"| {row.avg_edge:+.1%} | {row.roi_pct:+.1f}% |"
                )
        else:
            lines.append("| Month | n | Accuracy | Avg Edge |")
            lines.append("|-------|---|----------|----------|")
            for row in monthly.itertuples():
                lines.append(f"| {row.month} | {row.n} | {row.acc:.1%} | {row.avg_edge:+.1%} |")
        lines.append("")

    out = paths.report
    out.write_text("\n".join(lines) + "\n")
    return out


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_prev_card(paths: "Paths", tour: str) -> dict | None:
    parent = paths.artifacts.parent  # training/artifacts/
    for v in sorted(parent.glob("v*/"), reverse=True):
        card = v / tour / "model_card.json"
        if card.exists():
            try:
                return json.loads(card.read_text())
            except Exception:
                pass
    return None


def _feature_importance(paths: "Paths", feature_names: list[str]) -> list[tuple[str, float]] | None:
    try:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(str(paths.model_cbm))
        scores = model.get_feature_importance()
        return sorted(zip(feature_names, scores.tolist()), key=lambda x: -x[1])
    except Exception:
        return None


def _elo_top20(paths: "Paths") -> list[tuple[str, float]] | None:
    try:
        state = json.loads(paths.elo_state.read_text())
        players = state.get("players", {})
        ranked = sorted(
            ((name, float(data.get("elo_overall", 1500))) for name, data in players.items()),
            key=lambda x: -x[1],
        )
        return ranked[:20]
    except Exception:
        return None


def _monthly_stats(paths: "Paths") -> pd.DataFrame | None:
    """Compute per-month accuracy and avg edge on the test set."""
    try:
        from catboost import CatBoostClassifier, Pool
        from progno_train.train import apply_platt, get_feature_cols, TEST_START_YEAR, CAT_FEATURES

        model = CatBoostClassifier()
        model.load_model(str(paths.model_cbm))
        cal = json.loads(paths.calibration.read_text())
        a, b = cal["a"], cal["b"]

        df = pd.read_parquet(paths.featurized)
        test = df[df["year"] >= TEST_START_YEAR].copy()
        feature_cols = get_feature_cols(df)
        cat_idx = [i for i, c in enumerate(feature_cols) if c in CAT_FEATURES]
        pool = Pool(test[feature_cols].fillna(0), cat_features=cat_idx, feature_names=feature_cols)
        raw = model.predict_proba(pool)[:, 1]
        probs = apply_platt(raw, a, b)

        test = test.reset_index(drop=True)
        test["_prob"] = probs
        test["_month"] = pd.to_datetime(test["tourney_date"]).dt.to_period("M").astype(str)

        # Find any winner odds column for ROI calc
        odds_col = next(
            (c for c in ["odds_a_winner", "PSW", "B365W"] if c in test.columns and test[c].notna().any()),
            None,
        )

        rows = []
        for month, grp in test.groupby("_month"):
            p = grp["_prob"].values
            y = grp["label"].values
            acc = float(np.mean((p > 0.5) == (y == 1)))
            # avg edge = avg(p - 1/odds) for bets where p>0.5
            if odds_col:
                o = grp[odds_col].fillna(np.nan).values
                mask = (p > 0.5) & np.isfinite(o) & (o > 1)
                avg_edge = float(np.mean(p[mask] - 1.0 / o[mask])) if mask.any() else 0.0
                # simple flat-bet ROI: stake 1 on each bet where p>0.5
                n_bets = mask.sum()
                pnl = float(np.sum(np.where(mask, np.where(y == 1, o - 1, -1.0), 0)))
                roi_pct = float(pnl / n_bets * 100) if n_bets > 0 else float("nan")
            else:
                avg_edge = float(np.mean(p - 0.5))
                roi_pct = float("nan")
            rows.append({"month": str(month), "n": len(grp), "acc": acc,
                         "avg_edge": avg_edge, "roi_pct": roi_pct})

        return pd.DataFrame(rows)
    except Exception:
        return None
