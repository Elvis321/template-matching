"""
Base detection and feature extraction.

IMPORTANT: this logic must stay byte-for-byte identical to whatever produced the
training dataset in the research notebook. Any drift here (different rolling windows,
different fallback values, different rounding) silently shifts the live feature
distribution away from what the model was trained on -- a common and hard-to-notice
cause of live underperformance relative to backtest.

FIX (2026-09-11): removed an erroneous trailing "- atr_window" from the range in
detect_candidate_bases. It excluded any base from ending within the last atr_window
bars of the array -- harmless when scanning full multi-year history during training,
but fatal for live use, where "the end of the array" is always "right now." This made
get_most_recent_base's recency check mathematically impossible to satisfy, so no
setup could ever be detected live, on any symbol, regardless of market conditions.
"""
import numpy as np
import pandas as pd
from config import config


def detect_candidate_bases(prices, atr_window=14, base_len_range=(3, 10), contraction_ratio=0.5):
    returns = np.diff(prices) / prices[:-1]
    rolling_vol = pd.Series(returns).rolling(atr_window).std().values
    candidates = []
    for base_len in range(base_len_range[0], base_len_range[1] + 1):
        for i in range(atr_window, len(prices) - base_len):
            base_vol = np.nanstd(returns[i:i + base_len])
            surrounding_vol = rolling_vol[i]
            if np.isnan(surrounding_vol) or surrounding_vol == 0:
                continue
            if base_vol < contraction_ratio * surrounding_vol:
                candidates.append((i, i + base_len))
    return candidates


def dedupe_candidates(candidates, min_gap=5):
    candidates = sorted(candidates, key=lambda c: c[0])
    kept = []
    last_end = -999
    for start, end in candidates:
        if start - last_end >= min_gap:
            kept.append((start, end))
            last_end = end
    return kept


def extract_features(close, high, low, volume, leg_in_start, base_start, base_end,
                      leg_in_lookback=10, context_lookback=50):
    if leg_in_start < 0 or base_end >= len(close) - 1:
        return None
    if base_start - leg_in_start < 2:
        return None

    leg_in = close[leg_in_start:base_start]
    base = close[base_start:base_end]
    base_vol = volume[base_start:base_end]
    leg_in_vol = volume[leg_in_start:base_start]
    context_start = max(0, leg_in_start - context_lookback)
    context = close[context_start:leg_in_start]

    if len(leg_in) < 2 or len(base) < 2 or len(context) < 5:
        return None

    features = {}

    features["leg_in_return_pct"] = (leg_in[-1] / leg_in[0] - 1) * 100
    features["leg_in_bars"] = len(leg_in)
    features["leg_in_direction"] = 1 if leg_in[-1] > leg_in[0] else -1
    features["leg_in_volatility"] = np.std(np.diff(leg_in) / leg_in[:-1])
    features["leg_in_avg_vol_ratio"] = leg_in_vol.mean() / (volume[context_start:base_start].mean() + 1e-9)
    features["leg_in_monotonicity"] = np.mean(np.sign(np.diff(leg_in)) == features["leg_in_direction"])

    features["base_bars"] = len(base)
    features["base_range_pct"] = (base.max() - base.min()) / base.mean() * 100
    features["base_volatility"] = np.std(np.diff(base) / base[:-1]) if len(base) > 1 else 0
    features["base_vol_contraction"] = base_vol.mean() / (leg_in_vol.mean() + 1e-9)
    features["base_position_in_leg"] = (base.mean() - leg_in.min()) / (leg_in.max() - leg_in.min() + 1e-9)
    features["base_drift_pct"] = (base[-1] / base[0] - 1) * 100
    features["base_vol_trend"] = (
        np.polyfit(range(len(base_vol)), base_vol, 1)[0] / (base_vol.mean() + 1e-9)
        if len(base_vol) > 2 else 0
    )

    context_returns = np.diff(context) / context[:-1]
    features["context_volatility"] = np.std(context_returns)
    features["context_trend_pct"] = (context[-1] / context[0] - 1) * 100
    features["dist_from_context_high_pct"] = (context.max() - close[base_end - 1]) / context.max() * 100
    features["dist_from_context_low_pct"] = (close[base_end - 1] - context.min()) / context.min() * 100

    if len(context_returns) >= 14:
        gains = np.where(context_returns[-14:] > 0, context_returns[-14:], 0)
        losses = np.where(context_returns[-14:] < 0, -context_returns[-14:], 0)
        avg_gain, avg_loss = gains.mean(), losses.mean()
        rs = avg_gain / (avg_loss + 1e-9)
        features["rsi"] = 100 - (100 / (1 + rs))
    else:
        features["rsi"] = None  # insufficient history -- caller should skip, not fill 50

    return features


def get_most_recent_base(close, high, low, volume, recency_bars=3):
    """
    Find the most recently completed base (if any) that finished within the last
    `recency_bars` bars -- i.e. a genuinely 'live' setup worth predicting on right now,
    not a stale one from earlier in the session.
    """
    candidates = detect_candidate_bases(
        close,
        atr_window=config.ATR_WINDOW,
        base_len_range=(config.BASE_LEN_MIN, config.BASE_LEN_MAX),
        contraction_ratio=config.CONTRACTION_RATIO,
    )
    candidates = dedupe_candidates(candidates, min_gap=config.DEDUP_GAP)
    if not candidates:
        return None

    base_start, base_end = candidates[-1]
    if base_end < len(close) - recency_bars:
        return None  # most recent detected base is stale, not a live setup

    leg_in_start = max(0, base_start - config.LEG_IN_LOOKBACK)
    feats = extract_features(
        close, high, low, volume, leg_in_start, base_start, base_end,
        leg_in_lookback=config.LEG_IN_LOOKBACK, context_lookback=config.CONTEXT_LOOKBACK,
    )
    if feats is None or feats.get("rsi") is None:
        return None

    return {"base_start": base_start, "base_end": base_end, "features": feats}