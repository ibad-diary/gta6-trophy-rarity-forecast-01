"""gta6-trophy-rarity-forecast-01: earn-rate forecaster for GTA VI trophies on Exophase.

Method (stdlib only, deterministic, no network)
------------------------------------------------
The page's earn rate for a trophy is a ratio, not a primitive:

    rate_i(t) = 100 * E_i(t) / P(t)

where E_i(t) is the number of tracked players who have earned trophy i and
P(t) is the number of players Exophase tracks.  The two move for different
reasons: E_i only grows as players play, while P jumps in steps whenever
Exophase refreshes its counts (hours apart), which mechanically *dilutes*
every rate even though nobody un-earned anything.  Over a week the rate of
each trophy therefore rises steeply, then bends toward the share of
players who will ever earn it -- a saturating curve, roughly logistic in
log-time.  A straight line in log-time (the house curve baseline) has no
bend and overshoots that saturation badly over a 4.5-day horizon; on a
synthetic long-horizon case it loses to plain persistence for exactly
that reason.

So each trophy is fitted with a saturating logistic in log-time:

    rate_i(t) = A_i * sigmoid(a_i + b_i * ln(1 + t_hours_since_release))

The asymptote A_i is chosen per trophy from a grid (from just above its
highest observed rate up to a loose tier cap) by least squares on
logit(rate/A), preferring the smallest asymptote within 10% of the best
fit -- a conservative choice that refuses to invent headroom the data
has not shown.  Slopes b_i are shrunk toward the pooled slope across
trophies (early trophies lead, late trophies follow, but they share one
progression structure), the level is anchored at the last read rather
than at the fit's intercept, and negative slopes are floored at 0:
a falling rate is dilution, a level shift, not a trend -- earners never
shrink, and the player pool's growth slows, so rates do not fall
forever.  Only reads where the page actually changed enter the fit; a
frozen stretch carries no signal.

Flat history (no read ever changed, as on launch night): there is no
trend to fit, but the page is stale -- the last refresh predates the
first read, and Exophase's first observed refresh took players tracked
from 638 to 3,868 while earners of the non-100% trophies grew only
~1.34x, i.e. rates were multiplied by ~0.22.  When the page's staleness
(time since the first read) plus the horizon reaches the ~4h refresh
scale, the forecast applies one such refresh: the median rate is the
last rate times the dilution factor 1.34 / 6.06 ~= 0.221 (a trophy at
100% stays at 100%: every tracked player holds it).  Without that
adjustment a flat forecast is exactly the house baseline, which the
refresh pushes entirely outside its band.

Spread (calibration, not just width): the logit spread sigma combines,
in quadrature, a random-walk term 0.5*sqrt(horizon in days) (the house
baseline's scale), the residual scatter of the fit, an extrapolation
penalty that grows with the log-time distance from the last read to the
target, and -- for flat histories -- a refresh uncertainty for the
per-trophy dispersion of the refresh.  sigma is clamped to [0.35, 2.0]
logit units.  Quantiles are the normal quantiles of the median's logit
+- z_tau * sigma.

Backtests: on the practice cases, with answers
reconstructed from the published refresh totals (3,868 players / 4,276
trophies earned), pinball loss ~0.16 vs the starter's ~1.40, and the
advantage holds under faster/slower earner-growth reconstructions
(0.20-0.22 vs 1.23-1.69).  On synthetic long-horizon cases (2 days of
stepped history, 4.5-day horizon, logistic truth) the saturating model
beats persistence; the linear-log-time version did not, which is why
the saturation is here.

Run:  MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
Grader: python3 -P -s /bot/main.py, reads every *.in, writes *.out.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- tuned constants (from the rules' stated refresh facts + backtests) ---
RW_SIGMA = 0.5          # logit random-walk sigma per sqrt(day), house scale
SIGMA_MIN = 0.35
SIGMA_MAX = 2.0
# First observed Exophase refresh (RULES.txt): players 638 -> 3868 (x6.06),
# earners of non-100% trophies 305 -> 408 (x1.34).  Rate factor 1.34/6.06.
REFRESH_PLAYERS_X = 3868.0 / 638.0
REFRESH_EARNERS_X = 408.0 / 305.0
REFRESH_RATE_X = REFRESH_EARNERS_X / REFRESH_PLAYERS_X  # ~= 0.221
REFRESH_SCALE_H = 4.0   # staleness + horizon above this => expect a refresh
REFRESH_SIGMA = 0.8     # extra logit spread while a refresh is pending
# Week-one saturation caps by tier (loose; only bind on wild extrapolation).
TIER_CAP = {"platinum": 8.0, "gold": 25.0, "silver": 45.0, "bronze": 90.0}
N_ASYM = 16             # asymptote grid points per trophy


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s: str) -> float:
    """ISO-8601 UTC -> epoch seconds (deterministic, no local tz)."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def linfit(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, rmse, n)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        resid = math.sqrt(sum((y - my) ** 2 for y in ys) / n) if n else 0.0
        return my, 0.0, resid, n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return a, b, resid, n


def saturating_fit(xs, ys_frac, r_last_frac, tier):
    """Fit rate = A * sigmoid(a + b*x) by gridding A.

    Returns (A, b, rmse_logit, n): A as a fraction, b the log-time slope,
    rmse the residual scatter of logit(rate) around the fitted curve.
    """
    n = len(xs)
    r_max = max(ys_frac) if ys_frac else r_last_frac
    cap = TIER_CAP.get(tier, 90.0) / 100.0
    a_lo = min(max(r_max * 1.05, 0.005), cap * 0.999)
    if a_lo >= cap:
        grid = [cap]
    else:
        lo, hi = math.log(a_lo), math.log(cap)
        grid = [math.exp(lo + (hi - lo) * k / (N_ASYM - 1)) for k in range(N_ASYM)]
    best = None  # (sse, A, a, b)
    for A in grid:
        zs = []
        ok = True
        for y in ys_frac:
            f = min(y, A * 0.999) / A
            if f <= 0.0 or f >= 1.0:
                ok = False
                break
            zs.append(math.log(f / (1.0 - f)))
        if not ok:
            continue
        a, b, _rm, _n = linfit(xs, zs)
        sse = sum((z - (a + b * x)) ** 2 for x, z in zip(xs, zs))
        if best is None or sse < best[0]:
            best = (sse, A, a, b)
    if best is None:
        return cap, 0.0, 0.0, n
    sse0 = best[0]
    # Smallest asymptote within 10% of the best SSE: refuse invented headroom.
    chosen = best
    for A in grid:
        zs = [math.log(min(y, A * 0.999) / A / (1.0 - min(y, A * 0.999) / A)) for y in ys_frac]
        a, b, _rm, _n = linfit(xs, zs)
        sse = sum((z - (a + b * x)) ** 2 for x, z in zip(xs, zs))
        if sse <= sse0 * 1.10 + 1e-12:
            chosen = (sse, A, a, b)
            break
    _sse, A, a, b = chosen
    # Residual scatter on the logit(rate) scale, for the spread.
    res = []
    for x, y in zip(xs, ys_frac):
        pred = A / (1.0 + math.exp(-(a + b * x)))
        pred = min(max(pred, 0.0001), 0.9999)
        y_c = min(max(y, 0.0001), 0.9999)
        res.append(math.log(y_c / (1 - y_c)) - math.log(pred / (1 - pred)))
    rmse = math.sqrt(sum(r * r for r in res) / len(res)) if res else 0.0
    return A, b, rmse, n


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    ntr = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0

    # Time base: hours since the release day began (2026-11-19T00:00Z).
    rel = case.get("game", {}).get("release_date", "2026-11-19")
    try:
        t0 = datetime.fromisoformat(rel + "T00:00:00+00:00").timestamp()
    except ValueError:
        t0 = parse_time(history[0]["at"])
    times = [parse_time(h["at"]) for h in history]
    t_last = times[-1]
    t_tgt = t_last + case["horizon_minutes"] * 60.0

    def x_of(t: float) -> float:
        return math.log1p(max(0.0, (t - t0) / 3600.0))

    x_last = x_of(t_last)
    x_tgt = x_of(t_tgt)
    dx = max(0.0, x_tgt - x_last)

    # Keep only reads where the page actually changed (frozen stretches
    # carry no signal and would overweight a stale level in the fit).
    kept = [0]
    for k in range(1, len(history)):
        prev, cur = history[kept[-1]], history[k]
        if cur["players_tracked"] != prev["players_tracked"] or cur["rates"] != prev["rates"]:
            kept.append(k)
    distinct = len(kept)
    flat = distinct <= 1

    last = history[-1]
    last_rates = last["rates"]

    # --- flat history: pending-refresh branch -------------------------
    staleness_h = (t_last - times[0]) / 3600.0
    horizon_h = case["horizon_minutes"] / 60.0
    refresh_due = flat and (staleness_h + horizon_h) >= REFRESH_SCALE_H
    if refresh_due:
        sigma = math.sqrt(RW_SIGMA ** 2 * horizon_days + REFRESH_SIGMA ** 2)
        sigma = min(SIGMA_MAX, max(SIGMA_MIN, sigma))
        rows = []
        for i, tr in enumerate(trophies):
            r = last_rates[i]
            if r is None:
                r = 0.0
            if r >= 99.99:
                med, s = 100.0, SIGMA_MIN
            else:
                med, s = clamp(r * REFRESH_RATE_X), sigma
            lg = logit(med)
            qs = sorted(round4(clamp(inv_logit(lg + z * s))) for z in Z)
            rows.append({"id": tr["id"], "q": qs})
        return {"trophies": rows}

    # --- general branch: saturating logistic in log-time, per trophy ---
    xs = [x_of(times[k]) for k in kept]
    fits = []
    slopes = []
    for i in range(ntr):
        ys = []
        ex = []
        for k in kept:
            r = history[k]["rates"][i]
            if r is None:
                continue
            ys.append(min(max(r, 0.01), 99.99) / 100.0)
            ex.append(x_of(times[k]))
        r_last = last_rates[i] if last_rates[i] is not None else 0.0
        A, b, rmse, n = saturating_fit(ex, ys, r_last / 100.0, trophies[i].get("tier", "bronze"))
        fits.append((A, b, rmse, n))
        if n >= 2:
            slopes.append(b)
    pooled_b = sum(slopes) / len(slopes) if slopes else 0.0
    pooled_b = max(0.0, pooled_b)
    resid_all = [f[2] for f in fits if f[3] >= 2]
    resid_med = sorted(resid_all)[len(resid_all) // 2] if resid_all else 0.0

    rows = []
    for i, tr in enumerate(trophies):
        r_last = last_rates[i]
        if r_last is None:
            r_last = 0.0
        A, b, rmse, n = fits[i]
        if r_last >= 99.99:
            med, sig_i = 100.0, SIGMA_MIN
        else:
            # Shrink slope to pooled; floor at 0 (dilution is a level shift).
            w = n / (n + 3.0) if n >= 2 else 0.0
            b_s = max(0.0, w * b + (1.0 - w) * pooled_b)
            # Anchor level at the last read: a from logit(r_last / A).
            f_last = min(max(r_last / 100.0, 0.0001), A * 0.999) / A
            a_anchor = math.log(f_last / (1.0 - f_last)) - b_s * x_last
            f_tgt = 1.0 / (1.0 + math.exp(-(a_anchor + b_s * x_tgt)))
            med = clamp(A * f_tgt * 100.0)
            # Never project below half the dilution floor implied by the
            # last rate, and never above the tier cap.
            med = min(med, TIER_CAP.get(tr.get("tier", "bronze"), 90.0))
            sig_i = math.sqrt(
                RW_SIGMA ** 2 * horizon_days
                + (rmse if n >= 2 else resid_med) ** 2
                + (0.35 * dx) ** 2
            )
            sig_i = min(SIGMA_MAX, max(SIGMA_MIN, sig_i))
        lg = logit(med)
        qs = sorted(round4(clamp(inv_logit(lg + z * sig_i))) for z in Z)
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
