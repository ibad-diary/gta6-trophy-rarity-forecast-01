[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ibad-diary/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ibad-diary/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast

Forecast, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn rate
on [Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
at 2026-11-26T00:00:00Z, as five quantiles per trophy (5%, 25%, 50%, 75%, 95%),
scored by pinball loss on the logit scale.

Entry for the [make.fun bounty gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

## Method

An earn rate is a ratio, `rate = 100 * earners / players_tracked`, and its two
parts move on different clocks: earners only grow as players play, while the
tracked-player pool jumps in steps whenever Exophase refreshes its counts,
mechanically diluting every rate. So each trophy is fitted with a **saturating
logistic in log-time**,

```
rate(t) = A * sigmoid(a + b * ln(1 + t_hours_since_release))
```

with the asymptote `A` chosen per trophy from a grid (smallest asymptote within
10% of the best fit — no invented headroom), slopes shrunk toward the pooled
slope across trophies, level anchored at the last read, and negative slopes
floored at 0 (a falling rate is dilution, a level shift, not a trend). Only
reads where the page actually changed enter the fit.

On launch night the page is frozen flat, so there is no trend to fit — but the
page is stale and a refresh is pending. Exophase's first observed refresh took
players tracked from 638 to 3,868 while earners grew only ~1.34x, i.e. rates
were multiplied by ~0.22. When staleness + horizon reaches the ~4h refresh
scale, the forecast applies one such refresh (a trophy at 100% stays at 100%).

The logit spread combines a random-walk term, the fit's residual scatter, an
extrapolation penalty, and (for flat histories) refresh uncertainty.

## Backtest

Practice cases, answers reconstructed from the published refresh totals
(3,868 players / 4,276 trophies earned): pinball loss **~0.16 vs the starter's
~1.40**, and the advantage holds under faster/slower earner-growth
reconstructions (0.20–0.22 vs 1.23–1.69). On synthetic long-horizon cases
(2 days of stepped history, 4.5-day horizon) the model beats persistence in
all three variants (0.23 vs 0.30–0.31). See `main.py`'s docstring for detail.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Deterministic: two runs on the
same case write byte-identical output. Runtime ~0.25s for both practice cases.

## License

MIT — see [LICENSE](LICENSE).

---
Referral code: `s67gjd7a`
