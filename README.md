# CFB Spread Model

FBS college football spread and total model. Same layout as the NFL Spread Model:
This Week, Game Detail, Results, Ratings, and a Method & Testing page that shows how
the model was built and backtested.

The site is plain HTML/JS reading JSON files in `data/`. A GitHub Action rebuilds those
files every morning and commits them; Netlify redeploys on each commit.

## One-time setup

1. **Create the repo.** On GitHub (account `barton84`), create a new repository named
   `CFBPrediction`. Upload everything in this folder (keep the `.github` folder; it is
   hidden on Mac/Windows, so drag the whole folder or use GitHub Desktop).
2. **Add the API keys.** Repo **Settings > Secrets and variables > Actions > New repository secret**:
   - `ODDS_API_KEY` : your the-odds-api.com key (DraftKings lines)
   - `CFBD_API_KEY` : your CollegeFootballData.com key (stadium locations for neutral sites)
3. **Let the Action commit.** Repo **Settings > Actions > General > Workflow permissions**:
   choose **Read and write permissions** and save.
4. **Connect Netlify.** Netlify **Add new site > Import an existing project > GitHub >
   CFBPrediction**. Leave the build command empty and the publish directory as `.`
   (the included `netlify.toml` already says this). Deploy.
5. **First run.** Repo **Actions** tab:
   - Run **Retrain model** once (about 15-25 minutes). This pulls 2019 to now, refits
     everything, rebuilds the backtest and backfills weather history so the weather
     adjustment gets fitted on real games.
   - After that, **Daily update** runs on its own every morning. You can also run it by hand.

The repo already contains a complete build from Oct 12, 2026 (ratings through week 6),
so the site works the moment Netlify deploys. Week 7 lines appear after the first daily
run with your Odds API key.

## Schedule and API usage

| Job | When | Uses |
| --- | --- | --- |
| Daily update | Every day 6:45 AM Central, Friday 4 PM, Saturday 10:15 AM | 1 Odds API call (2 credits), Open-Meteo forecast (free), CFBD venues on Mondays |
| Retrain model | Aug 10 each year, or manually | CFBD venues, Open-Meteo history (free) |

About 80 Odds API credits a month, well under the free 500.

## How it works (short)

1. Play-by-play for every FBS game (SportsDataverse cfbfastR data, public, no key).
2. Garbage time removed; opponent-adjusted offense and defense ratings for success rate,
   EPA per play, yards per play, points and pace, rebuilt before every week.
3. Preseason prior: last season's rating plus returning production, QB continuity and
   transfers (weights fitted on 2021-2025).
4. Spread = regression on success rate edge, yards per play edge, ESPN FPI, home field and
   regressed fumble luck, fitted separately for weeks 1-3, 4-6 and 7+.
5. Total = adjusted scoring, pace, EPA matchup, dome and scoring environment, centered on
   the median. Weather adjusts the total.
6. Flags: 5+ points vs DraftKings (light green), 7+ (green).

Backtest (leave one season out, graded vs closing lines): spreads 2022-2025 51.9% on all
games, 53.7% at 5+, 54.8% at 7+; totals 2023-2025 53.4%, 55.1% at 5+. Full detail,
including year-by-year swings and everything that was tested and dropped, is on the
Method & Testing tab.

## Files

- `index.html`, `assets/` : the site
- `data/` : JSON the site reads (written by the jobs)
- `state/` : preseason prior, DraftKings line history, weather history, stadium list
- `pipeline/run_daily.py` : daily job
- `pipeline/retrain.py` : full rebuild and backtest
- `pipeline/model/` : ratings, features, training, feature tests
- `pipeline/odds.py`, `weather.py`, `cfbd.py`, `fetch.py` : data sources

Run locally: `pip install -r requirements.txt`, then `python pipeline/run_daily.py`
and open the folder with any static server (`python -m http.server`).
