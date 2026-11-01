# Bet Engine (free, runs on your own computer)

## Repository layout (two categories)
```
app.py              <- 1. the app (what you run)
requirements.txt    <- 1. its dependencies
core/               <- 2. everything the app imports
  engine.py  daily.py  risk.py  store.py  budget.py  oddsapi.py  oddspapi.py
tests/  docs/  legacy/  .streamlit/secrets.toml.example  .gitignore
```
Deploying (e.g. Streamlit Community Cloud): main file `app.py`; put keys in Secrets as `ODDSPAPI_KEY`. Note that
hosted storage is wiped on restart, so keep the bet log on your own PC (`data/engine.db` is git-ignored on purpose).

Daily match analysis (BET or SKIP) and the Sporty 13 jackpot ticket, with bankroll protection.
It never places bets. You copy the slip into SportyBet by hand.

## Setup (once)
```
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501. The first launch downloads results and fits models, which can take a few minutes;
after that it is cached for 6 hours. Your data is saved in `data/engine.db` (back it up from the Bankroll tab).

## What the tool filters, learns and tests
- Today tab lists only matches that qualify (BET); untick the box to see the SKIPs and why.
- Jackpot: all 13 matches must be filled on SportyBet, so it cannot skip matches. It marks each as strong single,
  risky single or covered, and gives a BUY or SKIP THIS WEEK verdict for the whole ticket.
- Results are picked up automatically when the app opens (free results file updates a few times a week) and settle bets.
- Learning tab, "Run the past": walk-forward backtest with closing-line value and bootstrap ranges. Run it before paper trading.

## Daily routine
1. Today tab: copy matches and 1 X 2 odds from SportyBet, one per line, paste, press Load matches.
2. Read the Call column. SKIP is the normal answer. A BET shows the stake and the lowest odds worth taking.
3. Press "Save this analysis to the log" (this is how the tool learns).
4. After the matches, Learning tab: "Update results and settle bets".

## Weekly routine (Saturday before 17:00 Kenya time)
1. Jackpot tab: paste the 13 matches in SportyBet order, enter the current Super Jackpot.
2. Compare the two tickets, check the monthly budget line, save the ticket, enter the picks in SportyBet.
3. Monday: enter last round's results under "Past rounds". After 6+ rounds the tool estimates tickets sold.
4. Learning tab: run the calibration now and then.

## Safety rules built in
- PAPER mode by default. LIVE unlocks only after the proof gate (Learning tab) passes.
- Stake caps, daily bet limit, daily and weekly loss stops, drawdown stop that returns to PAPER.
- Loosening any limit waits 7 days. Tightening is immediate.
- No accumulators. Jackpot spend is capped per month.
- Predictions and bets cannot be edited or deleted after they are logged.

## Optional: sharp reference odds, paced to last the month
Default provider: OddsPapi free tier (250 requests a month, Pinnacle prices). Pacing (core/budget.py): a safety
reserve is never touched; each Saturday still to come this month holds back a few requests for the jackpot; the rest
is shared evenly over the remaining days and unused allowance rolls forward. Odds come in batches of 3 competitions per
request and are cached for 6 hours. Teams names come from one cached lookup (about 2 requests a month in total for
the competition and team lists). Enter the provider dashboard's remaining figure to keep our count honest.

### Older notes: The Odds API (500 credits, no Pinnacle)
Free key from the-odds-api.com (500 credits a month). One call covers a whole competition for 1 credit, however many
matches it has. The tool therefore: caches every call for 6 hours (reloading is free, and it survives restarts);
spends credits only on competitions that contain a match worth confirming (most days that is none); stops at the
monthly cap you set; and shows a safe daily allowance. The key is not saved. Whether the free key includes Pinnacle
prices is not confirmed. No key? Type a sharp bookmaker's 1 X 2 prices into the Ref columns (More columns): zero credits.

## Tests
```
pip install pytest
python -m pytest tests -q
```

## Honest limits
- No edge over the bookmaker has been demonstrated. PAPER mode exists to find out.
- The live data download, the odds API call, and team-name matching on real SportyBet names are untested.
- Jackpot inputs (prize pool, tickets sold, crowd picks) are placeholders until you enter real rounds.
- This is a tool for analysis, not financial advice. Only use money you can afford to lose. If betting is causing
  stress or debt, stop and seek help.
