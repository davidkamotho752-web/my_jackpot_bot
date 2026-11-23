"""Tax logic. Reads config/tax.json; never hardcodes rates in code."""
import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parent.parent / "config" / "tax.json"

def load():
    with open(CONFIG) as f:
        cfg = json.load(f)
    active = cfg["regimes"][cfg["active"]]
    return {
        "regime": cfg["active"],
        "description": active["description"],
        "deposit_excise": active["deposit_excise"],
        "withdrawal_wht": active["withdrawal_wht"],
        "winnings_wht": active["winnings_wht"],
    }

def effective_odds(odds, tax=None):
    t = tax or load()
    return 1.0 + (odds - 1.0) * (1.0 - t["winnings_wht"])

def net_edge(p, odds, tax=None):
    return p * effective_odds(odds, tax) - 1.0

def min_odds_for(p_low, min_edge, tax=None):
    t = tax or load()
    req_eff = 1.0 + min_edge / max(p_low, 1e-9)
    return 1.0 + max(req_eff - 1.0, 0.01) / max(1.0 - t["winnings_wht"], 0.01)

def flow_hurdle(tax=None):
    t = tax or load()
    return (1.0 / ((1.0 - t["deposit_excise"]) * (1.0 - t["withdrawal_wht"]))) - 1.0
