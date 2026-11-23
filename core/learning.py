"""Analyze settled predictions, diagnose why, and adjust calibration."""
import core.db as db
import core.calibration as cal
import numpy as np

def run_learning():
    settled = db.q("SELECT * FROM predictions WHERE result IN ('1', 'X', '2') AND source='manual'")
    if not settled:
        return
    n = len(settled)
    correct = 0
    total_error = 0.0
    for p in settled:
        probs = [p['pF1'], p['pFX'], p['pF2']]
        actual_idx = ['1', 'X', '2'].index(p['result'])
        top_pick = np.argmax(probs)
        if top_pick == actual_idx:
            correct += 1
        error = abs(probs[actual_idx] - 1.0)
        total_error += error
        cal.record_outcome(probs[actual_idx], actual_idx, n)
    hit_rate = correct / n if n > 0 else 0
    if n >= 20 and n % 10 == 0:
        detail = f"{n} matches. Hit rate: {hit_rate:.1%}. Avg prob error: {(total_error/n):.2f}. Shrinkage: {cal.current_factor():.2f}"
        db.log_event("learning", detail)
