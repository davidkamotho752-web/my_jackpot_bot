"""Monitor for SportyBet restrictions and delays."""
import core.db as db

def log_health_warning(kind, detail, severity="Warning"):
    db.insert("account_health", {
        "ts": db.iso(db.now_local()),
        "kind": kind,
        "detail": detail,
        "severity": severity
    })
    db.log_event("health_warning", f"{kind}: {detail}")

def check_health():
    warnings = db.q("SELECT * FROM account_health ORDER BY id DESC LIMIT 3")
    return warnings

def simulate_health_check():
    return check_health()
