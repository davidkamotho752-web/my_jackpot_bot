"""Team name resolution. Applies user-defined aliases, then fuzzy matching against the known team list."""
import difflib
import core.db as db

def normalize(name):
    if not name:
        return ""
    s = str(name).lower().strip()
    for ch in [".", "'", "-", "&"]:
        s = s.replace(ch, " ")
    return " ".join(s.split())

def load_aliases():
    rows = db.q("SELECT typed, official FROM team_aliases")
    return {r["typed"]: r["official"] for r in rows}

def add_alias(typed, official):
    db.x("INSERT INTO team_aliases(typed, official, created) VALUES(?, ?, ?) ON CONFLICT(typed) DO UPDATE SET official=excluded.official", (typed, official, db.iso(db.now_local())))

def load_custom():
    return {r["name"] for r in db.q("SELECT name FROM custom_teams")}

def add_custom(name):
    db.x("INSERT OR IGNORE INTO custom_teams(name, created) VALUES(?, ?)", (name, db.iso(db.now_local())))

def resolve(name, known, alias_map=None):
    if not name:
        return ""
    n = str(name).strip()
    if n in known:
        return n
    alias_map = alias_map if alias_map is not None else load_aliases()
    if n in alias_map:
        return alias_map[n]
    norm = normalize(n)
    for k in known:
        if normalize(k) == norm:
            return k
    candidates = difflib.get_close_matches(n, list(known), n=1, cutoff=0.85)
    if candidates:
        return candidates[0]
    return n

def suggestions(name, known, n=5):
    if not name:
        return []
    norm = normalize(name)
    scored = []
    for k in known:
        ratio = difflib.SequenceMatcher(None, norm, normalize(k)).ratio()
        if ratio >= 0.6:
            scored.append((ratio, k))
    scored.sort(reverse=True)
    return [k for _, k in scored[:n]]
