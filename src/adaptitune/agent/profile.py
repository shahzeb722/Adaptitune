"""On-device user profile (proposal 5.5, FR7, NFR3).

The profile is the agent's personal posterior (its judgments and the fitted weights) plus a
history of feedback, stored locally in a single SQLite file. Nothing ever leaves the machine; export writes a JSON file the user
chooses to keep.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from typing import Callable

from .policies import posterior_from_dict

SCHEMA = """
CREATE TABLE IF NOT EXISTS profile (
    user_id TEXT PRIMARY KEY,
    created REAL NOT NULL,
    updated REAL NOT NULL,
    posterior TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    ts REAL NOT NULL,
    take TEXT,
    settings TEXT NOT NULL,
    context TEXT,
    kind TEXT NOT NULL,
    value REAL NOT NULL,
    accuracy REAL,
    natural REAL,
    penalty REAL,
    reward REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS manual (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    ts REAL NOT NULL,
    take TEXT,
    settings TEXT NOT NULL,
    seconds REAL,
    adjustments INTEGER,
    rating REAL
);
"""


def default_path() -> Path:
    import os

    base = Path(os.environ.get("ADAPTITUNE_HOME", Path.home() / ".adaptitune"))
    base.mkdir(parents=True, exist_ok=True)
    return base / "profile.sqlite"


class ProfileStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else default_path()
        self.db = sqlite3.connect(str(self.path))
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self):
        self.db.close()

    # ------------------------------------------------------------------ posterior
    def load(self, user: str, new: Callable[[], object]):
        """The user's stored posterior, or ``new()`` (saved) for a new user."""
        row = self.db.execute("SELECT posterior FROM profile WHERE user_id=?", (user,)).fetchone()
        if row:
            return posterior_from_dict(json.loads(row[0]))
        post = new()
        self.save(user, post)
        return post

    def save(self, user: str, post) -> None:
        now = time.time()
        self.db.execute(
            "INSERT INTO profile(user_id, created, updated, posterior) VALUES(?,?,?,?) "
            "ON CONFLICT(user_id) DO UPDATE SET updated=excluded.updated, posterior=excluded.posterior",
            (user, now, now, json.dumps(post.to_dict())),
        )
        self.db.commit()

    # ------------------------------------------------------------------ history
    def add_feedback(self, user, take, settings: dict, context, kind, value, accuracy, natural, penalty, reward):
        self.db.execute(
            "INSERT INTO history(user_id, ts, take, settings, context, kind, value, accuracy, natural, penalty, reward) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (user, time.time(), take, json.dumps(settings), json.dumps(list(map(float, context))) if context is not None else None,
             kind, float(value), accuracy, natural, penalty, float(reward)),
        )
        self.db.commit()

    def add_manual(self, user, take, settings: dict, seconds=None, adjustments=None, rating=None):
        self.db.execute(
            "INSERT INTO manual(user_id, ts, take, settings, seconds, adjustments, rating) VALUES(?,?,?,?,?,?,?)",
            (user, time.time(), take, json.dumps(settings), seconds, adjustments, rating),
        )
        self.db.commit()

    def last_manual(self, user) -> dict | None:
        row = self.db.execute("SELECT settings FROM manual WHERE user_id=? ORDER BY id DESC LIMIT 1", (user,)).fetchone()
        return json.loads(row[0]) if row else None

    def history(self, user) -> list[dict]:
        cur = self.db.execute("SELECT * FROM history WHERE user_id=? ORDER BY id", (user,))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def users(self) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT user_id FROM profile ORDER BY user_id")]

    # ------------------------------------------------------------------ FR7
    def reset(self, user: str) -> None:
        for t in ("profile", "history", "manual"):
            self.db.execute(f"DELETE FROM {t} WHERE user_id=?", (user,))
        self.db.commit()

    def export(self, user: str, path: str | Path) -> Path:
        row = self.db.execute("SELECT posterior, created, updated FROM profile WHERE user_id=?", (user,)).fetchone()
        manual = self.db.execute("SELECT ts, take, settings, seconds, adjustments, rating FROM manual WHERE user_id=?", (user,)).fetchall()
        data = {
            "user": user,
            "created": row[1] if row else None,
            "updated": row[2] if row else None,
            "posterior": json.loads(row[0]) if row else None,
            "history": self.history(user),
            "manual": [dict(ts=m[0], take=m[1], settings=json.loads(m[2]), seconds=m[3], adjustments=m[4], rating=m[5]) for m in manual],
        }
        p = Path(path)
        p.write_text(json.dumps(data, indent=2))
        return p

    def import_(self, path: str | Path) -> str:
        data = json.loads(Path(path).read_text())
        user = data["user"]
        if data.get("posterior"):
            self.save(user, posterior_from_dict(data["posterior"]))
        return user
