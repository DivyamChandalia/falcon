"""Local, expiring evidence of active MCP waits (not model availability)."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from pathlib import Path
from uuid import uuid4

HEARTBEAT_SECONDS = 10
PRESENCE_TTL_SECONDS = 30


def _path(context, namespace):
    root = Path(os.environ.get("XDG_STATE_HOME", "~/.local/state")).expanduser()
    key = hashlib.sha256(f"{context or 'default'}\0{namespace}".encode()).hexdigest()[:24]
    return root / "falcon" / "agent-presence" / key / "waits.sqlite3"


class WaitPresence:
    """Each waiter owns a lease, so concurrent waits cannot clear each other."""

    def __init__(self, context, namespace, uids):
        self.path = _path(context, namespace)
        self.token = uuid4().hex
        self.uids = set(filter(None, uids))

    def renew(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if self.path.parent.is_symlink() or self.path.is_symlink():
                return
            self.path.parent.chmod(0o700)
            with sqlite3.connect(self.path, timeout=0.1) as db:
                self.path.chmod(0o600)
                db.execute(
                    "CREATE TABLE IF NOT EXISTS waits "
                    "(token TEXT, uid TEXT, expires REAL, PRIMARY KEY (token, uid))"
                )
                now = time.time()
                db.execute("DELETE FROM waits WHERE expires <= ?", (now,))
                db.execute("DELETE FROM waits WHERE token = ?", (self.token,))
                db.executemany(
                    "INSERT OR REPLACE INTO waits VALUES (?, ?, ?)",
                    [(self.token, uid, now + PRESENCE_TTL_SECONDS) for uid in self.uids],
                )
        except (OSError, sqlite3.Error):
            # UI decoration must never break monitoring on read-only homes.
            pass

    def close(self):
        try:
            if not self.path.is_file() or self.path.is_symlink():
                return
            with sqlite3.connect(self.path, timeout=0.1) as db:
                db.execute("DELETE FROM waits WHERE token = ?", (self.token,))
        except (OSError, sqlite3.Error):
            pass


def connected_jobs(context, namespace):
    """Read without creating state; absent/expired evidence means disconnected."""
    path = _path(context, namespace)
    try:
        if not path.is_file() or path.is_symlink():
            return set()
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=0.1) as db:
            return {row[0] for row in db.execute(
                "SELECT DISTINCT uid FROM waits WHERE expires > ?", (time.time(),)
            )}
    except (OSError, sqlite3.Error):
        return set()
