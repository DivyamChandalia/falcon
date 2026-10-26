"""Private local archives for same-name Job restarts."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import zlib
from contextlib import closing
from pathlib import Path

HISTORY_ANNOTATION = "falcon.dev/restart-history"


def history_path(context, namespace, state_dir=None):
    if state_dir is None:
        root = Path(os.environ.get("XDG_STATE_HOME", "~/.local/state")).expanduser()
        key = hashlib.sha256(f"{context or 'default'}\0{namespace}".encode()).hexdigest()[:24]
        state_dir = root / "falcon" / "mcp" / key
    return Path(state_dir) / "restart-history.sqlite3"


class RestartHistory:
    """UID-keyed snapshots, committed before deleting Kubernetes objects.

    Reads do not create files. Logs are compressed on disk; metadata queries
    never decompress them, keeping dashboard inventory updates inexpensive.
    """

    def __init__(self, path):
        self.path = Path(path).absolute()

    def save(self, ref, root_uid, summary, entries):
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.path.is_symlink() or self.path.parent.is_symlink():
            raise OSError("refusing symlinked restart archive")
        with closing(sqlite3.connect(self.path, timeout=5)) as db, db:
            self.path.chmod(0o600)
            db.execute("PRAGMA synchronous=FULL")
            db.execute("CREATE TABLE IF NOT EXISTS jobs "
                       "(uid TEXT PRIMARY KEY, root TEXT, name TEXT, summary TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS logs "
                       "(job_uid TEXT, pod_uid TEXT, container TEXT, metadata TEXT, payload BLOB, "
                       "PRIMARY KEY(job_uid, pod_uid, container))")
            # First capture wins: retries must not replace archived output
            # with an empty snapshot after Pods have been deleted.
            if db.execute("SELECT 1 FROM jobs WHERE uid=?", (ref["uid"],)).fetchone():
                return
            db.execute("INSERT INTO jobs VALUES (?, ?, ?, ?)",
                       (ref["uid"], root_uid, ref["name"], json.dumps(summary)))
            for entry in entries:
                metadata = {k: v for k, v in entry.items() if k not in {"stdout", "stderr"}}
                db.execute("INSERT INTO logs VALUES (?, ?, ?, ?, ?)", (
                    ref["uid"], entry["pod_uid"], entry["container"], json.dumps(metadata),
                    zlib.compress(json.dumps(entry).encode()),
                ))

    def _read(self):
        if not self.path.is_file() or self.path.is_symlink():
            return None
        return sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=0.1)

    def contains(self, uid):
        db = self._read()
        if db is None:
            return False
        try:
            return db.execute("SELECT 1 FROM jobs WHERE uid=?", (uid,)).fetchone() is not None
        finally:
            db.close()

    def attempts(self, root_uid, name):
        db = self._read()
        if db is None:
            return []
        try:
            return [dict(json.loads(metadata), source_uid=uid, summary=json.loads(summary))
                    for uid, summary, metadata in db.execute(
                        "SELECT jobs.uid, jobs.summary, logs.metadata FROM jobs JOIN logs "
                        "ON logs.job_uid=jobs.uid WHERE jobs.root=? AND jobs.name=? "
                        "ORDER BY jobs.rowid, logs.rowid", (root_uid, name))]
        finally:
            db.close()

    def logs(self, uid):
        db = self._read()
        if db is None:
            return None
        try:
            job = db.execute("SELECT summary, name FROM jobs WHERE uid=?", (uid,)).fetchone()
            if job is None:
                return None
            entries = [json.loads(zlib.decompress(row[0])) for row in db.execute(
                "SELECT payload FROM logs WHERE job_uid=? ORDER BY rowid", (uid,))]
            return {"summary": json.loads(job[0]), "name": job[1], "logs": entries}
        finally:
            db.close()
