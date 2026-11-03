from unittest.mock import patch

import pytest

from falcon.agent_presence import PRESENCE_TTL_SECONDS, WaitPresence, connected_jobs


@pytest.fixture(autouse=True)
def state_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))


def test_waiters_are_independent_and_context_scoped():
    first = WaitPresence("cluster", "team", ["uid"])
    second = WaitPresence("cluster", "team", ["uid", "other"])
    assert connected_jobs("cluster", "team") == set()
    first.renew()
    second.renew()
    first.close()
    assert connected_jobs("cluster", "team") == {"uid", "other"}
    assert connected_jobs("different", "team") == set()
    assert connected_jobs("cluster", "different") == set()
    second.close()
    assert connected_jobs("cluster", "team") == set()


def test_crashed_waiter_expires_and_live_waiter_renews():
    lease = WaitPresence("cluster", "team", ["uid"])
    with patch("falcon.agent_presence.time.time", return_value=100):
        lease.renew()
    with patch("falcon.agent_presence.time.time", return_value=100 + PRESENCE_TTL_SECONDS):
        assert connected_jobs("cluster", "team") == set()
        lease.renew()
        assert connected_jobs("cluster", "team") == {"uid"}
    lease.close()


def test_unavailable_registry_does_not_break_waits():
    lease = WaitPresence(None, "team", ["uid"])
    with patch("falcon.agent_presence.sqlite3.connect", side_effect=OSError("read only")):
        lease.renew()
        lease.close()
        assert connected_jobs(None, "team") == set()
