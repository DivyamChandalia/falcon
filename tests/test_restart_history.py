from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from textual import events

from falcon.dashboard import DemoUsageCollector, PodAttempt
from falcon.dashboard_logs import DashboardLogManager
from falcon.dashboard_ui import FalconDashboard, RestartSummaryScroll, SelectedJobScroll
from falcon.restart_history import RestartHistory, history_path
from falcon.theme import BORDER, CYAN


def _entry(pod="old", uid="old-pod"):
    return {"pod": pod, "pod_uid": uid, "container": "main", "phase": "Failed",
            "created_at": "2026-01-01T00:00:00Z", "stdout": "failure trace\n", "stderr": "",
            "returncode": 0, "archived": True}


def test_archive_is_private_partitioned_and_does_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    path = history_path("ctx", "team")
    history = RestartHistory(path)
    assert history.logs("uid") is None
    assert not path.exists()
    ref = {"name": "train", "uid": "uid", "namespace": "team"}
    history.save(ref, "uid", {"issue": "bug", "fix": "patch"}, [_entry()])
    history.save(ref, "uid", None, [])
    assert path.stat().st_mode & 0o777 == 0o600
    assert history.logs("uid")["logs"][0]["stdout"] == "failure trace\n"
    assert history.attempts("uid", "train")[0]["summary"]["fix"] == "patch"
    assert RestartHistory(history_path("other-context", "team")).logs("uid") is None
    assert RestartHistory(history_path("ctx", "other-team")).logs("uid") is None
    assert history.attempts("uid", "unrelated-same-name") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(160, 50), (100, 40)])
async def test_dashboard_archived_pod_summary_and_current_logs(tmp_path, monkeypatch, size):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    row = replace(DemoUsageCollector("mixed").collect()[0], uid="replacement", job="train",
                  restart_history="original", active_pod="current", active_pod_uid="current-pod",
                  attempt_details=(PodAttempt("current", "current-pod", "Pending"),))
    history = RestartHistory(history_path("ctx", "team"))
    history.save({"uid": "original", "name": "train"}, "original",
                 {"issue": "Missing dependency", "fix": "Installed dependency"}, [_entry()])
    history.save({"uid": "second", "name": "train"}, "original", None,
                 [_entry("second-pod", "second-pod-uid")])
    manager = DashboardLogManager("team", client=SimpleNamespace(context="ctx"))

    class Collector:
        def collect(self):
            return [row]

        def close(self):
            pass

    app = FalconDashboard(Collector(), log_manager=manager, refresh_seconds=999)
    async with app.run_test(size=size) as pilot:
        await pilot.pause(0.5)
        app.state.focused_pane = "selected"
        app.state.expanded_pane = "selected"
        app.state.selected_attempt_index = 0
        app._apply_layout()
        app._render_all()
        await pilot.pause()
        attempts = app._selected_attempts(row)
        assert [attempt.name for attempt in attempts] == ["old", "second-pod", "current"]
        # Archive decompression runs off the UI thread. Wait for its result
        # rather than assuming a single Textual layout tick completes I/O.
        for _ in range(100):
            if manager.snapshot(row, attempts[0]).revision:
                break
            await pilot.pause(0.02)
        app._render_log_updates()
        await pilot.pause()
        summary = app.query_one("#selected-restart-summary", RestartSummaryScroll)
        logs = app.query_one("#selected-logs-scroll", SelectedJobScroll)
        assert summary.display
        assert logs.region.bottom <= summary.region.y
        assert 3 <= summary.region.height <= 6
        assert "Missing dependency" in "\n".join(summary._source_lines)
        assert "failure trace" in "\n".join(logs._source_lines)
        assert manager.snapshot(row, attempts[0]).status == "archived"
        assert not summary.query("Button")
        start = summary.content_offset
        await pilot.click(summary, offset=(start.x, start.y))
        assert app.state.selected_section == "summary"
        assert summary.has_class("selected")
        assert not logs.has_class("selected")
        assert summary.styles.border_top[1].hex == CYAN
        assert logs.styles.border_top[1].hex == BORDER
        await pilot.click(logs, offset=(logs.content_offset.x, logs.content_offset.y))
        assert app.state.selected_section == "logs"
        assert logs.has_class("selected")
        assert not summary.has_class("selected")
        assert logs.styles.border_top[1].hex == CYAN
        assert summary.styles.border_top[1].hex == BORDER
        await pilot.press("tab")
        assert app.focused is summary
        assert summary.has_class("selected")
        assert not logs.has_class("selected")
        await pilot.press("shift+tab")
        assert app.focused is logs
        assert logs.has_class("selected")
        assert not summary.has_class("selected")
        await pilot.mouse_down(summary, offset=(start.x, start.y))
        await pilot._post_mouse_events(
            [events.MouseMove], widget=summary, offset=(start.x + 8, start.y), button=1,
        )
        await pilot.mouse_up(summary, offset=(start.x + 8, start.y))
        selected = summary.selection_for_copy()
        assert selected and selected.startswith("Issue:")
        with patch.object(app, "copy_to_clipboard") as copy:
            await pilot.press("ctrl+c")
            copy.assert_called_once_with(selected)
            copy.reset_mock()
            await pilot.press("command+c")
            copy.assert_called_once_with(selected)
        await pilot.click(summary, offset=(start.x, start.y))
        with patch.object(app, "copy_to_clipboard") as copy:
            await pilot.press("ctrl+c")
            copy.assert_not_called()
        assert app.is_running
        long_text = "Issue: " + "very long explanation " * 50 + "\nFix: patch"
        summary.set_summary(long_text)
        await pilot.pause()
        assert summary.region.height == 6
        assert summary.max_scroll_y > 0
        app.set_focus(summary)
        await pilot._post_mouse_events([events.MouseScrollDown], widget=summary, offset=(2, 1))
        await pilot.pause()
        assert summary.scroll_y > 0
        await pilot.press("home")
        await pilot.pause()
        assert summary.scroll_y == 0
        await pilot.press("down")
        assert summary.scroll_y > 0
        summary.scroll_end(animate=False)
        await pilot.pause()
        assert summary.scroll_y > 0
        app._move_selected_window(1)
        await pilot.pause()
        assert not summary.display
        assert app._selected_attempt(row).name == "second-pod"
        for _ in range(100):
            if manager.snapshot(row, attempts[1]).revision:
                break
            await pilot.pause(0.02)
        app._render_log_updates()
        await pilot.pause()
        assert "failure trace" in "\n".join(logs._source_lines)
        app._move_selected_window(1)
        await pilot.pause()
        assert not summary.display
        assert app._selected_attempt(row).name == "current"
        app._move_selected_window(-2)
        await pilot.pause()
        assert summary.display
        app.save_screenshot(path=str(tmp_path), filename=f"restart-history-{size[0]}x{size[1]}.svg")
    manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(160, 40), (100, 40)])
@pytest.mark.parametrize("expanded", [False, True])
async def test_selected_pane_arrows_switch_logs_from_every_focus(tmp_path, monkeypatch, size, expanded):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    row = replace(DemoUsageCollector("mixed").collect()[0], uid="replacement", job="train",
                  restart_history="original", active_pod="current", active_pod_uid="current-pod",
                  attempt_details=(PodAttempt("current", "current-pod", "Pending"),))
    history = RestartHistory(history_path("ctx", "team"))
    history.save({"uid": "original", "name": "train"}, "original",
                 {"issue": "Crash", "fix": "Patched configuration"}, [_entry()])
    manager = DashboardLogManager("team", client=SimpleNamespace(context="ctx"))

    class Collector:
        def collect(self):
            return [row]

        def close(self):
            pass

    app = FalconDashboard(Collector(), log_manager=manager, refresh_seconds=999)
    async with app.run_test(size=size) as pilot:
        await pilot.pause(0.5)
        app.state.focused_pane = "selected"
        app.state.expanded_pane = "selected" if expanded else None
        app.state.selected_attempt_index = 0
        app._apply_layout()
        app._render_all()
        await pilot.pause()
        for selector in ("#selected-pane", "#selected-inspector", "#selected-restart-summary", "#selected-logs-scroll"):
            if selector != "#selected-pane" and not app._selected_inspector_visible():
                continue
            app.state.selected_attempt_index = 0
            app._render_selected()
            app.set_focus(app.query_one(selector), scroll_visible=False)
            await pilot.pause()
            if selector == "#selected-restart-summary":
                assert app.state.selected_section == "summary"
                assert app.query_one(selector).has_class("selected")
                assert not app.query_one("#selected-logs-scroll").has_class("selected")
            await pilot.press("right")
            assert app._selected_attempt(row).name == "current", selector
            assert app.state.cursor_job_uid == row.uid
            # Put keyboard focus back on the outer pane to check both
            # directions without relying on log-widget focus restoration.
            app.set_focus(app.query_one("#selected-pane"), scroll_visible=False)
            await pilot.pause()
            await pilot.press("left")
            assert app._selected_attempt(row).name == "old", selector
    manager.close()
