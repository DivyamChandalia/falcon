import unittest

from rich.console import Console
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual.strip import Strip
from textual.widgets import Input

from falcon.dashboard import DemoUsageCollector
from falcon.dashboard_ui import FalconDashboard, SelectedJobScroll
from falcon.output_search import highlight_strip, highlight_text, matches


class OutputSearchTests(unittest.TestCase):
    def test_literal_case_insensitive_matches_and_empty_query(self):
        self.assertEqual(matches(["Error ERROR", "a.b axb"], "error"), [(0, 0, 5), (0, 6, 11)])
        self.assertEqual(matches(["a.b axb"], "a.b"), [(0, 0, 3)])
        self.assertEqual(matches(["anything"], ""), [])
        self.assertEqual(matches(["anything"], "missing"), [])

    def test_highlighting_preserves_plain_text_and_unicode_cell_positions(self):
        text = Text("界 error\nERROR", style="green on black")
        highlighted = highlight_text(text, "error", (1, 0, 5))
        self.assertEqual(highlighted.plain, text.plain)
        self.assertEqual(text.spans, [])
        segments = list(highlighted.render(Console(width=40)))
        self.assertTrue(any(segment.text == "ERROR" and segment.style.bgcolor == Style.parse("black on #5de6e6").bgcolor
                            for segment in segments))
        strip = Strip([Segment("界 error", Style.parse("green on black"))])
        result = highlight_strip(strip, strip.text, "error", 0, 0, (0, 2, 7))
        self.assertEqual(result.text, strip.text)
        self.assertEqual(result.cell_length, strip.cell_length)
        error = result.crop(3, 8)
        self.assertTrue(all(segment.style.bgcolor == Style.parse("black on #5de6e6").bgcolor
                            for segment in error))


class DashboardOutputSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_logs_search_navigation_typing_no_matches_and_escape(self):
        app = FalconDashboard(DemoUsageCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause(0.5)
            logs = app.query_one("#selected-logs-scroll", SelectedJobScroll)
            logs.replace_content("test", tuple(["start"] * 40 + ["Error first", "ERROR second"]), follow=False)
            app.set_focus(logs, scroll_visible=False)
            await pilot.press("ctrl+f")
            self.assertIsInstance(app.focused, Input)
            self.assertEqual(app.focused.id, "output-search-input")
            app.focused.value = "error"
            await pilot.pause()
            self.assertEqual(len(app._output_matches), 2)
            self.assertFalse(app.state.logs_auto_follow)
            self.assertEqual(app._output_match_index, 0)
            self.assertGreater(logs.scroll_y, 0)
            row, start, end = logs.search_match
            rendered = logs.render_line(row - int(logs.scroll_y))
            self.assertTrue(all(segment.style.bgcolor == Style.parse("black on #5de6e6").bgcolor
                                for segment in rendered.crop(start, end)))
            await pilot.press("enter")
            self.assertEqual(app._output_match_index, 1)
            await pilot.press("enter")
            self.assertEqual(app._output_match_index, 0)
            await pilot.press("shift+enter")
            self.assertEqual(app._output_match_index, 1)
            self.assertEqual(app.state.search_query, "")
            app.focused.value = "not found"
            await pilot.pause()
            self.assertEqual(app._output_matches, [])
            self.assertIn("No matches", app.query_one("#output-search-count").render().plain)
            await pilot.press("escape")
            self.assertIs(app.focused, logs)
            self.assertFalse(app.query_one("#output-search-bar").display)
            self.assertEqual(logs.search_query, "")

    async def test_ctrl_f_does_not_open_output_search_in_jobs(self):
        app = FalconDashboard(DemoUsageCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.5)
            app.action_focus_jobs()
            await pilot.press("ctrl+f")
            self.assertIsNone(app._output_search_target)


if __name__ == "__main__":
    unittest.main()
