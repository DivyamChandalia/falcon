from __future__ import annotations

import unittest

from rich.console import Console

from falcon.resources_charts import (
    CHART_COLORS,
    GPUHistoryPoint,
    _series_colors,
    downsample_history,
    render_gpu_history,
    render_namespace_pie,
)


class GPUHistoryRendererTests(unittest.TestCase):
    @staticmethod
    def points(count: int = 40):
        return [
            GPUHistoryPoint.from_mapping(
                1_700_000_000 + index * 60,
                {
                    "team-2": (index // 3) % 4,
                    "team-10": (index // 7) % 3,
                },
            )
            for index in range(count)
        ]

    def test_downsampling_preserves_endpoints_and_width_bound(self) -> None:
        points = self.points()
        sampled = downsample_history(points, 9)
        self.assertLessEqual(len(sampled), 9)
        self.assertIs(sampled[0], points[0])
        self.assertIs(sampled[-1], points[-1])
        self.assertEqual(downsample_history(points, 1), [points[-1]])
        self.assertEqual(downsample_history(points, 2), [points[0], points[-1]])

    def test_empty_and_single_sample_show_collecting_state(self) -> None:
        self.assertIn(
            "Collecting history",
            render_gpu_history([], width=30, height=8).plain,
        )
        self.assertIn(
            "1 sample since launch",
            render_gpu_history(self.points(1), width=50, height=8).plain,
        )

    def test_narrow_and_wide_step_charts_fit_and_label_series(self) -> None:
        for width, height in ((24, 6), (80, 15), (160, 24)):
            with self.subTest(size=(width, height)):
                chart = render_gpu_history(self.points(), width=width, height=height)
                lines = chart.plain.splitlines()
                self.assertLessEqual(len(lines), height)
                self.assertTrue(all(len(line) <= width for line in lines))
                self.assertIn("Total", chart.plain)
                self.assertIn("team-2", chart.plain)
                self.assertRegex(chart.plain, r"[━┃]")

    def test_node_series_use_the_expanded_palette_without_collisions(self) -> None:
        names = [f"gpu-node-{index:02d}" for index in range(20)]
        colors = _series_colors(names)
        self.assertGreaterEqual(len(CHART_COLORS), 20)
        self.assertEqual(len(set(colors.values())), len(names))


class NamespacePieRendererTests(unittest.TestCase):
    def test_empty_and_tiny_states_are_explicit(self) -> None:
        self.assertIn(
            "No GPU allocation",
            render_namespace_pie([], width=30, height=8).plain,
        )
        self.assertEqual(
            render_namespace_pie(
                [("team", 2)], width=12, height=4, unit="G"
            ).plain,
            "2G",
        )

    def test_many_long_categories_fit_narrow_and_wide_panels(self) -> None:
        categories = tuple(
            (f"extraordinarily-long-namespace-{index}", index + 1)
            for index in range(8)
        )
        for width, height in ((28, 7), (52, 12), (80, 18)):
            with self.subTest(size=(width, height)):
                pie = render_namespace_pie(categories, width=width, height=height)
                lines = pie.plain.splitlines()
                self.assertEqual(len(lines), height)
                self.assertTrue(all(len(line) <= width for line in lines))
                self.assertIn("█", pie.plain)

                # The center is a colored block, not a hollow donut cell.
                pie_width = min(max(9, width // 2), max(9, 2 * height - 1), 25)
                center_x = round((pie_width - 1) / 2)
                center_y = round((height - 1) / 2)
                self.assertEqual(lines[center_y][center_x], "█")

        reconciled = (*categories[:7], ("System/hidden", 2))
        pie = render_namespace_pie(reconciled, width=44, height=8)
        self.assertIn("System/hidden", pie.plain)

        # At 44×8 the ring occupies 15 cells, followed by two spaces and the
        # legend marker. Every category must receive a distinct marker color.
        marker_offsets = [row * 45 + 17 for row in range(8)]
        console = Console(width=44, color_system="truecolor")
        marker_colors = {
            pie.get_style_at_offset(console, offset).color.triplet
            for offset in marker_offsets
        }
        self.assertEqual(len(marker_colors), 8)


if __name__ == "__main__":
    unittest.main()
