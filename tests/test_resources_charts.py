from __future__ import annotations

import unittest

from falcon.resources_charts import (
    CHART_COLORS,
    GPUHistoryPoint,
    allocation_colors,
    _pie_dimensions,
    _series_colors,
    downsample_history,
    render_allocation_legend,
    render_gpu_history,
    render_namespace_pie,
)
from falcon.theme import PALETTE


class GPUHistoryRendererTests(unittest.TestCase):
    def test_resources_series_use_the_shared_dashboard_palette(self) -> None:
        self.assertIs(CHART_COLORS, PALETTE.pie)
        self.assertEqual(CHART_COLORS, (
            "#56B4E9",
            "#E69F00",
            "#009E73",
            "#CC79A7",
            "#F0E442",
            "#0072B2",
            "#D55E00",
        ))

    def test_allocation_colors_use_the_requested_pie_palette(self) -> None:
        categories = [("alpha", 1), ("beta", 2), ("gamma", 3)]
        colors = allocation_colors(categories)
        self.assertEqual(set(colors.values()), set(PALETTE.pie[:3]))

    @staticmethod
    def points(count: int = 40):
        return [
            GPUHistoryPoint.from_mapping(
                1_700_000_000 + index * 60,
                {
                    "team-2": (index // 3) % 4,
                    "team-10": (index // 7) % 3,
                },
                {
                    "team-2": ((index // 3) % 4) * 80,
                    "team-10": ((index // 7) % 3) * 40,
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
                if width >= 80:
                    self.assertIn("team-10", chart.plain)
                self.assertRegex(chart.plain, r"[━┃]")

        vram = render_gpu_history(
            self.points(), width=80, height=15, basis="vram"
        )
        self.assertIn("G", vram.plain)

    def test_node_series_use_the_expanded_palette_without_collisions(self) -> None:
        names = [f"gpu-node-{index:02d}" for index in range(20)]
        colors = _series_colors(names)
        self.assertEqual(len(set(CHART_COLORS)), len(CHART_COLORS))
        self.assertEqual(set(colors.values()), set(CHART_COLORS))


class NamespacePieRendererTests(unittest.TestCase):
    def test_compact_shared_legend_keeps_all_categories(self) -> None:
        categories = [(f"namespace-{index}", index + 1) for index in range(8)]
        legend = render_allocation_legend(
            categories,
            width=40,
            height=4,
            columns=2,
        )
        self.assertEqual(len(legend.plain.splitlines()), 4)
        for name, _ in categories:
            self.assertIn(name, legend.plain)

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

    def test_legend_percentages_use_the_selected_allocation_total(self) -> None:
        gpu_pie = render_namespace_pie(
            (("team-a", 3), ("team-b", 1)), width=48, height=8
        )
        self.assertIn("team-a", gpu_pie.plain)
        self.assertIn("75%", gpu_pie.plain)
        self.assertIn("team-b", gpu_pie.plain)
        self.assertIn("25%", gpu_pie.plain)

        vram_pie = render_namespace_pie(
            (("team-a", 240), ("team-b", 60)),
            width=48,
            height=8,
            unit="G",
        )
        self.assertIn("240G", vram_pie.plain)
        self.assertIn("80%", vram_pie.plain)
        self.assertIn("60G", vram_pie.plain)
        self.assertIn("20%", vram_pie.plain)

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
                legend_width, pie_width = _pie_dimensions(
                    categories, width=width, height=height
                )
                center_x = round((pie_width - 1) / 2)
                center_y = round((height - 1) / 2)
                self.assertEqual(
                    lines[center_y][legend_width + 2 + center_x], "█"
                )

        reconciled = (*categories[:7], ("System/hidden", 2))
        pie = render_namespace_pie(reconciled, width=44, height=8)
        self.assertIn("System/hidden", pie.plain)

        # The legend is on the left and the pie occupies the right side.
        legend_width, pie_width = _pie_dimensions(
            reconciled, width=44, height=8
        )
        self.assertTrue(all(line.startswith("█ ") for line in pie.plain.splitlines()))
        self.assertGreater(pie_width, 0)
        self.assertGreater(legend_width, 0)


if __name__ == "__main__":
    unittest.main()
