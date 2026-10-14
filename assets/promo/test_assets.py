from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import generate  # noqa: E402


class PromoAssetTests(unittest.TestCase):
    def test_owner_content_and_brain_map_are_consistent(self) -> None:
        content = json.loads((HERE / "content.json").read_text(encoding="utf-8"))
        brain = json.loads((HERE / "brain.json").read_text(encoding="utf-8"))
        self.assertEqual(content["identity"]["name"], "Aria Raoufi")
        self.assertEqual(content["links"]["email"], "raoufiari@edu.bme.hu")
        self.assertEqual(len(content["skills"]), 9)
        self.assertEqual(len(brain["nodes"]), 40)
        self.assertEqual(brain["trained_nodes"], 19)
        self.assertEqual(
            {skill for region in content["brain_regions"] for skill in region["skills"]},
            set(content["skills"]),
        )

    def test_trace_meets_visual_contract(self) -> None:
        trace = json.loads((HERE / "trace.json").read_text(encoding="utf-8"))
        self.assertEqual(trace["architecture"], [2, 8, 8, 1])
        self.assertGreaterEqual(trace["final_accuracy"], 0.95)
        width = trace["dataset"]["grid"]["width"]
        height = trace["dataset"]["grid"]["height"]
        self.assertTrue(all(len(epoch["grid"]) == width * height for epoch in trace["epochs"]))

    def test_accessibility_gates(self) -> None:
        generate.check_contrast()
        generate.check_glyphs()

    def test_readme_has_only_local_hero_and_badges(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertEqual(readme.count("<p align=\"center\">"), 2)
        image_sources = re.findall(r'<img\b[^>]*\bsrc="([^"]+)"', readme)
        self.assertEqual(
            image_sources,
            [
                "assets/promo/lordjohan-terminal.gif",
                "assets/badges/github.svg",
                "assets/badges/email.svg",
            ],
        )
        self.assertIn('srcset="assets/promo/lordjohan-terminal-poster.png"', readme)
        self.assertIn('href="mailto:raoufiari@edu.bme.hu"', readme)
        self.assertNotIn("https://", " ".join(image_sources))

    def test_generated_assets_meet_size_and_dimension_budgets(self) -> None:
        expected_size = (generate.WIDTH * generate.SCALE, generate.HEIGHT * generate.SCALE)
        gif = HERE / "lordjohan-terminal.gif"
        webp = HERE / "lordjohan-terminal.webp"
        poster = HERE / "lordjohan-terminal-poster.png"
        for path in (gif, webp, poster):
            self.assertTrue(path.exists(), f"Missing generated asset: {path.name}")
            with Image.open(path) as image:
                self.assertEqual(image.size, expected_size)
        self.assertLessEqual(gif.stat().st_size, 5_000_000)
        self.assertLessEqual(webp.stat().st_size, 3_000_000)
        durations = {}
        for path in (gif, webp):
            with Image.open(path) as image:
                durations[path.suffix] = []
                for index in range(image.n_frames):
                    image.seek(index)
                    image.load()
                    durations[path.suffix].append(image.info.get("duration", 0))
        self.assertEqual(sum(durations[".gif"]), sum(durations[".webp"]))
        self.assertEqual(sum(durations[".gif"]), 30_960)
        self.assertLessEqual(sum(durations[".gif"]), 36_000)
        for badge in (ROOT / "assets" / "badges" / "github.svg", ROOT / "assets" / "badges" / "email.svg"):
            self.assertLess(badge.stat().st_size, 2048)


if __name__ == "__main__":
    unittest.main()
