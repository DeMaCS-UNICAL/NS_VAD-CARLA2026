from __future__ import annotations

import unittest
from pathlib import Path

from agent.vlm import VLMConfig


class VLMConfigTests(unittest.TestCase):
    def test_config_normalizes_paths_and_keeps_defaults(self) -> None:
        config = VLMConfig(video_path="video.mp4", output_dir="outputs")

        self.assertEqual(Path("video.mp4"), config.video_path)
        self.assertEqual(Path("outputs"), config.output_dir)
        self.assertEqual("video", config.vlm_input_mode)
        self.assertFalse(config.save_vlm_input)
        self.assertEqual(1, config.max_workers)
        self.assertEqual(32, config.max_frames_per_clip)
        self.assertEqual(1.0, config.worker_join_timeout_s)

    def test_config_rejects_invalid_input_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported VLM input mode: audio"):
            VLMConfig(video_path="video.mp4", vlm_input_mode="audio")

    def test_config_rejects_invalid_worker_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "VLM workers must be at least 1"):
            VLMConfig(video_path="video.mp4", max_workers=0)

    def test_config_rejects_invalid_max_frames(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_frames_per_clip must be at least 1"):
            VLMConfig(video_path="video.mp4", max_frames_per_clip=0)


if __name__ == "__main__":
    unittest.main()
