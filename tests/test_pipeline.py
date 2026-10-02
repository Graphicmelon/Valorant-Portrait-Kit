"""Integration checks for the published model and portable runtime."""
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
from valorant_portrait_kit.pipeline import Pipeline

BUNDLE = Path(__file__).resolve().parents[1] / 'models' / 'scoreboard'


class PublishedPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = Pipeline(BUNDLE)

    def test_published_artifact_integrity(self):
        manifest = json.loads((BUNDLE / 'manifest.json').read_text())
        for filename, expected in manifest['sha256'].items():
            with self.subTest(filename=filename):
                actual = hashlib.sha256((BUNDLE / filename).read_bytes()).hexdigest()
                self.assertEqual(actual, expected)

    def test_blank_frames_do_not_create_agent_annotations(self):
        for gray in (0, 114, 255):
            with self.subTest(gray=gray):
                image = np.full((720, 1280, 3), gray, dtype=np.uint8)
                self.assertEqual(self.pipeline.predict(image)['detections'], [])

    def test_bank_class_ids_agree_with_output_vocabulary(self):
        self.assertEqual(self.pipeline.names, self.pipeline.matcher.names)
        self.assertEqual(len(self.pipeline.names), 29)


if __name__ == '__main__':
    unittest.main()
