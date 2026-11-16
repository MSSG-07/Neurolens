"""Constant-depth BRISC manifest path resolver tests."""
import sys
import tempfile
import types
import unittest
from pathlib import Path


def _stub_missing_runtime_dependencies():
    """The resolver itself is stdlib-only; keep path tests runnable on lean envs."""
    try:
        import numpy  # noqa: F401
    except ImportError:
        sys.modules["numpy"] = types.ModuleType("numpy")
    try:
        import torch  # noqa: F401
    except Exception:
        torch = types.ModuleType("torch")
        torch_utils = types.ModuleType("torch.utils")
        torch_data = types.ModuleType("torch.utils.data")
        torch_data.Dataset = type("Dataset", (), {})
        torch.utils = torch_utils
        torch_utils.data = torch_data
        sys.modules.update({"torch": torch, "torch.utils": torch_utils, "torch.utils.data": torch_data})
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        pil = types.ModuleType("PIL")
        pil.Image = types.ModuleType("PIL.Image")
        sys.modules.update({"PIL": pil, "PIL.Image": pil.Image})


_stub_missing_runtime_dependencies()
from neurolens.datasets.brisc import _resolve_path


class ResolvePathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def put(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        return path.resolve()

    def test_existing_canonical_path_wins(self):
        expected = self.put("classification_task/classA/nested/image.png")
        self.put("classification_task (1)/classA/nested/image.png")
        self.assertEqual(_resolve_path(self.root, "classification_task/classA/nested/image.png"), expected)

    def test_classification_duplicate_variant_preserves_suffix(self):
        expected = self.put("classification_task (1)/classA/nested/image.png")
        actual = _resolve_path(self.root, "classification_task/classA/nested/image.png")
        self.assertEqual(actual, expected)

    def test_classification_reverse_variant(self):
        expected = self.put("classification_task/classA/nested/image.png")
        actual = _resolve_path(self.root, "classification_task (1)/classA/nested/image.png")
        self.assertEqual(actual, expected)

    def test_segmentation_duplicate_variant_and_windows_separators(self):
        expected = self.put("segmentation_task (1)/masks/classA/image_mask.png")
        actual = _resolve_path(self.root, r"segmentation_task\masks\classA\image_mask.png")
        self.assertEqual(actual, expected)

    def test_colab_reported_mask_and_linked_jpeg_paths(self):
        # Keep the canonical task directory present but incomplete, as in the
        # Colab export; both manifest paths must resolve into the (1) variant.
        (self.root / "segmentation_task").mkdir()
        mask_rel = r"segmentation_task\train\masks\brisc2025_train_04867_pi_sa_t1.png"
        image_rel = r"segmentation_task\train\images\brisc2025_train_04867_pi_sa_t1.jpg"
        expected_mask = self.put("segmentation_task (1)/train/masks/brisc2025_train_04867_pi_sa_t1.png")
        expected_image = self.put("segmentation_task (1)/train/images/brisc2025_train_04867_pi_sa_t1.jpg")

        self.assertEqual(_resolve_path(self.root, mask_rel), expected_mask)
        # linked_image goes through the same resolver as the mask's own path.
        self.assertEqual(_resolve_path(self.root, image_rel), expected_image)

    def test_segmentation_reverse_variant(self):
        expected = self.put("segmentation_task/masks/classA/image_mask.png")
        actual = _resolve_path(self.root, "segmentation_task (1)/masks/classA/image_mask.png")
        self.assertEqual(actual, expected)

    def test_missing_file_lists_original_and_candidates(self):
        original = r"classification_task\classA\missing.png"
        with self.assertRaises(FileNotFoundError) as caught:
            _resolve_path(self.root, original)
        message = str(caught.exception)
        self.assertIn(original, message)
        self.assertIn("classification_task/classA/missing.png", message.replace("\\", "/"))
        self.assertIn("classification_task (1)", message)


if __name__ == "__main__":
    unittest.main()
