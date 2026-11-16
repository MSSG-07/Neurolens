"""Constant-depth BRISC manifest path resolver tests."""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


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
from neurolens.datasets import brisc as brisc_module
from neurolens.datasets.brisc import (BRISCPathResolver, _manifest_image_index, _read_manifest,
                                    _resolve_linked_image, _resolve_path, _resolve_row_path,
                                    validate_brisc)


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

    def test_relative_path_preferred_over_filename(self):
        expected = self.put("classification_task/train/glioma/actual.jpg")
        self.put("filename.jpg")
        row = {"relative_path": r"classification_task\train\glioma\actual.jpg", "filename": "filename.jpg"}
        self.assertEqual(_resolve_row_path(BRISCPathResolver(self.root), row, "relative_path"), expected)

    def test_filename_fallback_when_relative_path_is_empty(self):
        expected = self.put("filename.jpg")
        row = {"relative_path": "", "filename": "filename.jpg"}
        self.assertEqual(_resolve_row_path(BRISCPathResolver(self.root), row, "relative_path"), expected)

    def test_manifest_reader_selects_relative_path_column(self):
        manifest = self.root / "manifest.csv"
        manifest.write_text("relative_path,filename,is_mask,linked_image,tumor_label\na/b.jpg,b.jpg,0,,glioma\n", encoding="utf-8")
        _, column = _read_manifest(manifest)
        self.assertEqual(column, "relative_path")

    def test_different_canonical_and_duplicate_files_are_ambiguous(self):
        self.put("segmentation_task/train/masks/example.png")
        different = self.put("segmentation_task (1)/train/masks/example.png")
        different.write_bytes(b"different-content")
        with self.assertRaisesRegex(RuntimeError, "Ambiguous BRISC manifest path"):
            _resolve_path(self.root, "segmentation_task/train/masks/example.png")

    def test_parent_traversal_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsafe BRISC manifest path"):
            _resolve_path(self.root, "../outside.png")

    def test_linked_image_basename_resolves_via_full_manifest_relative_path(self):
        image = self.put("segmentation_task/train/images/example.jpg")
        mask = self.put("segmentation_task/train/masks/example.png")
        rows = [
            {"relative_path": r"segmentation_task\train\images\example.jpg", "filename": "example.jpg",
             "is_mask": "0", "split": "train"},
            {"relative_path": r"segmentation_task\train\masks\example.png", "filename": "example.png",
             "is_mask": "1", "linked_image": "example.jpg", "split": "train"},
        ]
        resolver = BRISCPathResolver(self.root)
        paths = {index: _resolve_row_path(resolver, row, "relative_path") for index, row in enumerate(rows)}
        index = _manifest_image_index(rows, paths, "split", self.root)
        linked, resolver_value = _resolve_linked_image("example.jpg", mask, "train", resolver, index, True)
        self.assertEqual(linked, image)
        self.assertEqual(resolver_value, "segmentation_task/train/images/example.jpg")
        self.assertEqual(resolver.last_call["value"], "segmentation_task/train/images/example.jpg")
        self.assertEqual(resolver.last_call["type"], "str")
        self.assertTrue(("train", "segmentation", "example.jpg") in index)

    def test_linked_basename_with_multiple_manifest_matches_is_ambiguous(self):
        mask = self.put("segmentation_task/train/masks/example.png")
        first = self.put("segmentation_task/train/images/glioma/example.jpg")
        second = self.put("segmentation_task/train/images/meningioma/example.jpg")
        resolver = BRISCPathResolver(self.root)
        with self.assertRaisesRegex(RuntimeError, "Ambiguous linked_image basename"):
            _resolve_linked_image("example.jpg", mask, "train", resolver,
                                  {("train", "segmentation", "example.jpg"): {
                                      str(first): first, str(second): second
                                  }})

    def test_windows_linked_image_path_uses_same_resolver(self):
        expected = self.put("segmentation_task (1)/train/images/example.jpg")
        resolver = BRISCPathResolver(self.root)
        actual = _resolve_linked_image(r"segmentation_task\train\images\example.jpg",
                                       self.put("segmentation_task (1)/train/masks/example.png"),
                                       "train", resolver, {})
        self.assertEqual(actual, expected)

    def test_audit_trace_preserves_relative_path_instead_of_filename(self):
        relative = "classification_task/train/glioma/example.jpg"
        self.put(relative)
        manifest = self.root / "manifest.csv"
        manifest.write_text(
            "relative_path,filename,task,split,linked_image,is_mask,tumor_label\n"
            f"{relative},example.jpg,classification,train,,0,glioma\n", encoding="utf-8"
        )
        (self.root / "manifest.json").write_text("{}", encoding="utf-8")

        class FakePixels:
            ndim = 2
            def tobytes(self): return b"pixels"

        class FakeImage:
            mode = "L"
            size = (2, 2)
            def __enter__(self): return self
            def __exit__(self, *args): return False

        resolver_inputs = []
        original_resolve = BRISCPathResolver.resolve
        def recording_resolve(instance, value):
            resolver_inputs.append(value)
            return original_resolve(instance, value)

        with (mock.patch.object(BRISCPathResolver, "resolve", recording_resolve),
              mock.patch.object(brisc_module.Image, "open", return_value=FakeImage(), create=True),
              mock.patch.object(brisc_module.np, "asarray", return_value=FakePixels(), create=True)):
            report = validate_brisc(self.root, manifest, trace_filename="example.jpg")

        self.assertIn(relative, resolver_inputs)
        self.assertNotIn("example.jpg", resolver_inputs)
        self.assertEqual(report["trace_rows"][0]["manifest_fields"]["relative_path"], relative)
        self.assertEqual(report["trace_rows"][0]["resolver_value"], relative)
        self.assertEqual(report["trace_rows"][0]["resolver_value_type"], "str")


if __name__ == "__main__":
    unittest.main()
