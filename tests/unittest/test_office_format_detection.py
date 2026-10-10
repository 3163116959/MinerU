import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mineru.utils import guess_suffix_or_lang as detection


def ole_container(stream_name):
    """Build a minimal CFB directory without including private document content."""
    header = bytearray(512)
    header[:8] = detection.OLE_SIG_BYTES
    struct.pack_into("<HHHH", header, 24, 0x3E, 3, 0xFFFE, 9)
    struct.pack_into("<H", header, 32, 6)
    struct.pack_into(
        "<9I", header, 40, 0, 1, 0, 0, 4096, 0xFFFFFFFE, 0, 0xFFFFFFFE, 0
    )
    struct.pack_into("<109I", header, 76, 1, *([0xFFFFFFFF] * 108))

    directory = bytearray(512)
    for index, (name, kind) in enumerate([("Root Entry", 5), (stream_name, 2)]):
        offset = index * 128
        encoded = (name + "\0").encode("utf-16le")
        directory[offset:offset + len(encoded)] = encoded
        struct.pack_into(
            "<HBBIII", directory, offset + 64, len(encoded), kind, 1,
            0xFFFFFFFF, 0xFFFFFFFF, 1 if index == 0 else 0xFFFFFFFF,
        )
        struct.pack_into("<I", directory, offset + 116, 0xFFFFFFFE)

    fat = struct.pack("<128I", 0xFFFFFFFE, 0xFFFFFFFD, *([0xFFFFFFFF] * 126))
    return bytes(header + directory + fat)


class OfficeFormatDetectionTest(unittest.TestCase):
    def test_ole_structure_overrides_magika_and_filename(self):
        for stream_name, expected in [("WordDocument", "doc"), ("PowerPoint Document", "ppt")]:
            with self.subTest(format=expected), tempfile.TemporaryDirectory() as directory:
                data = ole_container(stream_name)
                path = Path(directory) / "misleading.xlsx"
                path.write_bytes(data)
                with patch.object(detection, "magika") as magika:
                    magika.identify_bytes.return_value.prediction.output.label = "ppt"
                    magika.identify_path.return_value.prediction.output.label = "ppt"
                    self.assertEqual(detection.guess_suffix_by_bytes(data), expected)
                    self.assertEqual(detection.guess_suffix_by_path(path), expected)
                    magika.identify_bytes.assert_not_called()
                    magika.identify_path.assert_not_called()

    def test_unknown_ole_and_non_ole_keep_magika_detection(self):
        for data in [ole_container("Unrelated"), b"ordinary text"]:
            with self.subTest(data=data[:8]), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "input.bin"
                path.write_bytes(data)
                with patch.object(detection, "magika") as magika:
                    magika.identify_bytes.return_value.prediction.output.label = "unknown"
                    magika.identify_path.return_value.prediction.output.label = "unknown"
                    self.assertEqual(detection.guess_suffix_by_bytes(data), "unknown")
                    self.assertEqual(detection.guess_suffix_by_path(path), "unknown")
                    magika.identify_bytes.assert_called_once()
                    magika.identify_path.assert_called_once()


if __name__ == "__main__":
    unittest.main()
