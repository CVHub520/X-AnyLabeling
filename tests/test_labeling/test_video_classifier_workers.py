import base64
import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from anylabeling.views.labeling.video_classifier import exporter, utils
from anylabeling.views.labeling.video_classifier.sidecar import (
    InvalidSidecarError,
)
from anylabeling.views.labeling.widgets import video_classifier_dialog


class TestStreamingVideoBody(unittest.TestCase):
    def test_streamed_body_contains_complete_video_data(self):
        source_data = b"video-payload!"
        with tempfile.NamedTemporaryFile(suffix=".mp4") as source:
            source.write(source_data)
            source.flush()
            worker = video_classifier_dialog.VideoDescriptionWorker(
                source.name, None, "describe"
            )
            data = {
                "video": "__XANYLABELING_VIDEO_DATA__",
                "prompt": "describe",
            }
            with mock.patch.object(
                video_classifier_dialog, "AI_VIDEO_CHUNK_BYTES", 6
            ):
                body = worker._request_body(data, source.name, "video/mp4")
                payload = b"".join(body)

        self.assertEqual(len(body), len(payload))
        request_data = json.loads(payload)
        video_url = request_data["video"]
        self.assertTrue(video_url.startswith("data:video/mp4;base64,"))
        encoded = video_url.split(",", 1)[1]
        self.assertEqual(base64.b64decode(encoded), source_data)

    def test_streaming_stops_when_cancelled(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as source:
            source.write(b"video")
            source.flush()
            body = video_classifier_dialog._StreamingVideoBody(
                source.name,
                b'{"video":"',
                b'"}',
                "video/mp4",
                lambda: True,
            )
            chunks = iter(body)
            next(chunks)
            next(chunks)
            with self.assertRaisesRegex(
                RuntimeError, video_classifier_dialog.AI_CANCELLED
            ):
                next(chunks)

    def test_ai_video_limits_are_checked_before_upload(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as source:
            worker = video_classifier_dialog.VideoDescriptionWorker(
                source.name,
                None,
                "describe",
                duration_ms=(
                    video_classifier_dialog.AI_VIDEO_MAX_DURATION_MS + 1
                ),
            )
            with self.assertRaisesRegex(RuntimeError, "10-minute"):
                worker._video_source()

            worker.duration_ms = 0
            with mock.patch.object(
                video_classifier_dialog.os.path,
                "getsize",
                return_value=video_classifier_dialog.AI_VIDEO_MAX_BYTES + 1,
            ):
                with self.assertRaisesRegex(RuntimeError, "100 MB"):
                    worker._video_source()

    def test_rejected_extracted_segment_is_removed(self):
        temporary = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temporary.close()
        segment = SimpleNamespace(start_ms=0, end_ms=1000)
        worker = video_classifier_dialog.VideoDescriptionWorker(
            "video.mp4", segment, "describe"
        )
        try:
            with (
                mock.patch.object(
                    worker, "_extract_segment", return_value=temporary.name
                ),
                mock.patch.object(
                    video_classifier_dialog.os.path,
                    "getsize",
                    return_value=video_classifier_dialog.AI_VIDEO_MAX_BYTES
                    + 1,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "100 MB"):
                    worker._video_source()
            self.assertFalse(os.path.exists(temporary.name))
        finally:
            if os.path.exists(temporary.name):
                os.remove(temporary.name)

    def test_cancel_closes_active_media_resources(self):
        worker = video_classifier_dialog.VideoDescriptionWorker(
            "video.mp4", None, "describe"
        )
        process = mock.Mock()
        process.poll.return_value = None
        response = mock.Mock()
        session = mock.Mock()
        worker._process = process
        worker._response = response
        worker._session = session

        worker.cancel()

        process.terminate.assert_called_once_with()
        response.close.assert_called_once_with()
        session.close.assert_called_once_with()


class TestVideoMediaWorkers(unittest.TestCase):
    def test_invalid_sidecar_does_not_replace_loaded_video_state(self):
        dialog = mock.Mock()
        dialog._backup_and_rebuild_sidecar.return_value = None
        error = InvalidSidecarError("video.json", "Invalid JSON")
        with mock.patch.object(
            video_classifier_dialog, "load_sidecar", side_effect=error
        ):
            video_classifier_dialog.VideoClassifierDialog._apply_loaded_video(
                dialog, "video.mp4", {}, []
            )

        dialog._backup_and_rebuild_sidecar.assert_called_once_with(
            "video.mp4", error
        )
        dialog.title_label.setText.assert_not_called()

    def test_probe_process_is_terminated_on_cancel(self):
        process = mock.Mock()
        process.poll.return_value = None
        process.communicate.return_value = (b"", None)
        with mock.patch.object(
            utils.subprocess, "Popen", return_value=process
        ):
            result = utils._run_probe_command(
                ["ffprobe"],
                time.monotonic() + 10,
                lambda: True,
            )

        self.assertIsNone(result)
        process.terminate.assert_called_once_with()

    def test_load_worker_runs_probe_and_thumbnails_off_ui_path(self):
        results = []
        worker = video_classifier_dialog.VideoLoadWorker("video.mp4")
        worker.resultReady.connect(lambda *args: results.append(args))
        with (
            mock.patch.object(
                video_classifier_dialog,
                "probe_video",
                return_value={"fps": 25.0},
            ) as probe_mock,
            mock.patch.object(
                video_classifier_dialog,
                "extract_video_thumbnails",
                return_value=["thumbnail"],
            ) as thumbnail_mock,
        ):
            worker.run()

        self.assertEqual(
            results, [("video.mp4", {"fps": 25.0}, ["thumbnail"], False)]
        )
        self.assertTrue(callable(probe_mock.call_args.kwargs["is_cancelled"]))
        self.assertTrue(
            callable(thumbnail_mock.call_args.kwargs["is_cancelled"])
        )

    def test_segment_export_falls_back_to_background_reencode(self):
        copy_process = mock.Mock(returncode=1)
        copy_process.communicate.return_value = (b"", b"copy failed")
        encode_process = mock.Mock(returncode=0)
        encode_process.communicate.return_value = (b"", b"")
        results = []
        with tempfile.TemporaryDirectory() as directory:
            worker = exporter.SegmentExporterWorker(
                "ffmpeg",
                "video.mp4",
                1000,
                2000,
                os.path.join(directory, "segment.mp4"),
            )
            worker.finished.connect(lambda *args: results.append(args))
            with mock.patch.object(
                exporter.subprocess,
                "Popen",
                side_effect=[copy_process, encode_process],
            ) as popen_mock:
                worker.run()

        self.assertEqual(results, [(True, "")])
        self.assertEqual(popen_mock.call_count, 2)
        self.assertIn("copy", popen_mock.call_args_list[0].args[0])
        self.assertIn("libx264", popen_mock.call_args_list[1].args[0])


if __name__ == "__main__":
    unittest.main()
