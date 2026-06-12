import importlib.util
import unittest
from pathlib import Path
from unittest import mock
import subprocess


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "anylabeling"
    / "views"
    / "common"
    / "device_manager.py"
)
SPEC = importlib.util.spec_from_file_location("device_manager_module", MODULE_PATH)
DEVICE_MANAGER_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DEVICE_MANAGER_MODULE)


class TestDeviceManager(unittest.TestCase):
    def setUp(self):
        DEVICE_MANAGER_MODULE.DeviceManager._instance = None

    def test_auto_detect_prefers_gpu_when_cuda_provider_exists(self):
        manager = DEVICE_MANAGER_MODULE.DeviceManager()
        with mock.patch.object(
            DEVICE_MANAGER_MODULE,
            "_probe_onnx_providers",
            return_value=["CPUExecutionProvider", "CUDAExecutionProvider"],
        ):
            self.assertEqual(manager.get_device(), "GPU")
            self.assertEqual(manager.get_available_devices(), ["CPU", "GPU"])

    def test_auto_detect_falls_back_to_cpu_when_probe_fails(self):
        manager = DEVICE_MANAGER_MODULE.DeviceManager()
        with mock.patch.object(
            DEVICE_MANAGER_MODULE,
            "_probe_onnx_providers",
            return_value=[],
        ):
            self.assertEqual(manager.get_device(), "CPU")
            self.assertEqual(manager.get_available_devices(), ["CPU"])

    def test_available_providers_are_cached_after_first_probe(self):
        manager = DEVICE_MANAGER_MODULE.DeviceManager()
        with mock.patch.object(
            DEVICE_MANAGER_MODULE,
            "_probe_onnx_providers",
            return_value=["CUDAExecutionProvider"],
        ) as probe:
            self.assertTrue(manager._is_gpu_available())
            self.assertTrue(manager._is_gpu_available())
        self.assertEqual(probe.call_count, 1)

    def test_probe_returns_payload_from_successful_worker(self):
        completed = subprocess.CompletedProcess(
            args=["probe-device"],
            returncode=0,
            stdout='{"providers": ["CUDAExecutionProvider"]}',
            stderr="",
        )
        with mock.patch.object(
            DEVICE_MANAGER_MODULE.subprocess,
            "run",
            return_value=completed,
        ):
            providers = DEVICE_MANAGER_MODULE._probe_onnx_providers(timeout=1)
        self.assertEqual(providers, ["CUDAExecutionProvider"])

    def test_probe_returns_empty_list_when_worker_times_out(self):
        with mock.patch.object(
            DEVICE_MANAGER_MODULE.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd=["probe-device"], timeout=1),
        ):
            providers = DEVICE_MANAGER_MODULE._probe_onnx_providers(timeout=1)
        self.assertEqual(providers, [])

    def test_probe_returns_empty_list_when_worker_crashes(self):
        completed = subprocess.CompletedProcess(
            args=["probe-device"],
            returncode=1,
            stdout="",
            stderr="boom",
        )
        with mock.patch.object(
            DEVICE_MANAGER_MODULE.subprocess,
            "run",
            return_value=completed,
        ):
            providers = DEVICE_MANAGER_MODULE._probe_onnx_providers(timeout=1)
        self.assertEqual(providers, [])

    def test_probe_returns_empty_list_when_worker_returns_invalid_json(self):
        completed = subprocess.CompletedProcess(
            args=["probe-device"],
            returncode=0,
            stdout="not-json",
            stderr="",
        )
        with mock.patch.object(
            DEVICE_MANAGER_MODULE.subprocess,
            "run",
            return_value=completed,
        ):
            providers = DEVICE_MANAGER_MODULE._probe_onnx_providers(timeout=1)
        self.assertEqual(providers, [])


if __name__ == "__main__":
    unittest.main()
