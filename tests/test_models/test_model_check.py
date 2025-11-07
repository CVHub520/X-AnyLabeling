import os
import unittest
import tempfile
import multiprocessing


def _check_onnx_model_worker(model_path):
    try:
        import onnx

        onnx.checker.check_model(model_path)
    except Exception as e:
        import sys

        print(f"ONNX model check failed: {e}", file=sys.stderr)
        sys.exit(1)


def safe_check_onnx_model(model_path, timeout=30):
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_check_onnx_model_worker, args=(model_path,))
    p.start()
    p.join(timeout)

    if p.exitcode == 0:
        return True
    elif p.exitcode is None:
        p.terminate()
        p.join(1)
        if p.is_alive():
            p.kill()
            p.join()
        return False
    else:
        return False


def _blocking_worker(path):
    import time

    time.sleep(10)


class TestONNXModelCheck(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def test_check_invalid_file(self):
        invalid_file = os.path.join(self.temp_dir, "invalid.onnx")
        with open(invalid_file, "w") as f:
            f.write("not a valid onnx model")

        result = safe_check_onnx_model(invalid_file, timeout=5)
        self.assertFalse(result)

    def test_check_nonexistent_file(self):
        nonexistent_file = os.path.join(self.temp_dir, "nonexistent.onnx")
        result = safe_check_onnx_model(nonexistent_file, timeout=5)
        self.assertFalse(result)

    def test_timeout_handling(self):
        dummy_file = os.path.join(self.temp_dir, "dummy.onnx")
        with open(dummy_file, "w") as f:
            f.write("dummy")

        ctx = multiprocessing.get_context("spawn")
        p = ctx.Process(target=_blocking_worker, args=(dummy_file,))
        p.start()
        p.join(1)

        if p.exitcode is None:
            p.terminate()
            p.join(1)
            if p.is_alive():
                p.kill()
                p.join()

        self.assertTrue(p.exitcode is None or p.exitcode != 0)

    def tearDown(self):
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


if __name__ == "__main__":
    unittest.main()
