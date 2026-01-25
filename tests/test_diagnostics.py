import zipfile

import logging

from anylabeling.diagnostics import export_diagnostics_zip, ensure_log_dir
from anylabeling.views.labeling.logger import logger


def test_logger_rotating_file_handler_writes(tmp_path):
    log_file = tmp_path / "test.log"
    logger.setLevel(logging.INFO)
    logger.add_rotating_file_handler(log_file, max_bytes=1024, backup_count=1)
    logger.info("hello-from-test")
    handlers = list(logger.logger.handlers)
    for h in handlers:
        try:
            h.flush()
        except Exception:
            pass
    assert log_file.exists()
    assert "hello-from-test" in log_file.read_text(encoding="utf-8", errors="ignore")
    for h in list(logger.logger.handlers):
        if getattr(h, "baseFilename", None) == str(log_file.resolve()):
            logger.logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass


def test_export_diagnostics_zip_redacts_secrets(tmp_path):
    log_dir = ensure_log_dir(tmp_path / "logs")
    (log_dir / "xanylabeling.log").write_text(
        "Authorization: Bearer token123\nsk-AAAAAAAAAAAAAAAAAAAA\n",
        encoding="utf-8",
    )
    cfg = tmp_path / ".xanylabelingrc"
    cfg.write_text("api_key: sk-BBBBBBBBBBBBBBBBBBBB\n", encoding="utf-8")

    out = export_diagnostics_zip(
        tmp_path / "diagnostics.zip",
        log_dir=log_dir,
        config_file=str(cfg),
        max_log_files=5,
    )

    assert out.exists()
    with zipfile.ZipFile(out, "r") as zf:
        log_text = zf.read("logs/xanylabeling.log").decode("utf-8", errors="ignore")
        assert "token123" not in log_text
        assert "sk-AAAAAAAAAAAAAAAAAAAA" not in log_text
        cfg_text = zf.read("config.json").decode("utf-8", errors="ignore")
        assert "sk-BBBBBBBBBBBBBBBBBBBB" not in cfg_text
