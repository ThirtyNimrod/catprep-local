import logging

from core.logger import get_logger


def test_get_logger_configures_file_and_console_handlers(tmp_path):
    logger = get_logger("test_logger_configures", log_file=tmp_path / "a.log")
    assert logger.name == "test_logger_configures"
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 2
    assert logger.propagate is False


def test_get_logger_returns_same_instance_without_duplicate_handlers(tmp_path):
    name = "test_logger_same_instance"
    logger1 = get_logger(name, log_file=tmp_path / "b.log")
    handler_count = len(logger1.handlers)

    logger2 = get_logger(name, log_file=tmp_path / "b.log")

    assert logger1 is logger2
    assert len(logger2.handlers) == handler_count


def test_get_logger_writes_to_custom_log_file(tmp_path):
    log_file = tmp_path / "c.log"
    logger = get_logger("test_logger_writes_custom_file", log_file=log_file, mode="w")
    logger.info("hello from test")
    for handler in logger.handlers:
        handler.flush()

    assert log_file.exists()
    assert "hello from test" in log_file.read_text(encoding="utf-8")
