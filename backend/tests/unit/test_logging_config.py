"""Unit tests for structured logging configuration."""
import json
import logging
import os
import tempfile
from io import StringIO
from unittest.mock import patch

import pytest

from app.core.logging_config import (
    JSONFormatter,
    TextFormatter,
    get_log_level,
    get_log_format,
    get_log_output,
    get_log_file_path,
    get_log_file_max_bytes,
    get_log_file_backup_count,
    get_module_log_levels,
    setup_logging,
    get_logger,
    log_performance,
    log_error,
)


class TestJSONFormatter:
    """Test JSON formatter."""
    
    def test_basic_format(self):
        """Test basic JSON formatting."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        output = formatter.format(record)
        parsed = json.loads(output)
        
        assert parsed["level"] == "INFO"
        assert parsed["module"] == "test.module"
        assert parsed["message"] == "Test message"
        assert "timestamp" in parsed
    
    def test_with_extra_fields(self):
        """Test JSON formatting with extra fields."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Add custom fields
        record.service = "TestService"
        record.duration_ms = 123.45
        record.stage = "test_stage"
        record.conversation_id = "conv-123"
        record.task_id = "task-456"
        
        output = formatter.format(record)
        parsed = json.loads(output)
        
        assert parsed["service"] == "TestService"
        assert parsed["duration_ms"] == 123.45
        assert parsed["stage"] == "test_stage"
        assert parsed["conversation_id"] == "conv-123"
        assert parsed["task_id"] == "task-456"
    
    def test_with_error_fields(self):
        """Test JSON formatting with error fields."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.module",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Error occurred",
            args=(),
            exc_info=None
        )
        
        record.error_type = "ValueError"
        record.error_message = "Invalid value"
        record.stack_trace = "Traceback..."
        
        output = formatter.format(record)
        parsed = json.loads(output)
        
        assert parsed["error_type"] == "ValueError"
        assert parsed["error_message"] == "Invalid value"
        assert parsed["stack_trace"] == "Traceback..."


class TestTextFormatter:
    """Test text formatter."""
    
    def test_basic_format(self):
        """Test basic text formatting."""
        formatter = TextFormatter()
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        output = formatter.format(record)
        
        assert "[INFO]" in output
        assert "[test.module]" in output
        assert "Test message" in output
    
    def test_with_context(self):
        """Test text formatting with context."""
        formatter = TextFormatter()
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        record.service = "TestService"
        record.stage = "test_stage"
        record.conversation_id = "conv-123"
        record.task_id = "task-456"
        record.duration_ms = 123.45
        
        output = formatter.format(record)
        
        assert "[TestService]" in output
        assert "[test_stage]" in output
        assert "conv:conv-123" in output
        assert "task:task-456" in output
        assert "(123.45ms)" in output


class TestEnvironmentFunctions:
    """Test environment variable functions."""
    
    def test_get_log_level_default(self):
        """Test get_log_level with default value."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_log_level() == "INFO"
    
    def test_get_log_level_from_env(self):
        """Test get_log_level from environment variable."""
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            assert get_log_level() == "DEBUG"
    
    def test_get_log_format_default(self):
        """Test get_log_format with default value."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_log_format() == "json"
    
    def test_get_log_format_from_env(self):
        """Test get_log_format from environment variable."""
        with patch.dict(os.environ, {"LOG_FORMAT": "text"}):
            assert get_log_format() == "text"
    
    def test_get_log_output_default(self):
        """Test get_log_output with default value."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_log_output() == "console"
    
    def test_get_log_output_from_env(self):
        """Test get_log_output from environment variable."""
        with patch.dict(os.environ, {"LOG_OUTPUT": "file"}):
            assert get_log_output() == "file"
    
    def test_get_log_file_path_default(self):
        """Test get_log_file_path with default value."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_log_file_path() == "/var/log/orangerag/app.log"
    
    def test_get_log_file_max_bytes_default(self):
        """Test get_log_file_max_bytes with default value."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_log_file_max_bytes() == 104857600  # 100MB
    
    def test_get_log_file_backup_count_default(self):
        """Test get_log_file_backup_count with default value."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_log_file_backup_count() == 10
    
    def test_get_module_log_levels(self):
        """Test get_module_log_levels."""
        env_vars = {
            "LOG_LEVEL_MODULE_OLLAMA": "WARNING",
            "LOG_LEVEL_MODULE_RETRIEVERS": "DEBUG",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            levels = get_module_log_levels()
            assert levels["ollama"] == "WARNING"
            assert levels["retrievers"] == "DEBUG"


class TestSetupLogging:
    """Test setup_logging function."""
    
    def test_setup_logging_json_console(self):
        """Test setup logging with JSON format to console."""
        with patch.dict(os.environ, {}, clear=True):
            setup_logging(level="INFO", log_format="json", output="console")
            
            logger = logging.getLogger("test")
            assert logger.level == logging.INFO
            assert len(logger.handlers) > 0 or len(logging.getLogger().handlers) > 0
    
    def test_setup_logging_text_console(self):
        """Test setup logging with text format to console."""
        with patch.dict(os.environ, {}, clear=True):
            setup_logging(level="DEBUG", log_format="text", output="console")
            
            logger = logging.getLogger("test")
            assert logger.level == logging.DEBUG
    
    def test_get_logger(self):
        """Test get_logger convenience function."""
        logger = get_logger("test.module")
        assert logger.name == "test.module"
        assert isinstance(logger, logging.Logger)


class TestLogHelpers:
    """Test log helper functions."""
    
    def test_log_performance(self, caplog):
        """Test log_performance helper."""
        logger = logging.getLogger("test")
        logger.setLevel(logging.INFO)
        
        with caplog.at_level(logging.INFO):
            log_performance(
                logger,
                service="TestService",
                stage="test_stage",
                duration_ms=123.45,
                conversation_id="conv-123",
                task_id="task-456",
                extra={"custom": "value"}
            )
        
        assert "TestService test_stage completed in 123.45ms" in caplog.text
    
    def test_log_error(self, caplog):
        """Test log_error helper."""
        logger = logging.getLogger("test")
        logger.setLevel(logging.ERROR)
        
        with caplog.at_level(logging.ERROR):
            try:
                raise ValueError("Test error")
            except Exception as e:
                log_error(
                    logger,
                    service="TestService",
                    error=e,
                    conversation_id="conv-123",
                    task_id="task-456"
                )
        
        assert "TestService error: Test error" in caplog.text
        assert "ValueError" in caplog.text


class TestIntegration:
    """Integration tests for logging system."""
    
    def test_end_to_end_json_logging(self):
        """Test end-to-end JSON logging."""
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(JSONFormatter())
        
        logger = logging.getLogger("test.integration")
        logger.handlers = []
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Log a performance metric
        log_performance(
            logger,
            service="HybridChat",
            stage="retrieval",
            duration_ms=250.5,
            conversation_id="test-conv-123"
        )
        
        output = log_stream.getvalue()
        parsed = json.loads(output.strip())
        
        assert parsed["service"] == "HybridChat"
        assert parsed["stage"] == "retrieval"
        assert parsed["duration_ms"] == 250.5
        assert parsed["conversation_id"] == "test-conv-123"
    
    def test_end_to_end_error_logging(self):
        """Test end-to-end error logging."""
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(JSONFormatter())
        
        logger = logging.getLogger("test.integration.error")
        logger.handlers = []
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)
        
        # Log an error
        try:
            raise RuntimeError("Something went wrong")
        except Exception as e:
            log_error(
                logger,
                service="IndexingService",
                error=e,
                task_id="task-789"
            )
        
        output = log_stream.getvalue()
        parsed = json.loads(output.strip())
        
        assert parsed["service"] == "IndexingService"
        assert parsed["error_type"] == "RuntimeError"
        assert parsed["error_message"] == "Something went wrong"
        assert parsed["task_id"] == "task-789"
        assert "stack_trace" in parsed
