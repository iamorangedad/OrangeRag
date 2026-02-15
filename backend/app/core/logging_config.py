"""Structured logging configuration for OrangeRag.

This module provides JSON structured logging with environment-based configuration.
Supports both JSON and text formats, file rotation, and module-level log level control.
"""
import json
import logging
import logging.config
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_obj: Dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        
        # Add standard fields if available
        if hasattr(record, "service"):
            log_obj["service"] = record.service
        if hasattr(record, "duration_ms"):
            log_obj["duration_ms"] = record.duration_ms
        if hasattr(record, "stage"):
            log_obj["stage"] = record.stage
        if hasattr(record, "conversation_id"):
            log_obj["conversation_id"] = record.conversation_id
        if hasattr(record, "task_id"):
            log_obj["task_id"] = record.task_id
        
        # Add error fields if available
        if hasattr(record, "error_type"):
            log_obj["error_type"] = record.error_type
        if hasattr(record, "error_message"):
            log_obj["error_message"] = record.error_message
        if hasattr(record, "stack_trace"):
            log_obj["stack_trace"] = record.stack_trace
        
        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "asctime", "timestamp", "level", "module", "service",
                "duration_ms", "stage", "conversation_id", "task_id",
                "error_type", "error_message", "stack_trace", "exception"
            }:
                log_obj[key] = value
        
        return json.dumps(log_obj, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """Enhanced text formatter with optional fields."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as text with optional fields."""
        # Build base message
        parts = [
            self.formatTime(record),
            f"[{record.levelname}]",
            f"[{record.name}]"
        ]
        
        # Add service prefix if available
        if hasattr(record, "service"):
            parts.append(f"[{record.service}]")
        
        # Add stage if available
        if hasattr(record, "stage"):
            parts.append(f"[{record.stage}]")
        
        # Add context IDs
        context_parts = []
        if hasattr(record, "conversation_id"):
            context_parts.append(f"conv:{record.conversation_id}")
        if hasattr(record, "task_id"):
            context_parts.append(f"task:{record.task_id}")
        if context_parts:
            parts.append(f"[{' '.join(context_parts)}]")
        
        # Add duration if available
        if hasattr(record, "duration_ms"):
            parts.append(f"({record.duration_ms}ms)")
        
        # Add message
        parts.append(record.getMessage())
        
        # Add error info if available
        if hasattr(record, "error_type"):
            parts.append(f"[Error: {record.error_type}]")
        
        result = " ".join(parts)
        
        # Add exception info if present
        if record.exc_info:
            result += "\n" + self.formatException(record.exc_info)
        
        return result


def get_log_level() -> str:
    """Get log level from environment variable."""
    return os.getenv("LOG_LEVEL", "INFO").upper()


def get_log_format() -> str:
    """Get log format from environment variable."""
    return os.getenv("LOG_FORMAT", "json").lower()


def get_log_output() -> str:
    """Get log output destination from environment variable."""
    return os.getenv("LOG_OUTPUT", "console").lower()


def get_log_file_path() -> str:
    """Get log file path from environment variable."""
    return os.getenv("LOG_FILE_PATH", "/var/log/orangerag/app.log")


def get_log_file_max_bytes() -> int:
    """Get log file max size from environment variable."""
    try:
        return int(os.getenv("LOG_FILE_MAX_BYTES", "104857600"))  # 100MB default
    except ValueError:
        return 104857600


def get_log_file_backup_count() -> int:
    """Get log file backup count from environment variable."""
    try:
        return int(os.getenv("LOG_FILE_BACKUP_COUNT", "10"))
    except ValueError:
        return 10


def get_module_log_levels() -> Dict[str, str]:
    """Get module-specific log levels from environment variables.
    
    Environment variable format: LOG_LEVEL_MODULE_{MODULE_NAME}
    Example: LOG_LEVEL_MODULE_OLLAMA=WARNING
    """
    module_levels = {}
    prefix = "LOG_LEVEL_MODULE_"
    
    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Convert LOG_LEVEL_MODULE_OLLAMA to module name
            module_name = key[len(prefix):].lower().replace("_", ".")
            module_levels[module_name] = value.upper()
    
    return module_levels


def setup_logging(
    level: Optional[str] = None,
    log_format: Optional[str] = None,
    output: Optional[str] = None,
    file_path: Optional[str] = None,
) -> None:
    """Setup structured logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to LOG_LEVEL env var.
        log_format: Log format (json or text). Defaults to LOG_FORMAT env var.
        output: Output destination (console or file). Defaults to LOG_OUTPUT env var.
        file_path: Log file path. Defaults to LOG_FILE_PATH env var.
    """
    # Get configuration
    log_level = (level or get_log_level()).upper()
    fmt = (log_format or get_log_format()).lower()
    out = (output or get_log_output()).lower()
    
    # Create formatters
    if fmt == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    # Create handlers
    handlers = []
    
    if out in ("console", "both"):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    
    if out in ("file", "both"):
        fp = file_path or get_log_file_path()
        # Ensure log directory exists
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        
        file_handler = RotatingFileHandler(
            fp,
            maxBytes=get_log_file_max_bytes(),
            backupCount=get_log_file_backup_count(),
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Add new handlers
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Apply module-specific log levels
    module_levels = get_module_log_levels()
    for module_name, module_level in module_levels.items():
        logging.getLogger(module_name).setLevel(getattr(logging, module_level))
    
    # Log configuration
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging configured: level={log_level}, format={fmt}, output={out}"
    )
    if module_levels:
        logger.info(f"Module log levels: {module_levels}")


def get_logger(name: str) -> logging.Logger:
    """Get logger instance with the given name.
    
    This is a convenience wrapper around logging.getLogger.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_performance(
    logger: logging.Logger,
    service: str,
    stage: str,
    duration_ms: float,
    conversation_id: Optional[str] = None,
    task_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """Log performance metric with structured fields.
    
    Args:
        logger: Logger instance
        service: Service name (e.g., "HybridChat", "Indexing")
        stage: Processing stage (e.g., "retrieval", "llm_generation")
        duration_ms: Duration in milliseconds
        conversation_id: Optional conversation ID
        task_id: Optional task ID
        extra: Optional extra fields
    """
    extra_fields = {
        "service": service,
        "stage": stage,
        "duration_ms": round(duration_ms, 2)
    }
    
    if conversation_id:
        extra_fields["conversation_id"] = conversation_id
    if task_id:
        extra_fields["task_id"] = task_id
    if extra:
        extra_fields.update(extra)
    
    logger.info(
        f"{service} {stage} completed in {duration_ms:.2f}ms",
        extra=extra_fields
    )


def log_error(
    logger: logging.Logger,
    service: str,
    error: Exception,
    conversation_id: Optional[str] = None,
    task_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """Log error with structured fields.
    
    Args:
        logger: Logger instance
        service: Service name
        error: Exception instance
        conversation_id: Optional conversation ID
        task_id: Optional task ID
        extra: Optional extra fields
    """
    import traceback
    
    extra_fields = {
        "service": service,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "stack_trace": traceback.format_exc()
    }
    
    if conversation_id:
        extra_fields["conversation_id"] = conversation_id
    if task_id:
        extra_fields["task_id"] = task_id
    if extra:
        extra_fields.update(extra)
    
    logger.error(
        f"{service} error: {error}",
        extra=extra_fields,
        exc_info=True
    )
