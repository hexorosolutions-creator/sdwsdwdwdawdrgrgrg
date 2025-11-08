"""
Global error handler for BSEE with automatic recovery strategies.
"""

import sys
import traceback
import time
import psutil
from enum import Enum
from typing import Dict, Any, Callable, Optional, List
from collections import defaultdict


class ErrorCategory(Enum):
    """Categories of errors for different handling strategies."""
    RECOVERABLE = "recoverable"      # Can continue with fallback
    FATAL = "fatal"                  # Must stop execution
    WARNING = "warning"              # Log and continue
    TIMEOUT = "timeout"              # Operation took too long
    MEMORY = "memory"                # Memory related errors
    IO_ERROR = "io_error"            # Input/output errors
    VALIDATION = "validation"        # Data validation errors


class ErrorContext:
    """Context information for error handling."""

    def __init__(self, component: str, operation: str = "", iteration: int = 0,
                 state_score: float = 0.0, additional_info: Dict[str, Any] = None):
        self.component = component
        self.operation = operation
        self.iteration = iteration
        self.state_score = state_score
        self.additional_info = additional_info or {}
        self.timestamp = time.time()


class RecoveryStrategy:
    """Base class for recovery strategies."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.success_count = 0
        self.failure_count = 0

    def execute(self, exception: Exception, context: ErrorContext) -> bool:
        """Execute recovery strategy. Return True if successful."""
        raise NotImplementedError

    def get_success_rate(self) -> float:
        """Get success rate of this strategy."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


class MemoryErrorRecovery(RecoveryStrategy):
    """Recovery strategy for memory errors."""

    def __init__(self):
        super().__init__("memory_recovery", "Handles memory-related errors")

    def execute(self, exception: Exception, context: ErrorContext) -> bool:
        """Try to free memory and continue."""
        try:
            import gc

            # Force garbage collection
            gc.collect()

            # Get memory usage before cleanup
            process = psutil.Process()
            memory_before = process.memory_info().rss

            # Clear any caches if available in context
            if hasattr(context, 'clear_caches'):
                context.clear_caches()

            # Force garbage collection again
            gc.collect()

            memory_after = process.memory_info().rss
            memory_freed = memory_before - memory_after

            if memory_freed > 0:
                self.success_count += 1
                return True
            else:
                self.failure_count += 1
                return False

        except Exception:
            self.failure_count += 1
            return False


class TimeoutErrorRecovery(RecoveryStrategy):
    """Recovery strategy for timeout errors."""

    def __init__(self):
        super().__init__("timeout_recovery", "Handles timeout errors by returning best known solution")

    def execute(self, exception: Exception, context: ErrorContext) -> bool:
        """Return best known solution and reduce complexity."""
        try:
            # Get best known solution from context if available
            if hasattr(context, 'get_best_solution'):
                best_solution = context.get_best_solution()
                if best_solution:
                    context.additional_info['fallback_solution'] = best_solution
                    self.success_count += 1
                    return True

            # Reduce computational complexity for next attempt
            if hasattr(context, 'reduce_complexity'):
                context.reduce_complexity()
                self.success_count += 1
                return True

            self.failure_count += 1
            return False

        except Exception:
            self.failure_count += 1
            return False


class ImportErrorHandler:
    """Handler for import errors with graceful degradation."""

    def __init__(self):
        self.missing_modules = set()
        self.fallback_implementations = {}

    def handle_import_error(self, module_name: str, fallback_func: Optional[Callable] = None) -> bool:
        """Handle import error with optional fallback."""
        self.missing_modules.add(module_name)

        if fallback_func:
            self.fallback_implementations[module_name] = fallback_func
            return True

        return False

    def get_fallback(self, module_name: str) -> Optional[Callable]:
        """Get fallback implementation for missing module."""
        return self.fallback_implementations.get(module_name)


class GlobalErrorHandler:
    """Centralized error handling with automatic recovery strategies."""

    def __init__(self, logger=None):
        """Initialize global error handler."""
        self.logger = logger
        self.error_counts = defaultdict(int)
        self.error_categories: Dict[type, ErrorCategory] = {}
        self.recovery_strategies: Dict[ErrorCategory, List[RecoveryStrategy]] = defaultdict(list)
        self.import_handler = ImportErrorHandler()

        # Statistics
        self.total_errors = 0
        self.recovered_errors = 0
        self.error_history = []
        self.max_history = 1000

        # Setup default strategies
        self._setup_default_strategies()

        # Setup default error categorization
        self._setup_default_categories()

    def _setup_default_strategies(self):
        """Setup default recovery strategies."""
        # Memory error strategies
        self.recovery_strategies[ErrorCategory.MEMORY].append(MemoryErrorRecovery())

        # Timeout error strategies
        self.recovery_strategies[ErrorCategory.TIMEOUT].append(TimeoutErrorRecovery())

        # IO error strategies
        self.recovery_strategies[ErrorCategory.IO_ERROR].append(
            RecoveryStrategy("retry_with_backoff", "Retry operation with exponential backoff")
        )

        # Validation error strategies
        self.recovery_strategies[ErrorCategory.VALIDATION].append(
            RecoveryStrategy("skip_invalid", "Skip invalid data and continue")
        )

    def _setup_default_categories(self):
        """Setup default error categorization."""
        self.error_categories.update({
            MemoryError: ErrorCategory.MEMORY,
            TimeoutError: ErrorCategory.TIMEOUT,
            OSError: ErrorCategory.IO_ERROR,
            IOError: ErrorCategory.IO_ERROR,
            ImportError: ErrorCategory.FATAL,
            ValueError: ErrorCategory.VALIDATION,
            TypeError: ErrorCategory.WARNING,
        })

    def register_recovery_strategy(self, error_category: ErrorCategory,
                                  strategy: RecoveryStrategy):
        """Register a custom recovery strategy for an error category."""
        self.recovery_strategies[error_category].append(strategy)

    def categorize_error(self, exception: Exception) -> ErrorCategory:
        """Categorize an exception for appropriate handling."""
        # Check direct mapping first
        exception_type = type(exception)
        if exception_type in self.error_categories:
            return self.error_categories[exception_type]

        # Check parent classes
        for exc_type, category in self.error_categories.items():
            if isinstance(exception, exc_type):
                self.error_categories[exception_type] = category  # Cache for future
                return category

        # Default categorization based on exception message
        error_message = str(exception).lower()
        if any(keyword in error_message for keyword in ['memory', 'out of memory']):
            return ErrorCategory.MEMORY
        elif any(keyword in error_message for keyword in ['timeout', 'timed out']):
            return ErrorCategory.TIMEOUT
        elif any(keyword in error_message for keyword in ['file', 'io', 'directory']):
            return ErrorCategory.IO_ERROR
        elif any(keyword in error_message for keyword in ['invalid', 'validation']):
            return ErrorCategory.VALIDATION
        elif any(keyword in error_message for keyword in ['import', 'module']):
            return ErrorCategory.FATAL
        else:
            return ErrorCategory.WARNING

    def handle_exception(self, exception: Exception, context: ErrorContext) -> bool:
        """Handle exception with appropriate recovery strategy.

        Returns:
            True if recovery was successful, False otherwise.
        """
        self.total_errors += 1
        self.error_counts[type(exception).__name__] += 1

        # Add to history
        error_record = {
            'exception': exception,
            'context': context,
            'timestamp': time.time(),
            'recovered': False
        }
        self.error_history.append(error_record)

        # Trim history if needed
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history:]

        # Log error
        if self.logger:
            self.logger.error(
                f"Error in {context.component}: {type(exception).__name__}: {str(exception)}",
                extra={
                    'context': context.__dict__,
                    'traceback': traceback.format_exc()
                }
            )

        # Categorize error
        category = self.categorize_error(exception)

        # Handle fatal errors immediately
        if category == ErrorCategory.FATAL:
            if self.logger:
                self.logger.critical(f"Fatal error in {context.component}: {str(exception)}")
            return False

        # Try recovery strategies
        strategies = self.recovery_strategies.get(category, [])

        for strategy in strategies:
            try:
                if strategy.execute(exception, context):
                    self.recovered_errors += 1
                    error_record['recovered'] = True

                    if self.logger:
                        self.logger.info(
                            f"Successfully recovered from {type(exception).__name__} "
                            f"in {context.component} using {strategy.name}"
                        )
                    return True

            except Exception as recovery_error:
                if self.logger:
                    self.logger.error(
                        f"Recovery strategy {strategy.name} failed: {str(recovery_error)}"
                    )

        # If all recovery strategies failed, handle based on category
        if category == ErrorCategory.WARNING:
            # Warnings can be ignored
            return True
        elif category == ErrorCategory.RECOVERABLE:
            # Try to continue with degraded functionality
            if hasattr(context, 'set_degraded_mode'):
                context.set_degraded_mode()
            return True
        else:
            return False

    def handle_import_error(self, module_name: str, fallback_func: Optional[Callable] = None) -> bool:
        """Handle import errors gracefully."""
        return self.import_handler.handle_import_error(module_name, fallback_func)

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about handled errors."""
        category_stats = defaultdict(int)
        for error_record in self.error_history:
            category = self.categorize_error(error_record['exception'])
            category_stats[category.value] += 1

        strategy_stats = {}
        for category, strategies in self.recovery_strategies.items():
            strategy_stats[category.value] = []
            for strategy in strategies:
                strategy_stats[category.value].append({
                    'name': strategy.name,
                    'success_rate': strategy.get_success_rate(),
                    'success_count': strategy.success_count,
                    'failure_count': strategy.failure_count
                })

        return {
            'total_errors': self.total_errors,
            'recovered_errors': self.recovered_errors,
            'recovery_rate': (self.recovered_errors / self.total_errors) if self.total_errors > 0 else 0,
            'error_counts': dict(self.error_counts),
            'category_distribution': dict(category_stats),
            'strategy_performance': strategy_stats,
            'missing_modules': list(self.import_handler.missing_modules)
        }

    def clear_statistics(self):
        """Clear all error statistics."""
        self.error_counts.clear()
        self.total_errors = 0
        self.recovered_errors = 0
        self.error_history.clear()

        # Reset strategy statistics
        for strategies in self.recovery_strategies.values():
            for strategy in strategies:
                strategy.success_count = 0
                strategy.failure_count = 0


# Global instance
_global_error_handler = None


def get_error_handler(logger=None) -> GlobalErrorHandler:
    """Get or create the global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = GlobalErrorHandler(logger)
    return _global_error_handler


def handle_exception(exception: Exception, component: str, operation: str = "",
                   iteration: int = 0, state_score: float = 0.0, **kwargs) -> bool:
    """Convenience function to handle exceptions globally."""
    context = ErrorContext(component, operation, iteration, state_score, kwargs)
    handler = get_error_handler()
    return handler.handle_exception(exception, context)


def handle_import_error(module_name: str, fallback_func: Optional[Callable] = None) -> bool:
    """Convenience function to handle import errors globally."""
    handler = get_error_handler()
    return handler.handle_import_error(module_name, fallback_func)


# Decorator for automatic error handling
def error_handler(component: str, operation: str = ""):
    """Decorator for automatic error handling on functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Try to extract iteration and state from kwargs if available
                iteration = kwargs.get('iteration', 0)
                state_score = kwargs.get('state_score', 0.0)

                if handle_exception(e, component, operation, iteration, state_score, **kwargs):
                    # Recovery successful, return None or fallback value
                    return kwargs.get('fallback_value', None)
                else:
                    # Recovery failed, re-raise the exception
                    raise
        return wrapper
    return decorator