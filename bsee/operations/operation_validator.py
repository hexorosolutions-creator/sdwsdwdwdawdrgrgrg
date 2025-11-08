"""
Operation validation system for BSEE operations.
"""

import time
import traceback
from typing import Dict, List, Any, Tuple, Optional
from bsee.operations.operations_registry import OperationsRegistry


class ValidationResult:
    """Result of operation validation."""

    def __init__(self, operation_name: str, is_valid: bool, details: Dict[str, Any]):
        self.operation_name = operation_name
        self.is_valid = is_valid
        self.details = details
        self.error_message = details.get('error_message', '')
        self.execution_time = details.get('execution_time', 0.0)
        self.memory_usage = details.get('memory_usage', 0)
        self.test_cases = details.get('test_cases', [])


class OperationValidator:
    """Comprehensive validation system for binary operations."""

    def __init__(self, operations_registry: OperationsRegistry):
        """Initialize validator with operations registry."""
        self.operations_registry = operations_registry
        self.validation_results: Dict[str, ValidationResult] = {}
        self.test_data = self._generate_test_data()

    def _generate_test_data(self) -> List[bytes]:
        """Generate various test data sets for validation."""
        test_data = []

        # Empty data
        test_data.append(b'')

        # Small data
        test_data.append(b'Hello, World!')
        test_data.append(bytes(range(10)))

        # Medium data
        test_data.append(bytes([i % 256 for i in range(1000)]))
        test_data.append(b'A' * 500 + b'B' * 500)

        # Large data
        test_data.append(bytes([i % 256 for i in range(10000)]))
        test_data.append(b'Pattern' * 1000)

        # Edge cases
        test_data.append(b'\x00' * 100)  # All zeros
        test_data.append(b'\xFF' * 100)  # All ones
        test_data.append(bytes(range(256)))  # All possible byte values

        # Repetitive data (good for compression tests)
        test_data.append(b'AAAAABBBBBCCCCC' * 20)
        test_data.append(b'1234567890' * 50)

        return test_data

    def validate_all_operations(self) -> Dict[str, ValidationResult]:
        """Validate all operations in the registry."""
        print("Starting comprehensive operation validation...")

        all_operations = self.operations_registry.list_operations()
        total_operations = len(all_operations)

        for i, operation_name in enumerate(all_operations, 1):
            print(f"Validating {operation_name} ({i}/{total_operations})...")

            try:
                result = self.validate_operation(operation_name)
                self.validation_results[operation_name] = result

                status = "✓ PASS" if result.is_valid else "✗ FAIL"
                print(f"  {status}: {operation_name}")
                if not result.is_valid:
                    print(f"    Error: {result.error_message}")

            except Exception as e:
                print(f"  ✗ ERROR: {operation_name} - {str(e)}")
                error_result = ValidationResult(
                    operation_name,
                    False,
                    {
                        'error_message': f"Validation failed with exception: {str(e)}",
                        'traceback': traceback.format_exc(),
                        'execution_time': 0.0,
                        'test_cases': []
                    }
                )
                self.validation_results[operation_name] = error_result

        return self.validation_results

    def validate_operation(self, operation_name: str) -> ValidationResult:
        """Validate a single operation comprehensively."""
        start_time = time.time()

        try:
            # Get operation function and metadata
            operation_func = self.operations_registry.get_operation(operation_name)
            metadata = self.operations_registry.get_operation_metadata(operation_name)

            test_results = []
            reversibility_tests = []
            performance_metrics = []

            # Test with various data sizes
            for test_data in self.test_data:
                test_result = self._test_operation_with_data(
                    operation_name, operation_func, metadata, test_data
                )
                test_results.append(test_result)

                # Performance metrics
                if test_result.get('execution_time', 0) > 0:
                    performance_metrics.append(test_result['execution_time'])

                # Reversibility test
                if metadata.get('reversible', False):
                    reversibility_result = self._test_reversibility(
                        operation_name, operation_func, test_data, test_result
                    )
                    reversibility_tests.append(reversibility_result)

            # Analyze results
            execution_time = time.time() - start_time
            avg_execution_time = sum(performance_metrics) / len(performance_metrics) if performance_metrics else 0

            # Check for consistency
            is_valid = self._evaluate_validation_results(test_results, reversibility_tests, metadata)

            details = {
                'execution_time': avg_execution_time,
                'test_cases': test_results,
                'reversibility_tests': reversibility_tests,
                'metadata_validation': self._validate_metadata(operation_name, metadata),
                'performance_metrics': {
                    'avg_time': avg_execution_time,
                    'max_time': max(performance_metrics) if performance_metrics else 0,
                    'min_time': min(performance_metrics) if performance_metrics else 0
                }
            }

            return ValidationResult(operation_name, is_valid, details)

        except Exception as e:
            return ValidationResult(
                operation_name,
                False,
                {
                    'error_message': f"Validation error: {str(e)}",
                    'traceback': traceback.format_exc(),
                    'execution_time': time.time() - start_time,
                    'test_cases': []
                }
            )

    def _test_operation_with_data(self, operation_name: str, operation_func: callable,
                                 metadata: Dict[str, Any], test_data: bytes) -> Dict[str, Any]:
        """Test operation with specific data."""
        start_time = time.time()

        try:
            # Generate parameters if needed
            params = self._generate_operation_params(operation_name, metadata)

            # Execute operation
            result_data, inverse_func, op_metadata = operation_func(test_data, **params)

            execution_time = time.time() - start_time

            # Validate result
            validation_info = {
                'input_size': len(test_data),
                'output_size': len(result_data),
                'execution_time': execution_time,
                'params_used': params,
                'success': True,
                'error_message': '',
                'metadata_produced': op_metadata
            }

            # Check for reasonable results
            if len(result_data) == 0 and len(test_data) > 0:
                validation_info['warning'] = 'Operation returned empty data for non-empty input'

            return validation_info

        except Exception as e:
            return {
                'input_size': len(test_data),
                'output_size': 0,
                'execution_time': time.time() - start_time,
                'params_used': {},
                'success': False,
                'error_message': str(e),
                'metadata_produced': {}
            }

    def _test_reversibility(self, operation_name: str, operation_func: callable,
                          original_data: bytes, forward_result: Dict[str, Any]) -> Dict[str, Any]:
        """Test if operation's inverse works correctly."""
        if not forward_result.get('success', False):
            return {'reversible': False, 'error': 'Forward operation failed'}

        try:
            # Get inverse function from forward result
            # Note: This would need to be stored from the forward operation
            # For now, we'll re-run the operation to get the inverse
            metadata = self.operations_registry.get_operation_metadata(operation_name)
            params = self._generate_operation_params(operation_name, metadata)

            result_data, inverse_func, _ = operation_func(original_data, **params)

            if not inverse_func:
                return {'reversible': False, 'error': 'No inverse function provided'}

            # Test inverse
            start_time = time.time()
            restored_data = inverse_func()
            execution_time = time.time() - start_time

            # Check if restoration worked
            is_restored = restored_data == original_data

            return {
                'reversible': is_restored,
                'original_size': len(original_data),
                'restored_size': len(restored_data),
                'execution_time': execution_time,
                'data_matches': is_restored,
                'error': '' if is_restored else 'Inverse did not restore original data'
            }

        except Exception as e:
            return {
                'reversible': False,
                'error': f'Inverse test failed: {str(e)}',
                'execution_time': 0
            }

    def _generate_operation_params(self, operation_name: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate appropriate parameters for operation testing."""
        params = {}

        # Get parameter definitions from metadata
        param_defs = metadata.get('parameters', {})

        for param_name, param_info in param_defs.items():
            if param_info.get('required', False):
                param_type = param_info.get('type', 'auto')
                default = param_info.get('default')

                if default is not None:
                    params[param_name] = default
                elif param_type == 'int':
                    min_val = param_info.get('min', 1)
                    max_val = param_info.get('max', 255)
                    params[param_name] = (min_val + max_val) // 2  # Use middle value
                elif param_type == 'float':
                    min_val = param_info.get('min', 0.0)
                    max_val = param_info.get('max', 1.0)
                    params[param_name] = (min_val + max_val) / 2.0  # Use middle value
                elif param_type == 'bool':
                    params[param_name] = True

        return params

    def _validate_metadata(self, operation_name: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate operation metadata."""
        validation_result = {
            'valid': True,
            'issues': [],
            'warnings': []
        }

        # Check required fields
        required_fields = ['category', 'description', 'reversible']
        for field in required_fields:
            if field not in metadata:
                validation_result['issues'].append(f"Missing required field: {field}")
                validation_result['valid'] = False

        # Check category validity
        valid_categories = ['transform', 'bitwise', 'reordering', 'delta', 'substitution', 'custom']
        if metadata.get('category') not in valid_categories:
            validation_result['warnings'].append(f"Unknown category: {metadata.get('category')}")

        # Check reversibility consistency
        reversible = metadata.get('reversible', False)
        if reversible and not metadata.get('description', '').lower().startswith(('reversible', 'inverse')):
            validation_result['warnings'].append("Operation marked as reversible but description doesn't indicate this")

        return validation_result

    def _evaluate_validation_results(self, test_results: List[Dict], reversibility_tests: List[Dict],
                                   metadata: Dict[str, Any]) -> bool:
        """Evaluate if operation passes validation based on test results."""

        # Check if any test completely failed
        failed_tests = [r for r in test_results if not r.get('success', False)]
        if failed_tests:
            # Allow some failures for edge cases, but not too many
            failure_rate = len(failed_tests) / len(test_results)
            if failure_rate > 0.3:  # More than 30% failures is bad
                return False

        # Check reversibility if claimed
        if metadata.get('reversible', False):
            reversible_tests_passed = [r for r in reversibility_tests if r.get('reversible', False)]
            if len(reversible_tests_passed) == 0:
                return False

            # Allow some reversibility failures for edge cases
            reversibility_rate = len(reversible_tests_passed) / len(reversibility_tests)
            if reversibility_rate < 0.7:  # Less than 70% reversibility is bad
                return False

        # Check performance (shouldn't take too long)
        execution_times = [r.get('execution_time', 0) for r in test_results if r.get('execution_time', 0) > 0]
        if execution_times:
            avg_time = sum(execution_times) / len(execution_times)
            if avg_time > 10.0:  # More than 10 seconds average is concerning
                return False

        return True

    def generate_validation_report(self) -> str:
        """Generate a comprehensive validation report."""
        if not self.validation_results:
            return "No validation results available. Run validate_all_operations() first."

        report_lines = ["BSEE Operation Validation Report", "=" * 50, ""]

        total_operations = len(self.validation_results)
        valid_operations = sum(1 for r in self.validation_results.values() if r.is_valid)
        invalid_operations = total_operations - valid_operations

        # Summary
        report_lines.extend([
            f"Total Operations: {total_operations}",
            f"Valid Operations: {valid_operations}",
            f"Invalid Operations: {invalid_operations}",
            f"Success Rate: {(valid_operations/total_operations)*100:.1f}%",
            ""
        ])

        # Valid operations
        if valid_operations > 0:
            report_lines.append("Valid Operations:")
            for name, result in self.validation_results.items():
                if result.is_valid:
                    avg_time = result.details.get('performance_metrics', {}).get('avg_time', 0)
                    report_lines.append(f"  ✓ {name} (avg: {avg_time:.3f}s)")
            report_lines.append("")

        # Invalid operations
        if invalid_operations > 0:
            report_lines.append("Invalid Operations:")
            for name, result in self.validation_results.items():
                if not result.is_valid:
                    report_lines.append(f"  ✗ {name}: {result.error_message}")
            report_lines.append("")

        # Detailed results
        report_lines.append("Detailed Results:")
        report_lines.append("-" * 30)

        for name, result in self.validation_results.items():
            report_lines.append(f"\n{name}:")
            report_lines.append(f"  Status: {'PASS' if result.is_valid else 'FAIL'}")

            if result.error_message:
                report_lines.append(f"  Error: {result.error_message}")

            metadata_val = result.details.get('metadata_validation', {})
            if metadata_val.get('issues'):
                report_lines.append(f"  Metadata Issues: {', '.join(metadata_val['issues'])}")

            if metadata_val.get('warnings'):
                report_lines.append(f"  Metadata Warnings: {', '.join(metadata_val['warnings'])}")

            perf_metrics = result.details.get('performance_metrics', {})
            if perf_metrics.get('avg_time', 0) > 0:
                report_lines.append(f"  Performance: avg={perf_metrics['avg_time']:.3f}s, "
                                f"max={perf_metrics['max_time']:.3f}s")

        return "\n".join(report_lines)

    def get_broken_operations(self) -> List[str]:
        """Get list of operations that failed validation."""
        return [name for name, result in self.validation_results.items() if not result.is_valid]

    def get_operation_summary(self) -> Dict[str, Any]:
        """Get summary statistics of validation results."""
        if not self.validation_results:
            return {}

        total = len(self.validation_results)
        valid = sum(1 for r in self.validation_results.values() if r.is_valid)

        # Performance stats
        all_times = []
        for result in self.validation_results.values():
            if result.is_valid:
                avg_time = result.details.get('performance_metrics', {}).get('avg_time', 0)
                if avg_time > 0:
                    all_times.append(avg_time)

        return {
            'total_operations': total,
            'valid_operations': valid,
            'invalid_operations': total - valid,
            'success_rate': (valid / total) * 100 if total > 0 else 0,
            'average_execution_time': sum(all_times) / len(all_times) if all_times else 0,
            'broken_operations': self.get_broken_operations()
        }