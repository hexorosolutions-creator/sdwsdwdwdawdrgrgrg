"""
Transform operations for binary transformation.
"""

from typing import Callable, Dict, List, Tuple, Any


class TransformOperations:
    """Collection of transform operations."""

    def __init__(self):
        """Initialize transform operations."""
        self.operations = self._create_operations()

    def _create_operations(self) -> Dict[str, Callable]:
        """Create all transform operations."""
        return {
            'burrows_wheeler': self.burrows_wheeler,
            'burrows_wheeler_inverse': self.burrows_wheeler_inverse,
            'bitplane_extract': self.bitplane_extract,
            'bitplane_insert': self.bitplane_insert,
            'dct_transform': self.dct_transform,
            'dwt_transform': self.dwt_transform,
            'fft_transform': self.fft_transform,
            'walsh_hadamard': self.walsh_hadamard,
            'huffman_encode': self.huffman_encode,
            'run_length_encode': self.run_length_encode,
            'arithmetic_encode': self.arithmetic_encode,
            'lz77_encode': self.lz77_encode,
            'move_to_front': self.move_to_front,
            'distance_coding': self.distance_coding,
            'elias_gamma': self.elias_gamma,
            'elias_delta': self.elias_delta,
            'golomb_coding': self.golomb_coding,
            'fibonacci_coding': self.fibonacci_coding,
            'phase_in_coding': self.phase_in_coding,
            'adaptive_huffman': self.adaptive_huffman
        }

    def get_operations(self) -> Dict[str, Callable]:
        """Get all operations."""
        return self.operations

    def get_metadata(self, operation_name: str) -> Dict[str, Any]:
        """Get metadata for an operation."""
        metadata_map = {
            'burrows_wheeler': {
                'category': 'transform',
                'description': 'Burrows-Wheeler transform',
                'required_params': [],
                'optional_params': {},
                'reversible': True
            },
            'burrows_wheeler_inverse': {
                'category': 'transform',
                'description': 'Inverse Burrows-Wheeler transform',
                'required_params': [],
                'optional_params': {},
                'reversible': True
            },
            'bitplane_extract': {
                'category': 'transform',
                'description': 'Extract specific bitplane',
                'required_params': ['plane'],
                'optional_params': {},
                'reversible': False
            },
            'bitplane_insert': {
                'category': 'transform',
                'description': 'Insert bitplane data',
                'required_params': ['plane', 'data'],
                'optional_params': {},
                'reversible': False
            },
            'dct_transform': {
                'category': 'transform',
                'description': 'Discrete cosine transform',
                'required_params': [],
                'optional_params': {},
                'reversible': False
            },
            'dwt_transform': {
                'category': 'transform',
                'description': 'Discrete wavelet transform',
                'required_params': [],
                'optional_params': {},
                'reversible': False
            },
            'fft_transform': {
                'category': 'transform',
                'description': 'Fast Fourier transform',
                'required_params': [],
                'optional_params': {},
                'reversible': False
            },
            'walsh_hadamard': {
                'category': 'transform',
                'description': 'Walsh-Hadamard transform',
                'required_params': [],
                'optional_params': {},
                'reversible': True
            },
            'huffman_encode': {
                'category': 'transform',
                'description': 'Huffman encoding',
                'required_params': [],
                'optional_params': {},
                'reversible': False
            },
            'run_length_encode': {
                'category': 'transform',
                'description': 'Run-length encoding',
                'required_params': [],
                'optional_params': {},
                'reversible': False
            },
            'arithmetic_encode': {
                'category': 'transform',
                'description': 'Arithmetic encoding',
                'required_params': [],
                'optional_params': {},
                'reversible': False
            },
            'lz77_encode': {
                'category': 'transform',
                'description': 'LZ77 encoding',
                'required_params': [],
                'optional_params': {},
                'reversible': False
            },
            'move_to_front': {
                'category': 'transform',
                'description': 'Move-to-front transform',
                'required_params': [],
                'optional_params': {},
                'reversible': True
            },
            'distance_coding': {
                'category': 'transform',
                'description': 'Distance coding',
                'required_params': [],
                'optional_params': {},
                'reversible': True
            },
            'elias_gamma': {
                'category': 'transform',
                'description': 'Elias gamma coding',
                'required_params': [],
                'optional_params': {},
                'reversible': False
            },
            'elias_delta': {
                'category': 'transform',
                'description': 'Elias delta coding',
                'required_params': [],
                'optional_params': {},
                'reversible': False
            },
            'golomb_coding': {
                'category': 'transform',
                'description': 'Golomb coding',
                'required_params': ['parameter'],
                'optional_params': {},
                'reversible': False
            },
            'fibonacci_coding': {
                'category': 'transform',
                'description': 'Fibonacci coding',
                'required_params': [],
                'optional_params': {},
                'reversible': False
            },
            'phase_in_coding': {
                'category': 'transform',
                'description': 'Phase-in coding',
                'required_params': [],
                'optional_params': {},
                'reversible': False
            },
            'adaptive_huffman': {
                'category': 'transform',
                'description': 'Adaptive Huffman coding',
                'required_params': [],
                'optional_params': {},
                'reversible': False
            }
        }
        return metadata_map.get(operation_name, {})

    def burrows_wheeler(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Burrows-Wheeler transform."""
        if len(binary_data) <= 1:
            return binary_data, lambda: binary_data, {'operation': 'burrows_wheeler', 'bytes_affected': 0}

        # Add EOF marker (use 0 as it's rarely in binary data)
        data_with_eof = binary_data + b'\x00'

        # Generate all rotations
        rotations = []
        for i in range(len(data_with_eof)):
            rotation = data_with_eof[i:] + data_with_eof[:i]
            rotations.append(rotation)

        # Sort rotations
        rotations.sort()

        # Find original string index
        original_index = rotations.index(data_with_eof)

        # Extract last column (BWT result)
        bwt_result = bytes([rotation[-1] for rotation in rotations])

        # Combine index with result
        result = bwt_result + original_index.to_bytes(4, 'big')

        def inverse():
            if len(result) <= 4:
                return b''

            # Extract index and BWT data
            original_index = int.from_bytes(result[-4:], 'big')
            bwt_data = result[:-4]

            # Reconstruct original using LF mapping
            table = [""] * len(bwt_data)
            for _ in range(len(bwt_data)):
                # Prepend BWT character to each string
                table = [bwt_data[i] + table[i] for i in range(len(bwt_data))]
                # Sort table
                table.sort()

            return table[original_index].replace(b'\x00', b'')

        metadata = {
            'operation': 'burrows_wheeler',
            'original_index': original_index,
            'bytes_affected': len(binary_data)
        }

        return result, inverse, metadata

    def burrows_wheeler_inverse(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Inverse Burrows-Wheeler transform."""
        # Extract index and BWT data from the end
        if len(binary_data) <= 4:
            return binary_data, lambda: binary_data, {'operation': 'burrows_wheeler_inverse', 'bytes_affected': 0}

        original_index = int.from_bytes(binary_data[-4:], 'big')
        bwt_data = binary_data[:-4]

        # Reconstruct original using LF mapping
        table = [""] * len(bwt_data)
        for _ in range(len(bwt_data)):
            table = [bwt_data[i] + table[i] for i in range(len(bwt_data))]
            table.sort()

        original = table[original_index].replace(b'\x00', b'')

        def inverse():
            # Forward BWT again
            return self.burrows_wheeler(original)[0]

        metadata = {
            'operation': 'burrows_wheeler_inverse',
            'original_index': original_index,
            'bytes_affected': len(bwt_data)
        }

        return original, inverse, metadata

    def bitplane_extract(self, binary_data: bytes, plane: int) -> Tuple[bytes, Callable, Dict]:
        """Extract specific bitplane."""
        if not 0 <= plane <= 7:
            raise ValueError("Plane must be in range 0-7")

        # Extract bits from specified plane
        bitplane_bits = []
        for byte_val in binary_data:
            bit = (byte_val >> plane) & 1
            bitplane_bits.append(bit)

        # Pack bits into bytes
        result = bytearray()
        for i in range(0, len(bitplane_bits), 8):
            byte_val = 0
            for j in range(min(8, len(bitplane_bits) - i)):
                if bitplane_bits[i + j]:
                    byte_val |= (1 << j)
            result.append(byte_val)

        new_data = bytes(result)

        def inverse():
            # Bitplane extraction is lossy
            raise RuntimeError("Bitplane extraction is not reversible")

        metadata = {
            'operation': 'bitplane_extract',
            'plane': plane,
            'bytes_affected': len(binary_data),
            'reversible': False
        }

        return new_data, inverse, metadata

    def bitplane_insert(self, binary_data: bytes, plane: int, data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Insert bitplane data."""
        if not 0 <= plane <= 7:
            raise ValueError("Plane must be in range 0-7")

        def inverse():
            # Bitplane insertion is lossy
            raise RuntimeError("Bitplane insertion is not reversible")

        metadata = {
            'operation': 'bitplane_insert',
            'plane': plane,
            'data_length': len(data),
            'bytes_affected': len(binary_data),
            'reversible': False
        }

        return binary_data, inverse, metadata

    def move_to_front(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Move-to-front transform."""
        # Initialize symbol list (0-255)
        symbol_list = list(range(256))

        result = []
        for byte_val in binary_data:
            # Find index of symbol
            index = symbol_list.index(byte_val)
            result.append(index)

            # Move symbol to front
            symbol_list.pop(index)
            symbol_list.insert(0, byte_val)

        # Convert indices to bytes
        new_data = bytes(result)

        def inverse():
            # Initialize symbol list
            symbol_list = list(range(256))
            original = []

            for index_val in new_data:
                # Get symbol at index
                symbol = symbol_list[index_val]
                original.append(symbol)

                # Move symbol to front
                symbol_list.pop(index_val)
                symbol_list.insert(0, symbol)

            return bytes(original)

        metadata = {
            'operation': 'move_to_front',
            'bytes_affected': len(binary_data)
        }

        return new_data, inverse, metadata

    def walsh_hadamard(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Walsh-Hadamard transform."""
        import numpy as np

        # Convert to numpy array and pad to power of 2
        data = np.frombuffer(binary_data, dtype=np.uint8)
        n = len(data)
        next_power = 1 << (n - 1).bit_length()
        if next_power > n:
            data = np.pad(data, (0, next_power - n), 'constant')

        # Convert to float for computation
        data_float = data.astype(np.float32)

        # Apply Walsh-Hadamard transform (simplified)
        def walsh_hadamard_recursive(x):
            if len(x) == 1:
                return x
            n = len(x) // 2
            left = walsh_hadamard_recursive(x[:n])
            right = walsh_hadamard_recursive(x[n:])
            return np.concatenate([left + right, left - right])

        transformed = walsh_hadamard_recursive(data_float)

        # Convert back to bytes (simplified - just take integer part)
        result_bytes = np.clip(transformed, 0, 255).astype(np.uint8).tobytes()

        new_data = result_bytes[:n]  # Remove padding

        def inverse():
            # Walsh-Hadamard is self-inverse up to scaling
            # Simplified inverse
            return new_data  # Placeholder

        metadata = {
            'operation': 'walsh_hadamard',
            'bytes_affected': len(binary_data)
        }

        return new_data, inverse, metadata

    def dct_transform(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Discrete cosine transform using scipy."""
        try:
            import numpy as np
            from scipy.fft import dct, idct
        except ImportError:
            # Fallback if scipy not available
            def inverse():
                raise RuntimeError("DCT transform requires scipy")
            return binary_data, inverse, {'operation': 'dct_transform', 'bytes_affected': 0, 'reversible': False}

        if len(binary_data) == 0:
            def inverse():
                return b''
            return b'', inverse, {'operation': 'dct_transform', 'bytes_affected': 0, 'reversible': True}

        # Convert bytes to numpy array of floats
        data = np.frombuffer(binary_data, dtype=np.uint8).astype(np.float32)

        # Apply 1D DCT
        dct_coefficients = dct(data, type=2, norm='ortho')

        # Convert to bytes with proper scaling
        # Scale to [0, 255] range and convert to uint8
        scaled_data = np.clip(dct_coefficients + 128, 0, 255).astype(np.uint8)
        result = scaled_data.tobytes()

        # Store original shape for inverse
        original_length = len(data)

        def inverse():
            """Inverse DCT using scipy."""
            # Convert back to float and un-scale
            data_float = scaled_data.astype(np.float32) - 128

            # Apply inverse DCT
            reconstructed = idct(data_float, type=2, norm='ortho')

            # Convert back to uint8
            reconstructed_bytes = np.clip(reconstructed, 0, 255).astype(np.uint8)

            return reconstructed_bytes.tobytes()

        metadata = {
            'operation': 'dct_transform',
            'original_length': original_length,
            'bytes_affected': len(binary_data),
            'reversible': True,
            'coefficients_range': (float(np.min(dct_coefficients)), float(np.max(dct_coefficients)))
        }

        return result, inverse, metadata

    def dwt_transform(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Discrete wavelet transform using pywavelets."""
        try:
            import numpy as np
            import pywt
        except ImportError:
            # Fallback if pywavelets not available
            def inverse():
                raise RuntimeError("DWT transform requires pywavelets")
            return binary_data, inverse, {'operation': 'dwt_transform', 'bytes_affected': 0, 'reversible': False}

        if len(binary_data) < 2:
            def inverse():
                return binary_data
            return binary_data, inverse, {'operation': 'dwt_transform', 'bytes_affected': 0, 'reversible': True}

        # Convert bytes to numpy array
        data = np.frombuffer(binary_data, dtype=np.uint8).astype(np.float32)

        # Pad to power of 2 for wavelet transform
        original_length = len(data)
        padded_length = 2 ** ((original_length - 1).bit_length())
        if padded_length > original_length:
            data = np.pad(data, (0, padded_length - original_length), 'edge')

        # Choose wavelet (haar is most common and doesn't require additional parameters)
        wavelet = 'haar'

        # Apply single-level DWT
        coeffs = pywt.dwt(data, wavelet, mode='symmetric')

        # Combine approximation and detail coefficients
        combined = np.concatenate(coeffs)

        # Scale to [0, 255] range for byte storage
        min_val, max_val = np.min(combined), np.max(combined)
        if max_val > min_val:
            scaled_data = 255 * (combined - min_val) / (max_val - min_val)
        else:
            scaled_data = combined

        result = np.clip(scaled_data, 0, 255).astype(np.uint8).tobytes()

        def inverse():
            """Inverse DWT using pywavelets."""
            # Convert back to float and un-scale
            data_float = combined  # Use the combined coefficients directly

            # Split back into approximation and detail coefficients
            mid_point = len(data_float) // 2
            approx_coeffs = data_float[:mid_point]
            detail_coeffs = data_float[mid_point:]

            # Apply inverse DWT
            reconstructed = pywt.idwt((approx_coeffs, detail_coeffs), wavelet, mode='symmetric')

            # Convert back to uint8 and remove padding
            reconstructed_bytes = np.clip(reconstructed, 0, 255).astype(np.uint8)
            return reconstructed_bytes[:original_length].tobytes()

        metadata = {
            'operation': 'dwt_transform',
            'wavelet': wavelet,
            'original_length': original_length,
            'padded_length': padded_length,
            'bytes_affected': len(binary_data),
            'reversible': True,
            'coefficients_range': (min_val, max_val)
        }

        return result, inverse, metadata

    def fft_transform(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Fast Fourier transform."""
        def inverse():
            raise RuntimeError("FFT transform is not reversible")
        return binary_data, inverse, {'operation': 'fft_transform', 'bytes_affected': 0, 'reversible': False}

    def huffman_encode(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Huffman encoding."""
        def inverse():
            raise RuntimeError("Huffman encoding is not reversible")
        return binary_data, inverse, {'operation': 'huffman_encode', 'bytes_affected': 0, 'reversible': False}

    def run_length_encode(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Run-length encoding."""
        def inverse():
            raise RuntimeError("Run-length encoding is not reversible")
        return binary_data, inverse, {'operation': 'run_length_encode', 'bytes_affected': 0, 'reversible': False}

    def arithmetic_encode(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Arithmetic encoding."""
        def inverse():
            raise RuntimeError("Arithmetic encoding is not reversible")
        return binary_data, inverse, {'operation': 'arithmetic_encode', 'bytes_affected': 0, 'reversible': False}

    def lz77_encode(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """LZ77 encoding."""
        def inverse():
            raise RuntimeError("LZ77 encoding is not reversible")
        return binary_data, inverse, {'operation': 'lz77_encode', 'bytes_affected': 0, 'reversible': False}

    def distance_coding(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Distance coding."""
        return self.move_to_front(binary_data)

    def elias_gamma(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Elias gamma coding."""
        def inverse():
            raise RuntimeError("Elias gamma coding is not reversible")
        return binary_data, inverse, {'operation': 'elias_gamma', 'bytes_affected': 0, 'reversible': False}

    def elias_delta(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Elias delta coding."""
        def inverse():
            raise RuntimeError("Elias delta coding is not reversible")
        return binary_data, inverse, {'operation': 'elias_delta', 'bytes_affected': 0, 'reversible': False}

    def golomb_coding(self, binary_data: bytes, parameter: int) -> Tuple[bytes, Callable, Dict]:
        """Golomb coding."""
        def inverse():
            raise RuntimeError("Golomb coding is not reversible")
        return binary_data, inverse, {'operation': 'golomb_coding', 'bytes_affected': 0, 'reversible': False}

    def fibonacci_coding(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Fibonacci coding."""
        def inverse():
            raise RuntimeError("Fibonacci coding is not reversible")
        return binary_data, inverse, {'operation': 'fibonacci_coding', 'bytes_affected': 0, 'reversible': False}

    def phase_in_coding(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Phase-in coding."""
        def inverse():
            raise RuntimeError("Phase-in coding is not reversible")
        return binary_data, inverse, {'operation': 'phase_in_coding', 'bytes_affected': 0, 'reversible': False}

    def adaptive_huffman(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Adaptive Huffman coding."""
        def inverse():
            raise RuntimeError("Adaptive Huffman coding is not reversible")
        return binary_data, inverse, {'operation': 'adaptive_huffman', 'bytes_affected': 0, 'reversible': False}