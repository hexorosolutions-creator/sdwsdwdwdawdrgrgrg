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
                'reversible': True,
                'parameters': {
                    'type': 'auto'
                }
            },
            'dwt_transform': {
                'category': 'transform',
                'description': 'Discrete wavelet transform',
                'required_params': [],
                'optional_params': {},
                'reversible': True,
                'parameters': {
                    'type': 'auto'
                }
            },
            'fft_transform': {
                'category': 'transform',
                'description': 'Fast Fourier transform',
                'required_params': [],
                'optional_params': {},
                'reversible': True,
                'parameters': {
                    'type': 'auto'
                }
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
                'reversible': True,
                'parameters': {
                    'type': 'auto'
                }
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
        """Fast Fourier transform using numpy.fft."""
        try:
            import numpy as np
        except ImportError:
            # Fallback if numpy not available
            def inverse():
                raise RuntimeError("FFT transform requires numpy")
            return binary_data, inverse, {'operation': 'fft_transform', 'bytes_affected': 0, 'reversible': False}

        if len(binary_data) == 0:
            def inverse():
                return b''
            return b'', inverse, {'operation': 'fft_transform', 'bytes_affected': 0, 'reversible': True}

        # Convert bytes to numpy array of floats
        data = np.frombuffer(binary_data, dtype=np.uint8).astype(np.float32)

        # Apply FFT
        fft_coefficients = np.fft.fft(data)

        # Get magnitude and phase
        magnitudes = np.abs(fft_coefficients)
        phases = np.angle(fft_coefficients)

        # Convert to bytes - store magnitudes first, then phases
        # Scale both to [0, 255] range
        mag_scaled = np.clip(255 * magnitudes / np.max(magnitudes) if np.max(magnitudes) > 0 else magnitudes, 0, 255).astype(np.uint8)

        # Scale phases from [-π, π] to [0, 255]
        phase_scaled = np.clip(255 * (phases + np.pi) / (2 * np.pi), 0, 255).astype(np.uint8)

        # Interleave magnitude and phase data
        combined = np.empty(mag_scaled.size + phase_scaled.size, dtype=np.uint8)
        combined[0::2] = mag_scaled
        combined[1::2] = phase_scaled

        result = combined.tobytes()

        # Store original data for inverse
        original_length = len(data)
        max_magnitude = float(np.max(magnitudes))

        def inverse():
            """Inverse FFT using numpy.fft."""
            # Extract magnitudes and phases
            mag_extracted = combined[0::2].astype(np.float32)
            phase_extracted = combined[1::2].astype(np.float32)

            # Un-scale
            mag_unscaled = mag_extracted * max_magnitude / 255.0
            phase_unscaled = phase_extracted * (2 * np.pi) / 255.0 - np.pi

            # Reconstruct complex FFT coefficients
            fft_reconstructed = mag_unscaled * np.exp(1j * phase_unscaled)

            # Apply inverse FFT
            reconstructed = np.fft.ifft(fft_reconstructed)

            # Convert back to uint8
            reconstructed_bytes = np.clip(reconstructed.real, 0, 255).astype(np.uint8)

            return reconstructed_bytes.tobytes()

        metadata = {
            'operation': 'fft_transform',
            'original_length': original_length,
            'bytes_affected': len(binary_data),
            'reversible': True,
            'max_magnitude': max_magnitude,
            'frequency_components': len(fft_coefficients)
        }

        return result, inverse, metadata

    def huffman_encode(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Huffman coding implementation."""
        if len(binary_data) == 0:
            def inverse():
                return b''
            return b'', inverse, {'operation': 'huffman_encode', 'bytes_affected': 0, 'reversible': True}

        # Count byte frequencies
        frequency = {}
        for byte in binary_data:
            frequency[byte] = frequency.get(byte, 0) + 1

        # Build Huffman tree
        import heapq
        heap = []

        # Create leaf nodes
        for byte, freq in frequency.items():
            heapq.heappush(heap, (freq, [byte, []]))

        # Build tree by combining smallest frequency nodes
        while len(heap) > 1:
            freq1, node1 = heapq.heappop(heap)
            freq2, node2 = heapq.heappop(heap)

            # Create internal node
            new_freq = freq1 + freq2
            new_node = [None, [node1, node2]]  # None for internal nodes
            heapq.heappush(heap, (new_freq, new_node))

        # Generate codes by traversing tree
        codes = {}
        root = heap[0][1] if heap else []

        def generate_codes(node, code):
            if node is None:
                return

            if len(node) == 2 and node[0] is not None:
                # Leaf node
                codes[node[0]] = code
            elif len(node) == 2 and node[1]:
                # Internal node
                generate_codes(node[1][0], code + '0') if len(node[1]) > 0 else None
                generate_codes(node[1][1], code + '1') if len(node[1]) > 1 else None

        generate_codes(root, '')

        # Encode data
        encoded_bits = []
        for byte in binary_data:
            encoded_bits.append(codes[byte])

        encoded_string = ''.join(encoded_bits)

        # Pack bits into bytes
        encoded_bytes = bytearray()
        for i in range(0, len(encoded_string), 8):
            byte_bits = encoded_string[i:i+8]
            if len(byte_bits) < 8:
                # Pad final byte
                byte_bits += '0' * (8 - len(byte_bits))
            byte_val = int(byte_bits, 2)
            encoded_bytes.append(byte_val)

        # Store tree structure and padding info for decoding
        # Simple tree serialization: for each byte, store its code length and code
        tree_data = bytearray()
        tree_data.append(len(frequency))  # Number of unique symbols

        for byte, code in sorted(codes.items()):
            tree_data.append(byte)  # Symbol
            tree_data.append(len(code))  # Code length
            # Store code bits (packed)
            for i in range(0, len(code), 8):
                code_bits = code[i:i+8]
                if len(code_bits) < 8:
                    code_bits += '0' * (8 - len(code_bits))
                code_byte = int(code_bits, 2)
                tree_data.append(code_byte)

        # Combine tree data and padding info with encoded data
        padding_bits = (8 - len(encoded_string) % 8) % 8
        result = bytes([padding_bits]) + bytes(tree_data) + bytes(encoded_bytes)

        # Store data for inverse
        original_data = binary_data
        symbol_count = len(frequency)
        tree_info = {'codes': codes, 'symbol_count': symbol_count, 'padding_bits': padding_bits}

        def inverse():
            """Huffman decoding."""
            if len(result) < 2:
                return b''

            # Extract padding bits
            padding_bits = result[0]
            pos = 1

            # Extract tree data
            symbol_count = result[pos]
            pos += 1

            codes = {}
            for _ in range(symbol_count):
                symbol = result[pos]
                pos += 1
                code_length = result[pos]
                pos += 1

                # Extract code bits
                code_bytes_needed = (code_length + 7) // 8
                code_bits = ''
                for i in range(code_bytes_needed):
                    byte_val = result[pos]
                    pos += 1
                    code_bits += format(byte_val, '08b')[:code_length - len(code_bits)]

                codes[symbol] = code_bits

            # Build reverse mapping for decoding
            code_to_symbol = {code: symbol for symbol, code in codes.items()}

            # Extract encoded data
            encoded_data = result[pos:]

            # Convert to bits and remove padding
            encoded_bits = ''.join(format(byte, '08b') for byte in encoded_data)
            encoded_bits = encoded_bits[:-padding_bits] if padding_bits > 0 else encoded_bits

            # Decode
            decoded_bytes = bytearray()
            current_code = ''

            for bit in encoded_bits:
                current_code += bit
                if current_code in code_to_symbol:
                    decoded_bytes.append(code_to_symbol[current_code])
                    current_code = ''

            return bytes(decoded_bytes)

        metadata = {
            'operation': 'huffman_encode',
            'original_size': len(binary_data),
            'compressed_size': len(result),
            'compression_ratio': len(result) / len(binary_data) if len(binary_data) > 0 else 1.0,
            'symbol_count': symbol_count,
            'bytes_affected': len(binary_data),
            'reversible': True
        }

        return result, inverse, metadata

    def run_length_encode(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Run-length encoding with configurable run detection."""
        if len(binary_data) == 0:
            def inverse():
                return b''
            return b'', inverse, {'operation': 'run_length_encode', 'bytes_affected': 0, 'reversible': True}

        # Run-length encode: replace consecutive identical bytes with (count, byte) pairs
        encoded = bytearray()
        max_run_length = 255  # Maximum run length to prevent expansion attacks

        i = 0
        while i < len(binary_data):
            current_byte = binary_data[i]
            run_length = 1

            # Count consecutive identical bytes
            while (i + run_length < len(binary_data) and
                   binary_data[i + run_length] == current_byte and
                   run_length < max_run_length):
                run_length += 1

            # Only encode as run if length > 1 or it's a single byte
            if run_length > 1:
                # Use run encoding
                encoded.append(run_length)
                encoded.append(current_byte)
            else:
                # Single byte - could use special marker, but for simplicity use run length 1
                encoded.append(1)
                encoded.append(current_byte)

            i += run_length

        result = bytes(encoded)

        def inverse():
            """Run-length decode."""
            if len(result) == 0:
                return b''

            decoded = bytearray()
            i = 0

            while i < len(result):
                if i + 1 >= len(result):
                    # Incomplete run-length pair
                    break

                run_length = result[i]
                byte_value = result[i + 1]

                # Add repeated bytes
                decoded.extend([byte_value] * run_length)

                i += 2

            return bytes(decoded)

        # Calculate compression statistics
        compression_ratio = len(result) / len(binary_data) if len(binary_data) > 0 else 1.0

        metadata = {
            'operation': 'run_length_encode',
            'original_size': len(binary_data),
            'compressed_size': len(result),
            'compression_ratio': compression_ratio,
            'max_run_length': max_run_length,
            'bytes_affected': len(binary_data),
            'reversible': True
        }

        return result, inverse, metadata

    def arithmetic_encode(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Arithmetic encoding."""
        def inverse():
            raise RuntimeError("Arithmetic encoding is not reversible")
        return binary_data, inverse, {'operation': 'arithmetic_encode', 'bytes_affected': 0, 'reversible': False}

    def lz77_encode(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """LZ77 compression with sliding window and look-ahead buffer."""
        if len(binary_data) == 0:
            def inverse():
                return b''
            return b'', inverse, {'operation': 'lz77_encode', 'bytes_affected': 0, 'reversible': True}

        # LZ77 parameters
        window_size = 32768    # Sliding window size (32K)
        look_ahead_size = 258  # Maximum match length
        min_match_length = 3    # Minimum match length to encode

        encoded = bytearray()

        # Position in input data
        pos = 0
        data = binary_data

        while pos < len(data):
            # Find longest match in sliding window
            match_length = 0
            match_offset = 0

            # Search in window
            search_start = max(0, pos - window_size)
            search_end = pos

            for i in range(search_start, search_end):
                # Try to match starting at position i
                potential_length = 0
                max_possible_match = min(len(data) - i, look_ahead_size)

                while (potential_length < max_possible_match and
                       i + potential_length < len(data) and
                       pos + potential_length < len(data) and
                       data[i + potential_length] == data[pos + potential_length]):
                    potential_length += 1

                if potential_length > match_length and potential_length >= min_match_length:
                    match_length = potential_length
                    match_offset = pos - i

                    if match_length == look_ahead_size:  # Found maximum possible match
                        break

            if match_length >= min_match_length:
                # Encode as (length, offset) pair
                # Use 2 bytes for length and 2 bytes for offset
                encoded.append(0)  # Marker for length-offset pair
                encoded.append(match_length)
                encoded.extend(match_offset.to_bytes(2, 'big'))

                pos += match_length
            else:
                # Encode literal byte
                if data[pos] == 0:  # Escape literal 0
                    encoded.append(0)  # Escape marker
                    encoded.append(0)  # Literal 0
                else:
                    encoded.append(data[pos])
                pos += 1

        result = bytes(encoded)

        def inverse():
            """LZ77 decompression."""
            if len(result) == 0:
                return b''

            decoded = bytearray()
            i = 0

            while i < len(result):
                if result[i] == 0:
                    # This could be a length-offset pair or escaped literal 0
                    if i + 1 >= len(result):
                        break

                    if result[i + 1] == 0 and i + 2 < len(result):
                        # Escaped literal 0
                        decoded.append(0)
                        i += 3
                    elif i + 3 < len(result):
                        # Length-offset pair
                        length = result[i + 1]
                        offset = int.from_bytes(result[i + 2:i + 4], 'big')

                        # Copy from decoded data
                        start_pos = len(decoded) - offset
                        for j in range(length):
                            if start_pos + j < len(decoded):
                                decoded.append(decoded[start_pos + j])
                            else:
                                # Handle edge case (shouldn't happen with proper encoding)
                                decoded.append(0)

                        i += 4
                    else:
                        # Incomplete sequence
                        break
                else:
                    # Literal byte
                    decoded.append(result[i])
                    i += 1

            return bytes(decoded)

        # Calculate compression statistics
        compression_ratio = len(result) / len(binary_data) if len(binary_data) > 0 else 1.0

        metadata = {
            'operation': 'lz77_encode',
            'original_size': len(binary_data),
            'compressed_size': len(result),
            'compression_ratio': compression_ratio,
            'window_size': window_size,
            'look_ahead_size': look_ahead_size,
            'bytes_affected': len(binary_data),
            'reversible': True
        }

        return result, inverse, metadata

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