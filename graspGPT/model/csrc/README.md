# C++ Parser Implementation (Pybind11)

This directory contains the C++ implementation of the GraspGPT parser using pybind11, providing significant performance improvements over the Python version.

## Files

- `parser.hpp` - Header file with class definitions
- `parser.cpp` - C++ implementation of the parser
- `parser_pybind.cpp` - Pybind11 wrapper for Python integration
- `setup.py` - Python setuptools configuration for building pybind11 module
- `Makefile` - Build configuration for compilation
- `test_parser.py` - Python script to test correctness and compare performance
- `compare_performance.py` - Detailed performance benchmarking script

## Usage

### Build Pybind11 Module
```bash
cd /home/wuminye/gitcode/GraspGPT/graspGPT/model/csrc
make pybind
```

### Use in Python
```python
from graspGPT.model.parser_and_serializer import parse_with_cpp

tokens = ["object01", (10, 20, 30), "end"]
result = parse_with_cpp(tokens)  # Uses C++ parser, raises error if unavailable
```

### Performance Comparison
```bash
python3 compare_performance.py
```

### Manual C++ Test
```bash
make test
```

## Features

- **Pybind11 Integration**: Seamless Python-C++ interop
- **Modern C++17**: Uses std::variant, std::optional, std::unique_ptr
- **Object-Oriented**: Each AST node is a separate class
- **Memory Safe**: Smart pointer memory management
- **Error Handling**: Raises exceptions on failure (no fallback)
- **Performance**: Significantly faster than Python implementation

## Requirements

- g++ with C++17 support
- Python 3 with pybind11
- Access to the original Python parser modules

## Installation

```bash
# Install pybind11 if not available
pip install pybind11

# Build the module
make pybind
```