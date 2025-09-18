CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -g -fPIC
LDFLAGS = 

# Source files
PARSER_SOURCES = parser.cpp
PARSER_OBJECTS = $(PARSER_SOURCES:.cpp=.o)
TEST_SOURCES = test_parser.cpp
TEST_OBJECTS = $(TEST_SOURCES:.cpp=.o)

# Targets
TEST_TARGET = test_parser
PYBIND_TARGET = parser_cpp

# Python comparison target
PYTHON_TARGET = compare_performance.py

.PHONY: all clean test performance install-deps pybind

all: pybind $(TEST_TARGET)

# Pybind11 module (primary target)
pybind: $(PYBIND_TARGET)

$(PYBIND_TARGET):
	@echo "Building pybind11 module..."
	cd . && python3 setup.py build_ext --inplace

# Test executable
$(TEST_TARGET): $(PARSER_OBJECTS) $(TEST_OBJECTS)
	$(CXX) $(PARSER_OBJECTS) $(TEST_OBJECTS) -o $(TEST_TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Install dependencies for Python comparison
install-deps:
	@echo "Installing Python dependencies..."
	@python3 -m pip install --user numpy matplotlib

# Run tests
test: $(TEST_TARGET)
	@echo "Running C++ parser tests..."
	./$(TEST_TARGET)

# Run performance comparison
performance: pybind $(PYTHON_TARGET)
	@echo "Running performance comparison..."
	python3 $(PYTHON_TARGET)

# Clean build files
clean:
	rm -f $(PARSER_OBJECTS) $(TEST_OBJECTS) $(TEST_TARGET)
	rm -f *.so parser_cpp*.so
	rm -rf build/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Dependencies
parser.o: parser.cpp parser.hpp
test_parser.o: test_parser.cpp parser.hpp