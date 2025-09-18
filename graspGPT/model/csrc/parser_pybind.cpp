#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
#include "parser.hpp"
#include <chrono>

namespace py = pybind11;

// Convert C++ AST to Python dict (simplified)
py::object convert_ast_to_python(const std::unique_ptr<SeqNode>& seq_node) {
    if (!seq_node) {
        return py::none();
    }
    
    py::dict result;
    result["type"] = "SEQ";
    result["item_count"] = seq_node->items.size();
    
    // Add basic statistics about items
    int sb_count = 0, grasp_count = 0, unseg_count = 0, inpaint_count = 0, amodal_count = 0;
    
    for (const auto& item : seq_node->items) {
        if (dynamic_cast<SBNode*>(item.get())) {
            sb_count++;
        } else if (dynamic_cast<GRASPNode*>(item.get())) {
            grasp_count++;
        } else if (dynamic_cast<UNSEGNode*>(item.get())) {
            unseg_count++;
        } else if (dynamic_cast<INPAINTNode*>(item.get())) {
            inpaint_count++;
        } else if (dynamic_cast<AMODALNode*>(item.get())) {
            amodal_count++;
        }
    }
    
    result["sb_count"] = sb_count;
    result["grasp_count"] = grasp_count;
    result["unseg_count"] = unseg_count;
    result["inpaint_count"] = inpaint_count;
    result["amodal_count"] = amodal_count;
    
    return result;
}

// Parse function exposed to Python
py::object parse_tokens_cpp(const std::vector<py::object>& tokens) {
    try {
        // Convert Python tokens to C++ tokens
        std::vector<Token> cpp_tokens;
        
        for (const auto& token : tokens) {
            if (py::isinstance<py::str>(token)) {
                cpp_tokens.push_back(token.cast<std::string>());
            } else if (py::isinstance<py::tuple>(token)) {
                auto tuple = token.cast<py::tuple>();
                if (tuple.size() == 3) {
                    cpp_tokens.push_back(std::make_tuple(
                        tuple[0].cast<int>(),
                        tuple[1].cast<int>(),
                        tuple[2].cast<int>()
                    ));
                } else {
                    throw std::runtime_error("Coordinate tuple must have 3 elements");
                }
            } else {
                throw std::runtime_error("Invalid token type");
            }
        }
        
        // Parse with C++ parser
        Parser parser(cpp_tokens);
        auto result = parser.parse();
        
        // Convert result back to Python
        return convert_ast_to_python(result);
        
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("C++ parser error: ") + e.what());
    }
}

// Benchmark function
double benchmark_parser_cpp(const std::vector<py::object>& tokens, int iterations = 100) {
    try {
        // Convert Python tokens to C++ tokens
        std::vector<Token> cpp_tokens;
        
        for (const auto& token : tokens) {
            if (py::isinstance<py::str>(token)) {
                cpp_tokens.push_back(token.cast<std::string>());
            } else if (py::isinstance<py::tuple>(token)) {
                auto tuple = token.cast<py::tuple>();
                if (tuple.size() == 3) {
                    cpp_tokens.push_back(std::make_tuple(
                        tuple[0].cast<int>(),
                        tuple[1].cast<int>(),
                        tuple[2].cast<int>()
                    ));
                }
            }
        }
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            Parser parser(cpp_tokens);
            auto result = parser.parse();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        return duration.count() / 1000.0 / iterations;  // Return average time in microseconds
        
    } catch (const std::exception& e) {
        return -1.0;  // Error
    }
}

// Check if C++ parser is available
bool is_cpp_parser_available() {
    return true;
}

PYBIND11_MODULE(parser_cpp, m) {
    m.doc() = "C++ parser module using pybind11";
    
    m.def("parse_tokens", &parse_tokens_cpp, 
          "Parse tokens using C++ parser",
          py::arg("tokens"));
    
    m.def("benchmark_parser", &benchmark_parser_cpp,
          "Benchmark C++ parser performance",
          py::arg("tokens"), py::arg("iterations") = 100);
    
    m.def("is_available", &is_cpp_parser_available,
          "Check if C++ parser is available");
    
    // Expose some basic info
    m.attr("__version__") = "1.0.0";
}