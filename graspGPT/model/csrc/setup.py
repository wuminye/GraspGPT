#!/usr/bin/env python3

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, Extension
import pybind11

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "parser_cpp",
        [
            "parser.cpp",
            "parser_pybind.cpp",
        ],
        include_dirs=[
            pybind11.get_cmake_dir() + "/../../../include",
        ],
        language="c++",
        cxx_std=17,
    ),
]

setup(
    name="parser_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)