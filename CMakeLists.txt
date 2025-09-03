# CMake minimum version
cmake_minimum_required(VERSION 3.16)

# Project name
project(LLamaCppTest1)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(MSVC)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# Include directories
include_directories(include)

# ONNX Runtime
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

# bge-small-en model path (must be set before adding src so compile definitions get a value)
set(BGE_SMALL_EN_MODEL "${CMAKE_SOURCE_DIR}/third_party/bge-small-en/model.onnx")

# Source directories
add_subdirectory(src)
