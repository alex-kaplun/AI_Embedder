# Build the QA app executable

set(SRC_FILES
    ${CMAKE_SOURCE_DIR}/src/main.cpp
    ${CMAKE_SOURCE_DIR}/src/chunker/SmartChunker.cpp
)

add_executable(qa_app ${SRC_FILES})

# Include project headers and ONNX Runtime headers
target_include_directories(qa_app PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/include
)

# Link ONNX Runtime libraries via absolute paths (robust with spaces in paths)
set(ORT_LIB_DIR ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib)
target_link_libraries(qa_app PRIVATE
    ${ORT_LIB_DIR}/onnxruntime.lib
    ${ORT_LIB_DIR}/onnxruntime_providers_shared.lib
)

# Organize output to bin/
set_target_properties(qa_app PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Copy required ONNX Runtime DLLs next to the executable after build
add_custom_command(TARGET qa_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:qa_app>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib/onnxruntime.dll
        ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib/onnxruntime_providers_shared.dll
        $<TARGET_FILE_DIR:qa_app>
)

# Pass the model path to the code so it can load via wide-char API
target_compile_definitions(qa_app PRIVATE BGE_MODEL_PATH="${BGE_SMALL_EN_MODEL}")

# Pass the vocab path similarly so Tokenizer can load reliably
target_compile_definitions(qa_app PRIVATE BGE_VOCAB_PATH="${CMAKE_SOURCE_DIR}/third_party/bge-small-en/vocab.txt")
