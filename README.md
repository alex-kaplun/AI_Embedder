# Modern C++ Question Answering App

## Structure
- src/
  - main.cpp
  - chunker/
  - embedder/
  - vector_store/
  - llm/
  - utils/
- include/
- third_party/

## Build
- CMakeLists.txt

## Modules
- Chunker: Splits text into readable chunks
- Embedder: Uses ONNX Runtime + bge-small-en
- Vector Store: C++ only (Faiss)
- LLM: Interface + llama.cpp implementation

## Next Steps
- Implement chunker interface and simple splitter
- Scaffold interfaces for embedder, vector store, and LLM
