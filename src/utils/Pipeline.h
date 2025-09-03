#pragma once
#include "chunker/SimpleChunker.h"
#include "chunker/SmartChunker.h"
#include "embedder/OnnxEmbedder.h"
#include "vector_store/SimpleVectorStore.h"
#include "llm/LocalLLM.h"
#include <memory>

struct Pipeline {
    std::unique_ptr<IChunker> chunker;
    std::unique_ptr<IEmbedder> embedder;
    std::unique_ptr<IVectorStore> vector_store;
    std::unique_ptr<ILLM> llm;

  Pipeline(bool use_smart_chunker = true, size_t max_tokens = 400, size_t overlap_tokens = 80)
    : embedder(std::make_unique<OnnxEmbedder>()),
      vector_store(std::make_unique<SimpleVectorStore>()),
      llm(std::make_unique<LocalLLM>()) {
    if (use_smart_chunker) {
      auto tokenizer = std::make_shared<Tokenizer>(
        #ifdef BGE_VOCAB_PATH
        BGE_VOCAB_PATH
        #else
        "../third_party/bge-small-en/vocab.txt"
        #endif
      );
      chunker = std::make_unique<SmartChunker>(tokenizer, max_tokens, overlap_tokens);
    } else {
      chunker = std::make_unique<SimpleChunker>();
    }
  }
};
