#include "utils/Pipeline.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <execution>

static std::string read_file_to_string(const std::string& path) {
    std::ifstream in(path, std::ios::in | std::ios::binary);
    if (!in) return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

int main(int argc, char** argv) {
    std::cout << "Modern C++ QA Demo (stub)\n";
    std::cout << "Usage: qa_app [optional_text_file] [optional_question]\n";

    // Input handling: if a file path is provided, read it; else use demo text
    std::string text;
    if (argc > 1) {
        text = read_file_to_string(argv[1]);
        if (text.empty()) {
            std::cerr << "Failed to read input file: " << argv[1] << "\n";
            return 1;
        }
    } else {
        text = "This is a test. Here is another sentence.\n\nAnd a new paragraph.";
    }

    std::string question;
    if (argc > 2) {
        question = argv[2];
    } else {
        question = "What is this text about?";
    }

    Pipeline pipeline;


    std::cout << "\n[1/5] Chunking..." << std::endl;
    auto chunks = pipeline.chunker->chunk(text);
    std::cout << "Created " << chunks.size() << " chunk(s)." << std::endl;
    for (size_t i = 0; i < chunks.size(); ++i) {
        std::cout << "  - Chunk " << i + 1 << ": " << chunks[i] << std::endl;
    }
    std::cout.flush();



    std::cout << "\n[2/5] Embedding chunks..." << std::endl;
    pipeline.vector_store->resize(chunks.size());
    const size_t chunks_pp = chunks.size() / 100;
    std::atomic<size_t> processed_chunks_pp{0};
    size_t progress{0};
    std::for_each(std::execution::par_unseq, chunks.begin(), chunks.end(), [&](const std::string& chunk) {
        try {
            std::string passage = std::string("passage: ") + chunk;
            auto emb = pipeline.embedder->embed(passage);
            pipeline.vector_store->add(emb, chunk);
            if (processed_chunks_pp++ == chunks_pp) {
                processed_chunks_pp = 0;
                size_t prog = ++progress;
                std::cout << "\r  - Processed " << prog << "% of chunks..." << std::flush;
            }
            // std::cout << "Chunk embedding: [";
            // for (size_t j = 0; j < std::min<size_t>(8, emb.size()); ++j) {
            //     std::cout << emb[j] << (j < 7 ? ", " : "");
            // }
            // std::cout << (emb.size() > 8 ? ", ..." : "") << "]" << std::endl;
        } catch (const std::exception& ex) {
            std::cerr << "Error embedding chunk: " << ex.what() << std::endl;
        }
    });
    std::cout << "Stored " << chunks.size() << " embedding(s)." << std::endl;
    std::cout.flush();



    std::cout << "\n[3/5] Embedding question..." << std::endl;
    std::vector<float> q_emb;
    try {
        std::string qtext = std::string("query: ") + question;
        q_emb = pipeline.embedder->embed(qtext);
        std::cout << "Question embedding: [";
        for (size_t j = 0; j < std::min<size_t>(8, q_emb.size()); ++j) {
            std::cout << q_emb[j] << (j < 7 ? ", " : "");
        }
        std::cout << (q_emb.size() > 8 ? ", ..." : "") << "]" << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error embedding question: " << ex.what() << std::endl;
    }
    std::cout.flush();


    std::cout << "\n[4/5] Retrieving relevant chunks..." << std::endl;
    const size_t k = 4;
    std::vector<std::string> relevant;
    try {
        relevant = pipeline.vector_store->query(q_emb, k);
        std::cout << "Top-" << k << " relevant chunk(s):" << std::endl;
        for (size_t i = 0; i < relevant.size(); ++i) {
            std::cout << "  #" << i + 1 << ": " << relevant[i] << std::endl;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error retrieving relevant chunks: " << ex.what() << std::endl;
    }
    std::cout.flush();


    std::cout << "\n[5/5] LLM inference (stub)..." << std::endl;
    std::string answer;
    try {
        answer = pipeline.llm->infer(question, relevant);
        std::cout << "Q: " << question << std::endl;
        std::cout << "A: " << answer << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error during LLM inference: " << ex.what() << std::endl;
    }
    std::cout.flush();

    std::cout << "\nStub run complete.\n";
    return 0;
}
