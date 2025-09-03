#pragma once
#include "embedder.h"
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <cmath>

#include <onnxruntime_cxx_api.h>
#include "Tokenizer.h"

class OnnxEmbedder : public IEmbedder {
public:
    OnnxEmbedder(const std::string& model_path_override = "")
    : env_(ORT_LOGGING_LEVEL_WARNING, "OnnxEmbedder"), session_(nullptr), allocator_(nullptr) {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    // Determine model path: prefer compile-time define, then override, then fallback relative path
    std::string model_path;
#ifdef BGE_MODEL_PATH
    model_path = BGE_MODEL_PATH;
#else
    model_path = model_path_override.empty() ? std::string("../third_party/bge-small-en/model.onnx") : model_path_override;
#endif
        std::cerr << "Using model path: " << model_path << "\n";
    // ONNX Runtime expects ORTCHAR_T*, which is wchar_t* on Windows. Convert to wide string.
        std::wstring wpath(model_path.begin(), model_path.end());
        try {
            session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), session_options);
            allocator_ = std::make_unique<Ort::AllocatorWithDefaultOptions>();
        } catch (const Ort::Exception& ex) {
            std::cerr << "ONNX Runtime failed to load model: " << ex.what() << "\nModel path: " << model_path << "\n";
            session_.reset();
        } catch (const std::exception& ex) {
            std::cerr << "Failed to initialize ONNX session: " << ex.what() << "\nModel path: " << model_path << "\n";
            session_.reset();
        }
    std::string vocab_path =
#ifdef BGE_VOCAB_PATH
        std::string(BGE_VOCAB_PATH);
#else
        std::string("../third_party/bge-small-en/vocab.txt");
#endif
    tokenizer_ = std::make_unique<Tokenizer>(vocab_path);
        if (!tokenizer_->ok()) {
            std::cerr << "Warning: failed to load vocab.txt for tokenizer at: " << vocab_path << "\n";
        } else {
            std::cerr << "Loaded vocab from: " << vocab_path << "\n";
        }
    }

    std::vector<float> embed(const std::string& text) const override {
        if (!session_) {
            // Fallback if model failed to load
            return std::vector<float>(384, 0.0f);
        }
        if (!tokenizer_ || !tokenizer_->ok()) {
            return std::vector<float>(384, 0.0f);
        }
        auto enc = tokenizer_->encode(text);
        const int64_t batch = 1;
        const int64_t seq = static_cast<int64_t>(enc.input_ids.size());
        std::array<int64_t,2> shape{batch, seq};
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_ids = Ort::Value::CreateTensor<int64_t>(mem, const_cast<int64_t*>(enc.input_ids.data()), enc.input_ids.size(), shape.data(), shape.size());
    Ort::Value attention_mask = Ort::Value::CreateTensor<int64_t>(mem, const_cast<int64_t*>(enc.attention_mask.data()), enc.attention_mask.size(), shape.data(), shape.size());
    // Some BERT-derived models require token_type_ids; provide zeros if not used
    std::vector<int64_t> token_type(seq, 0);
    Ort::Value token_type_ids = Ort::Value::CreateTensor<int64_t>(mem, token_type.data(), token_type.size(), shape.data(), shape.size());
    const char* input_names[] = {"input_ids", "attention_mask", "token_type_ids"};
    std::vector<Ort::Value> inputs;
    inputs.emplace_back(std::move(input_ids));
    inputs.emplace_back(std::move(attention_mask));
    inputs.emplace_back(std::move(token_type_ids));
        try {
            // Preferred for BGE: mean pooling over last_hidden_state with attention mask
            const char* hidden_out[] = {"last_hidden_state"};
            auto out2 = session_->Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(), hidden_out, 1);
            if (!out2.empty() && out2[0].IsTensor()) {
                auto info = out2[0].GetTensorTypeAndShapeInfo();
                auto dims = info.GetShape();
                if (dims.size() == 3 && dims[0] == 1 && dims[1] >= 1) {
                    const int64_t seq_len = dims[1];
                    const size_t hidden = static_cast<size_t>(dims[2]);
                    const float* h = out2[0].GetTensorData<float>(); // shape [1, seq, hidden]
                    // Compute masked mean across seq
                    std::vector<float> v(hidden, 0.f);
                    double denom = 0.0;
                    for (int64_t t = 0; t < seq_len; ++t) {
                        if (enc.attention_mask[static_cast<size_t>(t)] == 0) continue;
                        const float* row = h + static_cast<size_t>(t) * hidden;
                        for (size_t d = 0; d < hidden; ++d) v[d] += row[d];
                        denom += 1.0;
                    }
                    if (denom > 0.0) {
                        const float inv = static_cast<float>(1.0 / denom);
                        for (auto& x : v) x *= inv;
                    }
                    // L2 normalize
                    float s = 0.f; for (float x : v) s += x*x; if (s > 0) { s = std::sqrt(s); for (auto& x : v) x /= s; }
                    return v;
                }
            }
        } catch (const Ort::Exception& ex) {
            std::cerr << "ONNX inference error: " << ex.what() << "\n";
        } catch (const std::exception& ex) {
            std::cerr << "Inference error: " << ex.what() << "\n";
        }
        // Fallback: try pooler_output if available
        try {
            const char* pooled_out[] = {"pooler_output"};
            auto out1 = session_->Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(), pooled_out, 1);
            if (!out1.empty() && out1[0].IsTensor()) {
                auto info = out1[0].GetTensorTypeAndShapeInfo();
                size_t n = info.GetElementCount();
                const float* ptr = out1[0].GetTensorData<float>();
                std::vector<float> v(ptr, ptr + n);
                float s = 0.f; for (float x : v) s += x*x; if (s > 0) { s = std::sqrt(s); for (auto& x : v) x /= s; }
                return v;
            }
        } catch (...) {
        }
        return std::vector<float>(384, 0.0f);
    }

private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator_;
    std::unique_ptr<Tokenizer> tokenizer_;
};
