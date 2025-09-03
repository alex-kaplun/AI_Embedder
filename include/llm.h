#pragma once
#include <string>
#include <vector>

class ILLM {
public:
    virtual ~ILLM() = default;
    virtual std::string infer(const std::string& question, const std::vector<std::string>& context) const = 0;
};
