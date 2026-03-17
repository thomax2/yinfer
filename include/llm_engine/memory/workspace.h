#pragma once

#include <cstddef>

namespace llm_engine {

class Workspace {

public:

    Workspace(size_t bytes);

    void* data();
    size_t size();

private:

    void* ptr;
    size_t capacity;
};

}