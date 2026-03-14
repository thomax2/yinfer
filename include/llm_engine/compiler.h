#pragma once

#include <vector>

#include "llm_engine/graph.h"

namespace llm_engine {

class GraphCompiler {
public:
    std::vector<GraphNode*> compile(ComputationGraph& graph);
};

class GraphRuntime {
public:
    Status run(const std::vector<GraphNode*>& plan);
};

}