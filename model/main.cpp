#include "model.h"
#include <iostream>
#include <cstdlib>

using namespace llm_engine;

int main(int argc, const char** argv) {
    QwenConfig cfg;
    QwenModel model(cfg);

    if(!model.load_weights("weights")) {
        std::cerr << "模型权重加载失败，程序退出。" << std::endl;
        return EXIT_FAILURE;
    }
    return 0;
}