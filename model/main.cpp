#include "model.h"
#include <iostream>
#include <cstdlib>

using namespace llm_engine;

// 假装是 Tokenizer：等下一步我们实现了真正的 BPE 再来替换它
std::vector<int> mock_encode(const std::string& text) {
    // 无论输入什么，我们暂时固定返回两个打底的 Token，假装是 Prompt
    // (108386 和 10413 在 Qwen 词表里可能代表某些标点或常用字)
    return {108386, 10413}; 
}

std::string mock_decode(int token_id) {
    // 既然还没法变成中文，我们就直接把它格式化打印出来
    return " [Token:" + std::to_string(token_id) + "] ";
}

int main(int argc, const char** argv) {
    // 1. 初始化引擎最核心的全局内存池 (至少给 512MB)
    if (g_memory_pool == nullptr) {
        g_memory_pool = new MemoryPool(512ULL * 1024 * 1024);
    }

    QwenConfig cfg;
    QwenModel model(cfg);

    // 2. 加载权重
    if (!model.load_weights("../weights/qwen_0.5b_full_bins")) {
        std::cerr << "模型权重加载失败，程序退出。" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "=========================================" << std::endl;
    std::cout << "欢迎使用 Qwen-0.5B CPU 极速推理引擎！" << std::endl;
    std::cout << "提示：输入 'exit' 退出程序。" << std::endl;
    std::cout << "注意：真正的 Tokenizer 尚未接入，目前只能输出生成的 Token ID。" << std::endl;
    std::cout << "=========================================\n" << std::endl;

    // 3. 进入交互式对话大循环
    while (true) {
        std::cout << "\nUser: ";
        std::string input;
        
        // 读取用户输入的一整行
        if (!std::getline(std::cin, input) || input == "exit" || input == "quit") {
            std::cout << "Bye!" << std::endl;
            break;
        }

        if (input.empty()) continue;

        // 【编码】：将文字转成机器看得懂的数组
        std::vector<int> input_tokens = mock_encode(input);
        
        std::cout << "Qwen: " << std::flush; // flush 确保提示符立即打印
        
        // 【生成】：调用模型的 generate 进行推演
        model.generate(input_tokens, 50, [](int token_id) {
            // 【流式回调】：每当模型算出一个新 ID，立刻打印到屏幕！
            std::cout << mock_decode(token_id) << std::flush;
            return true; // 返回 true 表示继续生成下一个字
        });
        
        std::cout << std::endl; // 这一句结束换行
    }

    return EXIT_SUCCESS;
}