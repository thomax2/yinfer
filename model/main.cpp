#include "model.h"
#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include "tiktoken/encoding.h"

using namespace llm_engine;

// 全局 Tokenizer 指针
std::shared_ptr<GptEncoding> tokenizer;

// 局部 env helpers，避免依赖 model 内部 private 静态方法
static bool main_env_flag(const char* name) {
    const char* v = std::getenv(name);
    if (!v) return false;
    std::string s(v);
    return s == "1" || s == "true" || s == "TRUE" || s == "on" || s == "ON";
}

static int main_env_int(const char* name, int default_value) {
    const char* v = std::getenv(name);
    if (!v) return default_value;
    char* end = nullptr;
    long x = std::strtol(v, &end, 10);
    if (end == v) return default_value;
    return static_cast<int>(x);
}


std::vector<int> real_encode(const std::string& text) {
    // 前导 \n 是为了在多轮拼接时形成正确的 ChatML 边界：
    // 上一轮模型刚把 <|im_end|> 写进 KV 缓存，本轮拼上 "\n<|im_start|>user..." 后，
    // 整段 KV 序列就是 "...<|im_end|>\n<|im_start|>user\n..."，与训练格式一致。
    // 第一轮多出来一个前导换行，对 Qwen2.5 来说是无伤大雅的扰动。
    std::string prompt = "\n<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
    
    // 1. 定义一个集合，显式列出你允许模型处理的特殊 Token
    std::unordered_set<std::string> allowed_special_tokens;
    allowed_special_tokens.insert("<|im_start|>");
    allowed_special_tokens.insert("<|im_end|>");

    // 2. 定义一个空的禁用集合
    // 注意：根据你提供的源码，只要这里面不包含 "all"，就不会因为发现特殊 Token 而报错
    std::unordered_set<std::string> disallowed_special_tokens;

    // 3. 传入这两个集合
    return tokenizer->encode(prompt, allowed_special_tokens, disallowed_special_tokens);
}

// 真实的 Decode：将模型输出的单个 Token ID 解码为人类语言
std::string real_decode(int token_id) {
    // decode 接口通常接收一个 vector
    std::vector<int> tokens = {token_id};
    return tokenizer->decode(tokens);
}

int main(int argc, const char** argv) {
    // 0. 初始化真正的 Tokenizer
    try {
        tokenizer = GptEncoding::get_encoding(LanguageModel::QWEN_BASE);
    } catch (const std::exception& e) {
        std::cerr << "[错误] Tokenizer 加载失败: " << e.what() << "\n"
                  << "请检查是否正确集成了 cpp-tiktoken 并且配置了词表！" << std::endl;
        return EXIT_FAILURE;
    }

    // 1. 初始化引擎最核心的全局内存池 (至少给 512MB)
    if (g_memory_pool == nullptr) {
        g_memory_pool = new MemoryPool(512ULL * 1024 * 1024);
    }

    QwenConfig cfg;
    QwenModel model(cfg);

    // 2. 加载权重
    std::cout << "正在加载模型权重..." << std::endl;
    if (!model.load_weights("../weights/qwen_0.5b_full_bins")) {
        std::cerr << "模型权重加载失败，程序退出。" << std::endl;
        return EXIT_FAILURE;
    }


    bool is_first_turn = true;
    model.clear_history(); // 初始化时清空一次

    std::cout << "=========================================" << std::endl;
    std::cout << "🚀 欢迎使用 Qwen-0.5B CPU 极速推理引擎！" << std::endl;
    std::cout << "✅ 真实 Tokenizer (cpp-tiktoken) 已成功接入。" << std::endl;
    std::cout << "提示：输入 'exit' 或 'quit' 退出程序。" << std::endl;
    std::cout << "=========================================\n" << std::endl;

    // 3. 进入交互式对话大循环
    bool debug_text = main_env_flag("LLM_DEBUG_TEXT");
    bool stateless = main_env_flag("LLM_STATELESS");
    int max_new_tokens = main_env_int("LLM_MAX_NEW_TOKENS", 512);

    while (true) {
        std::cout << "\nUser: ";
        std::string input;

        // 读取用户输入的一整行
        if (!std::getline(std::cin, input) || input == "exit" || input == "quit") {
            std::cout << "Bye!" << std::endl;
            break;
        }
        if (input.empty()) continue;

        // 【如果用户输入 clear，手动清空记忆】
        if (input == "clear") {
            model.clear_history();
            is_first_turn = true;
            std::cout << "[System] History cleared." << std::endl;
            continue;
        }

        // LLM_STATELESS=1：每轮强制清空，单轮独立
        if (stateless) {
            model.clear_history();
            is_first_turn = true;
            std::cerr << "[MAIN_STATELESS_CLEAR]" << std::endl;
        }

        if (debug_text) {
            std::cerr << "[MAIN_INPUT] text=" << input << std::endl;
        }

        // 【真实编码】：将中文文字转成机器看得懂的 Token 数组
        std::vector<int> input_tokens = real_encode(input);

        if (debug_text) {
            std::cerr << "[MAIN_TOKENS] count=" << input_tokens.size() << " ids=";
            for (size_t i = 0; i < input_tokens.size(); ++i) {
                if (i) std::cerr << ",";
                std::cerr << input_tokens[i];
            }
            std::cerr << std::endl;
        }

        std::cout << "Qwen: " << std::flush;

        // 【生成】：调用模型的 generate 进行推演
        int token_count = 0;
        auto start = std::chrono::steady_clock::now();

        model.generate(input_tokens, max_new_tokens, [&](int token_id) {
            // 【真实流式解码】：每当模型算出一个新 ID，立刻解码成中文打印到屏幕！
            std::string piece = real_decode(token_id);
            if (debug_text) {
                std::cerr << "[TEXT_TOKEN]"
                          << " id=" << token_id
                          << " piece=" << piece
                          << std::endl;
            }
            std::cout << piece << std::flush;
            token_count++;
            return true; // 返回 true 表示继续生成下一个字
        });

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        double tokens_per_sec = token_count / elapsed.count();

        std::cout << std::endl; // 回答结束后换行
        std::cout << "[用时 " << elapsed.count() << " 秒，生成 " << token_count
                  << " 个 token，速度 " << tokens_per_sec << " tok/s]" << std::endl;
    }

    return EXIT_SUCCESS;
}