#include "model.h"
#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <mutex>
#include <thread>
#include <array>
#include <atomic>
#include "llm_engine/engine/llm_engine.h"
#include "llm_engine/engine/engine_service.h"
#include "llm_engine/server/tcp_jsonl_server.h"
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

static std::string main_env_string(const char* name, const std::string& default_value) {
    const char* v = std::getenv(name);
    if (!v || !*v) return default_value;
    return std::string(v);
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

static bool run_service_multi_submit_test(
    EngineService& service,
    int max_new_tokens,
    bool debug_text,
    int test_abort_after_tokens
) {
    std::array<std::string, 3> prompts = {
        "鲁迅是谁",
        "李大钊是谁",
        "陈独秀是谁"
    };
    std::array<std::vector<int>, 3> encoded_prompts = {
        real_encode(prompts[0]),
        real_encode(prompts[1]),
        real_encode(prompts[2])
    };
    std::array<RequestId, 3> request_ids = {0, 0, 0};
    std::array<int, 3> token_counts = {0, 0, 0};
    std::mutex output_mu;
    std::atomic<bool> ok{true};

    SamplingParams params;
    params.max_new_tokens = max_new_tokens;
    params.greedy = true;

    std::vector<std::thread> threads;
    for (int i = 0; i < 3; ++i) {
        threads.emplace_back([&, i] {
            try {
                SessionId session_id = static_cast<SessionId>(i + 1);
                RequestId id = service.submit(session_id, encoded_prompts[(size_t)i], params, [&, i](int token_id) {
                    std::string piece = real_decode(token_id);
                    {
                        std::lock_guard<std::mutex> lk(output_mu);
                        if (debug_text) {
                            std::cerr << "[SERVICE_MULTI_TOKEN]"
                                      << " worker=" << i
                                      << " id=" << token_id
                                      << " piece=" << piece
                                      << std::endl;
                        }
                        std::cout << piece << std::flush;
                    }
                    token_counts[(size_t)i]++;
                    if (test_abort_after_tokens > 0 &&
                        token_counts[(size_t)i] >= test_abort_after_tokens) {
                        std::cerr << "[MAIN_TEST_ABORT]"
                                  << " worker=" << i
                                  << " after_tokens=" << test_abort_after_tokens
                                  << std::endl;
                        return false;
                    }
                    return true;
                });
                request_ids[(size_t)i] = id;
                std::cerr << "[SERVICE_MULTI_SUBMIT]"
                          << " worker=" << i
                          << " session=" << session_id
                          << " request=" << id
                          << std::endl;
            } catch (const std::exception& e) {
                ok.store(false);
                std::cerr << "[SERVICE_MULTI_ERROR]"
                          << " worker=" << i
                          << " error=" << e.what()
                          << std::endl;
            }
        });
    }

    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    for (size_t i = 0; i < request_ids.size(); ++i) {
        if (request_ids[i] == 0) {
            ok.store(false);
            continue;
        }
        service.wait_until_finished(request_ids[i]);
        EngineRequestSnapshot snapshot;
        if (service.request_snapshot(request_ids[i], &snapshot)) {
            std::cerr << "[SERVICE_MULTI_RESULT]"
                      << " worker=" << i
                      << " request=" << request_ids[i]
                      << " status=" << static_cast<int>(snapshot.status)
                      << " generated=" << snapshot.num_generated_tokens
                      << " error=" << snapshot.error_message
                      << std::endl;
            if (snapshot.status == RequestStatus::FAILED) {
                ok.store(false);
            }
        } else {
            ok.store(false);
            std::cerr << "[SERVICE_MULTI_RESULT]"
                      << " worker=" << i
                      << " request=" << request_ids[i]
                      << " status=missing"
                      << std::endl;
        }
    }

    std::cout << std::endl;
    return ok.load();
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

    // 1. 初始化引擎最核心的全局内存池。
    // Qwen2.5-1.5B 的 GPTQ packed weights + FP16 embedding 约 2GB+，
    // 默认给 3GB；可用 LLM_MEMORY_POOL_MB 覆盖。
    if (g_memory_pool == nullptr) {
        int pool_mb = main_env_int("LLM_MEMORY_POOL_MB", 3072);
        g_memory_pool = new MemoryPool((size_t)pool_mb * 1024ULL * 1024ULL);
    }

    QwenConfig cfg;
    QwenModel model(cfg);

    // 2. 加载权重
    std::cout << "正在加载模型权重..." << std::endl;
    if (!model.load_weights("../weights/qwen2p5_1p5b_fp16_gptq_int8_neon")) {
        std::cerr << "模型权重加载失败，程序退出。" << std::endl;
        return EXIT_FAILURE;
    }

    LLMEngine engine(model);

    bool is_first_turn = true;
    engine.clear_history(); // 初始化时清空一次

    bool use_service = main_env_flag("LLM_ENABLE_SERVICE");
    bool use_tcp_server = main_env_flag("LLM_ENABLE_TCP_SERVER");
    if (use_tcp_server && !use_service) {
        std::cerr << "[SERVER] LLM_ENABLE_TCP_SERVER=1 requires LLM_ENABLE_SERVICE=1"
                  << std::endl;
        return EXIT_FAILURE;
    }
    if (use_tcp_server && !engine.scheduler_enabled()) {
        std::cerr << "[SERVER] LLM_ENABLE_TCP_SERVER=1 requires LLM_ENABLE_SCHEDULER=1"
                  << std::endl;
        return EXIT_FAILURE;
    }
    std::unique_ptr<EngineService> service;
    if (use_service) {
        try {
            service = std::make_unique<EngineService>(engine);
            service->start();
        } catch (const std::exception& e) {
            std::cerr << "[SERVICE] start failed: " << e.what() << std::endl;
            return EXIT_FAILURE;
        }
    }

    int max_new_tokens = main_env_int("LLM_MAX_NEW_TOKENS", 512);
    if (use_tcp_server) {
        std::string host = main_env_string("LLM_SERVER_HOST", "0.0.0.0");
        int port = main_env_int("LLM_SERVER_PORT", 8080);
        TcpJsonlServer server(
            *service,
            [](const std::string& text) { return real_encode(text); },
            [](int token_id) { return real_decode(token_id); },
            max_new_tokens);
        bool ok = server.run_forever(host, port);
        service->stop();
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    std::cout << "=========================================" << std::endl;
    std::cout << "欢迎使用 Qwen2.5-1.5B FP16+GPTQ-Int8 CPU 推理引擎！" << std::endl;
    std::cout << "✅ 真实 Tokenizer (cpp-tiktoken) 已成功接入。" << std::endl;
    std::cout << "提示：输入 'exit' 或 'quit' 退出程序。" << std::endl;
    std::cout << "=========================================\n" << std::endl;

    // 3. 进入交互式对话大循环
    bool debug_text = main_env_flag("LLM_DEBUG_TEXT");
    bool stateless = main_env_flag("LLM_STATELESS");
    int test_abort_after_tokens = main_env_int("LLM_TEST_ABORT_AFTER_TOKENS", 0);
    bool test_service_multi_submit = main_env_flag("LLM_TEST_SERVICE_MULTI_SUBMIT");

    if (test_service_multi_submit) {
        if (!service) {
            std::cerr << "[SERVICE_MULTI_ERROR] LLM_TEST_SERVICE_MULTI_SUBMIT requires LLM_ENABLE_SERVICE=1"
                      << std::endl;
            return EXIT_FAILURE;
        }
        bool ok = run_service_multi_submit_test(
            *service,
            max_new_tokens,
            debug_text,
            test_abort_after_tokens);
        service->stop();
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    }

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
            if (service) {
                service->clear_history();
            } else {
                engine.clear_history();
            }
            is_first_turn = true;
            std::cout << "[System] History cleared." << std::endl;
            continue;
        }

        // LLM_STATELESS=1：每轮强制清空，单轮独立
        if (stateless) {
            if (service) {
                service->clear_history();
            } else {
                engine.clear_history();
            }
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

        SamplingParams params;
        params.max_new_tokens = max_new_tokens;
        params.greedy = true;

        auto callback = [&](int token_id) {
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
            if (test_abort_after_tokens > 0 && token_count >= test_abort_after_tokens) {
                std::cerr << "[MAIN_TEST_ABORT]"
                          << " after_tokens=" << test_abort_after_tokens
                          << std::endl;
                return false;
            }
            return true; // 返回 true 表示继续生成下一个字
        };

        RequestId req_id = 0;
        if (service) {
            req_id = service->submit(input_tokens, params, callback);
            service->wait_until_finished(req_id);
        } else {
            req_id = engine.submit(input_tokens, params, callback);
        }

        if (service) {
            EngineRequestSnapshot snapshot;
            if (service->request_snapshot(req_id, &snapshot) &&
                snapshot.status == RequestStatus::FAILED) {
                std::cerr << "[ERROR] request failed"
                          << " id=" << req_id
                          << " error=" << snapshot.error_message
                          << std::endl;
            }
        } else {
            const RequestState* request = engine.get_request(req_id);
            if (request && request->status == RequestStatus::FAILED) {
                std::cerr << "[ERROR] request failed"
                          << " id=" << req_id
                          << " error=" << request->error_message
                          << std::endl;
            }
        }

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        double tokens_per_sec = token_count / elapsed.count();

        std::cout << std::endl; // 回答结束后换行
        std::cout << "[用时 " << elapsed.count() << " 秒，生成 " << token_count
                  << " 个 token，速度 " << tokens_per_sec << " tok/s]" << std::endl;
    }

    if (service) {
        service->stop();
    }
    return EXIT_SUCCESS;
}
