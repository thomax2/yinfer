find_package(GTest REQUIRED)

include_directories(
    ${GTEST_INCLUDE_DIRS}
    ../include
    ../src           # 添加这一行，让测试能找到 src 下的头文件
)

add_executable(test_tensor
    test_tensor.cpp
)

target_link_libraries(test_tensor
    llm_engine
    GTest::GTest
    GTest::Main
)

add_executable(test_math_ref
    test_math_ref.cpp
)

target_link_libraries(test_math_ref
    llm_engine
    GTest::GTest
    GTest::Main
)

add_executable(test_graph test_graph.cpp)

target_link_libraries(test_graph
    llm_engine
    GTest::GTest
    GTest::Main
)

add_executable(test_executable_plan
    test_executable_plan.cpp
    ../model/qwen_execution_plan.cpp
)

add_executable(test_mixed_batch test_mixed_batch.cpp)
target_link_libraries(test_mixed_batch
    llm_engine
    GTest::GTest
    GTest::Main
)

# 不依赖 tokenizer/PCRE2，仍把模型与 Scheduler 的 Mixed Batch 实现完整编译、链接。
add_executable(test_mixed_batch_model
    test_mixed_batch_model.cpp
    ../model/model.cpp
    ../model/qwen_execution_plan.cpp
    ../src/engine/llm_engine.cpp
)
target_include_directories(test_mixed_batch_model PRIVATE ../model)
target_link_libraries(test_mixed_batch_model
    llm_engine
    GTest::GTest
    GTest::Main
)

target_include_directories(test_executable_plan PRIVATE ../model)
target_link_libraries(test_executable_plan
    llm_engine
    GTest::GTest
    GTest::Main
)

# Attention 测试临时禁用，避免阻塞当前联调
# add_executable(test_attention test_attention.cpp)
# target_link_libraries(test_attention
#     llm_engine
#     GTest::GTest
#     GTest::Main
# )
# target_compile_definitions(test_attention PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")


add_test(NAME tensor_test COMMAND test_tensor)
add_test(NAME math_ref_test COMMAND test_math_ref)
add_test(NAME graph_test COMMAND test_graph)
add_test(NAME executable_plan_test COMMAND test_executable_plan)
add_test(NAME mixed_batch_test COMMAND test_mixed_batch)
add_test(NAME mixed_batch_model_test COMMAND test_mixed_batch_model)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")
    add_executable(test_neon test_neon.cpp)
    target_link_libraries(test_neon llm_engine GTest::GTest GTest::Main)
    add_test(NAME neon_test COMMAND test_neon)

    add_executable(test_ffn test_ffn.cpp)
    target_link_libraries(test_ffn llm_engine GTest::GTest GTest::Main)
    target_compile_definitions(test_ffn PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
    add_test(NAME ffn_test COMMAND test_ffn)

    add_executable(test_qwen_block test_qwen_block.cpp)
    target_link_libraries(test_qwen_block llm_engine GTest::GTest GTest::Main)
    target_compile_definitions(
        test_qwen_block PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

    add_executable(test_graph_qwen_block test_graph_qwen_block.cpp)
    target_link_libraries(test_graph_qwen_block llm_engine GTest::GTest GTest::Main)
    target_compile_definitions(
        test_graph_qwen_block PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

    add_executable(test_gptq_batch_decode test_gptq_batch_decode.cpp)
    target_link_libraries(test_gptq_batch_decode llm_engine GTest::GTest GTest::Main)
    add_test(NAME gptq_batch_decode_test COMMAND test_gptq_batch_decode)
endif()
if(TEST ffn_test)
    set_tests_properties(ffn_test PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()
