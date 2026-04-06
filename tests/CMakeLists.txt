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

add_executable(test_neon test_neon.cpp)

target_link_libraries(test_neon
    llm_engine
    GTest::GTest
    GTest::Main
)

add_executable(test_attention test_attention.cpp)

target_link_libraries(test_attention
    llm_engine
    GTest::GTest
    GTest::Main
)

target_compile_definitions(test_attention PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")


add_executable(test_ffn test_ffn.cpp)

target_link_libraries(test_ffn
    llm_engine
    GTest::GTest
    GTest::Main
)

target_compile_definitions(test_ffn PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

add_executable(test_qwen_block test_qwen_block.cpp)

target_link_libraries(test_qwen_block
    llm_engine
    GTest::GTest
    GTest::Main
)

target_compile_definitions(test_qwen_block PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")


add_executable(test_graph_qwen_block test_graph_qwen_block.cpp)

target_link_libraries(test_graph_qwen_block
    llm_engine
    GTest::GTest
    GTest::Main
)

target_compile_definitions(test_graph_qwen_block PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

add_test(NAME tensor_test COMMAND test_tensor)
add_test(NAME math_ref_test COMMAND test_math_ref)
add_test(NAME graph_test COMMAND test_graph)
add_test(NAME neon_test COMMAND test_neon)
add_test(NAME attention_test COMMAND test_attention)
add_test(NAME ffn_test COMMAND test_ffn)
set_tests_properties(attention_test PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set_tests_properties(ffn_test PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})