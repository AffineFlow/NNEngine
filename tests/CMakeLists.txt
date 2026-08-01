include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.17.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

add_executable(nne_tests 
    cpp/test_tensor_tape.cpp
    cpp/test_autograd_ops.cpp
    cpp/test_eager_tape.cpp
    cpp/test_dataloader.cpp
    cpp/test_jit_graph.cpp
    cpp/test_implicit_tape.cpp
)

target_link_libraries(nne_tests PRIVATE NNEngine::Core GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(nne_tests)