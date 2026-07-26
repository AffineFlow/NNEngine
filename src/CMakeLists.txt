add_library(NNEngine STATIC
    core/DataLoader.cpp
    core/JITGraph.cpp
    core/Module.cpp
    autograd/Tape.cpp
    autograd/ops/AddBiasOp.cpp
    autograd/ops/BatchNorm1dOp.cpp
    autograd/ops/Conv2dOp.cpp
    autograd/ops/DropoutOp.cpp
    autograd/ops/LeakyReLUOp.cpp
    autograd/ops/MatMulOp.cpp
    autograd/ops/ReLUOp.cpp
    layers/BatchNorm1dLayer.cpp
    layers/Conv2dLayer.cpp
    layers/DenseLayer.cpp
    layers/DropoutLayer.cpp
    layers/LeakyReLULayer.cpp
    layers/ReLULayer.cpp
    losses/MSELoss.cpp
    losses/SoftmaxCrossEntropyLoss.cpp
    optimizers/Adam.cpp
    optimizers/AdamW.cpp
    optimizers/SGD.cpp
    regularizers/L2Regularizer.cpp
)

set_target_properties(NNEngine PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(NNEngine::Core ALIAS NNEngine)

# Point to the include directory at the root
target_include_directories(NNEngine PUBLIC ${CMAKE_SOURCE_DIR}/include)

if(DEV_MODE)
    target_link_libraries(NNEngine PUBLIC Eigen3::Eigen)
else()
    target_include_directories(NNEngine SYSTEM PUBLIC ${eigen_SOURCE_DIR})
endif()

if(APPLE)
    target_link_libraries(NNEngine PUBLIC "-framework Accelerate")
    target_compile_definitions(NNEngine PUBLIC EIGEN_USE_BLAS)
elseif(NOT MSVC)
    if(DEV_MODE)
        target_link_libraries(NNEngine PUBLIC BLAS::BLAS)
    else()
        target_link_libraries(NNEngine PUBLIC openblas)
    endif()
    target_compile_definitions(NNEngine PUBLIC EIGEN_USE_BLAS)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(NNEngine PUBLIC OpenMP::OpenMP_CXX)
    if(MSVC)
        target_compile_options(NNEngine PUBLIC /openmp:experimental)
    endif()
endif()

# Delegate to the Python bindings directory
add_subdirectory(bindings)