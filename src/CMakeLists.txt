# 1. Collect all engine C++ sources using standard relative paths
set(ENGINE_SOURCES
    core/DataLoader.cpp
    core/JITGraph.cpp
    core/Module.cpp
    autograd/Tape.cpp
    autograd/ops/AddBiasOp.cpp
    autograd/ops/AddOp.cpp
    autograd/ops/SubOp.cpp
    autograd/ops/MulOp.cpp
    autograd/ops/DivOp.cpp
    autograd/ops/BatchNorm1dOp.cpp
    autograd/ops/Pool2dOp.cpp
    autograd/ops/Conv2dOp.cpp
    autograd/ops/FlattenOp.cpp
    autograd/ops/DropoutOp.cpp
    autograd/ops/LeakyReLUOp.cpp
    autograd/ops/MatMulOp.cpp
    autograd/ops/ReLUOp.cpp
    autograd/ops/ScalarOps.cpp
    autograd/ops/TransposeOp.cpp
    autograd/ops/ReductionOps.cpp
    autograd/ops/UnaryOps.cpp
    layers/BatchNorm1dLayer.cpp
    layers/Pool2dLayer.cpp
    layers/Conv2dLayer.cpp
    layers/FlattenLayer.cpp
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

# 2. Determine architecture target capabilities
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|amd64)")
    if(MSVC)
        set(MICROARCHS "generic" "avx2")
    else()
        set(MICROARCHS "generic" "avx2" "avx512")
    endif()
else()
    set(MICROARCHS "generic")
endif()

# 3. Create an OBJECT library per microarchitecture
foreach(ARCH IN LISTS MICROARCHS)

    add_library(engine_core_${ARCH} OBJECT ${ENGINE_SOURCES})

    set_target_properties(engine_core_${ARCH} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_include_directories(engine_core_${ARCH} PUBLIC ${CMAKE_SOURCE_DIR}/include)

    # Apply hardware optimization flags
    if(EMSCRIPTEN)
        target_compile_options(engine_core_${ARCH} PUBLIC "-msimd128")
    else()
        if(ARCH STREQUAL "avx2")
            if(MSVC)
                target_compile_options(engine_core_${ARCH} PUBLIC "/arch:AVX2")
            else()
                target_compile_options(engine_core_${ARCH} PUBLIC "-mavx2" "-mfma")
            endif()
        elseif(ARCH STREQUAL "avx512")
            if(MSVC)
                target_compile_options(engine_core_${ARCH} PUBLIC "/arch:AVX512")
            else()
                target_compile_options(engine_core_${ARCH} PUBLIC "-mavx512f" "-mavx512cd" "-mavx512vl" "-mavx512bw" "-mavx512dq")
            endif()
        endif()
    endif()

    # Link dependencies to the object target
    if(DEV_MODE)
        target_link_libraries(engine_core_${ARCH} PUBLIC Eigen3::Eigen)
    else()
        target_include_directories(engine_core_${ARCH} SYSTEM PUBLIC ${eigen_SOURCE_DIR})
    endif()

    if(APPLE)
        target_link_libraries(engine_core_${ARCH} PUBLIC "-framework Accelerate")
        target_compile_definitions(engine_core_${ARCH} PUBLIC EIGEN_USE_BLAS)
    elseif(NOT MSVC AND NOT EMSCRIPTEN)
        target_link_libraries(engine_core_${ARCH} PUBLIC BLAS::BLAS)
        target_compile_definitions(engine_core_${ARCH} PUBLIC EIGEN_USE_BLAS)
    endif()

    if(OpenMP_CXX_FOUND)
        target_link_libraries(engine_core_${ARCH} PUBLIC OpenMP::OpenMP_CXX)
        if(MSVC)
            target_compile_options(engine_core_${ARCH} PUBLIC /openmp:experimental)
        endif()
    endif()
endforeach()

# Convenient alias for tests linking against the baseline target
add_library(AffineFlow-NN::Core ALIAS engine_core_generic)

# Delegate to the Python bindings directory
add_subdirectory(bindings)