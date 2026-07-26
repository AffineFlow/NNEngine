# 1. Detect if we are building for x86_64 or ARM64
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|amd64)")
    set(MICROARCHS "generic" "avx2" "avx512")
else()
    # ARM64 (Apple Silicon, Linux AArch64) uses standard NEON, no AVX fragmentation
    set(MICROARCHS "generic")
endif()

# 2. Loop through and build the required versions
foreach(ARCH IN LISTS MICROARCHS)
    
    pybind11_add_module(_backend_${ARCH} NO_EXTRAS
        module.cpp
        core.cpp
        layers.cpp
        losses.cpp
        optimizers.cpp 
    )

    target_compile_definitions(_backend_${ARCH} PRIVATE BACKEND_NAME=_backend_${ARCH})

    target_link_libraries(_backend_${ARCH} PRIVATE NNEngine::Core)

    # 3. Apply the compiler-specific hardware optimization flags
    if(ARCH STREQUAL "avx2")
        if(MSVC)
            target_compile_options(_backend_${ARCH} PRIVATE "/arch:AVX2")
        else()
            target_compile_options(_backend_${ARCH} PRIVATE "-mavx2" "-mfma")
        endif()
    elseif(ARCH STREQUAL "avx512")
        if(MSVC)
            target_compile_options(_backend_${ARCH} PRIVATE "/arch:AVX512")
        else()
            target_compile_options(_backend_${ARCH} PRIVATE "-mavx512f" "-mavx512cd" "-mavx512vl" "-mavx512bw" "-mavx512dq")
        endif()
    endif()

    # 4. Install each compiled binary variant into the python package directory
    install(TARGETS _backend_${ARCH} DESTINATION nnengine)

endforeach()