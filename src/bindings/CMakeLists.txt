foreach(ARCH IN LISTS MICROARCHS)

    pybind11_add_module(_backend_${ARCH} NO_EXTRAS
        module.cpp
        core.cpp
        layers.cpp
        losses.cpp
        optimizers.cpp
    )

    target_compile_definitions(_backend_${ARCH} PRIVATE BACKEND_NAME=_backend_${ARCH})

    # Automatically includes object files AND inherits compiler flags (/arch:AVX2, Eigen, etc.)
    target_link_libraries(_backend_${ARCH} PRIVATE engine_core_${ARCH})

    install(TARGETS _backend_${ARCH} DESTINATION nnengine)

endforeach()