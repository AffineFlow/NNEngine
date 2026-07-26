pybind11_add_module(_backend NO_EXTRAS
    module.cpp
    core.cpp
    layers.cpp
    losses.cpp
    optimizers.cpp 
)

target_link_libraries(_backend PRIVATE NNEngine::Core)

install(TARGETS _backend DESTINATION ${CMAKE_SOURCE_DIR}/nnengine)