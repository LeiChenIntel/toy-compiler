# TODO: Generate libraries from Toy

add_library(AddTensors8192 STATIC add_tensors.o)
set_target_properties(AddTensors8192 PROPERTIES LINKER_LANGUAGE CXX)

add_executable(benchmark-add-tensors benchmark-add-tensors.cpp)
add_dependencies(benchmark-add-tensors toy-translate)
if(MSVC)
    target_compile_options(benchmark-add-tensors PUBLIC "/arch:AVX2")
endif()
target_link_libraries(benchmark-add-tensors PRIVATE AddTensors8192)
install(TARGETS benchmark-add-tensors RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION lib)
