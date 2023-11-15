cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(main-app src/main.cpp)
target_link_libraries(main-app "${TORCH_LIBRARIES}")
set_property(TARGET main-app PROPERTY CXX_STANDARD 17)