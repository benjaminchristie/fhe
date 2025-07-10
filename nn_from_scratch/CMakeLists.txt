cmake_minimum_required(VERSION 3.25)
project(nn_from_scratch)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-O0 -ggdb3 -g3
        -Wall -Werror -Wpedantic -Wextra 
        -Wconversion -Wdouble-promotion -Wno-unused-parameter -Wno-sign-conversion 
        # -fsanitize=address,undefined
    )
else()
    add_compile_options(-O3 
        -Wall -Werror -Wpedantic -Wextra 
        -Wconversion -Wdouble-promotion -Wno-unused-parameter -Wno-sign-conversion 
        # -fsanitize=address,undefined
    )
endif()

add_library(NN SHARED 
    src/nn/nn_functionals.cpp
    src/nn/nn_classes.cpp
)
target_include_directories(NN PUBLIC include)

add_executable(MyTest src/main.cpp)
target_link_libraries(MyTest NN)
target_include_directories(MyTest PUBLIC include)