cmake_minimum_required(VERSION 3.25)
project(nn_from_scratch)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE) 

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "-O0 -ggdb3 -g3 -fopenmp")
else()
    set(CMAKE_CXX_FLAGS "-O3 -march=native -fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fomit-frame-pointer -funroll-loops -funswitch-loops")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wno-parentheses -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wdouble-promotion -Wno-sign-conversion")

message(STATUS "CXX Flags are ${CMAKE_CXX_FLAGS}")

find_package(OpenMP REQUIRED)

include(ExternalProject)
set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

ExternalProject_Add(openfhe-development
    GIT_REPOSITORY https://github.com/openfheorg/openfhe-development
    GIT_TAG 02a8e9c76c3e2eff53392530199c63e4da53eb65
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} 
               -DBUILD_STATIC=ON 
               -DBUILD_SHARED=OFF
               -DBUILD_EXAMPLES=OFF 
               -DBUILD_EXTRAS=OFF 
               -DBUILD_BENCHMARKS=OFF 
               -DBUILD_UNITTESTS=OFF 
               -DWITH_NATIVEOPT=ON
               -DWITH_REDUCED_NOISE=ON
)

include_directories(${EXTERNAL_INSTALL_LOCATION}/include/openfhe/)
include_directories(${EXTERNAL_INSTALL_LOCATION}/include/openfhe/pke)
include_directories(${EXTERNAL_INSTALL_LOCATION}/include/openfhe/core)
include_directories(${EXTERNAL_INSTALL_LOCATION}/include/openfhe/binfhe)
include_directories(${EXTERNAL_INSTALL_LOCATION}/include/openfhe/cereal)
include_directories(${EXTERNAL_INSTALL_LOCATION}/include/openfhe/third-party/include)
link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)

add_library(NN SHARED 
    src/nn/nn_functionals.cpp
    src/nn/nn_classes.cpp
)
target_include_directories(NN PUBLIC include)
target_compile_options(NN PRIVATE -Wextra -Wpedantic -Wunused-parameter)

add_executable(Example 
    src/example.cpp
)
add_dependencies(Example openfhe-development)
target_include_directories(Example PUBLIC include)
target_include_directories(Example PUBLIC ${OPENMP_DIRECTORIES})
target_include_directories(Example PUBLIC ${EXTERNAL_INSTALL_LOCATION}/include/openfhe/)
target_include_directories(Example PUBLIC ${EXTERNAL_INSTALL_LOCATION}/include/openfhe/pke)
target_include_directories(Example PUBLIC ${EXTERNAL_INSTALL_LOCATION}/include/openfhe/core)
target_include_directories(Example PUBLIC ${EXTERNAL_INSTALL_LOCATION}/include/openfhe/binfhe)
target_include_directories(Example PUBLIC ${EXTERNAL_INSTALL_LOCATION}/include/openfhe/cereal)
target_include_directories(Example PUBLIC ${EXTERNAL_INSTALL_LOCATION}/include/openfhe/third-party/include)
target_link_directories(Example PUBLIC ${EXTERNAL_INSTALL_LOCATION}/lib)
target_link_directories(Example PUBLIC ${OPENMP_LIBRARIES})
target_link_libraries(Example PUBLIC OPENFHEpke_static OPENFHEcore_static OPENFHEbinfhe_static OpenMP::OpenMP_CXX)

add_library(NNFHE SHARED 
    src/nn_fhe/nn_functionals.cpp
    src/nn_fhe/nn_classes.cpp
)
add_dependencies(NNFHE openfhe-development)
target_include_directories(NNFHE PUBLIC include)
target_include_directories(NNFHE PUBLIC ${OPENMP_DIRECTORIES})
target_include_directories(NNFHE PUBLIC ${EXTERNAL_INSTALL_LOCATION}/include/openfhe/)
target_include_directories(NNFHE PUBLIC ${EXTERNAL_INSTALL_LOCATION}/include/openfhe/pke)
target_include_directories(NNFHE PUBLIC ${EXTERNAL_INSTALL_LOCATION}/include/openfhe/core)
target_include_directories(NNFHE PUBLIC ${EXTERNAL_INSTALL_LOCATION}/include/openfhe/binfhe)
target_include_directories(NNFHE PUBLIC ${EXTERNAL_INSTALL_LOCATION}/include/openfhe/cereal)
target_include_directories(NNFHE PUBLIC ${EXTERNAL_INSTALL_LOCATION}/include/openfhe/third-party/include)
target_link_directories(NNFHE PUBLIC ${EXTERNAL_INSTALL_LOCATION}/lib)
target_link_directories(NNFHE PUBLIC ${OPENMP_LIBRARIES})
target_link_libraries(NNFHE PUBLIC NN OPENFHEpke_static OPENFHEcore_static OPENFHEbinfhe_static OpenMP::OpenMP_CXX)
# target_compile_options(NNFHE PRIVATE -Wextra -Wpedantic -Wunused-parameter)

add_executable(MyTest src/main.cpp)
target_link_libraries(MyTest NN)
target_include_directories(MyTest PUBLIC include)

add_executable(MyTest_FHE src/main_fhe.cpp)
target_link_libraries(MyTest_FHE NNFHE)
target_include_directories(MyTest_FHE PUBLIC include)