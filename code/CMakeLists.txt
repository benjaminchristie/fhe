cmake_minimum_required(VERSION 3.25)
project(he_nn)

set(CMAKE_CXX_FLAGS "-O3 -Wall -Werror -fopenmp -Wno-parentheses")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE) 
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

include(ExternalProject)
set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

ExternalProject_Add(openfhe-development
    GIT_REPOSITORY https://github.com/openfheorg/openfhe-development
    GIT_TAG origin/main
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} 
               -DBUILD_STATIC=ON 
               -DBUILD_EXAMPLES=OFF 
               -DBUILD_EXTRAS=OFF 
               -DBUILD_BENCHMARKS=OFF 
               -DBUILD_UNITTESTS=OFF 
               -DWITH_NATIVEOPT=OFF
)

include_directories(${EXTERNAL_INSTALL_LOCATION}/include/openfhe/)
include_directories(${EXTERNAL_INSTALL_LOCATION}/include/openfhe/pke)
include_directories(${EXTERNAL_INSTALL_LOCATION}/include/openfhe/core)
include_directories(${EXTERNAL_INSTALL_LOCATION}/include/openfhe/binfhe)
include_directories(${EXTERNAL_INSTALL_LOCATION}/include/openfhe/cereal)
include_directories(${EXTERNAL_INSTALL_LOCATION}/include/openfhe/third-party/include)
link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)

add_executable(FirstTest src/main.cpp)
add_dependencies(FirstTest openfhe-development)
target_link_libraries(FirstTest PUBLIC OPENFHEpke_static OPENFHEcore_static)

add_library(HFENN src/fhenn.cpp)
target_include_directories(HFENN PUBLIC include)

add_executable(HFENN_test src/tests/fhenn.cpp)
target_include_directories(HFENN_test PUBLIC include)
target_link_libraries(HFENN_test PUBLIC HFENN)