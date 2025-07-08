cmake_minimum_required(VERSION 3.25)

project(fhe)

include(ExternalProject)

set(EXTERNAL_PROJECT_DIR external)

ExternalProject_Add(
	openfhe
	GIT_REPOSITORY "https://github.com/openfheorg/openfhe-development.git"
	GIT_TAG "main"
	GIT_PROGRESS ON
	# SOURCE_DIR "${CMAKE_BINARY_DIR}/3rdparty/openfhe/"
	# INSTALL_DIR "${CMAKE_BINARY_DIR}/external/include/"
	# BINARY_DIR "${CMAKE_BINARY_DIR}/external/lib/"
	CMAKE_ARGS -DBUILD_EXAMPLES=OFF 
			   -DBUILD_BENCHMARKS=OFF 
			   -DBUILD_UNITTESTS=OFF 
			   -DWITH_NATIVEOPT=ON 
			   -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/external/openfhe/
)
set(CMAKE_CXX_STANDARD 17)
option(BUILD_STATIC "Set to ON to include static versions of the library" OFF)

if(OpenFHE_FOUND)
    message(STATUS "FOUND PACKAGE OpenFHE")
    message(STATUS "OpenFHE Version: ${BASE_OPENFHE_VERSION}")
    message(STATUS "OpenFHE installed as shared libraries: ${OpenFHE_SHARED}")
    message(STATUS "OpenFHE include files location: ${OpenFHE_INCLUDE}")
    message(STATUS "OpenFHE lib files location: ${OpenFHE_LIBDIR}")
    message(STATUS "OpenFHE Native Backend size: ${OpenFHE_NATIVE_SIZE}")
	add_custom_target(Rescan)
	message(FATAL_ERROR "lol fuck you")
else()
	add_custom_target(Rescan ${CMAKE_COMMAND} ${CMAKE_SOURCE_DIR} DEPENDS openfhe)
	message(STATUS "alsj: ${CMAKE_BINARY_DIR}")
    # message(FATAL_ERROR "PACKAGE OpenFHE NOT FOUND")
endif()

set(CMAKE_CXX_FLAGS ${OpenFHE_CXX_FLAGS})

include_directories(${OPENMP_INCLUDES})
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/third-party/include)
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)

link_directories(${OpenFHE_LIBDIR})
link_directories(${OPENMP_LIBRARIES})
if(BUILD_STATIC)
    set(CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS} -static")
    link_libraries(${OpenFHE_STATIC_LIBRARIES})
else()
    set(CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS})
    link_libraries(${OpenFHE_SHARED_LIBRARIES})
endif()

# add_executable(MyTest src/main.cpp)
# add_dependencies(MyTest openfhe)
# add_dependencies(MyTest Rescan)
