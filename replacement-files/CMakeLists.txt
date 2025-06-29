cmake_minimum_required(VERSION 3.5.1)
project(polycircuit_examples CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_PREFIX_PATH 
    "${CMAKE_PREFIX_PATH}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../polycircuit-install"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../openfhe-development-install")

set(CMAKE_INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../openfhe-development-install/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Use a variable to store the OpenFHE include path
set(OPENFHE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../openfhe-development-install/include")
include_directories(${OPENFHE_INCLUDE_DIR})
link_directories(${OpenFHE_LIBDIR})
link_directories(${OPENMP_LIBRARIES})
link_directories(${CMAKE_SOURCE_DIR}/../../../openfhe-development-install/lib)

add_subdirectory(MatrixMultiplicationUsage)
add_subdirectory(CIFAR10Usage)
add_subdirectory(LogisticFunctionUsage)
add_subdirectory(SignEvaluationUsage) 