# CMake build configuration for Vietnamese KWS Native Android Shared Library.
# Compiles the canonical C++ Log-Mel frontend and JNI bridge into libkws_mobile_frontend_jni.so.

cmake_minimum_required(VERSION 3.22.1)
project("kws_mobile_frontend_jni" CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to existing canonical mobile frontend codebase
get_filename_component(DEMO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)
set(MOBILE_FRONTEND_DIR "${DEMO_ROOT}/mobile_frontend")

message(STATUS "KWS Mobile Frontend source dir: ${MOBILE_FRONTEND_DIR}")

# Define the JNI shared library
add_library(kws_mobile_frontend_jni SHARED
    kws_jni_bridge.cpp
    "${MOBILE_FRONTEND_DIR}/src/kws_mobile_frontend.cpp"
)

# Header include paths
target_include_directories(kws_mobile_frontend_jni PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${MOBILE_FRONTEND_DIR}/include"
)

# Target compile definitions
target_compile_definitions(kws_mobile_frontend_jni PRIVATE
    KWS_FRONTEND_BUILD_DLL
)

# Link Android logging library if available (NDK build)
find_library(LOG_LIB log)
if (LOG_LIB)
    target_link_libraries(kws_mobile_frontend_jni PRIVATE ${LOG_LIB})
endif()
