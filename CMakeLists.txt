cmake_minimum_required(VERSION 3.12)
project(edgecrafter_inference CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_WARNING_FLAGS
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Wcast-align
    -Wunused
    -Woverloaded-virtual
    -Wconversion
    -Wsign-conversion
    -Wnull-dereference
    -Wdouble-promotion
    -Wformat=2
    -Wimplicit-fallthrough
)

option(WERROR "Treat compiler warnings as errors" OFF)
if(WERROR)
    list(APPEND PROJECT_WARNING_FLAGS -Werror)
endif()

option(SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if(SANITIZERS)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
    message(STATUS "Sanitizers: AddressSanitizer + UndefinedBehaviorSanitizer enabled")
endif()

option(USE_ONNX_RUNTIME "Build with ONNX Runtime backend support" ON)
option(USE_TENSORRT "Build with TensorRT backend support" OFF)
if(NOT USE_ONNX_RUNTIME AND NOT USE_TENSORRT)
    message(FATAL_ERROR "Enable at least one backend with USE_ONNX_RUNTIME=ON or USE_TENSORRT=ON")
endif()

message(STATUS "Backend configuration:")
message(STATUS "  ONNX Runtime: ${USE_ONNX_RUNTIME}")
message(STATUS "  TensorRT: ${USE_TENSORRT}")

if(USE_ONNX_RUNTIME)
    message(STATUS "Configuring ONNX Runtime backend...")

    set(ORT_VERSION "1.21.0")
    set(ORT_ARCHIVE_BASENAME "onnxruntime-linux-x64-${ORT_VERSION}")
    set(ORT_ARCHIVE_NAME "${ORT_ARCHIVE_BASENAME}.tgz")
    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ARCHIVE_NAME}")

    set(ORT_DEPS_DIR "${CMAKE_BINARY_DIR}/_deps")
    set(ORT_DOWNLOAD_PATH "${ORT_DEPS_DIR}/${ORT_ARCHIVE_NAME}")
    set(ORT_EXTRACT_DIR "${ORT_DEPS_DIR}")
    set(ORT_INSTALL_DIR "${ORT_EXTRACT_DIR}/${ORT_ARCHIVE_BASENAME}")

    if(NOT EXISTS "${ORT_INSTALL_DIR}/include/onnxruntime_cxx_api.h")
        message(STATUS "ONNX Runtime not found in build directory. Downloading and extracting...")
        file(MAKE_DIRECTORY "${ORT_DEPS_DIR}")
        if(NOT EXISTS "${ORT_DOWNLOAD_PATH}")
            message(STATUS "Downloading ONNX Runtime from ${ORT_URL}...")
            file(DOWNLOAD ${ORT_URL} ${ORT_DOWNLOAD_PATH}
                 SHOW_PROGRESS
                 TIMEOUT 600
                 STATUS download_status)
            list(GET download_status 0 download_code)
            list(GET download_status 1 download_message)
            if(NOT download_code EQUAL 0)
                message(FATAL_ERROR "Download failed: ${download_message}")
            endif()
            message(STATUS "Download completed: ${ORT_DOWNLOAD_PATH}")
        else()
            message(STATUS "Using existing ONNX Runtime archive: ${ORT_DOWNLOAD_PATH}")
        endif()
        message(STATUS "Extracting ${ORT_DOWNLOAD_PATH} to ${ORT_EXTRACT_DIR}...")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf "${ORT_DOWNLOAD_PATH}"
            WORKING_DIRECTORY "${ORT_EXTRACT_DIR}"
            RESULT_VARIABLE extract_result
        )
        if(NOT extract_result EQUAL 0)
            message(FATAL_ERROR "Extraction failed with result: ${extract_result}")
        endif()
        message(STATUS "Extraction completed.")
        if(NOT EXISTS "${ORT_INSTALL_DIR}/include/onnxruntime_cxx_api.h")
            message(FATAL_ERROR "Extraction failed: expected directory not found at ${ORT_INSTALL_DIR}")
        endif()
    else()
        message(STATUS "Found existing ONNX Runtime at: ${ORT_INSTALL_DIR}")
    endif()

    set(ONNXRUNTIME_ROOTDIR "${ORT_INSTALL_DIR}" CACHE PATH "Path to ONNX Runtime root directory")
    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOTDIR}/include")
    set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.so.${ORT_VERSION}")

    if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
        message(FATAL_ERROR "ONNX Runtime include directory not found at: ${ONNXRUNTIME_INCLUDE_DIR}")
    endif()
    if(NOT EXISTS "${ONNXRUNTIME_LIBRARY}")
        message(FATAL_ERROR "ONNX Runtime library not found at: ${ONNXRUNTIME_LIBRARY}")
    endif()

    message(STATUS "ONNX Runtime include directory: ${ONNXRUNTIME_INCLUDE_DIR}")
    message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")
endif()

if(USE_TENSORRT)
    set(TENSORRT_DIR "" CACHE PATH "Path to TensorRT root directory")
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS "${TENSORRT_DIR}" ENV TENSORRT_DIR
        PATH_SUFFIXES include)
    find_library(TENSORRT_LIBRARY nvinfer
        HINTS "${TENSORRT_DIR}" ENV TENSORRT_DIR
        PATH_SUFFIXES lib lib64)
    find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h
        HINTS ENV CUDA_HOME ENV CUDA_PATH /usr/local/cuda
        PATH_SUFFIXES include)
    find_library(CUDART_LIBRARY cudart
        HINTS ENV CUDA_HOME ENV CUDA_PATH /usr/local/cuda
        PATH_SUFFIXES lib lib64)

    if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
        message(FATAL_ERROR "TensorRT headers or library not found. Set TENSORRT_DIR to your TensorRT installation.")
    endif()
    if(NOT CUDA_INCLUDE_DIR OR NOT CUDART_LIBRARY)
        message(FATAL_ERROR "CUDA headers or cudart library not found. Set CUDA_HOME or CUDA_PATH.")
    endif()

    message(STATUS "TensorRT include directory: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "TensorRT library: ${TENSORRT_LIBRARY}")
    message(STATUS "CUDA include directory: ${CUDA_INCLUDE_DIR}")
    message(STATUS "CUDA runtime library: ${CUDART_LIBRARY}")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)
find_package(Threads REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Install via: sudo apt-get install libopencv-dev")
endif()
message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/src")
set(DATA_DIR "${CMAKE_SOURCE_DIR}/data")
file(MAKE_DIRECTORY "${DATA_DIR}")

set(EDGECRAFTER_SOURCES
    "${SOURCE_DIR}/edgecrafter_inference.cpp"
    "${SOURCE_DIR}/processing_utils.cpp"
    "${SOURCE_DIR}/backends/inference_backend.cpp"
)

if(USE_ONNX_RUNTIME)
    list(APPEND EDGECRAFTER_SOURCES "${SOURCE_DIR}/backends/onnx_runtime_backend.cpp")
endif()
if(USE_TENSORRT)
    list(APPEND EDGECRAFTER_SOURCES "${SOURCE_DIR}/backends/tensorrt_backend.cpp")
endif()

add_library(edgecrafter_inference_lib STATIC ${EDGECRAFTER_SOURCES})
target_compile_options(edgecrafter_inference_lib PRIVATE ${PROJECT_WARNING_FLAGS})

target_include_directories(edgecrafter_inference_lib PUBLIC "${SOURCE_DIR}")
target_include_directories(edgecrafter_inference_lib SYSTEM PUBLIC "${OpenCV_INCLUDE_DIRS}")

if(USE_ONNX_RUNTIME)
    target_include_directories(edgecrafter_inference_lib SYSTEM PUBLIC "${ONNXRUNTIME_INCLUDE_DIR}")
    target_compile_definitions(edgecrafter_inference_lib PUBLIC USE_ONNX_RUNTIME)
endif()
if(USE_TENSORRT)
    target_include_directories(edgecrafter_inference_lib SYSTEM PUBLIC "${TENSORRT_INCLUDE_DIR}" "${CUDA_INCLUDE_DIR}")
    target_compile_definitions(edgecrafter_inference_lib PUBLIC USE_TENSORRT)
endif()

target_link_libraries(edgecrafter_inference_lib
    PUBLIC
        "${OpenCV_LIBS}"
        Threads::Threads
)

if(USE_ONNX_RUNTIME)
    target_link_libraries(edgecrafter_inference_lib PUBLIC "${ONNXRUNTIME_LIBRARY}")
endif()
if(USE_TENSORRT)
    target_link_libraries(edgecrafter_inference_lib PUBLIC "${TENSORRT_LIBRARY}" "${CUDART_LIBRARY}")
endif()

add_executable(inference_app "${SOURCE_DIR}/main.cpp")
target_compile_options(inference_app PRIVATE ${PROJECT_WARNING_FLAGS})
target_link_libraries(inference_app PRIVATE edgecrafter_inference_lib)

if(USE_ONNX_RUNTIME)
    add_custom_command(TARGET inference_app POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_LIBRARY}"
                $<TARGET_FILE_DIR:inference_app>
        COMMENT "Copying ONNX Runtime library to output directory"
    )
endif()
