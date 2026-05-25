cmake_minimum_required(VERSION 3.12)
project(edgecrafter_inference CXX)

# Set C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Compiler Warnings (applied to project targets only, not third-party) ---
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

# Treat warnings as errors in CI (opt-in via -DWERROR=ON)
option(WERROR "Treat compiler warnings as errors" OFF)
if(WERROR)
    list(APPEND PROJECT_WARNING_FLAGS -Werror)
endif()

# Enable sanitizers in debug builds (opt-in via -DSANITIZERS=ON)
option(SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if(SANITIZERS)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
    message(STATUS "Sanitizers: AddressSanitizer + UndefinedBehaviorSanitizer enabled")
endif()

# --- Backend Selection Options ---
option(USE_ONNX_RUNTIME "Build with ONNX Runtime backend support" ON)
if(NOT USE_ONNX_RUNTIME)
    message(FATAL_ERROR "EdgeCrafter inference currently requires ONNX Runtime. Set USE_ONNX_RUNTIME=ON")
endif()

message(STATUS "Backend configuration:")
message(STATUS "  ONNX Runtime: ${USE_ONNX_RUNTIME}")

# --- ONNX Runtime Configuration ---
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

# --- Find OpenCV ---
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)
find_package(Threads REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Install via: sudo apt-get install libopencv-dev")
endif()
message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# --- Source and Include Directories ---
set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/src")
set(TEST_DIR "${CMAKE_SOURCE_DIR}/tests")
set(DATA_DIR "${CMAKE_SOURCE_DIR}/data")

# Create the data directory if it doesn't exist
file(MAKE_DIRECTORY "${DATA_DIR}")

# --- Main Library ---
set(EDGECRAFTER_SOURCES
    "${SOURCE_DIR}/edgecrafter_inference.cpp"
    "${SOURCE_DIR}/processing_utils.cpp"
    "${SOURCE_DIR}/backends/inference_backend.cpp"
)

# Add backend-specific sources
list(APPEND EDGECRAFTER_SOURCES "${SOURCE_DIR}/backends/onnx_runtime_backend.cpp")

add_library(edgecrafter_inference_lib STATIC ${EDGECRAFTER_SOURCES})
target_compile_options(edgecrafter_inference_lib PRIVATE ${PROJECT_WARNING_FLAGS})

target_include_directories(edgecrafter_inference_lib
    PUBLIC
        "${SOURCE_DIR}"
)

# Third-party includes as SYSTEM to suppress warnings from their headers
target_include_directories(edgecrafter_inference_lib SYSTEM
    PUBLIC
        "${OpenCV_INCLUDE_DIRS}"
)

# Add backend-specific includes and compile definitions
target_include_directories(edgecrafter_inference_lib SYSTEM PUBLIC "${ONNXRUNTIME_INCLUDE_DIR}")
target_compile_definitions(edgecrafter_inference_lib PUBLIC USE_ONNX_RUNTIME)

# Link libraries
target_link_libraries(edgecrafter_inference_lib
    PUBLIC
        "${OpenCV_LIBS}"
        Threads::Threads
)

target_link_libraries(edgecrafter_inference_lib PUBLIC "${ONNXRUNTIME_LIBRARY}")

# --- Main Application ---
add_executable(inference_app "${SOURCE_DIR}/main.cpp")
target_compile_options(inference_app PRIVATE ${PROJECT_WARNING_FLAGS})
target_link_libraries(inference_app PRIVATE edgecrafter_inference_lib)

# Copy backend shared libraries to the executable directory
add_custom_command(TARGET inference_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_LIBRARY}"
            $<TARGET_FILE_DIR:inference_app>
    COMMENT "Copying ONNX Runtime library to output directory"
)
