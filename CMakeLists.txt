cmake_minimum_required(VERSION 3.10)
project(OnnxParser VERSION 1.0 LANGUAGES CXX)

# Стандарт C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Пути к исходникам
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

# Заголовочные файлы
set(HEADERS
    ${INCLUDE_DIR}/bin_reader.h
    ${INCLUDE_DIR}/parser.h
)

# Исходные файлы
set(SOURCES
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/parser.cpp
)

# Создаём исполняемый файл
add_executable(parser ${SOURCES} ${HEADERS})

# Подключаем директорию с заголовками
target_include_directories(parser PRIVATE ${INCLUDE_DIR})

# === Тесты (опционально) ===
enable_testing()

# Тест 1: simple_matmul.onnx
if(EXISTS ${CMAKE_SOURCE_DIR}/tests/simple_matmul.onnx)
    add_test(NAME TestSimpleMatmul 
             COMMAND parser ${CMAKE_SOURCE_DIR}/tests/simple_matmul.onnx)
endif()

# Тест 2: complex_net.onnx
if(EXISTS ${CMAKE_SOURCE_DIR}/tests/complex_net.onnx)
    add_test(NAME TestComplexNet 
             COMMAND parser ${CMAKE_SOURCE_DIR}/tests/complex_net.onnx)
endif()

# Тест 3: custom_net.onnx
if(EXISTS ${CMAKE_SOURCE_DIR}/tests/custom_net.onnx)
    add_test(NAME TestCustomNet 
             COMMAND parser ${CMAKE_SOURCE_DIR}/tests/custom_net.onnx)
endif()


# Вывод информации
message(STATUS "")
message(STATUS "=== OnnxParser ===")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "")