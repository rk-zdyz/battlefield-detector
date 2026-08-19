cmake_minimum_required(VERSION 3.14)
project(battlefield_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Static linking for MinGW DLL self-containment on Windows
if(MINGW)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-libgcc -static-libstdc++ -static")
endif()

# Find pybind11 include directory via Python
execute_process(
    COMMAND python -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

# Find Python include directory
execute_process(
    COMMAND python -c "import sysconfig; print(sysconfig.get_path('include'))"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

# Find Python library path for Windows MinGW linking
execute_process(
    COMMAND python -c "import sys, os; print(os.path.join(sys.base_prefix, 'libs'))"
    OUTPUT_VARIABLE PYTHON_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

file(GLOB PYTHON_LIBS "${PYTHON_LIB_DIR}/python*.lib")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PYBIND11_INCLUDE_DIR}
    ${PYTHON_INCLUDE_DIR}
)

add_library(battlefield_core SHARED
    src/pybind_wrapper.cpp
)

set_target_properties(battlefield_core PROPERTIES
    PREFIX ""
    SUFFIX ".pyd"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../tactical_ui"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../tactical_ui"
)

if(PYTHON_LIBS)
    target_link_libraries(battlefield_core PRIVATE ${PYTHON_LIBS})
endif()
