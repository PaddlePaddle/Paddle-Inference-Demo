cmake_minimum_required(VERSION 3.0)
project(cpp_inference_demo CXX C)
option(WITH_IPU        "Compile demo with IPU/CPU, default use CPU."                    OFF)
option(CUSTOM_OPERATOR_FILES "List of file names for custom operators" "")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(NOT WITH_STATIC_LIB)
  add_definitions("-DPADDLE_WITH_SHARED_LIB")
else()
  # PD_INFER_DECL is mainly used to set the dllimport/dllexport attribute in dynamic library mode.
  # Set it to empty in static library mode to avoid compilation issues.
  add_definitions("/DPD_INFER_DECL=")
endif()

macro(safe_set_static_flag)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
endmacro()

if(NOT DEFINED PADDLE_LIB)
  message(FATAL_ERROR "please set PADDLE_LIB with -DPADDLE_LIB=/path/paddle/lib")
endif()
if(NOT DEFINED DEMO_NAME)
  message(FATAL_ERROR "please set DEMO_NAME with -DDEMO_NAME=demo_name")
endif()

include_directories("${PADDLE_LIB}/paddle/include")
set(PADDLE_LIB_THIRD_PARTY_PATH "${PADDLE_LIB}/third_party/install/")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}glog/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}gflags/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}onnxruntime/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}paddle2onnx/include")

link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}glog/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}gflags/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/lib")
link_directories("${PADDLE_LIB}/paddle/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}onnxruntime/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}paddle2onnx/lib")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
if(WITH_IPU)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DONNX_NAMESPACE=onnx")
endif()

set(DEPS ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX})
set(EXTERNAL_LIB "-lrt -ldl -lpthread")
set(DEPS ${DEPS}
      glog gflags protobuf xxhash
      ${EXTERNAL_LIB})

add_library(pd_infer_custom_op SHARED ${CUSTOM_OPERATOR_FILES})
add_executable(${DEMO_NAME} ${DEMO_NAME}.cc)
set(DEPS ${DEPS} pd_infer_custom_op)
target_link_libraries(${DEMO_NAME} ${DEPS})
