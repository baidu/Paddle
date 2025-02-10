# Compiling with WITH_PYTHON=ON and WITH_TENSORRT=ON failed on windows.
# Temporarily add paddle_inference_api dependency to solve the problem
if(WIN32)
  nv_library(
    tensorrt_engine
    SRCS engine.cc trt_int8_calibrator.cc
    DEPS ${GLOB_OPERATOR_DEPS} phi paddle_inference_api)
else()
  nv_library(
    tensorrt_engine
    SRCS engine.cc trt_int8_calibrator.cc
    DEPS ${GLOB_OPERATOR_DEPS} phi)
endif()
nv_library(
  tensorrt_dynamic_shape_infermeta_factory
  SRCS dynamic_shape_infermeta.cc
  DEPS phi)
nv_library(
  tensorrt_plugin_arg_mapping_context
  SRCS plugin_arg_mapping_context.cc
  DEPS phi)
nv_library(
  tensorrt_op_teller
  SRCS op_teller.cc
  DEPS phi tensorrt_dynamic_shape_infermeta_factory)
nv_test(
  test_tensorrt
  SRCS test_tensorrt.cc
  DEPS phi common)
if(WIN32)
  nv_test(
    test_tensorrt_engine
    SRCS test_engine.cc test_dynamic_engine.cc
    DEPS phi common tensorrt_engine tensorrt_plugin)
elseif(WITH_CINN)
  nv_test(
    test_tensorrt_engine
    SRCS test_engine.cc test_dynamic_engine.cc
    DEPS phi common tensorrt_engine tensorrt_plugin python)
else()
  nv_test(
    test_tensorrt_engine
    SRCS test_engine.cc test_dynamic_engine.cc
    DEPS phi common tensorrt_engine tensorrt_plugin python)
endif()
nv_test(
  test_arg_mapping_context
  SRCS test_arg_mapping_context.cc
  DEPS phi tensorrt_plugin_arg_mapping_context)

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(test_tensorrt_engine)
endif()

add_subdirectory(plugin)
add_subdirectory(convert)
add_subdirectory(pir)
