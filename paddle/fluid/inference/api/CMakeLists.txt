if(APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=pessimizing-move")
endif()

add_subdirectory(details)

cc_library(
  paddle_infer_contrib
  SRCS paddle_infer_contrib.cc
  DEPS zero_copy_tensor)
cc_library(
  paddle_pass_builder
  SRCS paddle_pass_builder.cc
  DEPS framework_proto)

set(paddle_inference_api_deps
    reset_tensor_array
    paddle_infer_contrib
    paddle_pass_builder
    zero_copy_tensor
    trainer_desc_proto
    custom_operator
    lod_tensor
    scope
    drr)

if(WITH_CRYPTO)
  list(APPEND paddle_inference_api_deps framework_io)
endif()
if(WITH_CUSTOM_DEVICE)
  set(paddle_inference_api_deps ${paddle_inference_api_deps} phi common)
endif()

if(WIN32)
  cc_library(
    paddle_inference_api
    SRCS api.cc api_impl.cc helper.cc
    DEPS executor ${paddle_inference_api_deps})
else()
  cc_library(
    paddle_inference_api
    SRCS api.cc api_impl.cc helper.cc
    DEPS executor paddle_inference_io ${paddle_inference_api_deps})
endif()

cc_library(
  analysis_config
  SRCS analysis_config.cc
  DEPS paddle_inference_api paddle_pass_builder table_printer utf8proc)

if(WIN32)
  target_link_libraries(paddle_inference_api phi common)
endif()

set(inference_deps
    ${analysis_deps}
    paddle_inference_api
    analysis
    analysis_config
    naive_executor
    ${GLOB_PASS_LIB}
    pir_transforms)

if(WITH_GPU AND TENSORRT_FOUND)
  set(inference_deps ${inference_deps} tensorrt_engine tensorrt_converter)
endif()

if(WITH_OPENVINO)
  set(inference_deps ${inference_deps} openvino_engine)
endif()

set(ANALYSIS_PREDICTOR_SRCS analysis_predictor.cc resource_manager.cc
                            infer_context.cc)
set(ANALYSIS_PREDICTOR_DEPS ${inference_deps} zero_copy_tensor ir_pass_manager
                            op_compatible_info infer_io_utils model_utils)

if(WITH_ONNXRUNTIME)
  set(ANALYSIS_PREDICTOR_SRCS ${ANALYSIS_PREDICTOR_SRCS}
                              onnxruntime_predictor.cc)
  set(ANALYSIS_PREDICTOR_DEPS ${ANALYSIS_PREDICTOR_DEPS} onnxruntime
                              paddle2onnx)
elseif(WITH_CINN)
  set(ANALYSIS_PREDICTOR_DEPS ${ANALYSIS_PREDICTOR_DEPS} add_cinn_pass)
endif()
cc_library(
  analysis_predictor
  SRCS ${ANALYSIS_PREDICTOR_SRCS}
  DEPS ${ANALYSIS_PREDICTOR_DEPS})

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(test_paddle_inference_api)
endif()
