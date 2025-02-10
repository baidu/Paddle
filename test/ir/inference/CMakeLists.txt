file(
  GLOB TEST_INFERENCE_IR_PASSES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_INFERENCE_IR_PASSES "${TEST_INFERENCE_IR_PASSES}")

file(
  GLOB TEST_TRT_IR_PASSES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_trt_*.py")
string(REPLACE ".py" "" TEST_TRT_IR_PASSES "${TEST_TRT_IR_PASSES}")

file(
  GLOB TEST_TRT_CONVERTER
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_trt_convert_*.py")
string(REPLACE ".py" "" TEST_TRT_CONVERTER "${TEST_TRT_CONVERTER}")

list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES "test_use_optimized_model_api")
list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES "test_mkldnn_conv_gelu_fuse_pass")
list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES
     "test_mkldnn_conv_transpose_bias_fuse_pass")
list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES
     "test_onednn_batch_norm_act_fuse_pass")
list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES "test_onednn_conv_bn_fuse_pass")
list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES
     "test_onednn_conv_elementwise_add_fuse_pass")
list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES "test_onednn_multi_gru_fuse_pass")
list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES
     "test_onednn_reshape_transpose_matmul_fuse_pass")
list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES "test_save_optimized_model_pass")
list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES
     "test_trt_remove_amp_strategy_op_pass")
list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES "test_dist_llama_inference")

if(NOT WITH_DISTRIBUTE)
  list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES "test_delete_c_identity_op_pass")
  list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES "test_trt_convert_c_allreduce")
  list(REMOVE_ITEM TEST_TRT_IR_PASSES "test_trt_convert_c_allreduce")
  list(REMOVE_ITEM TEST_TRT_CONVERTER "test_trt_convert_c_allreduce")
endif()

if(WIN32)
  list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES
       "test_trt_convert_fused_token_prune")
  list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES
       "test_preln_groupnorm_act_fuse_pass")
  list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES
       "test_element_groupnorm_act_fuse_pass")
  list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES "test_groupnorm_act_pass_fuse_pass")
  list(REMOVE_ITEM TEST_TRT_IR_PASSES "test_trt_convert_trans_layernorm")
  list(REMOVE_ITEM TEST_TRT_CONVERTER "test_trt_convert_trans_layernorm")
  list(REMOVE_ITEM TEST_TRT_CONVERTER "test_trt_convert_argsort")
  list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES "test_trt_convert_trans_layernorm")
  list(REMOVE_ITEM TEST_TRT_IR_PASSES "test_trt_convert_fused_token_prune")
  list(REMOVE_ITEM TEST_TRT_CONVERTER "test_trt_convert_fused_token_prune")
  list(REMOVE_ITEM TEST_TRT_IR_PASSES
       "test_trt_convert_elementwiseadd_transpose")
  list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES
       "test_trt_convert_elementwiseadd_transpose")
  list(REMOVE_ITEM TEST_TRT_CONVERTER
       "test_trt_convert_elementwiseadd_transpose")

  list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES
       "test_split_layernorm_to_math_ops_pass")
  list(REMOVE_ITEM TEST_TRT_CONVERTER
       "test_trt_convert_quantize_dequantize_linear")
  list(REMOVE_ITEM TEST_TRT_IR_PASSES "test_trt_explicit_quantization")
  list(REMOVE_ITEM TEST_TRT_IR_PASSES "test_trt_explicit_quantization_resnet")
  list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES
       "test_trt_explicit_quantization_resnet")
  list(REMOVE_ITEM TEST_TRT_IR_PASSES
       "test_trt_explicit_quantization_mobilenet")
  list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES
       "test_trt_explicit_quantization_mobilenet")
endif()

foreach(TEST_INFERENCE_IR_PASS ${TEST_TRT_IR_PASSES})
  list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES ${TEST_INFERENCE_IR_PASS})
endforeach()

if(WITH_GPU AND TENSORRT_FOUND)
  list(REMOVE_ITEM TEST_TRT_IR_PASSES test_trt_multiclass_nms_op)

  foreach(TRT_CONVERT ${TEST_TRT_CONVERTER})
    list(REMOVE_ITEM TEST_TRT_IR_PASSES ${TRT_CONVERT})
  endforeach()

  foreach(target ${TEST_TRT_IR_PASSES})
    if(${target} STREQUAL "test_trt_slice_dynamic_plugin")
      if("${TENSORRT_MAJOR_VERSION}.${TENSORRT_MINOR_VERSION}" VERSION_GREATER
         "7.1")
        py_test_modules(${target} MODULES ${target})
        set_tests_properties(${target} PROPERTIES TIMEOUT 60)
        set_tests_properties(${target} PROPERTIES LABELS "RUN_TYPE=INFER")
      endif()
    else()
      py_test_modules(${target} MODULES ${target})
      set_tests_properties(${target} PROPERTIES LABELS "RUN_TYPE=INFER")
    endif()
  endforeach()

  foreach(target ${TEST_TRT_CONVERTER})
    py_test_modules(${target} MODULES ${target})
    set_tests_properties(${target} PROPERTIES TIMEOUT 300)
    set_tests_properties(${target} PROPERTIES LABELS "RUN_TYPE=INFER")
  endforeach()
endif()

file(
  GLOB TEST_MKLDNN_IR_PASSES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_mkldnn_*.py")
string(REPLACE ".py" "" TEST_MKLDNN_IR_PASSES "${TEST_MKLDNN_IR_PASSES}")
foreach(TEST_INFERENCE_IR_PASS ${TEST_MKLDNN_IR_PASSES})
  list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES ${TEST_INFERENCE_IR_PASS})
endforeach()

file(
  GLOB TEST_ONEDNN_IR_PASSES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_onednn_*.py")
string(REPLACE ".py" "" TEST_ONEDNN_IR_PASSES "${TEST_ONEDNN_IR_PASSES}")
foreach(TEST_INFERENCE_IR_PASS ${TEST_ONEDNN_IR_PASSES})
  list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES ${TEST_INFERENCE_IR_PASS})
endforeach()

if(WIN32 AND WIN_UNITTEST_LEVEL LESS 2)
  message(STATUS "Skip tests unrelated to CUDA/TRT")
elseif(WITH_ONEDNN)
  foreach(target ${TEST_MKLDNN_IR_PASSES})
    py_test_modules(${target} MODULES ${target})
    set_tests_properties(${target} PROPERTIES LABELS "RUN_TYPE=INFER")
  endforeach()

  foreach(target ${TEST_ONEDNN_IR_PASSES})
    py_test_modules(${target} MODULES ${target})
    set_tests_properties(${target} PROPERTIES LABELS "RUN_TYPE=INFER")
  endforeach()
endif()

file(
  GLOB TEST_XPU_IR_PASSES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_xpu_*.py")
string(REPLACE ".py" "" TEST_XPU_IR_PASSES "${TEST_XPU_IR_PASSES}")
foreach(TEST_XPU_IR_PASS ${TEST_XPU_IR_PASSES})
  list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES ${TEST_XPU_IR_PASS})
endforeach()

if(WITH_XPU)
  foreach(target ${TEST_XPU_IR_PASSES})
    py_test_modules(${target} MODULES ${target})
    set_tests_properties(${target} PROPERTIES LABELS "RUN_TYPE=INFER")
  endforeach()
endif()

if(WITH_ONEDNN
   AND TENSORRT_FOUND
   AND WITH_GPU)
  foreach(target ${TEST_INFERENCE_IR_PASSES})
    py_test_modules(${target} MODULES ${target})
    set_tests_properties(${target} PROPERTIES LABELS "RUN_TYPE=INFER")
  endforeach()
endif()

if(WITH_GPU AND TENSORRT_FOUND)
  # longer timeout for trt_activation_pass for longer trt optimization time in trt 8
  #set_tests_properties(test_trt_multiclass_nms_op PROPERTIES TIMEOUT 200)

  set_tests_properties(test_trt_ops_fp32_mix_precision PROPERTIES TIMEOUT 300)
  set_tests_properties(test_trt_convert_unary PROPERTIES TIMEOUT 600)
  set_tests_properties(test_trt_convert_temporal_shift_deprecated
                       PROPERTIES TIMEOUT 1000)

  if(NOT WIN32)

  else()
    set_tests_properties(test_trt_convert_fill_constant PROPERTIES TIMEOUT 450)
  endif()

  set_tests_properties(test_trt_conv3d_op PROPERTIES TIMEOUT 60)

  if(WITH_ONEDNN)
    set_tests_properties(test_merge_layernorm_fuse_pass PROPERTIES TIMEOUT 300)
    set_tests_properties(test_skip_merge_layernorm_fuse_pass PROPERTIES TIMEOUT
                                                                        180)
    set_tests_properties(test_trt_emb_eltwise_layernorm_fuse_pass
                         PROPERTIES TIMEOUT 180)

    set_tests_properties(test_fc_fuse_pass PROPERTIES TIMEOUT 240)
    set_tests_properties(test_reverse_roll_fuse_pass PROPERTIES TIMEOUT 120)
    set_tests_properties(test_inplace_op_pass PROPERTIES TIMEOUT 120)
    set_tests_properties(test_transfer_layout_elim_pass PROPERTIES TIMEOUT 300)

    set_tests_properties(test_simplify_with_basic_ops_pass_autoscan
                         PROPERTIES TIMEOUT 480)
    set_tests_properties(test_adaptive_pool2d_convert_global_pass_autoscan
                         PROPERTIES TIMEOUT 100)
    set_tests_properties(test_conv_act_onednn_fuse_pass PROPERTIES TIMEOUT 120)
    set_tests_properties(test_conv_elementwise_add2_act_fuse_pass
                         PROPERTIES TIMEOUT 120)
    set_tests_properties(test_conv_elementwise_add_act_fuse_pass
                         PROPERTIES TIMEOUT 120)
    set_tests_properties(test_flatten2_matmul_fuse_pass PROPERTIES TIMEOUT 240)
    set_tests_properties(test_squeeze2_matmul_fuse_pass PROPERTIES TIMEOUT 240)
    set_tests_properties(test_reshape2_matmul_fuse_pass PROPERTIES TIMEOUT 240)
    set_tests_properties(test_preln_layernorm_x_fuse_pass PROPERTIES TIMEOUT
                                                                     240)
    set_tests_properties(test_shuffle_channel_detect_pass PROPERTIES TIMEOUT
                                                                     120)
    set_tests_properties(test_conv_act_onednn_fuse_pass PROPERTIES TIMEOUT 120)
    set_tests_properties(test_conv_transpose_eltwiseadd_bn_fuse_pass
                         PROPERTIES TIMEOUT 250)
    set_tests_properties(test_conv_eltwiseadd_bn_fuse_pass PROPERTIES TIMEOUT
                                                                      300)
    if(WIN32)
      set_tests_properties(test_matmul_scale_fuse_pass PROPERTIES TIMEOUT 300)
      set_tests_properties(test_matmul_v2_scale_fuse_pass PROPERTIES TIMEOUT
                                                                     300)
      set_tests_properties(test_map_matmul_v2_to_matmul_pass PROPERTIES TIMEOUT
                                                                        360)
      set_tests_properties(test_map_matmul_v2_to_mul_pass PROPERTIES TIMEOUT
                                                                     360)
      set_tests_properties(test_map_matmul_to_mul_pass PROPERTIES TIMEOUT 360)
      set_tests_properties(test_layernorm_shift_partition_pass
                           PROPERTIES TIMEOUT 360)
      if(WIN_UNITTEST_LEVEL EQUAL 2)
        set_tests_properties(test_onednn_conv_bias_fuse_pass PROPERTIES TIMEOUT
                                                                        300)
        set_tests_properties(test_onednn_conv_concat_activation_fuse_pass
                             PROPERTIES TIMEOUT 300)
        set_tests_properties(test_onednn_multi_gru_seq_fuse_pass
                             PROPERTIES TIMEOUT 120)
      endif()
    else()
      set_tests_properties(test_matmul_scale_fuse_pass PROPERTIES TIMEOUT 60)
      set_tests_properties(test_matmul_v2_scale_fuse_pass PROPERTIES TIMEOUT 60)
      set_tests_properties(test_map_matmul_v2_to_matmul_pass PROPERTIES TIMEOUT
                                                                        120)
      set_tests_properties(test_map_matmul_v2_to_mul_pass PROPERTIES TIMEOUT
                                                                     120)
      set_tests_properties(test_map_matmul_to_mul_pass PROPERTIES TIMEOUT 120)
      set_tests_properties(test_element_groupnorm_act_fuse_pass
                           PROPERTIES TIMEOUT 120)
      set_tests_properties(test_preln_groupnorm_act_fuse_pass PROPERTIES TIMEOUT
                                                                         120)
      set_tests_properties(test_groupnorm_act_pass_fuse_pass PROPERTIES TIMEOUT
                                                                        120)
      set_tests_properties(test_split_layernorm_to_math_ops_pass
                           PROPERTIES TIMEOUT 240)
      set_tests_properties(test_inference_predictor_run_pir PROPERTIES TIMEOUT
                                                                       60)
    endif()
    if(WIN32 AND WIN_UNITTEST_LEVEL LESS 2)
      message(STATUS "Skip tests unrelated to CUDA/TRT")
    else()
      set_tests_properties(test_mkldnn_depthwise_conv_pass PROPERTIES TIMEOUT
                                                                      120)
      set_tests_properties(test_mkldnn_mish_op PROPERTIES TIMEOUT 300)
      set_tests_properties(test_mkldnn_conv3d_op PROPERTIES TIMEOUT 300)
      set_tests_properties(test_mkldnn_prelu_op PROPERTIES TIMEOUT 300)

      set_tests_properties(test_onednn_matmul_transpose_reshape_fuse_pass
                           PROPERTIES TIMEOUT 100)
      set_tests_properties(test_conv_transpose_bn_fuse_pass PROPERTIES TIMEOUT
                                                                       300)
      set_tests_properties(test_mkldnn_conv_hard_sigmoid_fuse_pass
                           PROPERTIES TIMEOUT 300)
      set_tests_properties(test_mkldnn_conv_hard_swish_fuse_pass
                           PROPERTIES TIMEOUT 300)
      set_tests_properties(test_mkldnn_matmul_v2_transpose_reshape_fuse_pass
                           PROPERTIES TIMEOUT 100)
      set_tests_properties(test_mkldnn_conv_mish_fuse_pass PROPERTIES TIMEOUT
                                                                      300)
      set_tests_properties(test_onednn_fc_activation_fuse_pass
                           PROPERTIES TIMEOUT 300)
      set_tests_properties(test_mkldnn_conv_affine_channel_fuse_pass
                           PROPERTIES TIMEOUT 60)
    endif()
  endif()
endif()

if(WIN32 AND WIN_UNITTEST_LEVEL LESS 2)
  message(STATUS "Skip tests unrelated to CUDA/TRT")
elseif(WITH_ONEDNN)

  set(PIR_COVERAGE_MKLDNN_TESTS
      test_mkldnn_conv_affine_channel_fuse_pass
      test_mkldnn_conv_gelu_fuse_pass
      test_mkldnn_conv_hard_sigmoid_fuse_pass
      test_mkldnn_conv_hard_swish_fuse_pass
      test_mkldnn_conv_mish_fuse_pass
      test_mkldnn_conv_transpose_bias_fuse_pass
      test_mkldnn_conv3d_op
      test_mkldnn_depthwise_conv_pass
      test_mkldnn_shape_op
      test_mkldnn_shuffle_channel_op)
  foreach(target ${PIR_COVERAGE_MKLDNN_TESTS})
    py_test_modules(${target}_pir MODULES ${target} ENVS FLAGS_enable_pir_api=1)
    set_tests_properties(${target} PROPERTIES LABELS "RUN_TYPE=INFER")
    message(STATUS "PIR Copied Test: ${target}_pir in inference test")
  endforeach()

  set(PIR_COVERAGE_ONEDNN_TESTS
      test_onednn_batch_norm_act_fuse_pass
      test_onednn_conv_bias_fuse_pass
      test_onednn_conv_bn_fuse_pass
      test_onednn_conv_concat_activation_fuse_pass
      test_onednn_conv_elementwise_add_fuse_pass
      test_onednn_elementwise_add_activation_fuse_pass
      test_onednn_fc_activation_fuse_pass
      test_onednn_fc_gru_fuse_pass
      test_onednn_fc_lstm_fuse_pass)
  foreach(target ${PIR_COVERAGE_ONEDNN_TESTS})
    py_test_modules(${target}_pir MODULES ${target} ENVS FLAGS_enable_pir_api=1)
    set_tests_properties(${target} PROPERTIES LABELS "RUN_TYPE=INFER")
    message(STATUS "PIR Copied Test: ${target}_pir in inference test")
  endforeach()

  set_tests_properties(test_mkldnn_shape_op_pir PROPERTIES TIMEOUT 300)
  set_tests_properties(test_mkldnn_shuffle_channel_op_pir PROPERTIES TIMEOUT
                                                                     300)
  set_tests_properties(test_onednn_conv_bias_fuse_pass_pir PROPERTIES TIMEOUT
                                                                      300)
  set_tests_properties(test_onednn_conv_concat_activation_fuse_pass_pir
                       PROPERTIES TIMEOUT 300)

endif()

if(WITH_GPU AND TENSORRT_FOUND)
  if(WITH_ONEDNN)
    py_test_modules(test_matmul_scale_fuse_pass_pir MODULES
                    test_matmul_scale_fuse_pass ENVS FLAGS_enable_pir_api=1)
    py_test_modules(
      test_multihead_matmul_fuse_pass_v3_pir MODULES
      test_multihead_matmul_fuse_pass_v3 ENVS FLAGS_enable_pir_api=1)
    py_test_modules(
      test_multihead_matmul_roformer_fuse_pass_pir MODULES
      test_multihead_matmul_roformer_fuse_pass ENVS FLAGS_enable_pir_api=1)
    py_test_modules(test_matmul_v2_scale_fuse_pass_pir MODULES
                    test_matmul_v2_scale_fuse_pass ENVS FLAGS_enable_pir_api=1)
    set_tests_properties(test_multihead_matmul_fuse_pass_v3_pir
                         PROPERTIES LABELS "RUN_TYPE=INFER")
    set_tests_properties(test_multihead_matmul_roformer_fuse_pass_pir
                         PROPERTIES LABELS "RUN_TYPE=INFER")
    set_tests_properties(test_multihead_matmul_fuse_pass_v3_pir
                         PROPERTIES TIMEOUT 300)
    set_tests_properties(test_multihead_matmul_roformer_fuse_pass_pir
                         PROPERTIES TIMEOUT 300)

    if(WIN32)
      set_tests_properties(test_matmul_scale_fuse_pass_pir PROPERTIES TIMEOUT
                                                                      300)
      set_tests_properties(test_matmul_v2_scale_fuse_pass_pir PROPERTIES TIMEOUT
                                                                         300)

      if(WIN_UNITTEST_LEVEL EQUAL 2)
        set_tests_properties(test_onednn_conv_bias_fuse_pass_pir
                             PROPERTIES TIMEOUT 600)
        set_tests_properties(test_onednn_conv_concat_activation_fuse_pass_pir
                             PROPERTIES TIMEOUT 600)
      endif()
    else()
      set_tests_properties(test_matmul_scale_fuse_pass_pir PROPERTIES TIMEOUT
                                                                      60)
      set_tests_properties(test_matmul_v2_scale_fuse_pass_pir PROPERTIES TIMEOUT
                                                                         120)
    endif()
    if(WIN32 AND WIN_UNITTEST_LEVEL LESS 2)
      message(STATUS "Skip tests unrelated to CUDA/TRT")
    else()

      set_tests_properties(test_mkldnn_conv_affine_channel_fuse_pass_pir
                           PROPERTIES TIMEOUT 120)
      set_tests_properties(test_mkldnn_conv_gelu_fuse_pass_pir
                           PROPERTIES TIMEOUT 300)
      set_tests_properties(test_mkldnn_conv_hard_sigmoid_fuse_pass_pir
                           PROPERTIES TIMEOUT 300)
      set_tests_properties(test_mkldnn_conv_hard_swish_fuse_pass_pir
                           PROPERTIES TIMEOUT 300)
      set_tests_properties(test_mkldnn_conv_mish_fuse_pass_pir
                           PROPERTIES TIMEOUT 300)
      set_tests_properties(test_mkldnn_conv_transpose_bias_fuse_pass_pir
                           PROPERTIES TIMEOUT 100)
      set_tests_properties(test_mkldnn_conv3d_op_pir PROPERTIES TIMEOUT 300)
      set_tests_properties(test_mkldnn_depthwise_conv_pass_pir
                           PROPERTIES TIMEOUT 120)

      set_tests_properties(test_onednn_conv_bn_fuse_pass_pir PROPERTIES TIMEOUT
                                                                        120)
      set_tests_properties(test_onednn_conv_elementwise_add_fuse_pass_pir
                           PROPERTIES TIMEOUT 120)
      set_tests_properties(test_onednn_batch_norm_act_fuse_pass_pir
                           PROPERTIES TIMEOUT 100)
      set_tests_properties(test_onednn_fc_activation_fuse_pass_pir
                           PROPERTIES TIMEOUT 300)
      set_tests_properties(test_onednn_elementwise_add_activation_fuse_pass_pir
                           PROPERTIES TIMEOUT 300)
      set_tests_properties(test_onednn_fc_gru_fuse_pass_pir PROPERTIES TIMEOUT
                                                                       300)
      set_tests_properties(test_onednn_fc_lstm_fuse_pass_pir PROPERTIES TIMEOUT
                                                                        300)
    endif()
  endif()
endif()

if(WITH_DISTRIBUTE
   AND WITH_GPU
   AND LINUX)
  py_test_modules(
    test_dist_llama_inference MODULES test_dist_llama_inference ENVS
    "http_proxy=;https_proxy=;PYTHONPATH=../..:${PADDLE_BINARY_DIR}/python")
  set_tests_properties(test_dist_llama_inference PROPERTIES TIMEOUT "300")
endif()
