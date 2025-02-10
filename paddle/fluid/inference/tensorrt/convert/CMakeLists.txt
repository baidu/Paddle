# Add TRT tests
list(
  APPEND
  CONVERT_FILES
  matrix_multiply_op.cc
  bmm_op.cc
  conv2d_op.cc
  pool2d_op.cc
  elementwise_op.cc
  batch_norm_op.cc
  activation_op.cc
  unary_op.cc
  softmax_op.cc
  concat_op.cc
  dropout_op.cc
  group_norm_op.cc
  pad3d_op.cc
  pad_op.cc
  split_op.cc
  square_op.cc
  prelu_op.cc
  leaky_relu_op.cc
  gelu_op.cc
  layer_norm_op.cc
  multihead_matmul_op.cc
  multihead_matmul_roformer_op.cc
  flash_multihead_matmul_op.cc
  cross_multihead_matmul_op.cc
  qk_multihead_matmul_op.cc
  grid_sampler_op.cc
  shuffle_channel_op.cc
  fill_any_like_op.cc
  where_op.cc
  bitwise_not_op.cc
  one_hot_op.cc
  swish_op.cc
  silu_op.cc
  instance_norm_op.cc
  stack_op.cc
  transpose_op.cc
  flatten_op.cc
  flatten_contiguous_range_op.cc
  skip_layernorm.cc
  scale_op.cc
  slice_op.cc
  hard_sigmoid_op.cc
  hard_swish_op.cc
  clip_op.cc
  gather_op.cc
  index_select_op.cc
  anchor_generator_op.cc
  yolo_box_op.cc
  yolo_box_head_op.cc
  arg_max_op.cc
  arg_min_op.cc
  roi_align_op.cc
  affine_channel_op.cc
  multiclass_nms_op.cc
  multiclass_nms3_op.cc
  nearest_interp_op.cc
  reshape_op.cc
  reduce_op.cc
  gather_nd_op.cc
  tile_op.cc
  conv3d_op.cc
  mish_op.cc
  nearest_interp_v2_op.cc
  bilinear_interp_v2_op.cc
  pool3d_op.cc
  deformable_conv_op.cc
  strided_slice_op.cc
  preln_skip_layernorm.cc
  roll_op.cc
  transformer_input_convert_op.cc
  cast_op.cc
  remove_padding_op.cc
  equal_op.cc
  recover_padding_op.cc
  preln_residual_bias.cc
  c_allreduce_op.cc
  top_k_op.cc
  range_op.cc
  squeeze2_op.cc
  unsqueeze2_op.cc
  rnn_op.cc
  fill_constant_batch_size_like_op.cc
  sum_op.cc
  bitwise_and_op.cc
  bitwise_or_op.cc
  size_op.cc
  shape_op.cc
  fill_constant_op.cc
  fused_token_prune_op.cc
  celu_op.cc
  layernorm_shift_partition_op.cc
  reverse_roll_op.cc
  tanhshrink_op.cc
  take_along_axis_op.cc
  linear_interp_v2_op.cc
  logsigmoid_op.cc
  preln_layernorm_shift_partition_op.cc
  trans_layernorm_op.cc
  merge_layernorm_op.cc
  skip_merge_layernorm_op.cc
  generic_and_custom_plugin_creater.cc
  lookup_table_op.cc
  elementwiseadd_transpose_op.cc
  skip_groupnorm_act_op.cc
  preln_groupnorm_act_op.cc
  expand_v2_op.cc
  argsort_op.cc
  cumsum_op.cc
  temporal_shift_op.cc
  einsum_op.cc
  unbind_op.cc
  index_put_op.cc
  isnan_v2_op.cc
  assign_op.cc
  p_norm_op.cc
  flip_op.cc
  quantize_linear_op.cc
  dequantize_linear_op.cc
  share_data_op.cc)

if(${TENSORRT_MAJOR_VERSION} GREATER_EQUAL 7)
  list(APPEND CONVERT_FILES emb_eltwise_layernorm.cc
       preln_emb_eltwise_layernorm.cc prompt_tuning_emb_eltwise_layernorm.cc)
endif()

if(CUSPARSELT_FOUND AND ${TENSORRT_MAJOR_VERSION} GREATER_EQUAL 8)
  list(APPEND CONVERT_FILES sparse_fc_op.cc sparse_multihead_matmul_op.cc)
endif()

set(TENSORRT_VERSION_NUMBER
    "${TENSORRT_MAJOR_VERSION}${TENSORRT_MINOR_VERSION}")
if(${TENSORRT_VERSION_NUMBER} GREATER_EQUAL 82)
  list(APPEND CONVERT_FILES set_value_op.cc)
endif()

nv_library(
  tensorrt_converter
  SRCS ${CONVERT_FILES}
  DEPS tensorrt_engine
       tensorrt_plugin
       operator
       scope
       phi
       tensorrt_op_teller
       op_registry)

nv_test(
  test_op_converter
  SRCS test_op_converter.cc
  DEPS paddle_framework ${GLOB_OPERATOR_DEPS} tensorrt_engine
       tensorrt_converter)
if(WIN32)
  nv_test(
    test_custom_plugin_creater
    SRCS test_custom_plugin_creater.cc
    DEPS paddle_framework tensorrt_converter phi common custom_operator
         init_phi)
elseif(WITH_CINN)
  nv_test(
    test_custom_plugin_creater
    SRCS test_custom_plugin_creater.cc
    DEPS paddle_framework
         tensorrt_converter
         phi
         common
         custom_operator
         init_phi
         python)
else()
  nv_test(
    test_custom_plugin_creater
    SRCS test_custom_plugin_creater.cc
    DEPS paddle_framework
         tensorrt_converter
         phi
         common
         custom_operator
         init_phi
         python)
endif()

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(test_op_converter)
endif()
