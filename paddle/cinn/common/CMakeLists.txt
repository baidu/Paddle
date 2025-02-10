core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  shared.cc
  cinn_value.cc
  type.cc
  target.cc
  arch_util.cc
  object.cc
  debug_manager.cc
  info_registry.cc
  graph_utils.cc
  context.cc
  axis.cc
  ir_util.cc
  test_helper.cc
  # cuda_test_helper.cc
  arithmetic.cc
  cas.cc
  union_find.cc
  python_interpreter_guard.cc
  nvgpu_dev_info.cc
  integer_set.cc
  dim_expr_converter.cc
  broadcast_tree.cc
  iter_simplify.cc
  simplify_special_pattern.cc)

cinn_cc_test(test_equation_graph_topo_walker SRCS
             equation_graph_topo_walker_test.cc DEPS gtest glog)
cinn_cc_test(test_is_reachable_predicator SRCS is_reachable_predicator_test.cc
             DEPS gtest glog)
cinn_cc_test(test_cinn_value SRCS cinn_value_test.cc DEPS cinncore)
cinn_cc_test(test_shared SRCS shared_test.cc DEPS cinncore)
cinn_cc_test(test_type SRCS type_test.cc DEPS cinncore)
cinn_cc_test(test_integer_set SRCS integer_set_test.cc DEPS cinncore)

cinn_cc_test(test_fp16_bf16_host SRCS float16_bfloat16_host_test.cc DEPS gtest
             glog)
if(WITH_CUDA)
  cinn_nv_test(test_fp16_bf16_cuda SRCS float16_bfloat16_cuda_test.cu DEPS
               gtest glog)
endif()
