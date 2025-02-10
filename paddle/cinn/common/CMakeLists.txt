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

cinn_cc_test(test_dfs_walker SRCS dfs_walker_test.cc DEPS gtest glog)
cinn_cc_test(test_dfs_topo_walker SRCS dfs_topo_walker_test.cc DEPS gtest glog)
cinn_cc_test(test_cinn_value SRCS cinn_value_test.cc DEPS cinncore)
cinn_cc_test(test_cas SRCS cas_test.cc DEPS cinncore)
cinn_cc_test(test_axis SRCS axis_test.cc DEPS cinncore)

cinn_cc_test(dim_expr_converter_test SRCS dim_expr_converter_test.cc DEPS
             cinncore)
cinn_cc_test(broadcast_tree_test SRCS broadcast_tree_test.cc DEPS cinncore)
