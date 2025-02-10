cinn_cc_test(test_dfs_walker SRCS dfs_walker_test.cc DEPS gtest glog)
cinn_cc_test(test_dfs_topo_walker SRCS dfs_topo_walker_test.cc DEPS gtest glog)
cinn_cc_test(test_cinn_value SRCS cinn_value_test.cc DEPS cinncore)
cinn_cc_test(test_cas SRCS cas_test.cc DEPS cinncore)
cinn_cc_test(test_axis SRCS axis_test.cc DEPS cinncore)

cinn_cc_test(dim_expr_converter_test SRCS dim_expr_converter_test.cc DEPS
             cinncore)
cinn_cc_test(broadcast_tree_test SRCS broadcast_tree_test.cc DEPS cinncore)
