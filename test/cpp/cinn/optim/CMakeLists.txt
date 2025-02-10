cinn_cc_test(test_replace_call_with_expr SRCS replace_call_with_expr_test.cc
             DEPS cinncore)
cinn_cc_test(test_cast_simplify SRCS cast_simplify_test.cc DEPS cinncore)
cinn_cc_test(test_replace_cross_thread_reduction SRCS
             replace_cross_thread_reduction_test.cc DEPS cinncore)
