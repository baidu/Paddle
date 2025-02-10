cinn_cc_test(test_sized_multi_set SRCS sized_multi_set_test.cc DEPS cinncore)
cinn_cc_test(test_multi_threading SRCS multi_threading_test.cc DEPS cinncore)

cinn_cc_test(test_functional SRCS functional_test.cc DEPS absl Threads::Threads)
