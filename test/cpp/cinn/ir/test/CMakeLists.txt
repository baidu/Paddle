# cinn_cc_test(test_ir SRCS ir_test.cc DEPS core)
# cinn_cc_test(test_ir_printer SRCS ir_printer_test.cc DEPS core)
# cinn_cc_test(test_ir_operators SRCS ir_operators_test.cc DEPS core)
# cinn_cc_test(test_tensor SRCS tensor_test.cc DEPS core)
cinn_cc_test(test_collect_ir_nodes SRCS collect_ir_nodes_test.cc DEPS cinncore)

cinn_cc_test(test_intrinsic_ops SRCS intrinsic_ops_test.cc DEPS cinncore)
cinn_cc_test(test_ir_verify SRCS ir_verify_test.cc DEPS cinncore)
cinn_cc_test(test_ir_copy SRCS ir_copy_test.cc DEPS cinncore)
