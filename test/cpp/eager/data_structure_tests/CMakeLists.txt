if(WITH_CINN)
  set(eager_deps ${eager_deps} python)
endif()
cc_test(
  test_egr_ds_eager_tensor
  SRCS eager_tensor_test.cc
  DEPS final_dygraph_function ${eager_deps})
cc_test(
  test_egr_ds_auotgrad_meta
  SRCS autograd_meta_test.cc
  DEPS final_dygraph_function ${eager_deps})

if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
  cc_test(
    test_egr_ds_grad_tensor_holder
    SRCS grad_tensor_holder_test.cc
    DEPS conditional_block_op ${eager_deps} ${generated_deps})
  cc_test(
    test_egr_ds_grad_node_info
    SRCS grad_node_info_test.cc
    DEPS conditional_block_op ${eager_deps} ${generated_deps})
  cc_test(
    test_egr_ds_accumulation_node
    SRCS accumulation_node_test.cc
    DEPS conditional_block_op ${eager_deps} ${generated_deps})
  cc_test(
    test_egr_ds_tensor_wrapper
    SRCS tensor_wrapper_test.cc
    DEPS conditional_block_op ${eager_deps} ${generated_deps})
endif()
