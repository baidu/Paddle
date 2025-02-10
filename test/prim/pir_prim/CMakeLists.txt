set(TEST_PRIM_PURE_PIR_CASES
    test_custom_vjp_trait
    test_decomp_op
    test_decompose_op
    test_vjp_prim
    test_batch_norm_shape_check
    test_builtin_slice
    test_prim_program
    test_prim_simpnet
    test_prim_custom_vjp
    test_prim_jit
    test_pir_prim_flags
    test_pir_prim_flags_v2
    test_sink_decomp
    test_prim_skip_dynamic
    test_prim_dynamic
    test_prim_jit_dynamic
    test_auto_recompute
    test_auto_recompute_dy2static
    test_prim_sub_graph_dynamic_shape
    test_decompose_control_flow
    test_decomp_whole_program
    test_dynamic_combine1
    test_dynamic_combine2
    test_decomp_fallback
    test_prim_amax_amin_op)

foreach(target ${TEST_PRIM_PURE_PIR_CASES})
  py_test_modules(
    ${target}
    MODULES
    ${target}
    ENVS
    GLOG_v=1
    FLAGS_enable_pir_api=true
    FLAGS_prim_enable_dynamic=true)
endforeach()

set(TEST_PRIM_DYNAMIC_SHAPE_BACKWARD_PIR_CASES
    test_prim_sub_graph_backward_dynamic_shape
    test_prim_sub_graph_abcde_backward_dynamic_shape
    test_prim_sub_graph_fghij_backward_dynamic_shape
    test_prim_sub_graph_klmno_backward_dynamic_shape
    test_prim_sub_graph_pqrst_backward_dynamic_shape
    test_prim_sub_graph_uvwxyz_backward_dynamic_shape)

foreach(target ${TEST_PRIM_DYNAMIC_SHAPE_BACKWARD_PIR_CASES})
  py_test_modules(
    ${target}
    MODULES
    ${target}
    ENVS
    GLOG_v=1
    FLAGS_prim_vjp_skip_default_ops=0
    FLAGS_enable_pir_api=true
    FLAGS_prim_enable_dynamic=true)
endforeach()

py_test_modules(
  test_pir_prim_flags_v3
  MODULES
  test_pir_prim_flags_v3
  ENVS
  GLOG_v=1
  FLAGS_enable_pir_api=true
  FLAGS_prim_vjp_skip_default_ops=0)

set_tests_properties(test_auto_recompute PROPERTIES TIMEOUT 40)
set_tests_properties(test_auto_recompute_dy2static PROPERTIES TIMEOUT 40)

set(TEST_PRIM_PURE_PIR_CINN test_prim_rms_norm_st_shape
                            test_prim_flags_check_ops)

if(WITH_CINN)
  foreach(target ${TEST_PRIM_PURE_PIR_CINN})
    py_test_modules(
      ${target}
      MODULES
      ${target}
      ENVS
      GLOG_v=1
      FLAGS_prim_check_ops=true
      FLAGS_enable_pir_api=true
      FLAGS_prim_enable_dynamic=true
      FLAGS_prim_vjp_skip_default_ops=false)
    set_tests_properties(${target} PROPERTIES LABELS "RUN_TYPE=CINN")
  endforeach()
endif()

foreach(target ${TEST_PRIM_TRANS_PIR_CASES})
  py_test_modules(${target} MODULES ${target} ENVS GLOG_v=1
                  FLAGS_enable_pir_in_executor=true)
endforeach()

set(TEST_PRIM_BACKWARD_BLACKLIST "test_pir_prim_flags_v4")

py_test_modules(
  ${TEST_PRIM_BACKWARD_BLACKLIST}
  MODULES
  ${TEST_PRIM_BACKWARD_BLACKLIST}
  ENVS
  GLOG_v=1
  FLAGS_enable_pir_api=true
  FLAGS_prim_enable_dynamic=true
  FLAGS_prim_backward_blacklist=pd_op.tanh_grad)
