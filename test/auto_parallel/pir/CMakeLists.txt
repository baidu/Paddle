if(WITH_DISTRIBUTE AND WITH_GPU)
  py_test_modules(test_to_static_pir_program MODULES test_to_static_pir_program)
  set_tests_properties(test_to_static_pir_program
                       PROPERTIES ENVIRONMENT "FLAGS_enable_pir_api=1")
  py_test_modules(test_ir_dist_attr MODULES test_ir_dist_attr ENVS
                  FLAGS_enable_pir_api=1)
  py_test_modules(test_static_pir_program MODULES test_static_pir_program)
  py_test_modules(test_op_role MODULES test_op_role)
  set_tests_properties(test_op_role PROPERTIES ENVIRONMENT
                                               "FLAGS_enable_pir_api=1")
  py_test_modules(test_pir_elementwise_spmd MODULES test_elementwise_spmd_rule
                  ENVS FLAGS_enable_pir_api=1)
  py_test_modules(test_pir_relu_spmd MODULES test_relu_spmd_rule ENVS
                  FLAGS_enable_pir_api=1)
  py_test_modules(test_pir_mse_spmd MODULES test_mse_spmd_rule ENVS
                  FLAGS_enable_pir_api=1)
  py_test_modules(test_pir_reshard_s_to_r MODULES test_pir_reshard_s_to_r)
  set_tests_properties(test_pir_reshard_s_to_r PROPERTIES TIMEOUT 120)
  py_test_modules(test_mlp MODULES test_mlp ENVS FLAGS_enable_pir_api=1)
  py_test_modules(test_local_layer MODULES test_local_layer ENVS
                  FLAGS_enable_pir_api=1)
  py_test_modules(
    test_semi_auto_parallel_dist_to_static_pir MODULES
    test_semi_auto_parallel_dist_to_static_pir ENVS FLAGS_enable_pir_api=1)
  py_test_modules(
    test_semi_auto_parallel_dist_to_static_pir_decomp MODULES
    test_semi_auto_parallel_dist_to_static_pir_decomp ENVS
    FLAGS_enable_pir_api=1 FLAGS_dist_prim_all=1)
  py_test_modules(
    test_semi_auto_parallel_sot_pir MODULES test_semi_auto_parallel_sot_pir
    ENVS FLAGS_enable_pir_api=1;MIN_GRAPH_SIZE=0)
  set_tests_properties(test_semi_auto_parallel_sot_pir PROPERTIES TIMEOUT 120)
  py_test_modules(
    test_auto_parallel_c_embedding_pass MODULES
    test_auto_parallel_c_embedding_pass ENVS FLAGS_enable_pir_api=1)
  py_test_modules(
    test_auto_parallel_replace_with_parallel_cross_entropy_pass MODULES
    test_auto_parallel_replace_with_parallel_cross_entropy_pass ENVS
    FLAGS_enable_pir_api=1 FLAGS_dist_prim_all=1)
  py_test_modules(
    test_auto_parallel_recompute_pir_pass MODULES
    test_auto_parallel_recompute_pir_pass ENVS FLAGS_enable_pir_api=1)
  py_test_modules(test_reshard MODULES test_reshard ENVS FLAGS_enable_pir_api=1)
  py_test_modules(test_learning_rate MODULES test_learning_rate ENVS
                  FLAGS_enable_pir_api=1)
  py_test_modules(test_fold_reshard_pass MODULES test_fold_reshard_pass ENVS
                  FLAGS_enable_pir_api=1)
  set_tests_properties(test_mlp PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT
                                           60)
  set_tests_properties(test_semi_auto_parallel_dist_to_static_pir
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 300)
  set_tests_properties(test_semi_auto_parallel_dist_to_static_pir_decomp
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 60)
  set_tests_properties(test_auto_parallel_c_embedding_pass
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 300)
  set_tests_properties(
    test_auto_parallel_replace_with_parallel_cross_entropy_pass
    PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 60)
  set_tests_properties(test_auto_parallel_recompute_pir_pass
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 200)
  py_test_modules(
    test_eliminate_transpose_pass MODULES test_eliminate_transpose_pass ENVS
    FLAGS_enable_pir_in_executor=1)
  py_test_modules(test_pipeline_scheduler_1f1b_pir MODULES
                  test_pipeline_scheduler_1f1b_pir ENVS FLAGS_enable_pir_api=1)
  set_tests_properties(test_pipeline_scheduler_1f1b_pir
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 50)
  py_test_modules(test_moe_api MODULES test_moe_api ENVS FLAGS_enable_pir_api=1)
  py_test_modules(test_pipeline_scheduler_vpp_pir MODULES
                  test_pipeline_scheduler_vpp_pir ENVS FLAGS_enable_pir_api=1)
  set_tests_properties(test_pipeline_scheduler_vpp_pir
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 50)
  py_test_modules(test_pir_stack_grad_spmd_rule MODULES
                  test_stack_grad_spmd_rule ENVS FLAGS_enable_pir_api=1)
  py_test_modules(
    test_semi_auto_parallel_simple_net_ep MODULES
    test_semi_auto_parallel_simple_net_ep ENVS FLAGS_enable_pir_api=1)
  set_tests_properties(test_semi_auto_parallel_simple_net_ep
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 60)
  py_test_modules(test_custom_spec_pir MODULES test_custom_spec_pir ENVS
                  FLAGS_enable_pir_api=1)
  py_test_modules(
    test_sharding_tensor_fusion_save_load MODULES
    test_sharding_tensor_fusion_save_load ENVS FLAGS_enable_pir_api=1)
  py_test_modules(test_while_pir MODULES test_while_pir ENVS
                  FLAGS_enable_pir_api=1)
  set_tests_properties(test_sharding_tensor_fusion_save_load
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 300)
  py_test_modules(test_pir_moe_utils_api MODULES test_pir_moe_utils_api ENVS
                  FLAGS_enable_pir_api=1)
  py_test_modules(
    test_pipeline_scheduler_vpp_local_view MODULES
    test_pipeline_scheduler_vpp_local_view ENVS FLAGS_enable_pir_api=1)
endif()
py_test_modules(test_pir_1f1b_plan MODULES test_pir_1f1b_plan ENVS
                FLAGS_enable_pir_api=1)
py_test_modules(test_pir_reshard_p_to_s MODULES test_pir_reshard_p_to_s ENVS
                FLAGS_enable_pir_api=1)
