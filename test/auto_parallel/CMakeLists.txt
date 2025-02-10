# file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
# string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

add_subdirectory(spmd_rules)
add_subdirectory(hybrid_strategy)
add_subdirectory(custom_op)
add_subdirectory(pir)

if(WITH_DISTRIBUTE AND WITH_GPU)

  # NOTE(zyl): unittests WITH multi cards and timeout
  py_test_modules(test_converter MODULES test_converter)
  set_tests_properties(test_converter PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE"
                                                 TIMEOUT 50)
  py_test_modules(test_high_order_grad MODULES test_high_order_grad)
  set_tests_properties(test_high_order_grad
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 50)
  py_test_modules(test_iterable_dataset MODULES test_iterable_dataset)
  set_tests_properties(test_iterable_dataset
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 80)

  py_test_modules(test_reshard_api MODULES test_reshard_api)
  set_tests_properties(test_reshard_api PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE"
                                                   TIMEOUT 150)
  py_test_modules(test_reshard_s_to_p MODULES test_reshard_s_to_p)
  set_tests_properties(test_reshard_s_to_p
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)
  py_test_modules(test_reshard_p_to_s MODULES test_reshard_p_to_s)
  set_tests_properties(test_reshard_p_to_s
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)
  py_test_modules(test_reshard_s_to_s MODULES test_reshard_s_to_s)
  set_tests_properties(test_reshard_s_to_s
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)
  py_test_modules(test_reshard_r_to_s MODULES test_reshard_r_to_s)
  set_tests_properties(test_reshard_r_to_s
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 320)
  py_test_modules(test_reshard_p_to_r MODULES test_reshard_p_to_r)
  set_tests_properties(test_reshard_p_to_r
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 160)
  py_test_modules(test_reshard_s_to_r MODULES test_reshard_s_to_r)
  set_tests_properties(test_reshard_s_to_r
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 150)
  if(NOT WITH_COVERAGE)
    py_test_modules(test_pipeline_scheduler MODULES test_pipeline_scheduler)
    set_tests_properties(test_pipeline_scheduler
                         PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 400)
  endif()
  py_test_modules(test_reshard_r_to_p MODULES test_reshard_r_to_p)
  set_tests_properties(test_reshard_r_to_p
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 200)
  py_test_modules(test_reshard_x_to_r MODULES test_reshard_x_to_r)
  set_tests_properties(test_reshard_x_to_r
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)
  py_test_modules(test_reshard_r_to_x MODULES test_reshard_r_to_x)
  set_tests_properties(test_reshard_r_to_x
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)
  py_test_modules(test_reshard_nd_mesh MODULES test_reshard_nd_mesh)
  set_tests_properties(test_reshard_nd_mesh
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)

  py_test_modules(test_reshard_same_status MODULES test_reshard_same_status)
  set_tests_properties(test_reshard_same_status
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)

  py_test_modules(test_semi_auto_parallel_basic MODULES
                  test_semi_auto_parallel_basic)
  set_tests_properties(test_semi_auto_parallel_basic
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 800)

  py_test_modules(test_semi_auto_parallel_for_llama_subnet MODULES
                  test_semi_auto_parallel_for_llama_subnet)
  set_tests_properties(test_semi_auto_parallel_for_llama_subnet
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 300)

  py_test_modules(test_semi_auto_parallel_softmax_basic MODULES
                  test_semi_auto_parallel_softmax_basic)
  set_tests_properties(test_semi_auto_parallel_softmax_basic
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 120)

  py_test_modules(test_semi_auto_parallel_compare_basic MODULES
                  test_semi_auto_parallel_compare_basic)
  set_tests_properties(test_semi_auto_parallel_compare_basic
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 120)
  py_test_modules(test_semi_auto_parallel_subgraph_embedding_basic MODULES
                  test_semi_auto_parallel_subgraph_embedding_basic)
  set_tests_properties(test_semi_auto_parallel_subgraph_embedding_basic
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 120)
  py_test_modules(test_semi_auto_parallel_pylayer MODULES
                  test_semi_auto_parallel_pylayer)
  set_tests_properties(test_semi_auto_parallel_pylayer
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)
  py_test_modules(test_semi_auto_parallel_single_strategy MODULES
                  test_semi_auto_parallel_single_strategy)
  set_tests_properties(test_semi_auto_parallel_single_strategy
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 400)
  py_test_modules(test_semi_auto_parallel_sharding_strategy MODULES
                  test_semi_auto_parallel_sharding_strategy)
  set_tests_properties(test_semi_auto_parallel_sharding_strategy
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 200)
  py_test_modules(test_semi_auto_parallel_lazy_init MODULES
                  test_semi_auto_parallel_lazy_init)
  set_tests_properties(test_semi_auto_parallel_lazy_init
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 120)
  py_test_modules(test_semi_auto_parallel_in_framework MODULES
                  test_semi_auto_parallel_in_framework)
  set_tests_properties(test_semi_auto_parallel_in_framework
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 300)
  py_test_modules(test_semi_auto_parallel_dygraph_inplace MODULES
                  test_semi_auto_parallel_dygraph_inplace)
  set_tests_properties(test_semi_auto_parallel_dygraph_inplace
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)
  py_test_modules(test_dist_tensor_api MODULES test_dist_tensor_api)
  set_tests_properties(test_dist_tensor_api
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 200)
  py_test_modules(test_semi_auto_parallel_saved_tensor_hook MODULES
                  test_semi_auto_parallel_saved_tensor_hook)
  set_tests_properties(test_semi_auto_parallel_saved_tensor_hook
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)
  py_test_modules(
    test_semi_auto_parallel_dist_to_static MODULES
    test_semi_auto_parallel_dist_to_static ENVS FLAGS_enable_pir_api=1)
  set_tests_properties(test_semi_auto_parallel_dist_to_static
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 300)
  py_test_modules(test_static_reshard_api MODULES test_static_reshard_api ENVS
                  FLAGS_enable_pir_api=1)
  set_tests_properties(test_static_reshard_api
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 300)
  py_test_modules(test_dist_checkpoint_utils MODULES test_dist_checkpoint_utils)
  set_tests_properties(test_dist_checkpoint_utils
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 120)
  py_test_modules(
    test_semi_auto_parallel_unshard_dtensor MODULES
    test_semi_auto_parallel_unshard_dtensor ENVS FLAGS_enable_pir_api=1)
  set_tests_properties(test_semi_auto_parallel_unshard_dtensor
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)
  py_test_modules(test_auto_parallel_backward_test MODULES
                  test_auto_parallel_backward_test ENVS)
  set_tests_properties(test_auto_parallel_backward_test
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" TIMEOUT 100)
  py_test_modules(test_moe_utils MODULES test_moe_utils)
  set_tests_properties(test_moe_utils PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE"
                                                 TIMEOUT 30)
  # End of unittests WITH multi cards and timeout

  # NOTE(zyl): unittests WITH multi cards and WITHOUT timeout
  # End of unittests WITH multi cards and WITHOUT timeout

  py_test_modules(test_semi_auto_parallel_functional_in_single_card MODULES
                  test_semi_auto_parallel_functional_in_single_card)
  # End of unittests WITH single card and timeout

  # NOTE(zyl): unittests WITH single card and WITHOUT timeout
  py_test_modules(test_align_mode MODULES test_align_mode)
  py_test_modules(test_tunable_variable MODULES test_tunable_variable)
  py_test_modules(test_tunable_space MODULES test_tunable_space)
  py_test_modules(test_recorder MODULES test_recorder)
  py_test_modules(test_trial MODULES test_trial)
  py_test_modules(test_cluster MODULES test_cluster)
  py_test_modules(test_comm_cost MODULES test_comm_cost)
  py_test_modules(test_comp_cost MODULES test_comp_cost)
  py_test_modules(test_cluster_v2 MODULES test_cluster_v2)
  py_test_modules(test_process_mesh_v2 MODULES test_process_mesh_v2)
  py_test_modules(test_strategy MODULES test_strategy)
  py_test_modules(test_cluster_partition MODULES test_cluster_partition)
  py_test_modules(test_convert_to_process_meshes MODULES
                  test_convert_to_process_meshes)
  py_test_modules(test_dist_tensor MODULES test_dist_tensor ENVS
                  FLAGS_enable_pir_api=1)
  py_test_modules(test_api_dist_branch MODULES test_api_dist_branch)
  py_test_modules(test_shard_tensor_api MODULES test_shard_tensor_api ENVS
                  FLAGS_enable_pir_api=1)
  py_test_modules(test_strategy_api MODULES test_strategy_api)
  py_test_modules(test_parallel_api MODULES test_parallel_api)
  py_test_modules(test_dtensor_to_local_api MODULES test_dtensor_to_local_api)
  py_test_modules(test_dtensor_from_local_api MODULES
                  test_dtensor_from_local_api)
  py_test_modules(test_dy_local_view_compute MODULES test_dy_local_view_compute)
  py_test_modules(test_local_view_compute MODULES test_local_view_compute)
  # End of unittests WITH single card WITHOUT timeout

endif()

py_test_modules(test_job_schedule_profiler_range MODULES
                test_job_schedule_profiler_range)

set_pir_tests_properties()
