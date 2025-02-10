file(
  GLOB TEST_INTERP_CASES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_INTERP_CASES "${TEST_INTERP_CASES}")

set(DISTRIBUTED_OP_TRANSLATOR_TEST test_all_reduce_translator)
set(DISTRIBUTED_OP_TRANSLATOR_TEST test_c_allreduce_translator)
list(APPEND DISTRIBUTED_OP_TRANSLATOR_TEST test_barrier_translator)
list(APPEND DISTRIBUTED_OP_TRANSLATOR_TEST test_c_reduce_min_translator)
list(APPEND DISTRIBUTED_OP_TRANSLATOR_TEST test_c_allreduce_min_translator)
list(APPEND DISTRIBUTED_OP_TRANSLATOR_TEST test_all_reduce_min_translator)
list(APPEND DISTRIBUTED_OP_TRANSLATOR_TEST test_c_allreduce_prod_translator)
list(APPEND DISTRIBUTED_OP_TRANSLATOR_TEST test_c_reduce_max_translator)
list(APPEND DISTRIBUTED_OP_TRANSLATOR_TEST test_c_reduce_prod_translator)
list(APPEND DISTRIBUTED_OP_TRANSLATOR_TEST test_c_scatter_translator)
list(APPEND DISTRIBUTED_OP_TRANSLATOR_TEST test_c_split_translator)
list(APPEND DISTRIBUTED_OP_TRANSLATOR_TEST test_distributed_fused_lamb_init)
list(APPEND DISTRIBUTED_OP_TRANSLATOR_TEST
     test_distributed_lookup_table_translate)
list(APPEND DISTRIBUTED_OP_TRANSLATOR_TEST
     test_distributed_push_sparse_translator)
list(APPEND DISTRIBUTED_OP_TRANSLATOR_TEST test_dgc_translator)
list(APPEND DISTRIBUTED_OP_TRANSLATOR_TEST test_nop_translator)
list(APPEND DISTRIBUTED_OP_TRANSLATOR_TEST test_partial_allgather_translator)
list(APPEND DISTRIBUTED_OP_TRANSLATOR_TEST test_partial_send_translator)
list(APPEND DISTRIBUTED_OP_TRANSLATOR_TEST test_push_dense_translator)
list(APPEND DISTRIBUTED_OP_TRANSLATOR_TEST test_partial_recv_translator)
list(APPEND DISTRIBUTED_OP_TRANSLATOR_TEST
     test_prune_gate_by_capacity_translator)

list(APPEND DISTRIBUTED_OP_TRANSLATOR_TEST test_pull_gpups_sparse_translator)
list(APPEND DISTRIBUTED_OP_TRANSLATOR_TEST test_pull_sparse_v2_translator)
list(APPEND DISTRIBUTED_OP_TRANSLATOR_TEST test_random_routing_translator)
list(APPEND DISTRIBUTED_OP_TRANSLATOR_TEST test_send_and_recv_translator)
list(APPEND DISTRIBUTED_OP_TRANSLATOR_TEST test_limit_by_capacity_translator)
list(APPEND DISTRIBUTED_OP_TRANSLATOR_TEST test_global_scatter_translator)
list(APPEND DISTRIBUTED_OP_TRANSLATOR_TEST test_global_gather_translator)

if(NOT WITH_DISTRIBUTE)
  list(REMOVE_ITEM TEST_INTERP_CASES ${DISTRIBUTED_OP_TRANSLATOR_TEST})
endif()

if(NOT WITH_DGC)
  list(REMOVE_ITEM TEST_INTERP_CASES test_dgc_translator)
endif()

foreach(target ${TEST_INTERP_CASES})
  py_test_modules(${target} MODULES ${target})
endforeach()
