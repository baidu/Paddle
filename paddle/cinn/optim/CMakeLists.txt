core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  replace_call_with_expr.cc
  replace_var_with_expr.cc
  ir_simplify.cc
  optimize.cc
  vectorize_loops.cc
  unroll_loops.cc
  transform_polyfor_to_for.cc
  eliminate_broadcast_in_forloop.cc
  eliminate_invariant_loop.cc
  fold_cinn_call_arguments.cc
  call_arg_list_to_pod_value.cc
  insert_debug_log_callee.cc
  lower_function_call_bind_vars.cc
  extern_call_process_pass.cc
  map_extern_call.cc
  compute_inline_expand.cc
  buffer_assign.cc
  replace_const_param_to_integer.cc
  lower_intrin.cc
  cast_bool_to_int8.cc
  var_mod_simplify.cc
  remove_schedule_block_pass.cc
  replace_cross_block_reduction.cc
  replace_cross_thread_reduction.cc
  reindex_transpose_buffer_pass.cc
  replace_mod_to_max.cc
  resize_buffer.cc
  update_buffer_axis_pass.cc
  trans_buffer_with_dynamic_shape.cc
  schedule_block_dce.cc
  eliminate_common_factor_of_local_index.cc
  if_fusion_pass.cc
  merge_block_utils.cc
  eliminate_common_global_memory_read.cc
  rearrange_load_instruction_pass.cc
  check_tensor_buffer_map.cc
  longlong2int_pass.cc
  vectorize_for_trans.cc
  if_fold_pass.cc)

if(WITH_CUDA OR WITH_ROCM)
  gather_srcs(cinnapi_src SRCS transform_gpu_forloop.cc)
endif()
