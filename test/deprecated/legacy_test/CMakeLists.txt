file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")
set(GC_ENVS FLAGS_eager_delete_tensor_gb=0.0 FLAGS_fast_eager_deletion_mode=1
            FLAGS_memory_fraction_of_eager_deletion=1.0)
set(dist_ENVS http_proxy="" https_proxy="")

# The following unittest is now in deprecated dir, we can delete this code when we move it from deprecated dir to this dir
###### start ######
list(REMOVE_ITEM TEST_OPS test_imperative_base)
###### end ######
list(REMOVE_ITEM TEST_OPS test_fleet_util)

file(
  GLOB DIST_TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_dist_*.py")
list(REMOVE_ITEM DIST_TEST_OPS "test_dist_op")

string(REPLACE ".py" "" DIST_TEST_OPS "${DIST_TEST_OPS}")

if(WITH_COVERAGE)
  list(REMOVE_ITEM TEST_OPS test_unique)
endif()
list(APPEND DIST_TEST_OPS test_auto_parallel_autoconvert_deprecated)
list(APPEND DIST_TEST_OPS test_auto_parallel_data_unshard_deprecated)
list(APPEND DIST_TEST_OPS test_auto_parallel_save_load_deprecated)
set(MIXED_DIST_TEST_OPS ${DIST_TEST_OPS})
#remove distribute unittests.

list(APPEND MIXED_DIST_TEST_OPS test_simple_dist_transpiler)
list(APPEND MIXED_DIST_TEST_OPS test_communicator_ps_gpu)
list(APPEND MIXED_DIST_TEST_OPS test_communicator_geo_deprecated)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_launch_ascend)
list(APPEND MIXED_DIST_TEST_OPS test_ascend_group)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_api_input)

list(APPEND MIXED_DIST_TEST_OPS test_fleet_base)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_base_2_deprecated)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_base_3_deprecated)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_auto)
list(APPEND MIXED_DIST_TEST_OPS test_auto_parallel_partitioner_deprecated)
list(APPEND MIXED_DIST_TEST_OPS test_auto_parallel_partitioner_gpt_deprecated)
list(APPEND MIXED_DIST_TEST_OPS test_auto_parallel_searcher_deprecated)
list(APPEND MIXED_DIST_TEST_OPS test_auto_parallel_reshard_deprecated)
list(APPEND MIXED_DIST_TEST_OPS test_auto_parallel_dist_tensor_deprecated)
list(APPEND MIXED_DIST_TEST_OPS test_auto_parallel_reshard_mppp_deprecated)
list(APPEND MIXED_DIST_TEST_OPS test_auto_parallel_reshard_dpmppp_deprecated)
list(APPEND MIXED_DIST_TEST_OPS test_auto_parallel_cost_model_deprecated)
list(APPEND MIXED_DIST_TEST_OPS test_dygraph_hybrid_dp)
list(APPEND MIXED_DIST_TEST_OPS test_auto_parallel_reshard_serial_deprecated)
foreach(TEST_OP ${MIXED_DIST_TEST_OPS})
  list(REMOVE_ITEM TEST_OPS ${TEST_OP})
endforeach()

if(NOT WITH_PYTHON AND ON_INFER)
  list(REMOVE_ITEM TEST_OPS test_eager_trace_op)
endif()

if(NOT WITH_GPU)
  list(REMOVE_ITEM TEST_OPS test_async_read_write)
  list(REMOVE_ITEM TEST_OPS test_fused_multi_transformer_op)
  list(REMOVE_ITEM TEST_OPS test_fused_transformer_encoder_layer)
  list(REMOVE_ITEM TEST_OPS test_fused_bias_dropout_residual_layer_norm_op)
  list(REMOVE_ITEM TEST_OPS test_fused_bias_dropout_residual_layer_norm_op_api)
  list(REMOVE_ITEM TEST_OPS test_rms_norm_op)
  list(REMOVE_ITEM TEST_OPS test_fused_attention_pass)
  list(REMOVE_ITEM TEST_OPS test_fused_comm_buffer)
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_hapi_model")
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_fleet_spmt")
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_fleet_minimize")
  list(REMOVE_ITEM TEST_OPS test_async_read_write)
endif()

list(REMOVE_ITEM TEST_OPS test_audio_logmel_feature test_audio_mel_feature)
list(REMOVE_ITEM TEST_OPS test_fuse_gemm_epilogue_pass_deprecated)

if(((NOT WITH_ROCM) AND (NOT WITH_GPU)) OR WIN32)
  list(REMOVE_ITEM TEST_OPS test_memcpy_op)
  list(REMOVE_ITEM TEST_OPS test_raw_program_optimizer)
  list(REMOVE_ITEM TEST_OPS test_disable_signal_handler)
  list(REMOVE_ITEM TEST_OPS test_auto_parallel_mapper_deprecated)
endif()

if(WIN32)
  list(REMOVE_ITEM TEST_OPS test_multiprocess_reader_exception_deprecated)
  list(REMOVE_ITEM TEST_OPS test_trainer_desc)
  list(REMOVE_ITEM TEST_OPS test_checkpoint_notify_op)
  list(REMOVE_ITEM TEST_OPS test_downpoursgd_deprecated)
  list(REMOVE_ITEM TEST_OPS test_fleet_deprecated)
  list(REMOVE_ITEM TEST_OPS test_fleet_nocvm_1_deprecated)
  list(REMOVE_ITEM TEST_OPS test_fleet_rolemaker)
  list(REMOVE_ITEM TEST_OPS test_fleet_rolemaker_3)
  list(REMOVE_ITEM TEST_OPS test_fleet_unitaccessor_deprecated)
  list(REMOVE_ITEM TEST_OPS test_ps_dispatcher)
  list(REMOVE_ITEM TEST_OPS test_ir_memory_optimize_nlp)
  list(REMOVE_ITEM TEST_OPS test_nvprof)

  # TODO: Fix these unittests failed on Windows
  list(REMOVE_ITEM TEST_OPS test_debugger)
endif()

if(NOT WITH_DISTRIBUTE OR WIN32)
  # DISTRIBUTE related
  list(REMOVE_ITEM TEST_OPS test_avoid_twice_initialization_deprecated)
  list(REMOVE_ITEM TEST_OPS test_fleet_metric_deprecated)
  list(REMOVE_ITEM TEST_OPS test_fleet_ps)
  list(REMOVE_ITEM TEST_OPS test_fleet_rolemaker_2)
  list(REMOVE_ITEM TEST_OPS test_delete_c_identity_op_pass)
  # TODO: Fix these unittests failed on Windows
  list(REMOVE_ITEM TEST_OPS test_fake_init_op)
endif()

if(NOT WITH_DISTRIBUTE)
  list(REMOVE_ITEM TEST_OPS test_desc_clone_dist)
endif()

if(WIN32)
  list(REMOVE_ITEM TEST_OPS test_complex_matmul)
  list(REMOVE_ITEM TEST_OPS test_trt_convert_preln_residual_bias)
  list(REMOVE_ITEM TEST_OPS test_masked_multihead_attention_op)
  list(REMOVE_ITEM TEST_OPS test_rms_norm_op)
  list(REMOVE_ITEM TEST_OPS test_matmul_int8_op)
  list(REMOVE_ITEM TEST_OPS test_variable_length_memory_efficient_attention)
endif()
list(REMOVE_ITEM TEST_OPS test_checkpoint_saver)

if(APPLE OR WIN32)
  list(REMOVE_ITEM TEST_OPS test_fs_interface)
  list(REMOVE_ITEM TEST_OPS test_fleet_metric_deprecated)
endif()

list(REMOVE_ITEM TEST_OPS test_parallel_dygraph_hybrid_parallel)

list(REMOVE_ITEM TEST_OPS test_parallel_dygraph_transformer_gloo)
# NOTE: @xiongkun03, cpu is too slow, fix it in next PR

if(NOT WITH_GLOO)
  list(REMOVE_ITEM TEST_OPS
       test_parallel_dygraph_sparse_embedding_diff_length_gloo)
endif()

if((NOT WITH_GPU) AND (NOT WITH_ROCM))
  list(REMOVE_ITEM TEST_OPS test_fused_conv2d_add_act_op)
  # TODO(shenliang03): rank_attention_op support CPU device in future
  list(REMOVE_ITEM TEST_OPS test_batch_fc_op)
  # TODO(shenliang03): batch_fc_op support CPU device in future
  # TODO(Yancey1989): parallel dygraph support CPU device in future
  list(REMOVE_ITEM TEST_OPS test_fleet_base_single)
  list(REMOVE_ITEM TEST_OPS test_auto_parallel_partitioner_deprecated)
  list(REMOVE_ITEM TEST_OPS test_auto_parallel_partitioner_gpt_deprecated)
  list(REMOVE_ITEM TEST_OPS test_auto_parallel_searcher_deprecated)
  list(REMOVE_ITEM TEST_OPS test_auto_parallel_reshard_deprecated)
  list(REMOVE_ITEM TEST_OPS test_auto_parallel_dist_tensor_deprecated)
  list(REMOVE_ITEM TEST_OPS test_auto_parallel_reshard_mppp_deprecated)
  list(REMOVE_ITEM TEST_OPS test_auto_parallel_reshard_dpmppp_deprecated)
  list(REMOVE_ITEM TEST_OPS test_auto_parallel_cost_model_deprecated)
  list(REMOVE_ITEM TEST_OPS test_auto_parallel_autoconvert_deprecated)
  list(REMOVE_ITEM TEST_OPS test_auto_parallel_data_unshard_deprecated)
  list(REMOVE_ITEM TEST_OPS test_auto_parallel_save_load_deprecated)
  list(REMOVE_ITEM TEST_OPS test_auto_parallel_reshard_serial_deprecated)

elseif(WITH_GPU)
  if(${CUDNN_VERSION} VERSION_LESS 7100)
    list(REMOVE_ITEM TEST_OPS test_fused_conv2d_add_act_op)
  endif()
endif()

if((NOT WITH_NCCL) AND (NOT WITH_RCCL))
  list(REMOVE_ITEM TEST_OPS test_imperative_group)
endif()

if(((NOT WITH_ROCM) AND (NOT WITH_GPU)) OR WIN32)
  list(REMOVE_ITEM TEST_OPS test_fused_gate_attention_op)
  list(REMOVE_ITEM TEST_OPS test_boxps)
  list(REMOVE_ITEM TEST_OPS test_reducescatter_api)
endif()
list(REMOVE_ITEM TEST_OPS test_seq_concat_op)
# FIXME(helin): https://github.com/PaddlePaddle/Paddle/issues/8290
list(REMOVE_ITEM TEST_OPS test_lstm_unit_op)
# # FIXME(qijun) https://github.com/PaddlePaddle/Paddle/issues/5185
list(REMOVE_ITEM TEST_OPS test_cond_op)

# FIXME(qijun): https://github.com/PaddlePaddle/Paddle/issues/5101#issuecomment-339814957

list(REMOVE_ITEM TEST_OPS op_test) # op_test is a helper python file, not a test
list(REMOVE_ITEM TEST_OPS decorator_helper)
# decorator_helper is a helper python file, not a test

if(APPLE)
  if(NOT WITH_DISTRIBUTE)
    list(REMOVE_ITEM TEST_OPS test_desc_clone_deprecated)
    list(REMOVE_ITEM TEST_OPS test_program_code_deprecated)
  endif()
  message(
    WARNING
      "These tests has been disabled in OSX before being fixed:\n test_fuse_elewise_add_act_pass_deprecated \n test_dist_se_resnext_*"
  )
  # this op is not support on mac
  list(REMOVE_ITEM TEST_OPS test_fuse_elewise_add_act_pass_deprecated)
endif()

if(NOT WITH_MKL OR NOT WITH_AVX)
  list(REMOVE_ITEM TEST_OPS test_match_matrix_tensor_op)
  list(REMOVE_ITEM TEST_OPS test_var_conv_2d)
endif()

list(REMOVE_ITEM TEST_OPS test_fleet_pyramid_hash)

if((WITH_ROCM OR WITH_GPU) OR NOT WITH_MKLML)
  # matmul with multiple heads need MKL support
  list(REMOVE_ITEM TEST_OPS test_matmul_op_with_head)
endif()

if(NOT WITH_CRYPTO)
  list(REMOVE_ITEM TEST_OPS test_crypto)
endif()

function(py_test_modules TARGET_NAME)
  if(WITH_TESTING)
    set(options SERIAL)
    set(oneValueArgs "")
    set(multiValueArgs MODULES DEPS ENVS)
    cmake_parse_arguments(py_test_modules "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})

    string(REGEX MATCH "_deprecated\.py$" DEPRECATED_MODULES
                 "${py_test_modules_MODULES}")
    string(REGEX MATCH "_deprecated$" DEPRECATED_TARGET_NAME "${TARGET_NAME}")
    set(FLAGS_PIR_MODE "")
    if((NOT "${DEPRECATED_MODULES}" STREQUAL "")
       OR (NOT "${DEPRECATED_TARGET_NAME}" STREQUAL ""))
      set(FLAGS_PIR_MODE FLAGS_enable_pir_api=0)
    endif()

    if(WITH_COVERAGE AND NOT (WITH_INCREMENTAL_COVERAGE
                              AND "$ENV{PADDLE_GIT_DIFF_PY_FILE}" STREQUAL ""))
      if(WITH_ASCEND_CL)
        add_test(
          NAME ${TARGET_NAME}
          COMMAND
            ${CMAKE_COMMAND} -E env
            PYTHONPATH=${PADDLE_BINARY_DIR}/python:$ENV{PYTHONPATH}
            ${py_test_modules_ENVS} ${FLAGS_PIR_MODE}
            COVERAGE_FILE=${PADDLE_BINARY_DIR}/python-coverage.data
            ${PYTHON_EXECUTABLE} -m coverage run --branch -p
            ${PADDLE_SOURCE_DIR}/tools/test_runner.py ${py_test_modules_MODULES}
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
      else()
        add_test(
          NAME ${TARGET_NAME}
          COMMAND
            ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python
            ${py_test_modules_ENVS} ${FLAGS_PIR_MODE}
            COVERAGE_FILE=${PADDLE_BINARY_DIR}/python-coverage.data
            ${PYTHON_EXECUTABLE} -m coverage run --branch -p
            ${PADDLE_SOURCE_DIR}/tools/test_runner.py ${py_test_modules_MODULES}
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
      endif()
    else()
      if(WITH_ASCEND_CL)
        add_test(
          NAME ${TARGET_NAME}
          COMMAND
            ${CMAKE_COMMAND} -E env
            PYTHONPATH=${PADDLE_BINARY_DIR}/python:$ENV{PYTHONPATH}
            ${py_test_modules_ENVS} ${FLAGS_PIR_MODE} ${PYTHON_EXECUTABLE}
            ${PADDLE_SOURCE_DIR}/tools/test_runner.py ${py_test_modules_MODULES}
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
      else()
        add_test(
          NAME ${TARGET_NAME}
          COMMAND
            ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python
            ${py_test_modules_ENVS} ${FLAGS_PIR_MODE} ${PYTHON_EXECUTABLE}
            ${PADDLE_SOURCE_DIR}/tools/test_runner.py ${py_test_modules_MODULES}
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
      endif()
    endif()

    if(py_test_modules_SERIAL)
      set_property(TEST ${TARGET_NAME} PROPERTY RUN_SERIAL 1)
    endif()
    if(WIN32 OR APPLE)
      set_tests_properties(${TARGET_NAME} PROPERTIES TIMEOUT 150)
    endif()
  endif()
endfunction()

function(bash_test_modules TARGET_NAME)
  if(NOT WITH_TESTING)
    return()
  endif()

  set(options SERIAL)
  set(oneValueArgs TIMEOUT START_BASH)
  set(multiValueArgs DEPS ENVS LABELS)
  cmake_parse_arguments(bash_test_modules "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  set(timeout 350)
  if(${bash_test_modules_TIMEOUT})
    set(timeout ${bash_test_modules_TIMEOUT})
  endif()

  string(REGEX MATCH "_deprecated$" DEPRECATED_TARGET_NAME "${TARGET_NAME}")
  set(FLAGS_PIR_MODE "")
  if(NOT "${DEPRECATED_TARGET_NAME}" STREQUAL "")
    set(FLAGS_PIR_MODE FLAGS_enable_pir_api=0)
  endif()

  if(WITH_COVERAGE)
    add_test(
      NAME ${TARGET_NAME}
      COMMAND
        ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python
        TEST_TARGET_NAME=${TARGET_NAME} TEST_TIMEOUT=${timeout}
        ${bash_test_modules_ENVS} WITH_COVERAGE=ON ${FLAGS_PIR_MODE}
        COVERAGE_FILE=${PADDLE_BINARY_DIR}/python-coverage.data bash
        ${CMAKE_CURRENT_BINARY_DIR}/${bash_test_modules_START_BASH}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  else()
    add_test(
      NAME ${TARGET_NAME}
      COMMAND
        ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python
        TEST_TARGET_NAME=${TARGET_NAME} TEST_TIMEOUT=${timeout}
        ${bash_test_modules_ENVS} ${FLAGS_PIR_MODE} bash
        ${CMAKE_CURRENT_BINARY_DIR}/${bash_test_modules_START_BASH}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif()

  if(bash_test_modules_SERIAL)
    set_property(TEST ${TARGET_NAME} PROPERTY RUN_SERIAL 1)
  endif()

  if(bash_test_modules_LABELS)
    set_tests_properties(${TARGET_NAME} PROPERTIES LABELS
                                                   ${bash_test_modules_LABELS})
  endif()
endfunction()

function(parallel_bash_test_modules TARGET_NAME)
  if(NOT WITH_TESTING)
    return()
  endif()

  set(options SERIAL)
  set(oneValueArgs TIMEOUT START_BASH)
  set(multiValueArgs DEPS ENVS LABELS UnitTests)
  cmake_parse_arguments(parallel_bash_test_modules "${options}"
                        "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(timeout 120)
  if(${parallel_bash_test_modules_TIMEOUT})
    set(timeout ${parallel_bash_test_modules_TIMEOUT})
  endif()

  list(JOIN parallel_bash_test_modules_UnitTests " " uts_string)

  if(WITH_COVERAGE)
    add_test(
      NAME ${TARGET_NAME}
      COMMAND
        ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python
        TEST_TARGET_NAME=${TARGET_NAME} TEST_TIMEOUT=${timeout}
        ${parallel_bash_test_modules_ENVS} UnitTests=${uts_string}
        WITH_COVERAGE=ON COVERAGE_FILE=${PADDLE_BINARY_DIR}/python-coverage.data
        bash
        ${CMAKE_CURRENT_BINARY_DIR}/${parallel_bash_test_modules_START_BASH}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  else()
    add_test(
      NAME ${TARGET_NAME}
      COMMAND
        ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python
        TEST_TARGET_NAME=${TARGET_NAME} TEST_TIMEOUT=${timeout}
        ${parallel_bash_test_modules_ENVS} UnitTests=${uts_string} bash
        ${CMAKE_CURRENT_BINARY_DIR}/${parallel_bash_test_modules_START_BASH}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif()

  if(parallel_bash_test_modules_SERIAL)
    set_property(TEST ${TARGET_NAME} PROPERTY RUN_SERIAL 1)
  endif()

  if(parallel_bash_test_modules_LABELS)
    set_tests_properties(${TARGET_NAME}
                         PROPERTIES LABELS ${parallel_bash_test_modules_LABELS})
  endif()
endfunction()

list(REMOVE_ITEM TEST_OPS test_feed_data_check_shape_type_deprecated)
list(REMOVE_ITEM TEST_OPS test_data_norm_op_deprecated)
list(REMOVE_ITEM TEST_OPS test_layers_deprecated)
list(REMOVE_ITEM TEST_OPS test_basic_gru_api)
list(REMOVE_ITEM TEST_OPS test_basic_gru_unit_op)
list(REMOVE_ITEM TEST_OPS test_basic_lstm_api)
list(REMOVE_ITEM TEST_OPS test_basic_lstm_unit_op)
list(REMOVE_ITEM TEST_OPS test_fuse_bn_act_pass_deprecated)

# disable this unittest temporarily
list(REMOVE_ITEM TEST_OPS test_imperative_data_loader_exception)
list(REMOVE_ITEM TEST_OPS test_dataset_dataloader_deprecated)

# disable sparse_attention which not in suitable env
if((NOT WITH_GPU)
   OR (WIN32)
   OR (PADDLE_WITH_ARM)
   OR (WITH_ROCM))
  list(REMOVE_ITEM TEST_OPS test_sparse_attention_op)
endif()

if(APPLE OR WIN32)
  list(REMOVE_ITEM TEST_OPS test_dataset)
  list(REMOVE_ITEM TEST_OPS test_dataset_deprecated)
  list(REMOVE_ITEM TEST_OPS test_dataset_dataloader)
  list(REMOVE_ITEM TEST_OPS test_imperative_data_loader_process)
  list(REMOVE_ITEM TEST_OPS test_imperative_data_loader_exit_func)
  list(REMOVE_ITEM TEST_OPS test_multiprocess_dataloader_iterable_dataset)
endif()

if(NOT WITH_GLOO)
  list(REMOVE_ITEM TEST_OPS test_cpuonly_spawn)
endif()

if(NOT WITH_GPU
   OR WIN32
   OR APPLE)
  list(REMOVE_ITEM TEST_OPS test_build_strategy_fusion_group_pass)
endif()

if(NOT WITH_CUDNN_FRONTEND)
  list(REMOVE_ITEM TEST_OPS test_fused_scale_bias_relu_conv_bn_op)
  list(REMOVE_ITEM TEST_OPS test_fused_scale_bias_add_relu_op)
  list(REMOVE_ITEM TEST_OPS test_fused_dconv_drelu_dbn_op)
endif()

# Some ops need to check results when gc is enabled
# Currently, only ops that register NoNeedBufferVarsInference need to do this test
set(TEST_OPS_WITH_GC test_slice_op_deprecated)

foreach(TEST_OP ${TEST_OPS_WITH_GC})
  list(REMOVE_ITEM TEST_OPS ${TEST_OP})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP} ENVS ${GC_ENVS})
endforeach()

if((NOT WITH_GPU)
   AND (NOT WITH_XPU)
   AND NOT (WITH_ASCEND OR WITH_ASCEND_CL))
  list(REMOVE_ITEM TEST_OPS "test_fleet_graph_execution_meta_optimizer")
  list(REMOVE_ITEM TEST_OPS "test_dist_fleet_grad_clip")
  list(REMOVE_ITEM TEST_OPS "test_dist_fleet_heter_ctr")
  list(REMOVE_ITEM TEST_OPS "test_dist_fleet_ps_gpu_ctr")
endif()

list(REMOVE_ITEM TEST_OPS "test_graph_reindex")
list(REMOVE_ITEM DIST_TEST_OPS test_dist_fleet_geo_deprecated)
list(REMOVE_ITEM TEST_OPS test_dist_fleet_geo_deprecated)
if(WITH_COVERAGE)
  list(REMOVE_ITEM TEST_OPS test_cuda_graphed_layer)
  list(REMOVE_ITEM TEST_OPS test_cuda_graph_partial_graph_static_run)
endif()
foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach()

set_tests_properties(test_conv2d_api_deprecated PROPERTIES LABELS
                                                           "RUN_TYPE=EXCLUSIVE")
if(WITH_DISTRIBUTE)
  # FIXME(typhoonzero): add these tests back
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_transformer")
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_transpiler")

  # TODO(sandyhouse): fix and add the ut back
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_mnist_hallreduce")

  #not need
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_base")
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_fleet_base")

  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_ctr")
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_mnist_lars")
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_mnist_train")
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_save_load")
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_text_classification")
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_train")
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_word2vec")
  list(REMOVE_ITEM DIST_TEST_OPS "test_dist_fleet_gloo")

  if(NOT WITH_HETERPS)
    list(REMOVE_ITEM DIST_TEST_OPS "test_communicator_ps_gpu")
  endif()

  py_test_modules(test_communicator_geo_deprecated MODULES
                  test_communicator_geo_deprecated ENVS ${dist_ENVS})
  if(NOT APPLE)
    py_test_modules(test_fleet_base MODULES test_fleet_base ENVS ${dist_ENVS}
                    FLAGS_enable_pir_api=0)
    py_test_modules(test_fleet_base_2_deprecated MODULES
                    test_fleet_base_2_deprecated ENVS ${dist_ENVS})
    py_test_modules(test_fleet_base_3_deprecated MODULES
                    test_fleet_base_3_deprecated ENVS ${dist_ENVS})
    if(NOT WIN32)
      py_test_modules(
        test_auto_parallel_partitioner_deprecated MODULES
        test_auto_parallel_partitioner_deprecated ENVS ${dist_ENVS})
      py_test_modules(
        test_auto_parallel_partitioner_gpt_deprecated MODULES
        test_auto_parallel_partitioner_gpt_deprecated ENVS ${dist_ENVS})
      py_test_modules(test_auto_parallel_searcher_deprecated MODULES
                      test_auto_parallel_searcher_deprecated ENVS ${dist_ENVS})
      py_test_modules(test_auto_parallel_reshard_deprecated MODULES
                      test_auto_parallel_reshard_deprecated ENVS ${dist_ENVS})
      py_test_modules(
        test_auto_parallel_dist_tensor_deprecated MODULES
        test_auto_parallel_dist_tensor_deprecated ENVS ${dist_ENVS})
      py_test_modules(
        test_auto_parallel_reshard_mppp_deprecated MODULES
        test_auto_parallel_reshard_mppp_deprecated ENVS ${dist_ENVS})
      py_test_modules(
        test_auto_parallel_reshard_dpmppp_deprecated MODULES
        test_auto_parallel_reshard_dpmppp_deprecated ENVS ${dist_ENVS})
      py_test_modules(
        test_auto_parallel_cost_model_deprecated MODULES
        test_auto_parallel_cost_model_deprecated ENVS ${dist_ENVS})
      py_test_modules(
        test_auto_parallel_reshard_serial_deprecated MODULES
        test_auto_parallel_reshard_serial_deprecated ENVS ${dist_ENVS})

    endif()
  endif()

  if(NOT APPLE)

    if(WITH_ASCEND OR WITH_ASCEND_CL)
      bash_test_modules(
        test_fleet_launch_ascend START_BASH test_fleet_launch_ascend.sh ENVS
        PADDLE_BINARY_DIR=${PADDLE_BINARY_DIR})
      bash_test_modules(test_ascend_group START_BASH test_ascend_group.sh ENVS
                        PADDLE_BINARY_DIR=${PADDLE_BINARY_DIR})
    endif()

    # port range (20000, 21200) is reserved for dist-ops
    set(dist_ut_port 20001)
    foreach(TEST_OP ${DIST_TEST_OPS})
      bash_test_modules(
        ${TEST_OP}
        START_BASH
        dist_test.sh
        LABELS
        "RUN_TYPE=EXCLUSIVE"
        ENVS
        "PADDLE_DIST_UT_PORT=${dist_ut_port}")
      math(EXPR dist_ut_port "${dist_ut_port}+10")
      if(dist_ut_port GREATER_EQUAL 21198)
        message(
          FATAL_ERROR "available ports have been exhausted:${dist_ut_port}")
      endif()
    endforeach()
  endif()
endif()

if((WITH_GPU) AND (CUDA_VERSION GREATER_EQUAL 11.6))
  py_test_modules(test_fuse_gemm_epilogue_pass_deprecated MODULES
                  test_fuse_gemm_epilogue_pass_deprecated)
endif()

if(WIN32)
  py_test_modules(
    test_feed_data_check_shape_type_deprecated MODULES
    test_feed_data_check_shape_type_deprecated ENVS CUDA_VISIBLE_DEVICES=0)
else()
  py_test_modules(test_feed_data_check_shape_type_deprecated MODULES
                  test_feed_data_check_shape_type_deprecated)
endif()

py_test_modules(test_data_norm_op_deprecated MODULES
                test_data_norm_op_deprecated)
py_test_modules(
  test_fuse_bn_act_pass_deprecated
  MODULES
  test_fuse_bn_act_pass_deprecated
  ENVS
  FLAGS_cudnn_deterministic=1
  FLAGS_cudnn_batchnorm_spatial_persistent=1
  FLAGS_conv_workspace_size_limit=1000)

if(NOT WIN32)
  # TODO: fix these unittests failure on Windows
  py_test_modules(test_layers_deprecated MODULES test_layers_deprecated ENVS
                  FLAGS_cudnn_deterministic=1)
endif()

set_tests_properties(
  test_data_norm_op_deprecated test_dataloader_keep_order_deprecated
  test_dataloader_unkeep_order_deprecated PROPERTIES LABELS "RUN_TYPE=DIST")

if(NOT WIN32)
  set_tests_properties(test_multiprocess_reader_exception_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
  set_tests_properties(test_layers_deprecated PROPERTIES TIMEOUT 120)
endif()

# setting timeout value as 15S
set_tests_properties(test_imperative_lod_tensor_to_selected_rows_deprecated
                     PROPERTIES TIMEOUT 200)

set_tests_properties(test_deformable_conv_op_deprecated PROPERTIES TIMEOUT 200)
set_tests_properties(test_regularizer_api_deprecated PROPERTIES TIMEOUT 150)

if(NOT WIN32)
  if(WITH_NV_JETSON)
    set_tests_properties(test_ir_memory_optimize_nlp PROPERTIES TIMEOUT 1200)
  endif()
endif()
set_tests_properties(test_add_reader_dependency_deprecated PROPERTIES TIMEOUT
                                                                      120)

if(WITH_NV_JETSON)
  set_tests_properties(test_conv3d_transpose_part2_op_deprecated
                       PROPERTIES TIMEOUT 1200)
  set_tests_properties(test_layer_norm_op_deprecated PROPERTIES TIMEOUT 1500)
else()
  set_tests_properties(test_conv3d_transpose_part2_op_deprecated
                       PROPERTIES TIMEOUT 120)
  set_tests_properties(test_layer_norm_op_deprecated PROPERTIES TIMEOUT 250)
endif()

set_tests_properties(test_argsort_op_deprecated PROPERTIES TIMEOUT 120)
set_tests_properties(test_sgd_op_deprecated PROPERTIES TIMEOUT 250)
set_tests_properties(test_generator_dataloader_deprecated PROPERTIES TIMEOUT
                                                                     120)
set_tests_properties(test_program_prune_backward_deprecated PROPERTIES TIMEOUT
                                                                       120)
set_tests_properties(test_decoupled_py_reader_deprecated PROPERTIES TIMEOUT 120)
set_tests_properties(test_fuse_bn_act_pass_deprecated PROPERTIES TIMEOUT 120)
set_tests_properties(test_conv2d_api_deprecated PROPERTIES TIMEOUT 120)
set_tests_properties(test_regularizer_deprecated PROPERTIES TIMEOUT 150)
set_tests_properties(test_slice_op_deprecated PROPERTIES TIMEOUT 120)
set_tests_properties(test_dataloader_keep_order_deprecated PROPERTIES TIMEOUT
                                                                      120)
set_tests_properties(test_dataloader_unkeep_order_deprecated PROPERTIES TIMEOUT
                                                                        120)
set_tests_properties(test_reader_reset_deprecated PROPERTIES TIMEOUT 120)
set_tests_properties(test_split_program_deprecated PROPERTIES TIMEOUT 120)
set_tests_properties(test_uniform_random_op_deprecated PROPERTIES TIMEOUT 60)

set_tests_properties(test_inplace_addto_strategy_deprecated PROPERTIES TIMEOUT
                                                                       120)
if(WITH_DISTRIBUTE
   AND WITH_GPU
   AND WITH_NCCL)
  set_tests_properties(test_auto_parallel_autoconvert_deprecated
                       PROPERTIES TIMEOUT 120)
  set_tests_properties(test_auto_parallel_data_unshard_deprecated
                       PROPERTIES TIMEOUT 120)
  set_tests_properties(test_auto_parallel_save_load_deprecated
                       PROPERTIES TIMEOUT 120)
endif()

set(TEST_CINN_OPS
    test_slice_op_deprecated test_layer_norm_op_deprecated
    test_instance_norm_op_deprecated test_group_norm_op_deprecated)

foreach(TEST_CINN_OP ${TEST_CINN_OPS})
  if(WITH_CINN)
    set_tests_properties(${TEST_CINN_OP} PROPERTIES LABELS "RUN_TYPE=CINN")

    get_test_property(${TEST_CINN_OP} TIMEOUT ORIGIN_TIME_OUT)
    if((NOT ${ORIGIN_TIME_OUT}) OR (${ORIGIN_TIME_OUT} LESS 200))
      set_tests_properties(${TEST_CINN_OP} PROPERTIES TIMEOUT 200)
    endif()
  endif()
endforeach()

# In test_conditional_block_deprecated, the sub block changes the dtype and place of the output variable.
# The changed variable is used in the following op. Static build is not supported for this case.
set_tests_properties(test_conditional_block_deprecated
                     PROPERTIES ENVIRONMENT "FLAGS_new_executor_static_build=0")

# These UTs are to temporarily test static build for standalone_executor, will be removed after static build is enabled by default.
set(STATIC_BUILD_TESTS
    test_batch_norm_op_deprecated
    test_decoupled_py_reader_deprecated
    test_fuse_bn_act_pass_deprecated
    test_layer_norm_op_deprecated
    test_lookup_table_v2_op_deprecated
    test_momentum_op_deprecated
    test_nce_deprecated
    test_sparse_conv_op
    test_tensor_array_to_tensor_deprecated
    test_unique
    test_one_hot_v2_op)

if(NOT WITH_GPU)
  list(REMOVE_ITEM STATIC_BUILD_TESTS test_fused_feedforward_op_pass)
endif()

if(WITH_COVERAGE)
  list(REMOVE_ITEM STATIC_BUILD_TESTS test_unique)
endif()

foreach(STATIC_BUILD_TEST ${STATIC_BUILD_TESTS})
  py_test_modules(
    ${STATIC_BUILD_TEST}_static_build MODULES ${STATIC_BUILD_TEST} ENVS
    FLAGS_new_executor_static_build=true FLAGS_enable_pir_api=0)
endforeach()

set_tests_properties(test_decoupled_py_reader_deprecated_static_build
                     PROPERTIES TIMEOUT 120)
set_tests_properties(test_fuse_bn_act_pass_deprecated_static_build
                     PROPERTIES TIMEOUT 120)
set_tests_properties(
  test_fuse_bn_act_pass_deprecated_static_build
  PROPERTIES
    ENVIRONMENT
    "FLAGS_cudnn_deterministic=1;FLAGS_cudnn_batchnorm_spatial_persistent=1;FLAGS_conv_workspace_size_limit=1000"
)
set_tests_properties(test_layer_norm_op_deprecated_static_build
                     PROPERTIES TIMEOUT 1500)

set_pir_tests_properties()

set_tests_properties(test_apply_pass_to_program_deprecated PROPERTIES TIMEOUT
                                                                      120)
set_tests_properties(test_conv3d_layer_deprecated PROPERTIES TIMEOUT 100)
set_tests_properties(test_attribute_var_deprecated PROPERTIES TIMEOUT 100)
set_tests_properties(test_inference_api_deprecated PROPERTIES TIMEOUT 100)
set_tests_properties(test_lbfgs_deprecated PROPERTIES TIMEOUT 100)
set_tests_properties(test_group_norm_op_deprecated PROPERTIES TIMEOUT 1000)
