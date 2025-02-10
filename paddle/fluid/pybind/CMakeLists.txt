set(PYBIND_DEPS
    init
    init_phi
    pybind
    python
    proto_desc
    standalone_executor
    executor
    fleet_wrapper
    box_wrapper
    metrics
    prune
    feed_fetch_method
    pass
    generate_pass
    pass_builder
    compiled_program
    layer
    tracer
    engine
    common
    scope_pool
    analysis_predictor
    imperative_profiler
    imperative_flag
    dlpack_tensor
    device_context
    gloo_wrapper
    infer_io_utils
    heter_wrapper
    op_version_registry
    ps_gpu_wrapper
    custom_operator
    cost_model
    global_utils
    phi_utils
    phi
    op_dialect_vjp
    program_translator
    pir_transforms
    pir_save_load
    new_profiler
    fluid_jit
    prim_utils
    detail_op_handle
    type_info
    auto_parallel)

if(WITH_GPU)
  list(APPEND PYBIND_DEPS gpu_event_timer)
endif()

if(WITH_CINN)
  set(PYBIND_DEPS ${PYBIND_DEPS} pir_transforms cinn_transforms
                  sub_graph_checker add_cinn_pass)
  list(REMOVE_ITEM PYBIND_DEPS imperative_flag)
endif()

if(WITH_PSCORE)
  set(PYBIND_DEPS ${PYBIND_DEPS} ps_service)
  if(WITH_HETERPS)
    set(PYBIND_DEPS ${PYBIND_DEPS} graph_gpu_wrapper)
  endif()
endif()
if(WITH_RPC)
  set(PYBIND_DEPS ${PYBIND_DEPS} paddle_rpc ${EXTERNAL_BRPC_DEPS} zlib phi
                  common)
endif()

if(WITH_IPU)
  set(PYBIND_DEPS ${PYBIND_DEPS} ipu_info)
endif()

if(WITH_NCCL OR WITH_RCCL)
  set(PYBIND_DEPS ${PYBIND_DEPS} nccl_wrapper)
  set(PYBIND_DEPS ${PYBIND_DEPS} reducer)
endif()

if(WITH_XPU_BKCL)
  set(PYBIND_DEPS ${PYBIND_DEPS} reducer)
  set(PYBIND_DEPS ${PYBIND_DEPS} bkcl_context)
  set(PYBIND_DEPS ${PYBIND_DEPS} heter_ccl_context)
endif()

if(WITH_CUSTOM_DEVICE)
  set(PYBIND_DEPS ${PYBIND_DEPS} xccl_context)
  if(NOT
     (WITH_NCCL
      OR WITH_RCCL
      OR WITH_XPU_BKCL))
    set(PYBIND_DEPS ${PYBIND_DEPS} reducer)
    set(PYBIND_DEPS ${PYBIND_DEPS} heter_ccl_context)
  endif()
endif()

if(NOT WIN32)
  set(PYBIND_DEPS ${PYBIND_DEPS} data_loader)
  if(WITH_NCCL OR WITH_RCCL)
    set(PYBIND_DEPS ${PYBIND_DEPS} nccl_context)
    set(PYBIND_DEPS ${PYBIND_DEPS} heter_ccl_context)
  endif()
endif()

if(WITH_PYTHON)
  # list(APPEND PYBIND_DEPS py_func_op)
  set(PYBIND_DEPS ${PYBIND_DEPS} py_func_op)
endif()

set(PYBIND_SRCS
    pybind.cc
    imperative.cc
    inference_api.cc
    control_flow_api.cc
    pir.cc
    pir_utils.cc
    graph.cc
    reader_py.cc
    protobuf.cc
    exception.cc
    op_function_common.cc
    compiled_program.cc
    tensor.cc
    place.cc
    const_value.cc
    global_value_getter_setter.cc
    fleet_wrapper_py.cc
    heter_wrapper_py.cc
    ps_gpu_wrapper_py.cc
    gloo_wrapper_py.cc
    box_helper_py.cc
    metrics_py.cc
    data_set_py.cc
    bind_cost_model.cc
    compatible.cc
    io.cc
    generator_py.cc
    communication.cc
    cuda_streams_py.cc
    custom_device_py.cc
    xpu_streams_py.cc
    jit.cc
    auto_parallel_py.cc
    sot/eval_frame_tools.cc
    sot/cpython_internals.c
    sot/frame_proxy.c
    sot/eval_frame.c
    sot/guards.cc
    op_callstack_utils.cc
    python_callable_registry.cc)

#ifdef PADDLE_WITH_DISTRIBUTE
set(PYBIND_SRCS ${PYBIND_SRCS} dist_api.cc)
#endif

if(NOT WITH_SHARED_IR)
  # Note: We want to compile pir source into paddle.so directly, because
  # we can't get all pir symbols when link pir static lib
  file(GLOB_RECURSE PIR_CPP_SOURCES "${CMAKE_SOURCE_DIR}/paddle/pir/*.cc")
  set(PYBIND_SRCS ${PYBIND_SRCS} ${PIR_CPP_SOURCES})
else()
  set(PYBIND_DEPS ${PYBIND_DEPS} pir)
endif()

if(WITH_CUSTOM_DEVICE)
  set(PYBIND_DEPS ${PYBIND_DEPS} custom_device_common_op_registry)
endif()

if(WITH_PYTHON)
  set(PYBIND_DEPS ${PYBIND_DEPS} process_group eager_reducer)
  if(WITH_NCCL OR WITH_RCCL)
    set(PYBIND_DEPS ${PYBIND_DEPS} process_group_nccl async_load)
  endif()
  if(WITH_XPU_BKCL)
    set(PYBIND_DEPS ${PYBIND_DEPS} process_group_bkcl)
  endif()
  if(WITH_GLOO)
    set(PYBIND_DEPS ${PYBIND_DEPS} process_group_gloo)
  endif()
  if(WITH_MPI)
    set(PYBIND_DEPS ${PYBIND_DEPS} process_group_mpi)
  endif()
  if(WITH_UCC)
    set(PYBIND_DEPS ${PYBIND_DEPS} process_group_ucc)
  endif()
  if(WITH_CUSTOM_DEVICE)
    set(PYBIND_DEPS ${PYBIND_DEPS} process_group_custom)
  endif()
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
    set(DISTRIBUTE_COMPILE_FLAGS "${DISTRIBUTE_COMPILE_FLAGS} -faligned-new")
    set_source_files_properties(
      distributed_py.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
  endif()
  set(PYBIND_SRCS ${PYBIND_SRCS} distributed_py.cc)
endif()
set(PYBIND_DEPS ${PYBIND_DEPS} processgroup_comm_utils)

if(WITH_GLOO)
  set(PYBIND_DEPS ${PYBIND_DEPS} gloo_context)
  set(PYBIND_SRCS ${PYBIND_SRCS} gloo_context_py.cc)
  set(PYBIND_DEPS ${PYBIND_DEPS} imperative_gloo_context)
  set(PYBIND_DEPS ${PYBIND_DEPS} reducer)
endif()

if(WITH_CRYPTO)
  set(PYBIND_DEPS ${PYBIND_DEPS} framework_io)
  set(PYBIND_SRCS ${PYBIND_SRCS} crypto.cc)
endif()

if(WITH_PSLIB)
  set(DISTRIBUTE_COMPILE_FLAGS
      "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor -Wno-error=return-type -Wno-error=unused-but-set-variable -Wno-error=type-limits -Wno-error=parentheses -Wno-error=unused-result"
  )
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
    set(DISTRIBUTE_COMPILE_FLAGS "${DISTRIBUTE_COMPILE_FLAGS} -faligned-new")
  endif()
  set_source_files_properties(
    heter_wrapper_py.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
endif()
if(WITH_PSCORE)
  if(WITH_ARM_BRPC)
    set(DISTRIBUTE_COMPILE_FLAGS
        "-faligned-new -Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor -Wno-error=return-type -Wno-error=unused-but-set-variable -Wno-error=parentheses -Wno-error=unused-result"
    )
  else()
    set(DISTRIBUTE_COMPILE_FLAGS
        "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor -Wno-error=return-type -Wno-error=unused-but-set-variable -Wno-error=parentheses -Wno-error=unused-result"
    )
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
      set(DISTRIBUTE_COMPILE_FLAGS "${DISTRIBUTE_COMPILE_FLAGS} -faligned-new")
    endif()
  endif()
  set_source_files_properties(
    fleet_py.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
  list(APPEND PYBIND_DEPS fleet index_wrapper index_sampler)
  list(APPEND PYBIND_SRCS)
  set(PYBIND_SRCS fleet_py.cc ${PYBIND_SRCS})
endif()

if(WITH_CINN)
  set(PYBIND_SRCS ${PYBIND_SRCS} test.cc)
endif()

if(WITH_RPC)
  if(WITH_ARM_BRPC)
    set(DISTRIBUTE_COMPILE_FLAGS
        "-faligned-new -Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor -Wno-error=return-type -Wno-error=unused-but-set-variable -Wno-error=parentheses -Wno-error=unused-result"
    )
  else()
    set(DISTRIBUTE_COMPILE_FLAGS
        "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor -Wno-error=return-type -Wno-error=unused-but-set-variable -Wno-error=parentheses -Wno-error=unused-result"
    )
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
      set(DISTRIBUTE_COMPILE_FLAGS "${DISTRIBUTE_COMPILE_FLAGS} -faligned-new")
    endif()
  endif()
  set_source_files_properties(rpc.cc PROPERTIES COMPILE_FLAGS
                                                ${DISTRIBUTE_COMPILE_FLAGS})
  set(PYBIND_SRCS rpc.cc ${PYBIND_SRCS})
endif()

if(WITH_NCCL OR WITH_RCCL)
  list(APPEND PYBIND_SRCS nccl_wrapper_py.cc)
endif()

if(WITH_PYTHON)
  # generate op pybind functions automatically for dygraph.

  set(OP_FUNCTION_GENERATOR_DEPS
      pybind
      proto_desc
      executor
      layer
      tracer
      engine
      imperative_profiler
      imperative_flag)
  list(APPEND OP_FUNCTION_GENERATOR_DEPS ${GLOB_OP_LIB})
  list(APPEND OP_FUNCTION_GENERATOR_DEPS ${GLOB_OPERATOR_DEPS})

  if(WITH_NCCL OR WITH_RCCL)
    list(APPEND OP_FUNCTION_GENERATOR_DEPS nccl_context)
  endif()

  if(WITH_XPU_BKCL)
    list(APPEND OP_FUNCTION_GENERATOR_DEPS bkcl_context)
  endif()

  if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
    list(APPEND OP_FUNCTION_GENERATOR_DEPS ${PYTHON_LIBRARIES})
  endif()

  if(WITH_CUSTOM_DEVICE)
    set(OP_FUNCTION_GENERATOR_DEPS ${OP_FUNCTION_GENERATOR_DEPS}
                                   custom_device_common_op_registry)
  endif()

  set(GENERATOR_DEPS ${PYBIND_DEPS})
  list(REMOVE_DUPLICATES GENERATOR_DEPS)
  if(WIN32)
    list(REMOVE_ITEM GENERATOR_DEPS python)
  endif()

  get_property(os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)

  set(op_function_output_path ${CMAKE_SOURCE_DIR}/paddle/fluid/pybind/)
  set(CODE_GEN_SPLIT_FILE_COUNT "8")
  set(eager_impl_file
      ${CMAKE_SOURCE_DIR}/paddle/fluid/pybind/eager_legacy_op_function.cc)
  set(tmp_eager_impl_file ${eager_impl_file}.tmp)

  execute_process(
    COMMAND
      "${PYTHON_EXECUTABLE}"
      "${PADDLE_SOURCE_DIR}/paddle/fluid/pybind/generate_file_structures.py"
      "${PADDLE_SOURCE_DIR}/paddle/fluid/pybind/"
      "${CODE_GEN_SPLIT_FILE_COUNT}")

  set(EAGER_GENERATOR_DEPS
      ${GLOB_OP_LIB}
      ${GLOB_OPERATOR_DEPS}
      pybind
      proto_desc
      executor
      layer
      tracer
      engine
      imperative_profiler
      imperative_flag)

  if(WITH_CINN)
    list(REMOVE_ITEM EAGER_GENERATOR_DEPS imperative_flag)
  endif()

  if(WITH_CUSTOM_DEVICE)
    set(EAGER_GENERATOR_DEPS ${EAGER_GENERATOR_DEPS}
                             custom_device_common_op_registry)
  endif()

  add_executable(
    eager_generator generator.cc eager_legacy_op_function_generator.cc
                    eager_generator.cc)
  target_link_libraries(eager_generator ${EAGER_GENERATOR_DEPS})
  target_link_libraries(eager_generator ${GENERATOR_DEPS})

  get_property(os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
  target_link_libraries(eager_generator ${os_dependency_modules})

  set(EAGER_OP_IMPL_DEPS eager_generator eager_python_c_codegen)

  if(WITH_ROCM)
    target_link_libraries(eager_generator ${ROCM_HIPRTC_LIB})
  endif()

  if(WITH_CINN)
    target_link_libraries(eager_generator ${PYTHON_LIBRARIES})
  endif()

  if(WIN32)
    set(EAGER_CODEGEN_DEPS eager_generator)
    if("${CMAKE_GENERATOR}" STREQUAL "Ninja")
      set(eager_generator_path "${CMAKE_CURRENT_BINARY_DIR}")
    else()
      set(eager_generator_path
          "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}")
    endif()

    if(WITH_SHARED_PHI)
      message("Copied phi.dll for Eager AutoCodeGen")
      add_custom_command(
        OUTPUT ${eager_generator_path}/phi.dll
        COMMAND ${CMAKE_COMMAND} -E copy ${PHI_LIB} ${eager_generator_path}
        DEPENDS phi)
      add_custom_command(
        OUTPUT ${eager_generator_path}/phi_core.dll
        COMMAND ${CMAKE_COMMAND} -E copy ${PHI_CORE_LIB} ${eager_generator_path}
        DEPENDS phi)
      if(WITH_GPU OR WITH_ROCM)
        add_custom_command(
          OUTPUT ${eager_generator_path}/phi_gpu.dll
          COMMAND ${CMAKE_COMMAND} -E copy ${PHI_GPU_LIB}
                  ${eager_generator_path}
          DEPENDS phi)
      endif()
      list(APPEND EAGER_CODEGEN_DEPS ${eager_generator_path}/phi.dll)
    endif()

    add_custom_command(
      OUTPUT ${eager_generator_path}/common.dll
      COMMAND ${CMAKE_COMMAND} -E copy ${COMMON_LIB} ${eager_generator_path}
      DEPENDS common)
    list(APPEND EAGER_CODEGEN_DEPS ${eager_generator_path}/common.dll)

    if(WITH_SHARED_IR)
      add_custom_command(
        OUTPUT ${eager_generator_path}/ir.dll
        COMMAND ${CMAKE_COMMAND} -E copy ${IR_LIB} ${eager_generator_path}
        DEPENDS pir)
      list(APPEND EAGER_CODEGEN_DEPS ${eager_generator_path}/ir.dll)
    endif()

    if(${CBLAS_PROVIDER} STREQUAL MKLML)
      message("Copied libiomp5md.dll for Eager AutoCodeGen")
      add_custom_command(
        OUTPUT ${eager_generator_path}/libiomp5md.dll
        COMMAND ${CMAKE_COMMAND} -E copy ${MKLML_SHARED_IOMP_LIB}
                ${eager_generator_path}
        DEPENDS mklml)
      list(APPEND EAGER_CODEGEN_DEPS ${eager_generator_path}/libiomp5md.dll)
    else()
      message("Copied openblas.dll for Eager AutoCodeGen")
      add_custom_command(
        OUTPUT ${eager_generator_path}/openblas.dll
        COMMAND ${CMAKE_COMMAND} -E copy ${OPENBLAS_SHARED_LIB}
                ${eager_generator_path}
        DEPENDS extern_openblas)
      list(APPEND EAGER_CODEGEN_DEPS ${eager_generator_path}/openblas.dll)
    endif()

    if(WITH_ONEDNN)
      message("Copied mkldnn.dll for Eager AutoCodeGen")
      add_custom_command(
        OUTPUT ${eager_generator_path}/mkldnn.dll
        COMMAND ${CMAKE_COMMAND} -E copy ${ONEDNN_SHARED_LIB}
                ${eager_generator_path}
        DEPENDS onednn)
      list(APPEND EAGER_CODEGEN_DEPS ${eager_generator_path}/mkldnn.dll)
    endif()

    if(WITH_ONNXRUNTIME)
      message("Copied onnxruntime for Eager AutoCodeGen")
      add_custom_command(
        OUTPUT ${eager_generator_path}/onnxruntime.dll
        COMMAND ${CMAKE_COMMAND} -E copy ${ONNXRUNTIME_SHARED_LIB}
                ${eager_generator_path}
        DEPENDS onnxruntime)
      list(APPEND EAGER_CODEGEN_DEPS ${eager_generator_path}/onnxruntime.dll)
      add_custom_command(
        OUTPUT ${eager_generator_path}/paddle2onnx.dll
        COMMAND ${CMAKE_COMMAND} -E copy ${PADDLE2ONNX_LIB}
                ${eager_generator_path}
        DEPENDS paddle2onnx)
      list(APPEND EAGER_CODEGEN_DEPS ${eager_generator_path}/paddle2onnx.dll)
    endif()

    add_custom_target(
      legacy_eager_codegen
      COMMAND
        "${eager_generator_path}/eager_generator.exe"
        "${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/fluid_generated"
        "${CODE_GEN_SPLIT_FILE_COUNT}"
      DEPENDS ${EAGER_CODEGEN_DEPS}
      VERBATIM)
  else()
    add_custom_target(
      legacy_eager_codegen
      COMMAND
        ${CMAKE_COMMAND} -E env
        "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${CMAKE_CURRENT_BINARY_DIR}/../../pybind:${PADDLE_BINARY_DIR}/third_party/install/mklml/lib"
        "${CMAKE_CURRENT_BINARY_DIR}/eager_generator"
        "${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/fluid_generated"
        "${CODE_GEN_SPLIT_FILE_COUNT}"
      DEPENDS eager_generator
      VERBATIM)
  endif()

  if(WIN32)
    if("${CMAKE_GENERATOR}" STREQUAL "Ninja")
      set(op_impl_path "${CMAKE_CURRENT_BINARY_DIR}")
    else()
      set(op_impl_path "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}")
    endif()

    file(
      WRITE
      ${CMAKE_BINARY_DIR}/paddle/fluid/pybind/eager_legacy_op_function_generator_retry.bat
      ""
      "set build_times=1\n"
      ":retry\n"
      "ECHO eager_legacy_op_function_generator run %build_times% time\n"
      "taskkill /f /im eager_generator.exe 2>NUL\n"
      "${op_impl_path}/eager_generator.exe ${tmp_eager_impl_file}\n"
      "if %ERRORLEVEL% NEQ 0 (\n"
      "    set /a build_times=%build_times%+1\n"
      "    if %build_times% GEQ 10 (\n"
      "        exit /b 1\n"
      "    ) else (\n"
      "        goto :retry\n"
      "    )\n"
      ")\n"
      "exit /b 0")

    if(WITH_SHARED_PHI)
      add_custom_command(
        OUTPUT ${op_impl_path}/phi.dll
        COMMAND ${CMAKE_COMMAND} -E copy ${PHI_LIB} ${op_impl_path}
        DEPENDS phi)
      add_custom_command(
        OUTPUT ${op_impl_path}/phi_core.dll
        COMMAND ${CMAKE_COMMAND} -E copy ${PHI_CORE_LIB} ${op_impl_path}
        DEPENDS phi)
      if(WITH_GPU OR WITH_ROCM)
        add_custom_command(
          OUTPUT ${op_impl_path}/phi_gpu.dll
          COMMAND ${CMAKE_COMMAND} -E copy ${PHI_GPU_LIB} ${op_impl_path}
          DEPENDS phi)
      endif()
      list(APPEND EAGER_OP_IMPL_DEPS ${op_impl_path}/phi.dll)
    endif()

    if(WITH_SHARED_IR)
      add_custom_command(
        OUTPUT ${op_impl_path}/ir.dll
        COMMAND ${CMAKE_COMMAND} -E copy ${IR_LIB} ${op_impl_path}
        DEPENDS pir)
      list(APPEND EAGER_OP_IMPL_DEPS ${op_impl_path}/ir.dll)
    endif()

    add_custom_command(
      OUTPUT ${op_impl_path}/common.dll
      COMMAND ${CMAKE_COMMAND} -E copy ${COMMON_LIB} ${op_impl_path}
      DEPENDS common)
    list(APPEND EAGER_OP_IMPL_DEPS ${op_impl_path}/common.dll)

    if(${CBLAS_PROVIDER} STREQUAL MKLML)
      add_custom_command(
        OUTPUT ${op_impl_path}/libiomp5md.dll
        COMMAND ${CMAKE_COMMAND} -E copy ${MKLML_SHARED_IOMP_LIB}
                ${op_impl_path}
        DEPENDS mklml)
      list(APPEND OP_IMPL_DEPS ${op_impl_path}/libiomp5md.dll)
      list(APPEND EAGER_OP_IMPL_DEPS ${op_impl_path}/libiomp5md.dll)
    else()
      add_custom_command(
        OUTPUT ${op_impl_path}/openblas.dll
        COMMAND ${CMAKE_COMMAND} -E copy ${OPENBLAS_SHARED_LIB} ${op_impl_path}
        DEPENDS extern_openblas)
      list(APPEND OP_IMPL_DEPS ${op_impl_path}/openblas.dll)
      list(APPEND EAGER_OP_IMPL_DEPS ${op_impl_path}/openblas.dll)
    endif()
    if(WITH_ONEDNN)
      add_custom_command(
        OUTPUT ${op_impl_path}/mkldnn.dll
        COMMAND ${CMAKE_COMMAND} -E copy ${ONEDNN_SHARED_LIB} ${op_impl_path}
        DEPENDS onednn)
      list(APPEND OP_IMPL_DEPS ${op_impl_path}/mkldnn.dll)
      list(APPEND EAGER_OP_IMPL_DEPS ${op_impl_path}/mkldnn.dll)
    endif()
    if(WITH_ONNXRUNTIME)
      add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/paddle2onnx.dll
        COMMAND ${CMAKE_COMMAND} -E copy ${PADDLE2ONNX_LIB}
                ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS paddle2onnx)
      list(APPEND OP_IMPL_DEPS ${CMAKE_CURRENT_BINARY_DIR}/paddle2onnx.dll)
      list(APPEND EAGER_OP_IMPL_DEPS
           ${CMAKE_CURRENT_BINARY_DIR}/paddle2onnx.dll)

      add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.dll
        COMMAND ${CMAKE_COMMAND} -E copy ${ONNXRUNTIME_SHARED_LIB}
                ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS onnxruntime)
      list(APPEND OP_IMPL_DEPS ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.dll)
      list(APPEND EAGER_OP_IMPL_DEPS
           ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.dll)
    endif()

    if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
      add_custom_command(
        OUTPUT ${eager_impl_file}
        COMMAND
          ${CMAKE_BINARY_DIR}/paddle/fluid/pybind/eager_legacy_op_function_generator_retry.bat
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_eager_impl_file}
                ${eager_impl_file}
        COMMENT "copy_if_different ${tmp_eager_impl_file} to ${eager_impl_file}"
        DEPENDS ${EAGER_OP_IMPL_DEPS} legacy_eager_codegen)
    endif()
  else()
    # If there are no *.so in /usr/lib or LD_LIBRARY_PATH,
    # copy these *.so to current directory and append current directory to
    # LD_LIBRARY_PATH. This is different with Windows platform, which search
    # *.dll in current directory automatically.
    if(WITH_ONNXRUNTIME)
      set(PADDLE2ONNX_PYBIND_OUT
          ${CMAKE_CURRENT_BINARY_DIR}/${PADDLE2ONNX_LIB_NAME})
      set(ONNXRUNTIME_PYBIND_OUT
          ${CMAKE_CURRENT_BINARY_DIR}/${ONNXRUNTIME_LIB_NAME})

      add_custom_command(
        OUTPUT ${PADDLE2ONNX_PYBIND_OUT}
        COMMAND ${CMAKE_COMMAND} -E copy ${PADDLE2ONNX_LIB}
                ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS paddle2onnx)
      list(APPEND OP_IMPL_DEPS ${PADDLE2ONNX_PYBIND_OUT})
      list(APPEND EAGER_OP_IMPL_DEPS ${PADDLE2ONNX_PYBIND_OUT})

      add_custom_command(
        OUTPUT ${ONNXRUNTIME_PYBIND_OUT}
        COMMAND ${CMAKE_COMMAND} -E copy ${ONNXRUNTIME_LIB}
                ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS onnxruntime)
      list(APPEND OP_IMPL_DEPS ${ONNXRUNTIME_PYBIND_OUT})
      list(APPEND EAGER_OP_IMPL_DEPS ${ONNXRUNTIME_PYBIND_OUT})
    endif()

    if(WITH_MKLML)
      add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libiomp5.so
        COMMAND ${CMAKE_COMMAND} -E copy ${MKLML_SHARED_IOMP_LIB}
                ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS mklml)
      list(APPEND OP_IMPL_DEPS ${CMAKE_CURRENT_BINARY_DIR}/libiomp5.so)
      list(APPEND EAGER_OP_IMPL_DEPS ${CMAKE_CURRENT_BINARY_DIR}/libiomp5.so)
    endif()
    if(WITH_ONEDNN)
      add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libdnnl.so.0
        COMMAND ${CMAKE_COMMAND} -E copy ${ONEDNN_SHARED_LIB}
                ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS onednn)
      list(APPEND OP_IMPL_DEPS ${CMAKE_CURRENT_BINARY_DIR}/libdnnl.so.0)
      list(APPEND EAGER_OP_IMPL_DEPS ${CMAKE_CURRENT_BINARY_DIR}/libdnnl.so.0)
    endif()
    if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
      add_custom_command(
        OUTPUT ${eager_impl_file}
        COMMAND
          ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:."
          "${CMAKE_CURRENT_BINARY_DIR}/eager_generator" "${tmp_eager_impl_file}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_eager_impl_file}
                ${eager_impl_file}
        COMMENT "copy_if_different ${tmp_eager_impl_file} to ${eager_impl_file}"
        DEPENDS ${EAGER_OP_IMPL_DEPS} legacy_eager_codegen
        VERBATIM)
    endif()
  endif()
  if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
    add_custom_target(eager_legacy_op_function_generator_cmd ALL
                      DEPENDS ${eager_impl_file})
  endif()

  if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
    set(PYBIND_SRCS eager.cc ${PYBIND_SRCS})
    set(PYBIND_SRCS eager_functions.cc ${PYBIND_SRCS})
    set(PYBIND_SRCS eager_method.cc ${PYBIND_SRCS})
    set(PYBIND_SRCS eager_properties.cc ${PYBIND_SRCS})
    set(PYBIND_SRCS eager_utils.cc ${PYBIND_SRCS})
    set(PYBIND_SRCS eager_py_layer.cc ${PYBIND_SRCS})
    set(PYBIND_SRCS eager_legacy_op_function.cc ${PYBIND_SRCS})
    set(PYBIND_SRCS eager_op_function.cc ${PYBIND_SRCS})
    set(PYBIND_SRCS eager_math_op_patch.cc ${PYBIND_SRCS})
    set(PYBIND_SRCS ops_api.cc ${PYBIND_SRCS})
    set(PYBIND_SRCS static_op_function.cc ${PYBIND_SRCS})
    list(APPEND PYBIND_DEPS eager_api)
    list(APPEND PYBIND_DEPS autograd_meta)
    list(APPEND PYBIND_DEPS backward)
    list(APPEND PYBIND_DEPS grad_node_info)
    list(APPEND PYBIND_DEPS phi)
    list(APPEND PYBIND_DEPS common)
    list(APPEND PYBIND_DEPS final_dygraph_function)
    list(APPEND PYBIND_DEPS final_dygraph_node)
    list(APPEND PYBIND_DEPS dygraph_function)
    list(APPEND PYBIND_DEPS dygraph_node)
    list(APPEND PYBIND_DEPS accumulation_node)
    list(APPEND PYBIND_DEPS py_layer_node)
    list(APPEND PYBIND_DEPS global_utils)
    list(APPEND PYBIND_DEPS utils)
    list(APPEND PYBIND_DEPS python)
    list(APPEND PYBIND_DEPS custom_operator)
    list(APPEND PYBIND_DEPS custom_operator_node)
    list(APPEND PYBIND_DEPS eager_tensor_operants)
    list(APPEND PYBIND_DEPS pybind_util)
  endif()

  if(TENSORRT_FOUND)
    list(APPEND PYBIND_DEPS pir_tensorrt_plugin)
  endif()

  # On Linux, cc_library(paddle SHARED ..) will generate the libpaddle.so,
  # add a prefix `lib` by default, but on Windows, cc_library(paddle SHARED ..)
  # will not add prefix, so it generate paddle.lib and paddle.pyd,
  # we need to pay attention to the difference
  set(SHARD_LIB_NAME paddle)
  if(WIN32)
    set(SHARD_LIB_NAME libpaddle)
  endif()
  set_property(GLOBAL PROPERTY PADDLE_LIB_NAME ${SHARD_LIB_NAME})
  cc_library(
    ${SHARD_LIB_NAME} SHARED
    SRCS ${PYBIND_SRCS}
    DEPS ${PYBIND_DEPS} ${GLOB_OP_LIB} ${GLOB_OPERATOR_DEPS})

  if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    target_compile_options(${SHARD_LIB_NAME} PRIVATE -Wno-maybe-uninitialized)
  endif()

  # cc_test do not respect deps, whole archive to link symbols that may need by test
  if(WITH_TESTING)
    #set_target_properties(${SHARD_LIB_NAME} PROPERTIES LINK_FLAGS "-Wl,--whole-archive")
  endif()

  # TODO(zhiqiu): some symbols not exported even setting the following
  # property. Need to find a better way.

  # if(WIN32)
  #   set_property(TARGET ${SHARD_LIB_NAME}
  #                PROPERTY WINDOWS_EXPORT_ALL_SYMBOLS ON)
  # endif()

  add_dependencies(${SHARD_LIB_NAME} ops_api_gen)
  if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
    add_dependencies(${SHARD_LIB_NAME} legacy_eager_codegen)
    add_dependencies(${SHARD_LIB_NAME} eager_legacy_op_function_generator_cmd)
  endif()

  if(NOT APPLE AND NOT WIN32)
    target_link_libraries(${SHARD_LIB_NAME} rt)
  endif()

  if(WITH_ROCM)
    target_link_libraries(${SHARD_LIB_NAME} ${ROCM_HIPRTC_LIB})
  endif()

  if(WITH_MPI)
    target_link_libraries(${SHARD_LIB_NAME} ${MPI_CXX_LIBRARIES})
  endif()

  if(WITH_UCC)
    target_link_libraries(${SHARD_LIB_NAME} ${UCC_LIBRARIES})
  endif()

  get_property(os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
  target_link_libraries(${SHARD_LIB_NAME} ${os_dependency_modules})

endif()
