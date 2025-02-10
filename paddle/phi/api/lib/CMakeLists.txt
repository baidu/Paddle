set(api_gen_base ${CMAKE_SOURCE_DIR}/paddle/phi/api/generator/api_base.py)

# forward api file
set(api_gen_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/generator/api_gen.py)
set(api_yaml_file ${CMAKE_SOURCE_DIR}/paddle/phi/ops/yaml/ops.yaml)
set(legacy_api_yaml_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/ops/yaml/inconsistent/dygraph_ops.yaml)
set(api_header_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/include/api.h)
set(api_source_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/lib/api.cc)
set(api_header_file_tmp ${api_header_file}.tmp)
set(api_source_file_tmp ${api_source_file}.tmp)
# dist forward api file
set(dist_api_gen_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/generator/dist_api_gen.py)

# backward api file
set(bw_api_gen_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/generator/backward_api_gen.py)
set(bw_api_yaml_file ${CMAKE_SOURCE_DIR}/paddle/phi/ops/yaml/backward.yaml)
set(legacy_bw_api_yaml_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/ops/yaml/inconsistent/dygraph_backward.yaml)
set(bw_api_header_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/backward/backward_api.h)
set(bw_api_source_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/lib/backward_api.cc)
set(bw_api_header_file_tmp ${bw_api_header_file}.tmp)
set(bw_api_source_file_tmp ${bw_api_source_file}.tmp)
# dist backward api file
set(dist_bw_api_gen_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/generator/dist_bw_api_gen.py)

# dygraph(intermediate) api file
set(im_api_gen_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/generator/intermediate_api_gen.py)
set(dygraph_api_header_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/lib/dygraph_api.h)
set(dygraph_api_source_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/lib/dygraph_api.cc)
set(dygraph_api_header_file_tmp ${dygraph_api_header_file}.tmp)
set(dygraph_api_source_file_tmp ${dygraph_api_source_file}.tmp)

# fused_op forward api file
set(fused_api_yaml_file ${CMAKE_SOURCE_DIR}/paddle/phi/ops/yaml/fused_ops.yaml)
set(fused_api_header_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/include/fused_api.h)
set(fused_api_source_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/lib/fused_api.cc)
set(fused_api_header_file_tmp ${fused_api_header_file}.tmp)
set(fused_api_source_file_tmp ${fused_api_source_file}.tmp)

# fused_op backward api file
set(fused_bw_api_gen_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/generator/backward_api_gen.py)
set(fused_bw_api_yaml_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/ops/yaml/fused_backward.yaml)
set(fused_bw_api_header_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/backward/fused_backward_api.h)
set(fused_bw_api_source_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/lib/fused_backward_api.cc)
set(fused_bw_api_header_file_tmp ${fused_bw_api_header_file}.tmp)
set(fused_bw_api_source_file_tmp ${fused_bw_api_source_file}.tmp)

# sparse api file
set(sparse_api_gen_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/generator/sparse_api_gen.py)
set(sparse_api_yaml_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/ops/yaml/sparse_ops.yaml)
set(sparse_api_header_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/include/sparse_api.h)
set(sparse_api_source_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/lib/sparse_api.cc)
set(sparse_api_header_file_tmp ${sparse_api_header_file}.tmp)
set(sparse_api_source_file_tmp ${sparse_api_source_file}.tmp)

# sparse bw api file
set(sparse_bw_api_gen_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/generator/sparse_bw_api_gen.py)
set(sparse_bw_api_yaml_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/ops/yaml/sparse_backward.yaml)
set(sparse_bw_api_header_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/backward/sparse_bw_api.h)
set(sparse_bw_api_source_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/lib/sparse_bw_api.cc)
set(sparse_bw_api_header_file_tmp ${sparse_bw_api_header_file}.tmp)
set(sparse_bw_api_source_file_tmp ${sparse_bw_api_source_file}.tmp)

# strings api file
set(strings_api_gen_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/generator/strings_api_gen.py)
set(strings_api_yaml_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/ops/yaml/strings_ops.yaml)
set(strings_api_header_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/include/strings_api.h)
set(strings_api_source_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/lib/strings_api.cc)
set(strings_api_header_file_tmp ${strings_api_header_file}.tmp)
set(strings_api_source_file_tmp ${strings_api_source_file}.tmp)

# wrapped infermeta file
set(wrapped_infermeta_gen_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/generator/wrapped_infermeta_gen.py)
set(wrapped_infermeta_header_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/infermeta/generated.h)
set(wrapped_infermeta_source_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/infermeta/generated.cc)

# tensor and tensor operants file
set(tensor_api_yaml_path
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/lib/tensor_operants.yaml)
set(tensor_gen_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/generator/tensor_operants_gen.py)
set(operants_base_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/include/operants_base.h)
set(tensor_api_source_file ${CMAKE_SOURCE_DIR}/paddle/phi/api/lib/tensor_api.cc)
set(phi_tensor_operants_header_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/include/tensor_operants.h)
set(phi_tensor_operants_source_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/lib/tensor_operants.cc)
set(operants_manager_header_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/include/operants_manager.h)
set(operants_manager_source_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/lib/operants_manager.cc)
set(operants_base_file_tmp ${operants_base_file}.tmp)
set(tensor_api_source_file_tmp ${tensor_api_source_file}.tmp)
set(phi_tensor_operants_header_file_tmp ${phi_tensor_operants_header_file}.tmp)
set(phi_tensor_operants_source_file_tmp ${phi_tensor_operants_source_file}.tmp)
set(operants_manager_header_file_tmp ${operants_manager_header_file}.tmp)
set(operants_manager_source_file_tmp ${operants_manager_source_file}.tmp)

if(NOT PYTHONINTERP_FOUND)
  find_package(PythonInterp REQUIRED)
endif()

execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pip install pyyaml)

if(WITH_DISTRIBUTE)
  # generate dist forward api
  execute_process(
    COMMAND
      ${PYTHON_EXECUTABLE} ${dist_api_gen_file} --api_yaml_path
      ${api_yaml_file} ${legacy_api_yaml_file} ${bw_api_yaml_file}
      ${legacy_bw_api_yaml_file} --api_header_path ${api_header_file_tmp}
      --api_source_path ${api_source_file_tmp})

  # generate dist backward api
  execute_process(
    COMMAND
      ${PYTHON_EXECUTABLE} ${dist_bw_api_gen_file} --backward_yaml_path
      ${bw_api_yaml_file} ${legacy_bw_api_yaml_file} --backward_header_path
      ${bw_api_header_file_tmp} --backward_source_path
      ${bw_api_source_file_tmp})
else()
  # generate forward api
  execute_process(
    COMMAND
      ${PYTHON_EXECUTABLE} ${api_gen_file} --api_yaml_path ${api_yaml_file}
      ${legacy_api_yaml_file} ${bw_api_yaml_file} ${legacy_bw_api_yaml_file}
      --api_header_path ${api_header_file_tmp} --api_source_path
      ${api_source_file_tmp})

  # generate backward api
  execute_process(
    COMMAND
      ${PYTHON_EXECUTABLE} ${bw_api_gen_file} --backward_yaml_path
      ${bw_api_yaml_file} ${legacy_bw_api_yaml_file} --backward_header_path
      ${bw_api_header_file_tmp} --backward_source_path
      ${bw_api_source_file_tmp})
endif()

if(WITH_DISTRIBUTE)
  # generate fused_op api
  execute_process(
    COMMAND
      ${PYTHON_EXECUTABLE} ${dist_api_gen_file} --api_yaml_path
      ${fused_api_yaml_file} ${fused_bw_api_yaml_file} --is_fused_ops_yaml
      --api_header_path ${fused_api_header_file_tmp} --api_source_path
      ${fused_api_source_file_tmp})

  # generate fused_op backward api
  execute_process(
    COMMAND
      ${PYTHON_EXECUTABLE} ${dist_bw_api_gen_file} --backward_yaml_path
      ${fused_bw_api_yaml_file} --is_fused_backward_yaml --backward_header_path
      ${fused_bw_api_header_file_tmp} --backward_source_path
      ${fused_bw_api_source_file_tmp})
else()
  # generate fused_op api
  execute_process(
    COMMAND
      ${PYTHON_EXECUTABLE} ${api_gen_file} --api_yaml_path
      ${fused_api_yaml_file} ${fused_bw_api_yaml_file} --is_fused_ops_yaml
      --api_header_path ${fused_api_header_file_tmp} --api_source_path
      ${fused_api_source_file_tmp})

  # generate fused_op backward api
  execute_process(
    COMMAND
      ${PYTHON_EXECUTABLE} ${bw_api_gen_file} --backward_yaml_path
      ${fused_bw_api_yaml_file} --is_fused_backward_yaml --backward_header_path
      ${fused_bw_api_header_file_tmp} --backward_source_path
      ${fused_bw_api_source_file_tmp})
endif()

# generate sparse api
execute_process(
  COMMAND
    ${PYTHON_EXECUTABLE} ${sparse_api_gen_file} --api_yaml_path
    ${sparse_api_yaml_file} ${sparse_bw_api_yaml_file} --api_header_path
    ${sparse_api_header_file_tmp} --api_source_path
    ${sparse_api_source_file_tmp})

# generate backward sparse api
execute_process(
  COMMAND
    ${PYTHON_EXECUTABLE} ${sparse_bw_api_gen_file} --api_yaml_path
    ${sparse_bw_api_yaml_file} --api_header_path
    ${sparse_bw_api_header_file_tmp} --api_source_path
    ${sparse_bw_api_source_file_tmp})

# generate strings api
execute_process(
  COMMAND
    ${PYTHON_EXECUTABLE} ${strings_api_gen_file} --api_yaml_path
    ${strings_api_yaml_file} --api_header_path ${strings_api_header_file_tmp}
    --api_source_path ${strings_api_source_file_tmp})

# generate dygraph(intermediate) api
if(WITH_DISTRIBUTE)
  execute_process(
    COMMAND
      ${PYTHON_EXECUTABLE} ${im_api_gen_file} --api_yaml_path ${api_yaml_file}
      ${bw_api_yaml_file} ${legacy_api_yaml_file} ${legacy_bw_api_yaml_file}
      --sparse_api_yaml_path ${sparse_api_yaml_file} ${sparse_bw_api_yaml_file}
      --dygraph_api_header_path ${dygraph_api_header_file_tmp}
      --dygraph_api_source_path ${dygraph_api_source_file_tmp}
      --gen_dist_branch)
else()
  execute_process(
    COMMAND
      ${PYTHON_EXECUTABLE} ${im_api_gen_file} --api_yaml_path ${api_yaml_file}
      ${bw_api_yaml_file} ${legacy_api_yaml_file} ${legacy_bw_api_yaml_file}
      --sparse_api_yaml_path ${sparse_api_yaml_file} ${sparse_bw_api_yaml_file}
      --dygraph_api_header_path ${dygraph_api_header_file_tmp}
      --dygraph_api_source_path ${dygraph_api_source_file_tmp})
endif()

# generate tensor and tensor operants file
message("create or copy auto-generated tensor files")
execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/paddle/phi/api/generator
  COMMAND
    ${PYTHON_EXECUTABLE} ${tensor_gen_file} --api_yaml_path ${api_yaml_file}
    ${legacy_api_yaml_file} --operants_base_path ${operants_base_file_tmp}
    --tensor_api_source_path ${tensor_api_source_file_tmp}
    --phi_tensor_operants_header_path ${phi_tensor_operants_header_file_tmp}
    --phi_tensor_operants_source_path ${phi_tensor_operants_source_file_tmp}
    --operants_manager_header_path ${operants_manager_header_file_tmp}
    --operants_manager_source_path ${operants_manager_source_file_tmp}
    --tensor_api_yaml_path ${tensor_api_yaml_path}
  RESULT_VARIABLE _result)
if(${_result})
  message(FATAL_ERROR "tensor codegen failed, exiting.")
endif()

set(generated_files
    "${operants_base_file}"
    "${tensor_api_source_file}"
    "${phi_tensor_operants_header_file}"
    "${phi_tensor_operants_source_file}"
    "${operants_manager_header_file}"
    "${operants_manager_source_file}"
    "${wrapped_infermeta_source_file}"
    "${api_source_file}"
    "${api_header_file}"
    "${bw_api_source_file}"
    "${bw_api_header_file}"
    "${fused_api_source_file}"
    "${fused_api_header_file}"
    "${fused_bw_api_source_file}"
    "${fused_bw_api_header_file}"
    "${sparse_api_source_file}"
    "${sparse_api_header_file}"
    "${sparse_bw_api_source_file}"
    "${sparse_bw_api_header_file}"
    "${dygraph_api_source_file}"
    "${dygraph_api_header_file}"
    "${strings_api_source_file}"
    "${strings_api_header_file}")

foreach(generated_file ${generated_files})
  if(EXISTS "${generated_file}.tmp" AND EXISTS "${generated_file}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            "${generated_file}.tmp" "${generated_file}")
    message("copy if different ${generated_file}.tmp ${generated_file}")
  elseif(EXISTS "${generated_file}.tmp")
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy "${generated_file}.tmp"
                            "${generated_file}")
    message("copy ${generated_file}.tmp ${generated_file}")
  endif()
endforeach()

collect_srcs(
  api_srcs
  SRCS
  tensor.cc
  op_meta_info.cc
  context_pool.cc
  tensor_utils.cc
  kernel_dispatch.cc
  api_gen_utils.cc
  data_transform.cc
  api_custom_impl.cc
  tensor_method.cc
  tensor_copy.cc
  scalar.cc
  int_array.cc)
collect_generated_srcs(
  api_srcs
  SRCS
  ${api_source_file}
  ${bw_api_source_file}
  ${fused_api_source_file}
  ${fused_bw_api_source_file}
  ${sparse_api_source_file}
  ${sparse_bw_api_source_file}
  ${dygraph_api_source_file}
  ${strings_api_source_file}
  ${phi_tensor_operants_source_file}
  ${operants_manager_source_file}
  ${tensor_api_source_file})
