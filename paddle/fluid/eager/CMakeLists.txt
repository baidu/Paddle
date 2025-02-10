set(eager_deps
    phi
    common
    hook_utils
    utils
    global_utils
    backward
    tracer
    layer
    autograd_meta
    eager_nan_inf_utils
    grad_node_info
    grad_tensor_holder
    custom_operator_node
    fwd_utils)

if(WITH_GPU OR WITH_ROCM)
  set(eager_deps ${eager_deps} phi_gpu)
endif()

if(NOT (NOT WITH_PYTHON AND ON_INFER))
  set(eager_deps ${eager_deps} accumulation_node prim_utils fwd_utils)
endif()

set(fluid_deps tracer layer proto_desc operator op_registry variable_helper)
set(generated_deps final_dygraph_function final_dygraph_node dygraph_function
                   dygraph_node)

if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
  message("Performing Eager Dygraph Auto Code Generation")
  add_subdirectory(auto_code_generator)
endif()

add_subdirectory(api)
add_subdirectory(custom_operator)
add_subdirectory(fwd)
if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
  add_subdirectory(accumulation)
  add_subdirectory(pylayer)
  cc_library(
    grad_tensor_holder
    SRCS grad_tensor_holder.cc
    DEPS grad_node_info gradient_accumulator)
  add_dependencies(grad_tensor_holder eager_codegen)
  cc_library(
    backward
    SRCS backward.cc fwd/forward.cc fwd/forward_grad.cc fwd/forward_grad.h
    DEPS grad_tensor_holder utils autograd_meta grad_node_info phi common)
endif()

cc_library(
  eager_nan_inf_utils
  SRCS nan_inf_utils.cc
  DEPS phi common)
cc_library(
  grad_node_info
  SRCS grad_node_info.cc
  DEPS phi common)

cc_library(
  autograd_meta
  SRCS autograd_meta.cc
  DEPS phi common)
cc_library(
  utils
  SRCS utils.cc
  DEPS phi
       common
       global_utils
       layer
       proto_desc
       operator
       op_registry
       variable_helper
       generated_op
       autograd_meta
       hook_utils)

# FIXME(Aurelius84): It seems utils library is depended in cycle, but
# CMake only find it twice to deal cycle depend problem. If it is still
# not found, ld error will be raised.
set_target_properties(utils PROPERTIES LINK_INTERFACE_MULTIPLICITY 3)
