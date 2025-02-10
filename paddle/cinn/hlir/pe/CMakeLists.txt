cinn_proto_library(param_proto SRCS schedule_param.proto)

core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  broadcast.cc
  elementwise.cc
  nn.cc
  nn_util.cc
  reduction.cc
  load_x86_params.cc
  schedule.cc
  ir_schedule_pe.cc
  transform.cc
  vision.cc)

gather_srcs(cinnapi_src SRCS map_expr_to_ir.cc)

foreach(header ${param_proto_HDRS})
  set(core_proto_includes
      "${core_proto_includes};${header}"
      CACHE INTERNAL "")
endforeach()
