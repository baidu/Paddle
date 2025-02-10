core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  dim.cc
  domain.cc
  domain_add_unit_loop_mutator.cc
  map.cc
  stage.cc
  isl_utils.cc
  schedule.cc
  naive_scheduler.cc
  poly_scheduler.cc
  ast_gen.cc
  graph.cc
  compute_at_transform.cc)
