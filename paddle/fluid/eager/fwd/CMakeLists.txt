cc_library(
  fwd_utils
  SRCS forward_grad.cc forward_grad.h forward.cc
  DEPS phi common grad_node_info)
