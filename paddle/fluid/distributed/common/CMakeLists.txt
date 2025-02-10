cc_library(
  afs_wrapper
  SRCS afs_wrapper.cc
  DEPS framework_io ps_framework_proto)

#set_property(GLOBAL PROPERTY COMMON_DEPS afs_wrapper)
