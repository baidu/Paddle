if(WITH_TESTING)
  cinn_cc_library(cinn_gtest_main SRCS gtest_main.cc DEPS gtest common)
endif()

add_subdirectory(adt)
add_subdirectory(ast_gen_ius)
add_subdirectory(common)
add_subdirectory(utils)
add_subdirectory(poly)
add_subdirectory(runtime)
add_subdirectory(ir)
add_subdirectory(pass)
add_subdirectory(backends)
add_subdirectory(lang)
add_subdirectory(optim)
add_subdirectory(hlir)
add_subdirectory(pybind)
add_subdirectory(operator_fusion)

# Download a model
download_and_uncompress("${DOWNLOAD_MODEL_DIR}" "${PADDLE_RESOURCE_URL}"
                        "lite_naive_model.tar.gz")

core_gather_headers()
