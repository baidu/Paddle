if(WITH_GPU)
  file(
    GLOB CINN_PIR_SYMBOLIC_TEST
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "test_*.py")

  list(
    REMOVE_ITEM
    CINN_PIR_SYMBOLIC_TEST
    test_cinn_reduce_symbolic_demo.py
    test_if_st.py
    test_if_dy.py
    test_llama_if_dy.py
    test_decomp_inference_predictor_run.py
    test_unary_op_infer_sym_shape.py
    test_sub_graph_for_backend.py
    test_sub_graph_for_frontend.py
    test_check_infer_symbolic.py
    test_multiple_subgraph_st.py
    test_multiple_subgraph_dy.py
    test_llama_mlp_st.py
    test_llama_mlp_dy.py
    test_while_st.py
    test_infer_sym_shape_utils.py
    test_dyshape_cast.py
    test_dyshape_split_with_num.py)

  foreach(cinn_pir_test_name ${CINN_PIR_SYMBOLIC_TEST})
    string(REGEX REPLACE ".py" "" cinn_pir_test_name ${cinn_pir_test_name})
    add_test(
      NAME ${cinn_pir_test_name}
      COMMAND
        ${CMAKE_COMMAND} -E env
        PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
        FLAGS_check_infer_symbolic=1 FLAGS_enable_pir_api=1
        FLAGS_prim_enable_dynamic=true FLAGS_prim_all=True
        FLAGS_cinn_new_group_scheduler=1 ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/${cinn_pir_test_name}.py
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set_tests_properties(${cinn_pir_test_name} PROPERTIES LABELS
                                                          "RUN_TYPE=CINN")
  endforeach()

  add_test(
    NAME test_if_st
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
      FLAGS_cinn_convert_static_dim_to_dynamic_dim=2048:S0
      FLAGS_enable_pir_api=1 ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/test_if_st.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(test_if_st PROPERTIES LABELS "RUN_TYPE=CINN")

  add_test(
    NAME test_if_dy
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
      FLAGS_enable_pir_api=1 ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/test_if_dy.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(test_if_dy PROPERTIES LABELS "RUN_TYPE=CINN")

  add_test(
    NAME test_llama_if_dy
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
      FLAGS_prim_all=true FLAGS_enable_pir_api=true
      FLAGS_prim_enable_dynamic=true ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/test_llama_if_dy.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(test_llama_if_dy PROPERTIES LABELS "RUN_TYPE=CINN")

  add_test(
    NAME test_cinn_reduce_symbolic_demo
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
      FLAGS_enable_pir_api=1 ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/test_cinn_reduce_symbolic_demo.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(test_cinn_reduce_symbolic_demo
                       PROPERTIES LABELS "RUN_TYPE=CINN")

  add_test(
    NAME test_sub_graph_for_backend
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
      FLAGS_cinn_convert_static_dim_to_dynamic_dim=64:S0 FLAGS_enable_pir_api=1
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/test_sub_graph_for_backend.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(test_sub_graph_for_backend PROPERTIES LABELS
                                                             "RUN_TYPE=CINN")

  add_test(
    NAME test_sub_graph_for_frontend
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
      FLAGS_enable_pir_api=1 ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/test_sub_graph_for_frontend.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(test_sub_graph_for_frontend PROPERTIES LABELS
                                                              "RUN_TYPE=CINN")

  add_test(
    NAME test_check_infer_symbolic
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
      FLAGS_check_infer_symbolic=True FLAGS_enable_pir_api=1
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/test_check_infer_symbolic.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(test_check_infer_symbolic PROPERTIES LABELS
                                                            "RUN_TYPE=CINN")

  add_test(
    NAME test_multiple_subgraph_st
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
      FLAGS_cinn_convert_static_dim_to_dynamic_dim=2048:S0
      FLAGS_enable_pir_api=1 ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/test_multiple_subgraph_st.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(test_multiple_subgraph_st PROPERTIES LABELS
                                                            "RUN_TYPE=CINN")

  add_test(
    NAME test_multiple_subgraph_dy
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
      FLAGS_enable_pir_api=1 ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/test_multiple_subgraph_dy.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(test_multiple_subgraph_dy PROPERTIES LABELS
                                                            "RUN_TYPE=CINN")

  add_test(
    NAME test_llama_mlp_st
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
      FLAGS_cinn_convert_static_dim_to_dynamic_dim=2048:S0
      FLAGS_enable_pir_api=1 ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/test_llama_mlp_st.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(test_llama_mlp_st PROPERTIES LABELS "RUN_TYPE=CINN")

  add_test(
    NAME test_llama_mlp_dy
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
      FLAGS_prim_all=true FLAGS_enable_pir_api=1 ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/test_llama_mlp_dy.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(test_llama_mlp_dy PROPERTIES LABELS "RUN_TYPE=CINN")

  add_test(
    NAME test_while_st
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
      FLAGS_cinn_convert_static_dim_to_dynamic_dim=2048:S0
      FLAGS_enable_pir_api=1 ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/test_while_st.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(test_while_st PROPERTIES LABELS "RUN_TYPE=CINN")

  add_test(
    NAME test_dyshape_cast
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
      FLAGS_prim_all=true FLAGS_enable_pir_api=1 ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/test_dyshape_cast.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(test_dyshape_cast PROPERTIES LABELS "RUN_TYPE=CINN")

  add_test(
    NAME test_dyshape_split_with_num
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
      FLAGS_prim_all=true FLAGS_enable_pir_api=1 ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/test_dyshape_split_with_num.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(test_dyshape_split_with_num PROPERTIES LABELS
                                                              "RUN_TYPE=CINN")

endif()
