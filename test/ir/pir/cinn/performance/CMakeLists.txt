if(WITH_GPU)
  file(
    GLOB CINN_PERFORMANCE_TEST
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "test_*.py")

  foreach(cinn_pir_test_name ${CINN_PERFORMANCE_TEST})
    string(REGEX REPLACE ".py" "" cinn_pir_test_name ${cinn_pir_test_name})
    add_test(
      NAME ${cinn_pir_test_name}
      COMMAND
        ${CMAKE_COMMAND} -E env
        PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
        FLAGS_check_infer_symbolic=1 FLAGS_enable_pir_api=1
        FLAGS_prim_enable_dynamic=true FLAGS_cinn_new_group_scheduler=1
        ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/${cinn_pir_test_name}.py
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    if(cinn_pir_test_name STREQUAL "test_cinn_transpose_sub_graph")
      set_tests_properties(${cinn_pir_test_name} PROPERTIES TIMEOUT 120)
    endif()

    set_tests_properties(${cinn_pir_test_name} PROPERTIES LABELS
                                                          "RUN_TYPE=CINN")
  endforeach()

endif()
