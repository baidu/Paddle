if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
  add_subdirectory(fluid_manual)
  set(fluid_manual_functions
      ${fluid_manual_functions}
      PARENT_SCOPE)
  set(fluid_manual_nodes
      ${fluid_manual_nodes}
      PARENT_SCOPE)

  add_subdirectory(eager_manual)
  set(eager_manual_functions
      ${eager_manual_functions}
      PARENT_SCOPE)
  set(eager_manual_nodes
      ${eager_manual_nodes}
      PARENT_SCOPE)
  set(eager_manual_jvps
      ${eager_manual_jvps}
      PARENT_SCOPE)
endif()
