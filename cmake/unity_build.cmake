# Add the following code before all include to avoid compilation failure.
set(UNITY_CC_BEFORE_CODE
    [[
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif]])
set(UNITY_CU_BEFORE_CODE
    [[
#ifndef __CUDACC_VER_MAJOR__
#define __CUDACC_VER_MAJOR__ CUDA_COMPILER_MAJOR_VERSION
#endif
#ifndef __CUDACC_VER_MINOR__
#define __CUDACC_VER_MINOR__ CUDA_COMPILER_MINOR_VERSION
#endif]])
if(WITH_GPU)
  string(REPLACE "." ";" CUDA_COMPILER_VERSION ${CMAKE_CUDA_COMPILER_VERSION})
  list(GET CUDA_COMPILER_VERSION 0 CUDA_COMPILER_MAJOR_VERSION)
  list(GET CUDA_COMPILER_VERSION 1 CUDA_COMPILER_MINOR_VERSION)
  string(REPLACE "CUDA_COMPILER_MAJOR_VERSION" ${CUDA_COMPILER_MAJOR_VERSION}
                 UNITY_CU_BEFORE_CODE ${UNITY_CU_BEFORE_CODE})
  string(REPLACE "CUDA_COMPILER_MINOR_VERSION" ${CUDA_COMPILER_MINOR_VERSION}
                 UNITY_CU_BEFORE_CODE ${UNITY_CU_BEFORE_CODE})
endif()

# Group a list of source files that can be included together.
# This combination is just a guiding rule, and the source file of group
# do not have to exist.
# Here you need to specify the source type which belongs to cc or cu.
function(register_unity_group TYPE)
  # Get UNITY_TARGET from CMAKE_CURRENT_SOURCE_DIR.
  string(REPLACE "${PADDLE_SOURCE_DIR}/paddle/fluid/" "" UNITY_TARGET
                 ${CMAKE_CURRENT_SOURCE_DIR})
  string(REPLACE "/" "_" UNITY_TARGET ${UNITY_TARGET})
  set(UNITY_TARGET "paddle_${UNITY_TARGET}_unity")

  # Variable unity_group_index is used to record the number of UNITY_TARGET groups.
  get_property(unity_group_index GLOBAL
               PROPERTY ${UNITY_TARGET}_${TYPE}_group_index)
  if("${unity_group_index}" STREQUAL "")
    set(unity_group_index 0)
  endif()

  # Variable unity_group_sources is used to record the sources of one group.
  set(unity_group_sources
      ${UNITY_TARGET}_${TYPE}_group_${unity_group_index}_sources)
  set_property(GLOBAL PROPERTY ${unity_group_sources} "")
  foreach(src ${ARGN})
    # UB use absolute path of source.
    if(NOT IS_ABSOLUTE ${src})
      set(src ${CMAKE_CURRENT_SOURCE_DIR}/${src})
    endif()
    set_property(GLOBAL APPEND PROPERTY ${unity_group_sources} ${src})
  endforeach()

  # If unity_file does not exists, nv_library or cc_library will use
  # dummy_file. Touch unity_file to avoid to use dummy file.
  set(unity_file
      ${CMAKE_CURRENT_BINARY_DIR}/${UNITY_TARGET}_${unity_group_index}_${TYPE}.${TYPE}
  )
  if(NOT EXISTS ${unity_file})
    file(TOUCH ${unity_file})
  endif()

  math(EXPR unity_group_index "${unity_group_index} + 1")
  set_property(GLOBAL PROPERTY ${UNITY_TARGET}_${TYPE}_group_index
                               ${unity_group_index})
endfunction()

# Combine the original source files used by `TARGET`, then use
# `unity_target_${TYPE}_sources` to get the combined source files.
# If the source file does not hit any registered groups, use itself.
# This function put the actual combination relationship in variables instead of
# writing the unity source file. The reason is that writing unity source file
# will change the timestamp and affect the effect of retaining the build
# directory on Windows.
# Here you need to specify the source type which belongs to cc or cu.
function(compose_unity_target_sources TARGET TYPE)
  # Variable unity_target_sources represents the source file used in TARGET
  set(unity_target_sources "")
  get_property(unity_group_index_max GLOBAL
               PROPERTY ${TARGET}_${TYPE}_group_index)
  foreach(src ${ARGN})
    set(unity_file "")
    # Note(zhouwei25): UB use the path relative to CMAKE_SOURCE_DIR.
    # If use absolute path, sccache/ccache hit rate will be reduced.
    if(IS_ABSOLUTE ${src})
      set(src_absolute_path ${src})
      file(RELATIVE_PATH src_relative_path ${CMAKE_SOURCE_DIR} ${src})
    else()
      set(src_absolute_path ${CMAKE_CURRENT_SOURCE_DIR}/${src})
      file(RELATIVE_PATH src_relative_path ${CMAKE_SOURCE_DIR}
           ${src_absolute_path})
    endif()
    # If `unity_group_index_max` is empty, there is no combination
    # relationship.
    # TODO(Avin0323): Whether use target property `UNITY_BUILD` of CMAKE to
    # combine source files.
    if(NOT "${unity_group_index_max}" STREQUAL "")
      # Search in each registered group.
      foreach(unity_group_index RANGE ${unity_group_index_max})
        if(${unity_group_index} GREATER_EQUAL ${unity_group_index_max})
          break()
        endif()
        get_property(
          unity_group_sources GLOBAL
          PROPERTY ${TARGET}_${TYPE}_group_${unity_group_index}_sources)
        if(${src_absolute_path} IN_LIST unity_group_sources)
          set(unity_file
              ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_${unity_group_index}_${TYPE}.${TYPE}
          )
          set(unity_file_sources
              ${TARGET}_${TYPE}_file_${unity_group_index}_sources)
          get_property(
            set_unity_file_sources GLOBAL
            PROPERTY ${unity_file_sources}
            SET)
          if(NOT ${set_unity_file_sources})
            # Add macro before include source files.
            set_property(GLOBAL PROPERTY ${unity_file_sources}
                                         "// Generate by Unity Build")
            set_property(GLOBAL APPEND PROPERTY ${unity_file_sources}
                                                ${UNITY_CC_BEFORE_CODE})
            if(WITH_GPU AND "${TYPE}" STREQUAL "cu")
              set_property(GLOBAL APPEND PROPERTY ${unity_file_sources}
                                                  ${UNITY_CU_BEFORE_CODE})
            endif()
          endif()
          set_property(
            GLOBAL APPEND PROPERTY ${unity_file_sources}
                                   "#include \"${src_relative_path}\"")
          set(unity_target_sources ${unity_target_sources} ${unity_file})
          break()
        endif()
      endforeach()
    endif()
    # Use original source file.
    if("${unity_file}" STREQUAL "")
      set(unity_target_sources ${unity_target_sources} ${src})
    endif()
  endforeach()

  set(unity_target_${TYPE}_sources
      ${unity_target_sources}
      PARENT_SCOPE)
endfunction()

# Write the unity files used by `UNITY_TARGET`.
# Write dependent on whether the contents of the unity file have changed, which
# protects incremental compilation speed.
function(finish_unity_target TYPE)
  # Get UNITY_TARGET from CMAKE_CURRENT_SOURCE_DIR.
  string(REPLACE "${PADDLE_SOURCE_DIR}/paddle/fluid/" "" UNITY_TARGET
                 ${CMAKE_CURRENT_SOURCE_DIR})
  string(REPLACE "/" "_" UNITY_TARGET ${UNITY_TARGET})
  set(UNITY_TARGET "paddle_${UNITY_TARGET}_unity")

  get_property(unity_group_index_max GLOBAL
               PROPERTY ${UNITY_TARGET}_${TYPE}_group_index)
  if(NOT "${unity_group_index_max}" STREQUAL "")
    foreach(unity_group_index RANGE ${unity_group_index_max})
      if(${unity_group_index} GREATER_EQUAL ${unity_group_index_max})
        break()
      endif()
      get_property(
        unity_file_sources GLOBAL
        PROPERTY ${UNITY_TARGET}_${TYPE}_file_${unity_group_index}_sources)
      set(unity_file_read_content "")
      string(JOIN "\n" unity_file_write_content ${unity_file_sources})
      set(unity_file
          ${CMAKE_CURRENT_BINARY_DIR}/${UNITY_TARGET}_${unity_group_index}_${TYPE}.${TYPE}
      )
      file(READ ${unity_file} unity_file_read_content)
      if(NOT "${unity_file_read_content}" STREQUAL
         "${unity_file_write_content}")
        file(WRITE ${unity_file} ${unity_file_write_content})
      endif()
    endforeach()
  endif()
endfunction()
