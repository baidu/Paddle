# In Windows, c_api test link must link both 2 shared to avoid symbols redefinition,
# in Linux, c_api test cant do like this or graph_to_program register more than once.
# Both Windows and Linux can only use paddle_inference_c, but this will increase size
# of build folder by 30G.
set(inference_api_tester_deps paddle_inference_api analysis_config)

if(WITH_TESTING AND WITH_INFERENCE_API_TEST)
  function(download_data install_dir data_file check_sum)
    string(REGEX MATCH "[^/\\]+$" file_name ${data_file})
    if(NOT EXISTS ${install_dir}/${file_name})
      inference_download_and_uncompress(${install_dir} ${INFERENCE_URL}
                                        ${data_file} ${check_sum})
    endif()
  endfunction()

  function(download_data_without_verify install_dir data_file)
    string(REGEX MATCH "[^/\\]+$" file_name ${data_file})
    if(NOT EXISTS ${install_dir}/${file_name})
      inference_download_and_uncompress_without_verify(
        ${install_dir} ${INFERENCE_URL} ${data_file})
    endif()
  endfunction()

  function(download_int8_data install_dir data_file check_sum)
    if(NOT EXISTS ${install_dir}/${data_file})
      inference_download_and_uncompress(${install_dir} ${INFERENCE_URL}/int8
                                        ${data_file} ${check_sum})
    endif()
  endfunction()

  function(download_int8_data_without_verify install_dir data_file)
    if(NOT EXISTS ${install_dir}/${data_file})
      inference_download_and_uncompress_without_verify(
        ${install_dir} ${INFERENCE_URL}/int8 ${data_file})
    endif()
  endfunction()

  function(download_bfloat16_data install_dir data_file check_sum)
    if(NOT EXISTS ${install_dir}/${data_file})
      inference_download_and_uncompress(
        ${install_dir} ${INFERENCE_URL}/bfloat16 ${data_file} ${check_sum})
    endif()
  endfunction()

  function(download_bfloat16_data_without_verify install_dir data_file)
    if(NOT EXISTS ${install_dir}/${data_file})
      inference_download_and_uncompress_without_verify(
        ${install_dir} ${INFERENCE_URL}/bfloat16 ${data_file})
    endif()
  endfunction()

  function(download_GRU_data install_dir data_file check_sum)
    if(NOT EXISTS ${install_dir}/${data_file})
      inference_download_and_uncompress(${install_dir} ${INFERENCE_URL}/gru
                                        ${data_file} ${check_sum})
    endif()
  endfunction()

  function(download_GRU_data_without_verify install_dir data_file)
    if(NOT EXISTS ${install_dir}/${data_file})
      inference_download_and_uncompress_without_verify(
        ${install_dir} ${INFERENCE_URL}/gru ${data_file})
    endif()
  endfunction()

  function(download_quant_data install_dir data_file check_sum)
    if(NOT EXISTS ${install_dir}/${data_file})
      inference_download_and_uncompress(
        ${install_dir} ${INFERENCE_URL}/int8/QAT_models ${data_file}
        ${check_sum})
    endif()
  endfunction()

  function(download_quant_data_without_verify install_dir data_file)
    if(NOT EXISTS ${install_dir}/${data_file})
      inference_download_and_uncompress_without_verify(
        ${install_dir} ${INFERENCE_URL}/int8/QAT_models ${data_file})
    endif()
  endfunction()

  function(download_model_and_data install_dir model_name model_check_sum
           data_name data_check_sum)
    download_data(${install_dir} ${model_name} ${model_check_sum})
    download_data(${install_dir} ${data_name} ${data_check_sum})
  endfunction()

  function(download_model_and_data_without_verify install_dir model_name
           data_name)
    download_data_without_verify(${install_dir} ${model_name})
    download_data_without_verify(${install_dir} ${data_name})
  endfunction()

  function(download_result install_dir result_name check_sum)
    download_data(${install_dir} ${result_name} ${check_sum})
  endfunction()

  function(download_result_without_verify install_dir result_name)
    download_data_without_verify(${install_dir} ${result_name})
  endfunction()

  function(inference_analysis_api_test target install_dir filename)
    inference_analysis_test(
      ${target}
      SRCS
      ${filename}
      EXTRA_DEPS
      common
      paddle_inference_shared
      ARGS
      --infer_model=${install_dir}/model
      --infer_data=${install_dir}/data.txt
      --refer_result=${install_dir}/result.txt)
  endfunction()

  function(inference_analysis_api_int8_test target install_dir filename)
    inference_analysis_test(
      ${target}
      SRCS
      ${filename}
      EXTRA_DEPS
      common
      paddle_inference_shared
      ARGS
      --infer_model=${install_dir}/model
      --infer_data=${install_dir}/data.txt
      --refer_result=${install_dir}/result.txt
      --accuracy=0.8
      --batch_size=5
      --enable_int8_ptq=true)
  endfunction()

  function(inference_multiple_models_analysis_api_test target install_dir
           filename)
    inference_analysis_test(
      ${target}
      SRCS
      ${filename}
      EXTRA_DEPS
      common
      paddle_inference_shared
      ARGS
      --infer_model=${install_dir}/mobilenet_v2_models/1
      --infer_model2=${install_dir}/mobilenet_v2_models/xx
      --infer_model3=${install_dir}/mobilenet_v2_models/3)
  endfunction()

  function(inference_analysis_api_test_build TARGET_NAME filename)
    inference_analysis_test_build(${TARGET_NAME} SRCS ${filename} EXTRA_DEPS
                                  common paddle_inference_shared)
  endfunction()

  function(inference_analysis_api_int8_test_run TARGET_NAME test_binary
           model_dir data_path)
    inference_analysis_test_run(
      ${TARGET_NAME}
      COMMAND
      ${test_binary}
      ARGS
      --infer_model=${model_dir}/model
      --infer_data=${data_path}
      --warmup_batch_size=${WARMUP_BATCH_SIZE}
      --batch_size=50
      --enable_int8_ptq=true
      --cpu_num_threads=${CPU_NUM_THREADS_ON_CI}
      --iterations=2)
  endfunction()

  function(inference_analysis_api_int8_test_run_custom_warmup_batch_size
           TARGET_NAME test_binary model_dir data_path warmup_batch_size)
    set(WARMUP_BATCH_SIZE ${warmup_batch_size})
    inference_analysis_api_int8_test_run(${TARGET_NAME} ${test_binary}
                                         ${model_dir} ${data_path})
  endfunction()

  function(inference_analysis_api_bfloat16_test_run TARGET_NAME test_binary
           model_dir data_path)
    inference_analysis_test_run(
      ${TARGET_NAME}
      COMMAND
      ${test_binary}
      ARGS
      --infer_model=${model_dir}/model
      --infer_data=${data_path}
      --batch_size=50
      --enable_bf16=true
      --paddle_num_threads=${CPU_NUM_THREADS_ON_CI}
      --iterations=2)
  endfunction()

  function(inference_analysis_api_object_detection_int8_test_run TARGET_NAME
           test_binary model_dir data_path)
    inference_analysis_test_run(
      ${TARGET_NAME}
      COMMAND
      ${test_binary}
      ARGS
      --infer_model=${model_dir}/model
      --infer_data=${data_path}
      --warmup_batch_size=10
      --batch_size=300
      --enable_int8_ptq=true
      --cpu_num_threads=${CPU_NUM_THREADS_ON_CI}
      --iterations=1)
  endfunction()

  function(inference_analysis_api_test_with_fake_data_build TARGET_NAME
           filename)
    inference_analysis_test_build(${TARGET_NAME} SRCS ${filename} EXTRA_DEPS
                                  common paddle_inference_shared)
  endfunction()

  function(inference_analysis_api_test_with_fake_data_run TARGET_NAME
           test_binary model_dir disable_fc)
    inference_analysis_test_run(
      ${TARGET_NAME} COMMAND ${test_binary} ARGS
      --infer_model=${model_dir}/model --disable_mkldnn_fc=${disable_fc})
  endfunction()

  function(
    inference_analysis_api_quant_test_run
    TARGET_NAME
    test_binary
    fp32_model_dir
    int8_model_dir
    data_path
    enable_int8_qat)
    inference_analysis_test_run(
      ${TARGET_NAME}
      COMMAND
      ${test_binary}
      ARGS
      --fp32_model=${fp32_model_dir}
      --int8_model=${int8_model_dir}
      --infer_data=${data_path}
      --batch_size=50
      --enable_int8_qat=${enable_int8_qat}
      --cpu_num_threads=${CPU_NUM_THREADS_ON_CI}
      --with_accuracy_layer=false
      --iterations=2)
  endfunction()

  function(inference_analysis_api_lexical_test_run TARGET_NAME test_binary
           infer_model data_path)
    inference_analysis_test_run(
      ${TARGET_NAME}
      COMMAND
      ${test_binary}
      ARGS
      --infer_model=${infer_model}
      --infer_data=${data_path}
      --batch_size=50
      --cpu_num_threads=${CPU_NUM_THREADS_ON_CI}
      --with_accuracy_layer=true
      --use_analysis=true
      --iterations=2)
  endfunction()

  function(inference_analysis_api_lexical_bfloat16_test_run TARGET_NAME
           test_binary infer_model data_path)
    inference_analysis_test_run(
      ${TARGET_NAME}
      COMMAND
      ${test_binary}
      ARGS
      --infer_model=${infer_model}
      --infer_data=${data_path}
      --batch_size=50
      --cpu_num_threads=${CPU_NUM_THREADS_ON_CI}
      --with_accuracy_layer=true
      --use_analysis=true
      --enable_bf16=true
      --iterations=2)
  endfunction()

  function(
    inference_analysis_api_lexical_int8_test_run
    TARGET_NAME
    test_binary
    infer_model
    data_path
    enable_int8_ptq
    enable_int8_qat
    fuse_multi_gru)
    inference_analysis_test_run(
      ${TARGET_NAME}
      COMMAND
      ${test_binary}
      ARGS
      --infer_model=${infer_model}
      --infer_data=${data_path}
      --batch_size=100
      --cpu_num_threads=${CPU_NUM_THREADS_ON_CI}
      --with_accuracy_layer=true
      --use_analysis=true
      --enable_int8_ptq=${enable_int8_ptq}
      --enable_int8_qat=${enable_int8_qat}
      --quantized_accuracy=0.015
      --fuse_multi_gru=${fuse_multi_gru}
      --iterations=4)
  endfunction()

  function(preprocess_data2bin_test_run target py_script_source data_dir
           output_file)
    py_test(${target}
            SRCS ${CMAKE_CURRENT_SOURCE_DIR}/${py_script_source} ARGS
                 --data_dir=${data_dir} --output_file=${output_file} --local)
  endfunction()

  # transformer, the dataset only works on batch_size=8 now
  set(TRANSFORMER_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/transformer")
  download_model_and_data_without_verify(
    ${TRANSFORMER_INSTALL_DIR} "temp/transformer_model.tar.gz"
    "temp/transformer_data.txt.tar.gz")

  if(WITH_GPU
     AND TENSORRT_FOUND
     AND NOT WIN32)
    set(TRT_MODEL_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/trt_models")
    set(TEST_TRT_TRANSFORMER_PRUNE_MODEL
        "${TRT_MODEL_INSTALL_DIR}/transformer_prune")
    if(NOT EXISTS ${TEST_TRT_TRANSFORMER_PRUNE_MODEL}/transformer_prune.tar.gz)
      inference_download_and_uncompress(
        ${TEST_TRT_TRANSFORMER_PRUNE_MODEL} ${INFERENCE_URL}/tensorrt_test
        "transformer_prune.tar.gz" 77b56dc73ff0cf44ddb1ce9ca0b0f471)
    endif()
    inference_analysis_test(
      test_trt_dynamic_shape_transformer_prune_deprecated
      SRCS
      trt_dynamic_shape_transformer_prune_test.cc
      EXTRA_DEPS
      paddle_inference_shared
      common
      ARGS
      --infer_model=${TEST_TRT_TRANSFORMER_PRUNE_MODEL}/transformer_prune)
    set_tests_properties(test_trt_dynamic_shape_transformer_prune_deprecated
                         PROPERTIES TIMEOUT 300)
  endif()

  # Image classification tests with fake data
  set(IMG_CLASS_TEST_APP "test_analyzer_image_classification")
  set(IMG_CLASS_TEST_APP_SRC "analyzer_image_classification_tester.cc")

  # build test binary to be used in subsequent tests
  inference_analysis_api_test_with_fake_data_build(${IMG_CLASS_TEST_APP}
                                                   ${IMG_CLASS_TEST_APP_SRC})

  # googlenet
  set(GOOGLENET_MODEL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/googlenet")
  download_data_without_verify(${GOOGLENET_MODEL_DIR} "googlenet.tar.gz")
  inference_analysis_api_test_with_fake_data_run(
    test_analyzer_googlenet ${IMG_CLASS_TEST_APP} ${GOOGLENET_MODEL_DIR} false)

  # mobilenet with depthwise_conv op
  set(MOBILENET_MODEL_DIR
      "${INFERENCE_DEMO_INSTALL_DIR}/mobilenet_depthwise_conv")
  download_data_without_verify(${MOBILENET_MODEL_DIR} "mobilenet_model.tar.gz")
  inference_analysis_api_test_with_fake_data_run(
    test_analyzer_mobilenet_depthwise_conv ${IMG_CLASS_TEST_APP}
    ${MOBILENET_MODEL_DIR} false)

  set_tests_properties(test_analyzer_googlenet PROPERTIES TIMEOUT 120)
  set_tests_properties(test_analyzer_mobilenet_depthwise_conv PROPERTIES TIMEOUT
                                                                         120)

  if(WITH_TESTING AND TEST test_api_impl)
    if(NOT APPLE)
      set_tests_properties(test_api_impl PROPERTIES TIMEOUT 120)
    endif()
  endif()
endif()
