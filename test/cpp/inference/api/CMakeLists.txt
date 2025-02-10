# In Windows, c_api test link must link both 2 shared to avoid symbols redefinition,
# in Linux, c_api test cant do like this or graph_to_program register more than once.
# Both Windows and Linux can only use paddle_inference_c, but this will increase size
# of build folder by 30G.
set(inference_api_tester_deps paddle_inference_api analysis_config)

cc_test(
  test_paddle_inference_api
  SRCS api_tester.cc
  DEPS ${inference_api_tester_deps} common)

cc_test(
  inference_api_helper_test
  SRCS helper_test.cc
  DEPS ${inference_api_tester_deps} common)

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(test_paddle_inference_api)
endif()

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

  function(download_int8_data_without_verify_pir install_dir data_file)
    if(NOT EXISTS ${install_dir}/${data_file})
      inference_download_and_uncompress_without_verify(
        ${install_dir} ${INFERENCE_URL}/int8/pir ${data_file})
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
      --iterations=2
      --with_accuracy_layer=false)
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

  if(NOT APPLE AND WITH_MKLML)
    # seq_pool1
    set(SEQ_POOL1_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/seq_pool")
    download_model_and_data_without_verify(
      ${SEQ_POOL1_INSTALL_DIR} "seq_pool1_model_.tar.gz"
      "seq_pool1_data.txt.tar.gz")
    inference_analysis_api_test(
      test_analyzer_seq_pool1_compare_determine ${SEQ_POOL1_INSTALL_DIR}
      analyzer_seq_pool1_compare_determine_tester.cc EXTRA_DEPS common)
    inference_analysis_api_test(
      test_analyzer_seq_pool1 ${SEQ_POOL1_INSTALL_DIR}
      analyzer_seq_pool1_compare_tester.cc EXTRA_DEPS common)
    inference_analysis_api_test(
      test_analyzer_seq_pool1_fuse_compare_zero_copy ${SEQ_POOL1_INSTALL_DIR}
      analyzer_seq_pool1_fuse_compare_zero_copy_tester.cc EXTRA_DEPS common)
    inference_analysis_api_test(
      test_analyzer_seq_pool1_profile ${SEQ_POOL1_INSTALL_DIR}
      analyzer_seq_pool1_profile_tester.cc EXTRA_DEPS common)
    if(NOT WIN32)
      set_tests_properties(test_analyzer_seq_pool1_compare_determine
                           PROPERTIES TIMEOUT 120)
      set_tests_properties(test_analyzer_seq_pool1 PROPERTIES TIMEOUT 120)
      set_tests_properties(test_analyzer_seq_pool1_fuse_compare_zero_copy
                           PROPERTIES TIMEOUT 120)
      set_tests_properties(test_analyzer_seq_pool1_profile PROPERTIES TIMEOUT
                                                                      120)
    endif()
  else()
    # TODO: fix this test on MACOS and OPENBLAS, the reason is that
    # fusion_seqexpand_concat_fc_op is not supported on MACOS and OPENBLAS
    message(
      WARNING
        "These tests has been disabled in OSX or WITH_MKL=OFF before being fixed: \n test_analyzer_rnn1"
    )
    message(
      WARNING
        "These tests has been disabled in OSX or WITH_MKL=OFF before being fixed: \n test_analyzer_seq_pool1"
    )
  endif()

  # TODO(luotao, Superjom) Disable DAM test, temporarily fix
  # https://github.com/PaddlePaddle/Paddle/issues/15032#issuecomment-455990914.
  # After inference framework refactor, will reopen it.
  # normal DAM
  set(DAM_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/dam")
  download_model_and_data_without_verify(${DAM_INSTALL_DIR} "DAM_model.tar.gz"
                                         "DAM_data.txt.tar.gz")
  #inference_analysis_api_test(test_analyzer_dam ${DAM_INSTALL_DIR} analyzer_dam_tester.cc EXTRA_DEPS legacy_allocator common)

  # Ernie
  set(ERNIE_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/Ernie")
  download_model_and_data(
    ${ERNIE_INSTALL_DIR} "Ernie_model.tar.gz" aa59192dd41ed377f9f168e3a1309fa6
    "Ernie_data.txt.tar.gz" 5396e63548edad7ca561e7e26a9476d1)
  download_result(${ERNIE_INSTALL_DIR} "Ernie_result.txt.tar.gz"
                  73beea65abda2edb61c1662cd3180c62)
  if(WITH_GPU)
    inference_analysis_api_test(gpu_ernie_half_test ${ERNIE_INSTALL_DIR}
                                gpu_ernie_half_test.cc EXTRA_DEPS common)
    set_tests_properties(gpu_ernie_half_test PROPERTIES TIMEOUT 60)
  endif()

  # text_classification
  set(TEXT_CLASSIFICATION_INSTALL_DIR
      "${INFERENCE_DEMO_INSTALL_DIR}/text_classification")
  download_model_and_data(
    ${TEXT_CLASSIFICATION_INSTALL_DIR} "text-classification-Senta.tar.gz"
    3f0f440313ca50e26184e65ffd5809ab "text_classification_data.txt.tar.gz"
    36ae620020cc3377f45ed330dd36238f)
  inference_analysis_api_test(
    test_analyzer_text_classification ${TEXT_CLASSIFICATION_INSTALL_DIR}
    analyzer_text_classification_tester.cc EXTRA_DEPS common)

  # mobilenet
  set(MOBILENET_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/mobilenet_pir")
  if(NOT EXISTS ${MOBILENET_INSTALL_DIR}/mobilenetv1.tar.gz)
    inference_download_and_uncompress_without_verify(
      ${MOBILENET_INSTALL_DIR} "http://paddlemodels.bj.bcebos.com/"
      "inference-vis-demos/mobilenetv1.tar.gz")
  endif()
  inference_analysis_api_test(
    test_analyzer_mobilenet_transpose ${MOBILENET_INSTALL_DIR}
    analyzer_vis_tester.cc EXTRA_DEPS common)
  set_tests_properties(test_analyzer_mobilenet_transpose PROPERTIES TIMEOUT 120)

  inference_analysis_test(
    test_analyzer_capi_exp_pd_tensor
    SRCS
    analyzer_capi_exp_pd_tensor_tester.cc
    EXTRA_DEPS
    paddle_inference_c_shared
    ARGS
    --infer_model=${MOBILENET_INSTALL_DIR}/model)

  if(NOT APPLE AND NOT WIN32)
    inference_analysis_test(
      test_analyzer_capi_exp_pd_threads
      SRCS
      analyzer_capi_exp_pd_threads_tester.cc
      EXTRA_DEPS
      paddle_inference_c_shared
      ARGS
      --infer_model=${MOBILENET_INSTALL_DIR}/model)
  endif()

  if(WITH_ONEDNN)

    ### INT8 tests

    set(INT8_DATA_DIR "${INFERENCE_DEMO_INSTALL_DIR}/int8v2")

    ## Image classification models

    # ImageNet small dataset
    # It may be already downloaded for Quant & INT8 unit tests
    set(IMAGENET_DATA_ARCHIVE "imagenet_val_100_tail.tar.gz")
    set(IMAGENET_DATA_DIR "${INFERENCE_DEMO_INSTALL_DIR}/imagenet")
    set(IMAGENET_DATA_PATH "${IMAGENET_DATA_DIR}/data.bin")
    download_int8_data_without_verify(${IMAGENET_DATA_DIR}
                                      ${IMAGENET_DATA_ARCHIVE})

    # build test binary to be used in subsequent tests

    # new resnet50 int8
    set(INT8_RESNET50_MODEL_DIR "${INT8_DATA_DIR}/resnet50_pir")
    download_int8_data_without_verify_pir(${INT8_RESNET50_MODEL_DIR}
                                          "resnet50_int8_model.tar.gz")

    # mobilenetv1 int8
    set(INT8_MOBILENETV1_MODEL_DIR "${INT8_DATA_DIR}/mobilenetv1_pir")
    download_int8_data_without_verify_pir(${INT8_MOBILENETV1_MODEL_DIR}
                                          "mobilenetv1_int8_model.tar.gz")

    # mobilenetv2 int8
    set(INT8_MOBILENETV2_MODEL_DIR "${INT8_DATA_DIR}/mobilenetv2_pir")
    download_int8_data_without_verify_pir(${INT8_MOBILENETV2_MODEL_DIR}
                                          "mobilenet_v2_int8_model.tar.gz")

    # resnet101 int8
    set(INT8_RESNET101_MODEL_DIR "${INT8_DATA_DIR}/resnet101")
    download_int8_data_without_verify(${INT8_RESNET101_MODEL_DIR}
                                      "Res101_int8_model.tar.gz")

    # vgg16 int8
    set(INT8_VGG16_MODEL_DIR "${INT8_DATA_DIR}/vgg16")
    download_int8_data_without_verify(${INT8_VGG16_MODEL_DIR}
                                      "VGG16_int8_model.tar.gz")

    # vgg19 int8
    set(INT8_VGG19_MODEL_DIR "${INT8_DATA_DIR}/vgg19")
    download_int8_data_without_verify(${INT8_VGG19_MODEL_DIR}
                                      "VGG19_int8_model.tar.gz")

    # googlenet int8
    set(INT8_GOOGLENET_MODEL_DIR "${INT8_DATA_DIR}/googlenet_pir")
    download_int8_data_without_verify_pir(${INT8_GOOGLENET_MODEL_DIR}
                                          "GoogleNet_int8_model.tar.gz")

    # mobilenetv3_large_x1_0 int8
    set(INT8_MOBILENETV3_LARGE_MODEL_DIR "${INT8_DATA_DIR}/mobilenetv3_large")
    set(INT8_MOBILENETV3_FILE_NAME "MobileNetV3_large_x1_0_infer.tar")
    if(NOT EXISTS
       ${INT8_MOBILENETV3_LARGE_MODEL_DIR}/${INT8_MOBILENETV3_FILE_NAME})
      inference_download_and_uncompress_without_verify(
        ${INT8_MOBILENETV3_LARGE_MODEL_DIR}
        "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/"
        ${INT8_MOBILENETV3_FILE_NAME})
    endif()

    ### BFLOAT16 tests

    # build test binary to be used in subsequent tests
    set(BF16_IMG_CLASS_TEST_APP "test_analyzer_bfloat16_image_classification")
    set(BF16_IMG_CLASS_TEST_APP_SRC
        "analyzer_bfloat16_image_classification_tester.cc")
    inference_analysis_api_test_build(${BF16_IMG_CLASS_TEST_APP}
                                      ${BF16_IMG_CLASS_TEST_APP_SRC})

    # resnet50 bfloat16
    inference_analysis_api_bfloat16_test_run(
      test_analyzer_bfloat16_resnet50 ${BF16_IMG_CLASS_TEST_APP}
      ${INT8_RESNET50_MODEL_DIR} ${IMAGENET_DATA_PATH})

    # googlenet bfloat16
    inference_analysis_api_bfloat16_test_run(
      test_analyzer_bfloat16_googlenet ${BF16_IMG_CLASS_TEST_APP}
      ${INT8_GOOGLENET_MODEL_DIR} ${IMAGENET_DATA_PATH})

    # mobilenetv1 bfloat16
    inference_analysis_api_bfloat16_test_run(
      test_analyzer_bfloat16_mobilenetv1 ${BF16_IMG_CLASS_TEST_APP}
      ${INT8_MOBILENETV1_MODEL_DIR} ${IMAGENET_DATA_PATH})

    # mobilenetv2 bfloat16
    inference_analysis_api_bfloat16_test_run(
      test_analyzer_bfloat16_mobilenetv2 ${BF16_IMG_CLASS_TEST_APP}
      ${INT8_MOBILENETV2_MODEL_DIR} ${IMAGENET_DATA_PATH})

    # mobilenetv3_large
    inference_analysis_test_run(
      test_analyzer_bfloat16_mobilenetv3_large
      COMMAND
      ${BF16_IMG_CLASS_TEST_APP}
      ARGS
      --infer_model=${INT8_MOBILENETV3_LARGE_MODEL_DIR}/MobileNetV3_large_x1_0_infer
      --infer_data=${IMAGENET_DATA_PATH}
      --batch_size=1
      --enable_bf16=true
      --paddle_num_threads=${CPU_NUM_THREADS_ON_CI}
      --iterations=100
      --with_accuracy_layer=false)

    ### Object detection models
    set(PASCALVOC_DATA_PATH "${INT8_DATA_DIR}/pascalvoc_val_head_300.bin")

    # download dataset if necessary
    download_int8_data_without_verify(${INT8_DATA_DIR}
                                      "pascalvoc_val_head_300.tar.gz")

    ### optimized FP32 vs. Quant INT8 tests

    set(QUANT_DATA_DIR "${INFERENCE_DEMO_INSTALL_DIR}/quant")
    set(QUANT_IMG_CLASS_TEST_APP "test_analyzer_quant_image_classification")
    set(QUANT_IMG_CLASS_TEST_APP_SRC
        "analyzer_quant_image_classification_tester.cc")

    # build test binary to be used in subsequent tests
    inference_analysis_api_test_build(${QUANT_IMG_CLASS_TEST_APP}
                                      ${QUANT_IMG_CLASS_TEST_APP_SRC})

    # MobileNetV1 FP32 vs. Quant INT8
    # The FP32 model should already be downloaded for slim Quant unit tests on Linux
    set(QUANT2_MobileNetV1_MODEL_DIR "${QUANT_DATA_DIR}/MobileNetV1_quant2")
    set(QUANT2_INT8_MobileNetV1_MODEL_DIR
        "${QUANT_DATA_DIR}/MobileNetV1_quant2_int8")
    if(NOT LINUX)
      download_quant_data_without_verify(${QUANT2_MobileNetV1_MODEL_DIR}
                                         "MobileNet_qat_perf.tar.gz")
    endif()
    download_quant_data_without_verify(${QUANT2_INT8_MobileNetV1_MODEL_DIR}
                                       "MobileNet_qat_perf_int8.tar.gz")
    inference_analysis_api_quant_test_run(
      test_analyzer_quant_performance_benchmark
      ${QUANT_IMG_CLASS_TEST_APP}
      ${QUANT2_MobileNetV1_MODEL_DIR}/MobileNet_qat_perf/float
      ${QUANT2_INT8_MobileNetV1_MODEL_DIR}/MobileNet_qat_perf_int8
      ${IMAGENET_DATA_PATH}
      false)

    # Quant2 MobileNetV1
    inference_analysis_api_quant_test_run(
      test_analyzer_quant2_mobilenetv1_mkldnn
      ${QUANT_IMG_CLASS_TEST_APP}
      ${QUANT2_MobileNetV1_MODEL_DIR}/MobileNet_qat_perf/float
      ${QUANT2_MobileNetV1_MODEL_DIR}/MobileNet_qat_perf/float
      ${IMAGENET_DATA_PATH}
      true)

    # Quant2 ResNet50 with input/output scales in `fake_quantize_range_abs_max` operators and the `out_threshold` attributes,
    # with weight scales in `fake_channel_wise_dequantize_max_abs` operators
    set(QUANT2_RESNET50_CHANNELWISE_MODEL_DIR
        "${QUANT_DATA_DIR}/ResNet50_quant2_channelwise")
    set(QUANT2_RESNET50_CHANNELWISE_MODEL_ARCHIVE
        "ResNet50_qat_channelwise.tar.gz")
    if(NOT LINUX)
      download_quant_data_without_verify(
        ${QUANT2_RESNET50_CHANNELWISE_MODEL_DIR}
        ${QUANT2_RESNET50_CHANNELWISE_MODEL_ARCHIVE})
    endif()
    set(QUANT2_RESNET50_MODEL
        ${QUANT2_RESNET50_CHANNELWISE_MODEL_DIR}/ResNet50_qat_channelwise)
    inference_analysis_api_quant_test_run(
      test_analyzer_quant2_resnet50_channelwise_mkldnn
      ${QUANT_IMG_CLASS_TEST_APP} ${QUANT2_RESNET50_MODEL}
      ${QUANT2_RESNET50_MODEL} ${IMAGENET_DATA_PATH} true)

    # preprocess data2bin imagenet
    download_int8_data_without_verify(${INT8_DATA_DIR} "imagenet_small.tar.gz")
    set(IMAGENET_SMALL_DATA_DIR "${INT8_DATA_DIR}/imagenet_small")
    set(IMAGENET_SMALL_OUTPUT_FILE "imagenet_small.bin")
    preprocess_data2bin_test_run(
      preprocess_local_imagenet "full_ILSVRC2012_val_preprocess.py"
      ${IMAGENET_SMALL_DATA_DIR} ${IMAGENET_SMALL_OUTPUT_FILE})

    # preprocess data2bin pascalvoc
    download_int8_data_without_verify(${INT8_DATA_DIR} "pascalvoc_small.tar.gz")
    set(PASCALVOC_SMALL_DATA_DIR "${INT8_DATA_DIR}/pascalvoc_small")
    set(PASCALVOC_SMALL_OUTPUT_FILE "pascalvoc_small.bin")
    preprocess_data2bin_test_run(
      preprocess_local_pascalvoc "full_pascalvoc_test_preprocess.py"
      ${PASCALVOC_SMALL_DATA_DIR} ${PASCALVOC_SMALL_OUTPUT_FILE})

  endif()

  if(WITH_GPU AND TENSORRT_FOUND)
    set(TRT_MODEL_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/trt_models")
    if(NOT EXISTS ${TRT_MODEL_INSTALL_DIR}/trt_inference_test_models.tar.gz)
      inference_download_and_uncompress(
        ${TRT_MODEL_INSTALL_DIR} ${INFERENCE_URL}/tensorrt_test
        "trt_inference_test_models.tar.gz" 3dcccdc38b549b6b1b4089723757bd98)
    endif()
    set(TEST_SPLIT_CONVERTER_MODEL
        "${TRT_MODEL_INSTALL_DIR}/trt_split_op_converter_test")
    if(NOT EXISTS ${TEST_SPLIT_CONVERTER_MODEL}/split_converter.tgz)
      inference_download_and_uncompress_without_verify(
        ${TEST_SPLIT_CONVERTER_MODEL} ${INFERENCE_URL}/tensorrt_test
        "split_converter.tgz")
    endif()
    inference_analysis_test(
      trt_mobilenet_test
      SRCS
      trt_mobilenet_test.cc
      EXTRA_DEPS
      common
      paddle_inference_shared
      ARGS
      --infer_model=${TRT_MODEL_INSTALL_DIR}/trt_inference_test_models)
    inference_analysis_test(
      trt_mark_trt_engine_outputs_test
      SRCS
      trt_mark_trt_engine_outputs_test.cc
      EXTRA_DEPS
      common
      paddle_inference_shared
      ARGS
      --infer_model=${TRT_MODEL_INSTALL_DIR}/trt_inference_test_models)
    inference_analysis_test(
      trt_disable_tensorrt_half_ops_test
      SRCS
      trt_disable_tensorrt_half_ops_test.cc
      EXTRA_DEPS
      common
      paddle_inference_shared
      ARGS
      --infer_model=${TRT_MODEL_INSTALL_DIR}/trt_inference_test_models)
    inference_analysis_test(
      trt_split_converter_test
      SRCS
      trt_split_converter_test.cc
      EXTRA_DEPS
      paddle_inference_shared
      common
      ARGS
      --infer_model=${TEST_SPLIT_CONVERTER_MODEL}/)
    inference_analysis_test(
      test_analyzer_capi_exp_gpu
      SRCS
      analyzer_capi_exp_gpu_tester.cc
      EXTRA_DEPS
      common
      paddle_inference_c_shared
      ARGS
      --infer_model=${TRT_MODEL_INSTALL_DIR}/trt_inference_test_models)
    inference_analysis_test(
      test_analyzer_capi_exp_xpu
      SRCS
      analyzer_capi_exp_xpu_tester.cc
      EXTRA_DEPS
      common
      paddle_inference_c_shared
      ARGS
      --infer_model=${TRT_MODEL_INSTALL_DIR}/trt_inference_test_models)
    #TODO(inference): windows encounter a SEH error, we need to fix it.
    if(NOT WIN32)
      inference_analysis_test(
        trt_rebind_stream_test
        SRCS
        trt_rebind_stream_test.cc
        EXTRA_DEPS
        common
        paddle_inference_shared
        ARGS
        --infer_model=${TRT_MODEL_INSTALL_DIR}/trt_inference_test_models)
    endif()

    set(TRT_MODEL_QUANT_RESNET_DIR
        "${INFERENCE_DEMO_INSTALL_DIR}/small_quant_model")
    if(NOT EXISTS ${INFERENCE_DEMO_INSTALL_DIR}/small_quant_model.tgz)
      inference_download_and_uncompress_without_verify(
        ${INFERENCE_DEMO_INSTALL_DIR} ${INFERENCE_URL}/tensorrt_test
        "small_quant_model.tgz")
    endif()
    inference_analysis_test(
      trt_quant_int8_test
      SRCS
      trt_quant_int8_test.cc
      EXTRA_DEPS
      common
      paddle_inference_shared
      ARGS
      --infer_model=${TRT_MODEL_QUANT_RESNET_DIR})

    set(TRT_MODEL_QUANT_YOLOV3_DIR
        "${INFERENCE_DEMO_INSTALL_DIR}/yolov3_r50_quant_aware")
    if(NOT EXISTS ${INFERENCE_DEMO_INSTALL_DIR}/yolov3_r50_quant_aware.tgz)
      inference_download_and_uncompress_without_verify(
        ${INFERENCE_DEMO_INSTALL_DIR} ${INFERENCE_URL}/tensorrt_test
        "yolov3_r50_quant_aware.tgz")
    endif()

    set(TEST_TRT_DYNAMIC_MODEL2
        "${TRT_MODEL_INSTALL_DIR}/complex_model_dynamic")
    if(NOT EXISTS ${TEST_TRT_DYNAMIC_MODEL2}/complex_model_dynamic2.tar.gz)
      inference_download_and_uncompress_without_verify(
        ${TEST_TRT_DYNAMIC_MODEL2} ${INFERENCE_URL}/tensorrt_test
        "complex_model_dynamic2.tar.gz")
    endif()

    set(TEST_TRT_DYNAMIC_MODEL
        "${TRT_MODEL_INSTALL_DIR}/conv_bn_swish_split_gelu")
    if(NOT EXISTS ${TEST_TRT_DYNAMIC_MODEL}/conv_bn_swish_split_gelu.tar.gz)
      inference_download_and_uncompress(
        ${TEST_TRT_DYNAMIC_MODEL} ${INFERENCE_URL}/tensorrt_test
        "conv_bn_swish_split_gelu.tar.gz" 2a5e8791e47b221b4f782151d76da9c6)
    endif()

    inference_analysis_test(
      trt_dynamic_shape_test
      SRCS
      trt_dynamic_shape_test.cc
      EXTRA_DEPS
      common
      paddle_inference_shared
      ARGS
      --infer_model=${TRT_MODEL_INSTALL_DIR})
    set_tests_properties(trt_dynamic_shape_test PROPERTIES TIMEOUT 300)

    set(TEST_TRT_ERNIE_MODEL "${TRT_MODEL_INSTALL_DIR}/ernie_test")
    if(NOT EXISTS ${TEST_TRT_ERNIE_MODEL}/ernie_model_4.tar.gz)
      inference_download_and_uncompress(
        ${TEST_TRT_ERNIE_MODEL} ${INFERENCE_URL}/tensorrt_test
        "ernie_model_4.tar.gz" 5fa371efa75706becbaad79195d2ca68)
    endif()

    inference_analysis_test(
      test_trt_dynamic_shape_ernie
      SRCS
      trt_dynamic_shape_ernie_test.cc
      EXTRA_DEPS
      common
      paddle_inference_shared
      ARGS
      --infer_model=${TEST_TRT_ERNIE_MODEL}/ernie_model_4)

    if(NOT EXISTS ${TEST_TRT_ERNIE_MODEL}/ernie_model_4_unserialized.tgz)
      inference_download_and_uncompress(
        ${TEST_TRT_ERNIE_MODEL} ${INFERENCE_URL}/tensorrt_test
        "ernie_model_4_unserialized.tgz" 833d73fc6a7f7e1ee4a1fd6419209e55)
    endif()

    inference_analysis_test(
      test_trt_dynamic_shape_ernie_ser_deser
      SRCS
      trt_dynamic_shape_ernie_serialize_deserialize_test.cc
      EXTRA_DEPS
      common
      paddle_inference_shared
      ARGS
      --infer_model=${TEST_TRT_ERNIE_MODEL}/ernie_model_4_unserialized)

    if(NOT EXISTS ${TEST_TRT_ERNIE_MODEL}/ernie_model_4_fp16_unserialized.tgz)
      inference_download_and_uncompress(
        ${TEST_TRT_ERNIE_MODEL} ${INFERENCE_URL}/tensorrt_test
        "ernie_model_4_fp16_unserialized.tgz" c5ff2d0cad79953ffbf2b8b9e2fae6e4)
    endif()

    inference_analysis_test(
      test_trt_dynamic_shape_ernie_fp16_ser_deser
      SRCS
      trt_dynamic_shape_ernie_fp16_serialize_deserialize_test.cc
      EXTRA_DEPS
      common
      paddle_inference_shared
      ARGS
      --infer_model=${TEST_TRT_ERNIE_MODEL}/ernie_model_4_fp16_unserialized)

  endif()

  set(LITE_MODEL_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/lite")
  download_data_without_verify(${LITE_MODEL_INSTALL_DIR} "mul_model_fp32.tgz")

  inference_analysis_test(
    lite_mul_model_test
    SRCS
    lite_mul_model_test.cc
    EXTRA_DEPS
    common
    paddle_inference_shared
    ARGS
    --infer_model=${LITE_MODEL_INSTALL_DIR})

  inference_analysis_test(
    test_analyzer_capi_exp_pd_config
    SRCS
    analyzer_capi_exp_pd_config_tester.cc
    EXTRA_DEPS
    paddle_inference_c_shared
    ARGS
    --infer_model=${MOBILENET_INSTALL_DIR}/model)

  # ch_PP-OCRv3_det
  set(OCR_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/ch_PP-OCRv3_det_pir")
  if(NOT EXISTS ${OCR_INSTALL_DIR}/ch_PP-OCRv3_det.tar.gz)
    inference_download_and_uncompress_without_verify(
      ${OCR_INSTALL_DIR} "http://paddle-inference-dist.bj.bcebos.com/"
      "int8/pir/ch_PP-OCRv3_det.tar.gz")
  endif()

  inference_analysis_test(
    test_analyzer_zerocopytensor_tensor
    SRCS
    analyzer_zerocopy_tensor_tester.cc
    EXTRA_DEPS
    common
    paddle_inference_shared
    ARGS
    --infer_model=${OCR_INSTALL_DIR}/model)

  if(WITH_DISTRIBUTE
     AND WITH_PSCORE
     AND WITH_XPU
     AND WITH_XPU_BKCL)
    inference_analysis_test(
      test_analyzer_dist_model_xpu
      SRCS
      analyzer_dist_model_xpu_tester.cc
      EXTRA_DEPS
      common
      paddle_inference_shared
      ARGS
      --infer_model=${OCR_INSTALL_DIR}/model)
  endif()

  # chinese_ner
  set(CHINESE_NER_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/chinese_ner")
  download_model_and_data_without_verify(
    ${CHINESE_NER_INSTALL_DIR} "chinese_ner_model.tar.gz"
    "chinese_ner-data.txt.tar.gz")
  inference_analysis_test(
    test_analyzer_capi_exp_ner
    SRCS
    analyzer_capi_exp_ner_tester.cc
    EXTRA_DEPS
    paddle_inference_c_shared
    ARGS
    --infer_model=${CHINESE_NER_INSTALL_DIR}/model)

  # small DAM
  set(DAM_SMALL_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/small_dam")
  download_model_and_data_without_verify(
    ${DAM_SMALL_INSTALL_DIR} "dam_small_model.tar.gz"
    "dam_small_data.txt.tar.gz")
  inference_analysis_test(
    test_analyzer_small_dam
    SRCS
    analyzer_dam_tester.cc
    EXTRA_DEPS
    paddle_inference_shared
    common
    ARGS
    --infer_model=${DAM_SMALL_INSTALL_DIR}/model
    --infer_data=${DAM_SMALL_INSTALL_DIR}/data.txt)

  set_tests_properties(test_analyzer_small_dam PROPERTIES TIMEOUT 120)

  # seq_conv1
  set(SEQ_CONV1_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/seq_conv1")
  download_model_and_data_without_verify(
    ${SEQ_CONV1_INSTALL_DIR} "seq_conv1_model.tar.gz"
    "seq_conv1_data.txt.tar.gz")
  inference_analysis_api_test(test_analyzer_seq_conv1 ${SEQ_CONV1_INSTALL_DIR}
                              analyzer_seq_conv1_tester.cc EXTRA_DEPS common)

  # RNN2
  set(RNN2_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/rnn2")
  download_model_and_data_without_verify(
    ${RNN2_INSTALL_DIR} "rnn2_model.tar.gz" "rnn2_data.txt.tar.gz")
  inference_analysis_api_test(test_analyzer_rnn2 ${RNN2_INSTALL_DIR}
                              analyzer_rnn2_tester.cc EXTRA_DEPS common)

  if(WITH_ONNXRUNTIME AND WIN32)
    # Copy onnxruntime for some c++ test in Windows, since the test will
    # be build only in CI, so suppose the generator in Windows is Ninja.
    copy_onnx(test_analyzer_rnn2)
  endif()

  # chinese_ner
  set(CHINESE_NER_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/chinese_ner")
  download_model_and_data_without_verify(
    ${CHINESE_NER_INSTALL_DIR} "chinese_ner_model.tar.gz"
    "chinese_ner-data.txt.tar.gz")
  inference_analysis_api_test(test_analyzer_ner ${CHINESE_NER_INSTALL_DIR}
                              analyzer_ner_tester.cc EXTRA_DEPS common)
  set_tests_properties(test_analyzer_ner PROPERTIES TIMEOUT 120)

  # multiple models prediction
  set(MMP_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/multi_model_prediction")
  download_data_without_verify(${MMP_INSTALL_DIR}
                               PaddleInference/mobilenet_v2_models.tar.gz)
  inference_multiple_models_analysis_api_test(
    test_analyzer_multi_model_prediction ${MMP_INSTALL_DIR}
    analyzer_mmp_tester.cc EXTRA_DEPS common)

  # lac
  set(LAC_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/lac")
  download_model_and_data(
    ${LAC_INSTALL_DIR} "lac_model.tar.gz" 419ca6eb85f57a01bfe173591910aec5
    "lac_data.txt.tar.gz" 9983539cd6b34fbdc411e43422776bfd)
  inference_analysis_api_test(test_analyzer_lac ${LAC_INSTALL_DIR}
                              analyzer_lac_tester.cc EXTRA_DEPS common)
  ### Image classification tests with fake data
  set(IMG_CLASS_TEST_APP "analyzer_image_classification")
  set(IMG_CLASS_TEST_APP_SRC "analyzer_image_classification_tester.cc")

  # build test binary to be used in subsequent tests
  inference_analysis_api_test_with_fake_data_build(${IMG_CLASS_TEST_APP}
                                                   ${IMG_CLASS_TEST_APP_SRC})

  # resnet50
  set(RESNET50_MODEL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/resnet50")
  download_data_without_verify(${RESNET50_MODEL_DIR} "resnet50_model.tar.gz")
  if(NOT WIN32)
    inference_analysis_api_test_with_fake_data_run(
      test_analyzer_resnet50 ${IMG_CLASS_TEST_APP} ${RESNET50_MODEL_DIR} true)
    set_tests_properties(test_analyzer_resnet50 PROPERTIES TIMEOUT 120)
  endif()
  if(WITH_OPENVINO
     AND NOT WIN32
     AND NOT WITH_XPU
     AND NOT WITH_ROCM)
    set(OPENVINO_RESNET50_MODEL_DIR
        "${INFERENCE_DEMO_INSTALL_DIR}/openvino_resnet50")
    download_data_without_verify(${OPENVINO_RESNET50_MODEL_DIR}
                                 "resnet50_pdmodel.tar.gz")
    set(OPENVINO_TEST_APP "analyzer_openvino")
    set(OPENVINO_TEST_APP_SRC "analyzer_openvino_tester.cc")
    inference_analysis_api_test_with_fake_data_build(${OPENVINO_TEST_APP}
                                                     ${OPENVINO_TEST_APP_SRC})
    inference_analysis_api_test_with_fake_data_run(
      test_analyzer_openvino_resnet50 ${OPENVINO_TEST_APP}
      ${OPENVINO_RESNET50_MODEL_DIR} true)
    set_tests_properties(test_analyzer_openvino_resnet50 PROPERTIES TIMEOUT 120)
  endif()

  inference_analysis_test(
    test_analyzer_capi_exp
    SRCS
    analyzer_capi_exp_tester.cc
    EXTRA_DEPS
    paddle_inference_c_shared
    ARGS
    --infer_model=${RESNET50_MODEL_DIR}/model)

  cc_test(
    paddle_infer_api_errors_test
    SRCS paddle_infer_api_errors_tester.cc
    DEPS ${inference_api_tester_deps} common)

  if(WITH_GPU)
    inference_analysis_test(
      paddle_infer_api_test
      SRCS
      paddle_infer_api_test.cc
      EXTRA_DEPS
      common
      paddle_inference_shared
      ARGS
      --infer_model=${RESNET50_MODEL_DIR})
    set_tests_properties(paddle_infer_api_test PROPERTIES TIMEOUT 300)

    inference_analysis_test(
      paddle_infer_api_copy_tensor_tester
      SRCS
      paddle_infer_api_copy_tensor_tester.cc
      EXTRA_DEPS
      common
      paddle_inference_shared
      ARGS
      --infer_model=${RESNET50_MODEL_DIR})
    set_tests_properties(paddle_infer_api_copy_tensor_tester PROPERTIES TIMEOUT
                                                                        300)
  endif()

  if(WITH_GPU AND TENSORRT_FOUND)
    set_tests_properties(test_trt_dynamic_shape_ernie_ser_deser
                         PROPERTIES TIMEOUT 300)
    set_tests_properties(test_trt_dynamic_shape_ernie_fp16_ser_deser
                         PROPERTIES TIMEOUT 300)
    set_tests_properties(test_trt_dynamic_shape_ernie PROPERTIES TIMEOUT 480)
    set_tests_properties(trt_mark_trt_engine_outputs_test PROPERTIES TIMEOUT
                                                                     300)
    set_tests_properties(trt_disable_tensorrt_half_ops_test PROPERTIES TIMEOUT
                                                                       300)
  endif()

  if(WITH_ONEDNN)
    set_tests_properties(test_analyzer_quant_performance_benchmark
                         PROPERTIES TIMEOUT 120)
    set_tests_properties(test_analyzer_quant2_mobilenetv1_mkldnn
                         PROPERTIES TIMEOUT 120)
    set_tests_properties(test_analyzer_quant2_resnet50_channelwise_mkldnn
                         PROPERTIES TIMEOUT 120)
  endif()

  if(WITH_GPU AND TENSORRT_FOUND)
    set_tests_properties(trt_mobilenet_test PROPERTIES TIMEOUT 240)
    if(NOT WIN32)
      set_tests_properties(trt_rebind_stream_test
                           PROPERTIES TIMEOUT 360 LABELS "RUN_TYPE=EXCLUSIVE")
    endif()
    if(WITH_ONEDNN)
      set_tests_properties(test_analyzer_bfloat16_resnet50 PROPERTIES TIMEOUT
                                                                      120)
    endif()
  endif()

  if(WITH_IPU)
    #word2vec sample
    set(WORD2VEC_INSTALL_DIR
        "${INFERENCE_DEMO_INSTALL_DIR}/word2vec/word2vec.inference.model")
    inference_analysis_test(
      ipu_word2vec_sample
      SRCS
      ipu_word2vec_sample.cc
      EXTRA_DEPS
      common
      paddle_inference_shared
      ARGS
      --infer_model=${WORD2VEC_INSTALL_DIR})

    # ERNIE
    set(ERNIE_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/Ernie")
    inference_analysis_api_test(
      ipu_ernie_test
      ${ERNIE_INSTALL_DIR}
      ipu_ernie_test.cc
      ARGS
      --warmup=true
      --repeat=10
      EXTRA_DEPS
      common)
    inference_analysis_api_test(
      ipu_ernie_fp16_test
      ${ERNIE_INSTALL_DIR}
      ipu_ernie_fp16_test.cc
      ARGS
      --warmup=true
      --repeat=10
      EXTRA_DEPS
      common)

    # Resnet50
    set(RESNET50_MODEL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/resnet50")
    inference_analysis_test(
      ipu_resnet50_test
      SRCS
      ipu_resnet50_test.cc
      EXTRA_DEPS
      common
      paddle_inference_shared
      ARGS
      --infer_model=${RESNET50_MODEL_DIR}
      --warmup=true
      --repeat=10)
    inference_analysis_test(
      ipu_resnet50_fp16_test
      SRCS
      ipu_resnet50_fp16_test.cc
      EXTRA_DEPS
      common
      paddle_inference_shared
      ARGS
      --infer_model=${RESNET50_MODEL_DIR}
      --warmup=true
      --repeat=10)

    # Only support Resnet50 and Ernie currently
    inference_analysis_api_test(
      ipu_multi_model_profile
      SRCS
      ipu_multi_model_profile.cc
      ARGS
      --model_name="Resnet50"
      --infer_model=${RESNET50_MODEL_DIR}
      --warmup=true
      --repeat=10
      EXTRA_DEPS
      common)
  endif()

  if(WITH_XPU)
    inference_analysis_test(
      xpu_config_resnet50_test
      SRCS
      xpu_config_resnet50_test.cc
      EXTRA_DEPS
      common
      paddle_inference_shared
      ARGS
      --infer_model=${RESNET50_MODEL_DIR})
    inference_analysis_test(
      xpu_runtime_config_resnet50_test
      SRCS
      xpu_runtime_config_resnet50_test.cc
      EXTRA_DEPS
      common
      paddle_inference_shared
      ARGS
      --infer_model=${RESNET50_MODEL_DIR})
  endif()

  if(WITH_TESTING)
    if(NOT APPLE)
      if(NOT EXISTS ${WORD2VEC_INSTALL_DIR}/word2vec.inference.model.tar.gz)
        inference_download_and_uncompress_without_verify(
          ${WORD2VEC_INSTALL_DIR} ${INFERENCE_URL}
          "word2vec.inference.model.tar.gz")
      endif()
      inference_base_test(
        test_api_impl
        SRCS
        api_impl_tester.cc
        DEPS
        common
        paddle_inference_shared
        ARGS
        --word2vec_dirname=${WORD2VEC_MODEL_DIR}
        --book_dirname=${IMG_CLS_RESNET_INSTALL_DIR})
    endif()
  endif()

  set(VIT_OCR_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/vit")
  if(NOT EXISTS ${VIT_OCR_INSTALL_DIR}/vit_ocr.tgz)
    inference_download_and_uncompress_without_verify(
      ${VIT_OCR_INSTALL_DIR} ${INFERENCE_URL} "ocr/vit_ocr.tgz")
  endif()
  inference_analysis_test(
    test_analyzer_vit_ocr
    SRCS
    analyzer_vit_ocr_tester.cc
    EXTRA_DEPS
    common
    paddle_inference_shared
    ARGS
    --infer_model=${VIT_OCR_INSTALL_DIR}/vit_ocr/model
    --infer_data=${VIT_OCR_INSTALL_DIR}/vit_ocr/datavit.txt)

endif()
