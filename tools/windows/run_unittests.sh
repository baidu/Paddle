# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# /*================Fixed Disabled Windows CUDA10.x MKL(PR-CI-Windows) unittests===========================*/
# TODO: fix these unittest that is bound to fail
disable_wingpu_test="^test_model$|\
^test_dataloader_early_reset_deprecated$|\
^test_add_reader_dependency_deprecated$|\
^test_add_reader_dependency_for_interpretercore$|\
^test_decoupled_py_reader_deprecated$|\
^test_decoupled_py_reader_deprecated_static_build$|\
^test_generator_dataloader_deprecated$|\
^test_parallel_dygraph_sync_batch_norm$|\
^test_py_reader_using_executor$|\
^test_program_prune_backward_deprecated$|\
^test_decoupled_py_reader_data_check_deprecated$|\
^test_fleet_base_single$|\
^test_multiprocess_dataloader_iterable_dataset_dynamic$|\
^test_py_reader_combination$|\
^test_py_reader_pin_memory$|\
^test_py_reader_push_pop$|\
^test_reader_reset_deprecated$|\
^test_imperative_se_resnext$|\
^test_dataloader_keep_order_deprecated$|\
^test_dataloader_unkeep_order_deprecated$|\
^test_multiprocess_dataloader_iterable_dataset_static$|\
^test_fuse_bn_act_pass_deprecated$|\
^test_fuse_bn_act_pass_deprecated_static_build$|\
^test_fuse_bn_add_act_pass$|\
^test_gather_op$|\
^test_activation_op$|\
^test_norm_nn_grad$|\
^test_bilinear_interp_op$|\
^test_fused_moe_op$|\
^disable_wingpu_test$"

# /*================Fixed Disabled Windows CUDA12.0 MKL(PR-CI-Windows) unittests===========================*/
# TODO: fix these unittest that is bound to fail
disable_wingpu_cuda12_test="^test_cholesky_op$|\
^test_cholesky_solve_op$|\
^test_eigh_op$|\
^test_eigvalsh_op$|\
^test_lu_op$|\
^test_math_op_patch_var_base$|\
^test_cudnn_bn_add_relu$|\
^test_linalg_pinv_op$|\
^test_pca_lowrank$|\
^test_qr_op$|\
^test_sparse_matmul_op$|\
^test_svd_op$|\
^test_mul_op$|\
^test_bmn$|\
^test_memory_efficient_attention$|\
^test_fuse_gemm_epilogue_pass_deprecated$|\
^test_tril_triu_op$|\
^test_train_step_resnet18_adam$|\
^test_train_step_resnet18_sgd$|\
^test_elementwise_add_mkldnn_op$|\
^test_comp_high_grad$|\
^test_multi_precision_fp16_train$|\
^test_imperative_skip_op$|\
^test_qat$|\
^test_standalone_cuda_graph_multi_stream_deprecated$|\
^test_standalone_cuda_graph_multi_stream_deprecated_static_build$|\
^test_save_load$|\
^test_conv_transpose_nn_grad$|\
^test_dygraph_spectral_norm$|\
^test_lambv2_op$|\
^test_retain_graph$|\
^test_switch_autotune$|\
^test_elementwise_div_op$|\
^test_elementwise_mul_op$|\
^test_conv2d_api_deprecated$|\
^test_fused_gemm_epilogue_pass$|\
^test_cuda_graphed_layer$|\
^test_quant_linear_op$|\
^test_sot_resnet$|\
^test_sot_resnet50_backward$|\
^test_amp_api$|\
^test_asp_optimize_dynamic$|\
^test_asp_optimize_static$|\
^test_asp_save_load$|\
^test_fuse_resnet_unit$|\
^test_conv2d_transpose_op$|\
^test_dygraph_multi_forward$|\
^test_instance_norm_op_v2$|\
^test_rnn_op$|\
^test_composite_batch_norm_deprecated$|\
^test_prim_amp$|\
^test_cumprod_op$|\
^test_elementwise_sub_op$|\
^test_amp_decorate$|\
^test_amp_promote$|\
^test_cudnn_norm_conv$|\
^test_basic_api_transformation$|\
^test_conv2d_transpose_op_depthwise_conv$|\
^test_conv3d_transpose_part2_op$|\
^test_dygraph_mnist_fp16$|\
^test_sparse_conv_op$|\
^test_sparse_conv_op_static_build$|\
^test_conv2d_transpose_mkldnn_op$|\
^test_ptq$|\
^test_stub$|\
^test_lu_unpack_op$|\
^test_softmax_with_cross_entropy_op$|\
^test_fused_gemm_epilogue_op$|\
^test_fused_gemm_epilogue_grad_op$|\
^test_fused_gemm_epilogue_op_with_es$|\
^test_fused_gemm_epilogue_grad_op_with_es$|\
^test_matmul_op_static_build$|\
^test_matmul_v2_op_static_build$|\
^test_image_classification$|\
^test_apply_pass_to_program_deprecated$|\
^test_mobile_net$|\
^test_IntermediateLayerGetter$|\
^test_pad3d_op$|\
^paddle_infer_api_test$|\
^device_context_test_cuda_graph$|\
^test_fused_linear_param_grad_add$|\
^test_fused_matmul_bias$|\
^test_tensordot$|\
^test_cuda_graph$|\
^test_cuda_graph_partial_graph_static_run$|\
^test_cuda_graph_static_mode$|\
^test_matrix_rank_op$|\
^test_sparse_pca_lowrank$|\
^test_zero_dim_no_backward_api$|\
^test_zero_dim_sundry_dygraph_api$|\
^test_zero_dim_sundry_static_api_part1$|\
^test_zero_dim_sundry_static_api_part2$|\
^test_zero_dim_sundry_static_api_part3$|\
^test_zero_dim_sundry_static_api_part4$|\
^paddle_infer_api_copy_tensor_tester$|\
^cudnn_helper_test$|\
^test_analyzer_small_dam$|\
^test_analyzer_transformer_deprecated$|\
^test_analyzer_int8_mobilenetv3_large$|\
^test_analyzer_bfloat16_mobilenetv3_large$|\
^test_api_impl$|\
^test_mkldnn_conv_affine_channel_fuse_pass$|\
^test_mkldnn_conv_gelu_fuse_pass$|\
^test_mkldnn_conv_hard_sigmoid_fuse_pass$|\
^test_mkldnn_conv_hard_swish_fuse_pass$|\
^test_mkldnn_conv_mish_fuse_pass$|\
^test_mkldnn_conv_transpose_bias_fuse_pass$|\
^test_mkldnn_depthwise_conv_pass$|\
^test_mkldnn_matmul_elementwise_add_fuse_pass$|\
^test_mkldnn_matmul_v2_elementwise_add_fuse_pass$|\
^test_mkldnn_matmul_v2_transpose_reshape_fuse_pass$|\
^test_mkldnn_mish_op$|\
^test_mkldnn_pad3d_op$|\
^test_mkldnn_prelu_op$|\
^test_mkldnn_shuffle_channel_detect_pass$|\
^test_onednn_batch_norm_act_fuse_pass$|\
^test_onednn_conv_bias_fuse_pass$|\
^test_onednn_conv_bn_fuse_pass$|\
^test_onednn_conv_concat_activation_fuse_pass$|\
^test_onednn_conv_elementwise_add_fuse_pass$|\
^test_onednn_matmul_transpose_reshape_fuse_pass$|\
^test_onednn_multi_gru_fuse_pass$|\
^test_onednn_multi_gru_seq_fuse_pass$|\
^test_onednn_reshape_transpose_matmul_fuse_pass$|\
^test_conv2d_layer_deprecated$|\
^test_conv3d_layer$|\
^test_decorator$|\
^test_flash_attention$|\
^test_flash_attention_deterministic$|\
^test_conv3d_mkldnn_op$|\
^test_functional_conv2d$|\
^test_functional_conv2d_transpose$|\
^test_functional_conv3d$|\
^test_functional_conv3d_transpose$|\
^test_imperative_layer_children$|\
^test_inference_api_deprecated$|\
^test_trans_layout_op$|\
^test_pool2d_op$|\
^test_conv3d_transpose_op$|\
^test_linalg_cond$|\
^test_eigh_op_static_build$|\
^test_einsum_op$|\
^test_sequence_pool$|\
^test_conv2d_op$|\
^test_graph_send_ue_recv_op$|\
^test_recognize_digits$|\
^test_mnist$|\
^test_mnist_amp$|\
^test_hapi_amp$|\
^test_imperative_mnist_sorted_gradient$|\
^test_imperative_qat_lsq$|\
^test_argsort_op$|\
^test_image_classification_fp16$|\
^test_imperative_double_grad$|\
^test_se_resnet$|\
^test_standalone_executor_aot_choose_kernel$|\
^test_imperative_qat_user_defined$|\
^test_mnist_pure_fp16$|\
^test_callback_reduce_lr_on_plateau$|\
^test_callback_visualdl$|\
^test_callback_wandb$|\
^test_user_defined_quantization$|\
^test_quantization_scale_pass_deprecated$|\
^test_quantization_pass$|\
^test_imperative_qat$|\
^test_graph$|\
^test_executor_and_mul$|\
^test_gru_unit_op$|\
^test_matmul_op$|\
^test_decoupled_py_reader_data_check_deprecated$|\
^test_decoupled_py_reader_deprecated$|\
^test_generator_dataloader_deprecated$|\
^test_py_reader_combination$|\
^test_reader_reset_deprecated$|\
^test_decoupled_py_reader_deprecated_static_build$|\
^test_multiprocess_dataloader_iterable_dataset_dynamic$|\
^test_multiprocess_dataloader_iterable_dataset_static$|\
^test_dataloader_keep_order_deprecated$|\
^test_dataloader_unkeep_order_deprecated$|\
^test_add_reader_dependency_deprecated$|\
^test_fuse_bn_act_pass_deprecated$|\
^test_fuse_bn_act_pass_deprecated_static_build$|\
^test_fuse_bn_add_act_pass$|\
^test_model$|\
^test_dataloader_early_reset_deprecated$|\
^test_add_reader_dependency_deprecated$|\
^test_conv2d_fusion_op$|\
^test_fused_conv2d_add_act_op$|\
^test_audio_datasets$|\
^test_signal$|\
^test_stft_op$|\
^test_trt_convert_flatten_contiguous_range$|\
^test_trt_convert_gather$|\
^test_trt_convert_index_select$|\
^test_trt_convert_lookup_table$|\
^test_trt_convert_prelu$|\
^test_trt_convert_bilinear_interp_v2$|\
^test_trt_convert_leaky_relu$|\
^test_reverse_roll_fuse_pass$|\
^test_trt_convert_einsum$|\
^test_trt_convert_roi_align$|\
^test_trt_convert_temporal_shift$|\
^test_trt_convert_mish$|\
^test_trt_convert_pad3d$|\
^test_trt_convert_yolo_box$|\
^test_merge_layernorm_fuse_pass$|\
^test_trt_convert_instance_norm$|\
^test_skip_merge_layernorm_fuse_pass$|\
^test_trt_float64$|\
^test_trt_convert_arg_max$|\
^test_trt_convert_arg_min$|\
^test_trt_convert_assign$|\
^test_trt_convert_cast$|\
^test_trt_convert_compare_and_logical$|\
^test_trt_convert_concat$|\
^test_preln_layernorm_x_fuse_pass$|\
^test_trt_convert_argsort$|\
^test_trt_remove_amp_strategy_op_pass$|\
^test_trt_convert_bitwise_and$|\
^test_trt_convert_bitwise_or$|\
^test_trt_convert_scatter$|\
^test_trt_convert_solve$|\
^test_quant_linear_fuse_pass$|\
^test_trt_explicit_quantization$|\
^test_trt_nearest_interp_v2_op$|\
^test_trt_pool3d_op$|\
^test_trt_convert_anchor_generator$|\
^test_trt_convert_softmax$|\
^test_trt_convert_strided_slice$|\
^test_layernorm_shift_partition_pass$|\
^test_trt_convert_multihead_matmul$|\
^test_trt_convert_reshape$|\
^test_trt_convert_split$|\
^test_trt_convert_squeeze2$|\
^test_trt_convert_sum$|\
^test_trt_convert_transpose$|\
^test_trt_convert_unsqueeze2$|\
^test_simplify_with_basic_ops_pass_autoscan$|\
^test_trt_convert_nearest_interp$|\
^test_trt_pool_op$|\
^test_trt_convert_clip$|\
^test_trt_convert_grid_sampler$|\
^test_trt_convert_p_norm$|\
^test_fused_moe_op$|\
^disable_wingpu_cuda12_test$"

# /*=================Fixed Disabled Windows TRT MKL unittests=======================*/
# TODO: fix these unittest that is bound to fail
disable_win_trt_test="^test_trt_convert_conv2d$|\
^test_trt_convert_fused_conv2d_add_act$|\
^test_trt_convert_conv2d_transpose$|\
^test_trt_convert_depthwise_conv2d$|\
^test_trt_convert_emb_eltwise_layernorm$|\
^test_trt_convert_pool2d$|\
^test_trt_conv3d_op$|\
^test_trt_subgraph_pass$|\
^test_trt_convert_dropout$|\
^test_trt_convert_hard_sigmoid$|\
^test_trt_convert_reduce_mean$|\
^test_trt_convert_reduce_sum$|\
^test_trt_convert_group_norm$|\
^test_trt_convert_batch_norm$|\
^test_trt_convert_activation$|\
^test_trt_convert_depthwise_conv2d_transpose$|\
^test_trt_convert_elementwise$|\
^test_trt_convert_matmul$|\
^test_trt_convert_scale$"

# /*==========Fixed Disabled Windows CUDA11.x inference_api_test(PR-CI-Windows-Inference) unittests=============*/
disable_win_inference_test="^trt_quant_int8_yolov3_r50_test$|\
^test_trt_dynamic_shape_ernie$|\
^test_trt_dynamic_shape_ernie_fp16_ser_deser$|\
^lite_resnet50_test$|\
^test_trt_dynamic_shape_transformer_prune$|\
^lite_mul_model_test$|\
^trt_split_converter_test$|\
^paddle_infer_api_copy_tensor_tester$|\
^test_trt_deformable_conv$|\
^test_imperative_triple_grad$|\
^test_full_name_usage$|\
^test_trt_convert_unary$|\
^test_eigh_op$|\
^test_eigh_op_static_build$|\
^test_fc_op$|\
^test_stack_op$|\
^trt_split_converter_test$|\
^paddle_infer_api_copy_tensor_tester$|\
^test_eager_tensor$|\
^test_einsum_v2$|\
^test_tensor_scalar_type_promotion_static$|\
^test_matrix_power_op$|\
^test_deformable_conv_v1_op$|\
^test_where_index$|\
^test_custom_grad_input$|\
^test_conv3d_transpose_op$|\
^test_conv_elementwise_add_act_fuse_pass$|\
^test_conv_eltwiseadd_bn_fuse_pass$|\
^test_custom_relu_op_setup$|\
^test_conv3d_transpose_part2_op$|\
^test_deform_conv2d$|\
^test_deform_conv2d_deprecated$|\
^test_matmul_op$|\
^test_matmul_op_static_build$|\
^test_basic_api_transformation$|\
^test_deformable_conv_op$|\
^test_variable$|\
^test_mkldnn_conv_hard_sigmoid_fuse_pass$|\
^test_mkldnn_conv_hard_swish_fuse_pass$|\
^test_conv_act_mkldnn_fuse_pass$|\
^test_matmul_scale_fuse_pass$|\
^test_addmm_op$|\
^test_inverse_op$|\
^test_set_value_op$|\
^test_fused_multihead_matmul_op$|\
^test_cudnn_bn_add_relu$|\
^test_cond$|\
^test_conv_bn_fuse_pass$|\
^test_graph_khop_sampler$|\
^test_gru_rnn_op$|\
^test_masked_select_op$|\
^test_ir_fc_fuse_pass_deprecated$|\
^test_fc_elementwise_layernorm_fuse_pass$|\
^test_linalg_pinv_op$|\
^test_math_op_patch_var_base$|\
^test_slice$|\
^test_conv_elementwise_add_fuse_pass$|\
^test_executor_and_mul$|\
^test_analyzer_int8_resnet50$|\
^test_analyzer_int8_mobilenetv1$|\
^test_trt_conv_pass$|\
^test_roll_op$|\
^test_lcm$|\
^test_elementwise_floordiv_op$|\
^test_autograd_functional_dynamic$|\
^test_corr$|\
^test_trt_convert_deformable_conv$|\
^test_conv_elementwise_add2_act_fuse_pass$|\
^test_tensor_scalar_type_promotion_dynamic$|\
^test_model$|\
^test_py_reader_combination$|\
^test_py_reader_push_pop$|\
^test_reader_reset_deprecated$|\
^test_py_reader_pin_memory$|\
^test_multiprocess_dataloader_iterable_dataset_dynamic$|\
^test_multiprocess_dataloader_iterable_dataset_static$|\
^test_add_reader_dependency_deprecated$|\
^test_add_reader_dependency_for_interpretercore$|\
^test_compat$|\
^test_decoupled_py_reader_deprecated$|\
^test_decoupled_py_reader_deprecated_static_build$|\
^test_generator_dataloader_deprecated$|\
^test_py_reader_using_executor$|\
^test_dataloader_keep_order_deprecated$|\
^test_dataloader_unkeep_order_deprecated$|\
^test_fuse_bn_act_pass_deprecated$|\
^test_fuse_bn_act_pass_deprecated_static_build$|\
^test_fuse_bn_add_act_pass$|\
^test_decoupled_py_reader_data_check_deprecated$|\
^test_parallel_dygraph_sync_batch_norm$|\
^test_dataloader_early_reset_deprecated$|\
^test_fleet_base_single$|\
^test_sequence_pool$|\
^test_simplify_with_basic_ops_pass_autoscan$|\
^test_trt_activation_pass$|\
^test_trt_convert_hard_swish$|\
^test_trt_convert_leaky_relu$|\
^test_trt_convert_multihead_matmul$|\
^test_trt_convert_prelu$|\
^test_trt_fc_fuse_quant_dequant_pass$|\
^test_api_impl$|\
^test_tensordot$|\
^disable_win_inference_test$|\
^test_imperative_double_grad$|\
^test_comp_eager_matmul_double_grad$|\
^test_cuda_graph_partial_graph_static_run$|\
^test_imperative_triple_grad$"


# /*==========Fixed Disabled Windows CPU OPENBLAS((PR-CI-Windows-OPENBLAS)) unittests==============================*/
# TODO: fix these unittest that is bound to fail
disable_wincpu_test="^jit_kernel_test$|\
^test_analyzer_transformer_deprecated$|\
^test_vision_models$|\
^test_dygraph_multi_forward$|\
^test_imperative_transformer_sorted_gradient$|\
^test_program_prune_backward_deprecated$|\
^test_imperative_resnet$|\
^test_imperative_resnet_sorted_gradient$|\
^test_imperative_se_resnext$|\
^test_bmn$|\
^test_mobile_net$|\
^test_build_strategy$|\
^test_se_resnet$|\
^disable_wincpu_test$"

# these unittest that cost long time, disabled temporarily, Maybe moved to the night
long_time_test="^test_gru_op$|\
^decorator_test$|\
^test_dataset_imdb$|\
^test_datasets$|\
^test_pretrained_model$|\
^test_gather_op$|\
^test_gather_nd_op$|\
^test_sequence_conv$|\
^test_activation_nn_grad$|\
^test_activation_op$|\
^test_bicubic_interp_v2_op$|\
^test_bilinear_interp_v2_op$|\
^test_crop_tensor_op$|\
^test_cross_entropy2_op$|\
^test_cross_op$|\
^test_elementwise_nn_grad$|\
^test_fused_elemwise_activation_op$|\
^test_imperative_lod_tensor_to_selected_rows_deprecated$|\
^test_imperative_selected_rows_to_lod_tensor$|\
^test_layer_norm_op$|\
^test_layer_norm_op_static_build$|\
^test_multiclass_nms_op$|\
^test_nearest_interp_v2_op$|\
^test_nn_grad$|\
^test_norm_nn_grad$|\
^test_normal$|\
^test_pool3d_op$|\
^test_static_save_load$|\
^test_trilinear_interp_op$|\
^test_trilinear_interp_v2_op$|\
^test_bilinear_interp_op$|\
^test_nearest_interp_op$|\
^test_sequence_conv$|\
^test_sgd_op$|\
^test_transformer$|\
^test_trt_matmul_quant_dequant$|\
^test_strided_slice_op$"


# /*============================================================================*/

set -e
set +x
NIGHTLY_MODE=$1
PRECISION_TEST=$2
WITH_GPU=$3

# Step1: Print disable_ut_quickly
export PADDLE_ROOT="$(cd "$PWD/../" && pwd )"
if [ ${NIGHTLY_MODE:-OFF} == "ON" ]; then
    nightly_label=""
else
    nightly_label="(RUN_TYPE=NIGHTLY|RUN_TYPE=DIST:NIGHTLY|RUN_TYPE=EXCLUSIVE:NIGHTLY)"
    echo "========================================="
    echo "Unittests with nightly labels  are only run at night"
    echo "========================================="
fi

if disable_ut_quickly=$(python ${PADDLE_ROOT}/tools/get_quick_disable_lt.py); then
    echo "========================================="
    echo "The following unittests have been disabled:"
    echo ${disable_ut_quickly}
    echo "========================================="
else
    disable_ut_quickly=''
fi

# Step2: Check added ut
set +e
cp $PADDLE_ROOT/tools/check_added_ut.sh $PADDLE_ROOT/tools/check_added_ut_win.sh
bash $PADDLE_ROOT/tools/check_added_ut_win.sh
rm -rf $PADDLE_ROOT/tools/check_added_ut_win.sh
if [ -f "$PADDLE_ROOT/added_ut" ];then
    added_uts=^$(awk BEGIN{RS=EOF}'{gsub(/\n/,"$|^");print}' $PADDLE_ROOT/added_ut)$
    ctest -R "(${added_uts})" -E "${disable_win_inference_test}" --output-on-failure -C Release --repeat-until-fail 3;added_ut_error=$?
    #rm -f $PADDLE_ROOT/added_ut
    if [ "$added_ut_error" != 0 ];then
        echo "========================================"
        echo "Added UT should pass three additional executions"
        echo "========================================"
        exit 8;
    fi
fi


# Step3: Get precision UT and intersect with parallel UT, generate tools/*_new file
set -e
if [ ${WITH_GPU:-OFF} == "ON" ];then
    export CUDA_VISIBLE_DEVICES=0

    ctest -N | awk -F ': ' '{print $2}' | sed '/^$/d' | sed '$d' > all_ut_list
    num=$(ctest -N | awk -F ': ' '{print $2}' | sed '/^$/d' | sed '$d' | wc -l)
    echo "Windows 1 card TestCases count is $num"
    if [ ${PRECISION_TEST:-OFF} == "ON" ]; then
        python ${PADDLE_ROOT}/tools/get_pr_ut.py || echo "Failed to obtain ut_list !"
    fi

    python ${PADDLE_ROOT}/tools/group_case_for_parallel.py ${PADDLE_ROOT}

fi

failed_test_lists=''
tmp_dir=`mktemp -d`
function collect_failed_tests() {
    set +e
    for file in `ls $tmp_dir`; do
        grep -q 'The following tests FAILED:' $tmp_dir/$file
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            failuretest=''
        else
            failuretest=`grep -A 10000 'The following tests FAILED:' $tmp_dir/$file | sed 's/The following tests FAILED://g'|sed '/^$/d'`
            failed_test_lists="${failed_test_lists}
            ${failuretest}"
        fi
    done
    set -e
}

function run_unittest_cpu() {
    tmpfile=$tmp_dir/$RANDOM
    (ctest -E "$disable_ut_quickly|$disable_wincpu_test" -LE "${nightly_label}" --output-on-failure -C Release -j 8 | tee $tmpfile) &
    wait;
}

function run_unittest_gpu() {
    test_case=$1
    parallel_job=$2
    parallel_level_base=${CTEST_PARALLEL_LEVEL:-1}
    if [ "$2" == "" ]; then
        parallel_job=$parallel_level_base
    else
        # set parallel_job according to CUDA memory and suggested parallel num,
        # the latter is derived in linux server with 16G CUDA memory.
        cuda_memory=$(nvidia-smi --query-gpu=memory.total --format=csv | tail -1 | awk -F ' ' '{print $1}')
        parallel_job=$(($2 * $cuda_memory / 16000))
        if [ $parallel_job -lt 1 ]; then
            parallel_job=1
        fi
    fi
    echo "************************************************************************"
    echo "********These unittests run $parallel_job job each time with 1 GPU**********"
    echo "************************************************************************"
    export CUDA_VISIBLE_DEVICES=0

    if nvcc --version | grep 11.2; then
        disable_wingpu_test=${disable_win_inference_test}
    fi

    if nvcc --version | grep 12.0; then
        disable_wingpu_test=${disable_wingpu_cuda12_test}
    fi

    tmpfile=$tmp_dir/$RANDOM
    (ctest -R "$test_case" -E "$disable_ut_quickly|$disable_wingpu_test|$disable_win_trt_test|$long_time_test" -LE "${nightly_label}" --output-on-failure -C Release -j $parallel_job | tee $tmpfile ) &
    wait;
}

function unittests_retry(){
    is_retry_execute=0
    wintest_error=1
    retry_time=3
    exec_times=0
    exec_retry_threshold=30
    retry_unittests=$(echo "${failed_test_lists}" | grep -oEi "\-.+\(" | sed 's/(//' | sed 's/- //' )
    need_retry_ut_counts=$(echo "$retry_unittests" |awk -F ' ' '{print }'| sed '/^$/d' | wc -l)
    retry_unittests_regular=$(echo "$retry_unittests" |awk -F ' ' '{print }' | awk 'BEGIN{ all_str=""}{if (all_str==""){all_str=$1}else{all_str=all_str"$|^"$1}} END{print "^"all_str"$"}')
    tmpfile=$tmp_dir/$RANDOM

    if [ $need_retry_ut_counts -lt $exec_retry_threshold ];then
            retry_unittests_record=''
            while ( [ $exec_times -lt $retry_time ] )
                do
                    retry_unittests_record="$retry_unittests_record$failed_test_lists"
                    if ( [[ "$exec_times" == "0" ]] );then
                        cur_order='first'
                    elif ( [[ "$exec_times" == "1" ]] );then
                        cur_order='second'
                        if [[ "$failed_test_lists" == "" ]]; then
                            break
                        else
                            retry_unittests=$(echo "${failed_test_lists}" | grep -oEi "\-.+\(" | sed 's/(//' | sed 's/- //' )
                            retry_unittests_regular=$(echo "$retry_unittests" |awk -F ' ' '{print }' | awk 'BEGIN{ all_str=""}{if (all_str==""){all_str=$1}else{all_str=all_str"$|^"$1}} END{print "^"all_str"$"}')
                        fi
                    elif ( [[ "$exec_times" == "2" ]] );then
                        cur_order='third'
                    fi
                    echo "========================================="
                    echo "This is the ${cur_order} time to re-run"
                    echo "========================================="
                    echo "The following unittest will be re-run:"
                    echo "${retry_unittests}"
                    echo "========================================="
                    rm -f $tmp_dir/*
                    failed_test_lists=''
                    (ctest -R "($retry_unittests_regular)" --output-on-failure -C Release -j 1 | tee $tmpfile ) &
                    wait;
                    collect_failed_tests
                    exec_times=$(echo $exec_times | awk '{print $0+1}')
                done
    else
        # There are more than 30 failed unit tests, so no unit test retry
        is_retry_execute=1
    fi
    rm -f $tmp_dir/*
}

function show_ut_retry_result() {
    if [[ "$is_retry_execute" != "0" ]];then
        failed_test_lists_ult=`echo "${failed_test_lists}"`
        echo "========================================="
        echo "There are more than 30 failed unit tests, so no unit test retry!!!"
        echo "========================================="
        echo "${failed_test_lists_ult}"
        exit 8;
    else
        retry_unittests_ut_name=$(echo "$retry_unittests_record" | grep -oEi "\-.+\(" | sed 's/(//' | sed 's/- //' )
        retry_unittests_record_judge=$(echo ${retry_unittests_ut_name}| tr ' ' '\n' | sort | uniq -c | awk '{if ($1 >=3) {print $2}}')
        if [ -z "${retry_unittests_record_judge}" ];then
            echo "========================================"
            echo "There are failed tests, which have been successful after re-run:"
            echo "========================================"
            echo "The following tests have been re-run:"
            echo "${retry_unittests_record}"
        else
            failed_ut_re=$(echo "${retry_unittests_record_judge}" | awk 'BEGIN{ all_str=""}{if (all_str==""){all_str=$1}else{all_str=all_str"|"$1}} END{print all_str}')
            echo "========================================"
            echo "There are failed tests, which have been executed re-run,but success rate is less than 50%:"
            echo "Summary Failed Tests... "
            echo "========================================"
            echo "The following tests FAILED: "
            echo "${retry_unittests_record}" | grep -E "$failed_ut_re"
            exit 8;
        fi
    fi
}

# Step4: Run UT gpu or cpu
set +e
export FLAGS_call_stack_level=2
if [ "${WITH_GPU:-OFF}" == "ON" ];then

    single_ut_mem_0_startTime_s=`date +%s`
    if [ ${WIN_UNITTEST_LEVEL:-2} == "0" ]; then
        echo "ipipe_log_param_1_mem_0_TestCases_Total_Time: 0 s"
    else
        while read line
        do
            run_unittest_gpu "$line" 16
        done < $PADDLE_ROOT/tools/single_card_tests_mem0_new
        single_ut_mem_0_endTime_s=`date +%s`
        single_ut_mem_0_Time_s=`expr $single_ut_mem_0_endTime_s - $single_ut_mem_0_startTime_s`
        echo "ipipe_log_param_1_mem_0_TestCases_Total_Time: $single_ut_mem_0_Time_s s"
    fi

    single_ut_startTime_s=`date +%s`
    while read line
    do
        num=`echo $line | awk -F"$" '{print NF-1}'`
        para_num=`expr $num / 2`
        if [ $para_num -eq 0 ]; then
            para_num=4
        fi
        run_unittest_gpu "$line" $para_num
    done < $PADDLE_ROOT/tools/single_card_tests_new
    single_ut_endTime_s=`date +%s`
    single_ut_Time_s=`expr $single_ut_endTime_s - $single_ut_startTime_s`
    echo "ipipe_log_param_1_TestCases_Total_Time: $single_ut_Time_s s"

    multiple_ut_mem_0_startTime_s=`date +%s`
    while read line
    do
        run_unittest_gpu "$line" 10
    done < $PADDLE_ROOT/tools/multiple_card_tests_mem0_new
    multiple_ut_mem_0_endTime_s=`date +%s`
    multiple_ut_mem_0_Time_s=`expr $multiple_ut_mem_0_endTime_s - $multiple_ut_mem_0_startTime_s`
    echo "ipipe_log_param_2_mem0_TestCases_Total_Time: $multiple_ut_mem_0_Time_s s"

    multiple_ut_startTime_s=`date +%s`
    while read line
    do
        num=`echo $line | awk -F"$" '{print NF-1}'`
        para_num=`expr $num / 2`
        if [ $para_num -eq 0 ]; then
            para_num=4
        fi
        run_unittest_gpu "$line" $para_num

    done < $PADDLE_ROOT/tools/multiple_card_tests_new
    multiple_ut_endTime_s=`date +%s`
    multiple_ut_Time_s=`expr $multiple_ut_endTime_s - $multiple_ut_startTime_s`
    echo "ipipe_log_param_2_TestCases_Total_Time: $multiple_ut_Time_s s"


    exclusive_ut_mem_0_startTime_s=`date +%s`
    while read line
    do
        run_unittest_gpu "$line" 10
    done < $PADDLE_ROOT/tools/exclusive_card_tests_mem0_new
    exclusive_ut_mem_0_endTime_s=`date +%s`
    exclusive_ut_mem_0_Time_s=`expr $exclusive_ut_mem_0_endTime_s - $exclusive_ut_mem_0_startTime_s`
    echo "ipipe_log_param_-1_mem0_TestCases_Total_Time: $exclusive_ut_mem_0_Time_s s"

    exclusive_ut_startTime_s=`date +%s`
    while read line
    do
        num=`echo $line | awk -F"$" '{print NF-1}'`
        para_num=`expr $num / 2`
        if [ $para_num -eq 0 ]; then
            para_num=4
        fi
        run_unittest_gpu "$line" $para_num
    done < $PADDLE_ROOT/tools/exclusive_card_tests_new
    exclusive_ut_endTime_s=`date +%s`
    exclusive_ut_Time_s=`expr $exclusive_ut_endTime_s - $exclusive_ut_startTime_s`
    echo "ipipe_log_param_-1_TestCases_Total_Time: $exclusive_ut_Time_s s"

    noparallel_ut_startTime_s=`date +%s`
    while read line
    do
        run_unittest_gpu "$line" 8
    done < $PADDLE_ROOT/tools/no_parallel_case_file
    noparallel_ut_endTime_s=`date +%s`
    noparallel_ut_Time_s=`expr $noparallel_ut_endTime_s - $noparallel_ut_startTime_s`
    echo "ipipe_log_param_noparallel_TestCases_Total_Time: $noparallel_ut_Time_s s"
else
    run_unittest_cpu
fi
collect_failed_tests
set -e
rm -f $tmp_dir/*
if [[ "$failed_test_lists" != "" ]]; then
    unittests_retry
    show_ut_retry_result
fi
