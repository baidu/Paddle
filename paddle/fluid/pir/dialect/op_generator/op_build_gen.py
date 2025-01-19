# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# generator build function
_INFERMETA_NEED_META_CONFIG = {
    # binary.h
    'AllValueCompareInferMeta',
    'KLDivInferMeta',
    'ArrayWriteInferMeta',
    'ArrayReadInferMeta',
    'BCELossInferMeta',
    'BinomialInferMeta',
    'ConvInferMeta',
    'Conv3DInferMeta',
    'ConvTransposeInferMeta',
    'Conv2dTransposeInferMeta',
    'CrossEntropyWithSoftmaxInferMeta',
    'CSoftmaxWithCrossEntropyInferMeta',
    'CSoftmaxWithMultiLabelCrossEntropyInferMeta',
    'DepthwiseConvInferMeta',
    'DistributeFpnProposalsInferMeta',
    'ElementwiseRawInferMeta',
    'GridSampleBaseInferMeta',
    'HuberLossInferMeta',
    'IndexSampleInferMeta',
    'LogLossInferMeta',
    'MarginCrossEntropyInferMeta',
    'MatrixNMSInferMeta',
    'PReluInferMeta',
    'SegmentPoolInferMeta',
    'YoloBoxInferMeta',
    'ValueCompareInferMeta',
    'UnpoolInferMeta',
    'Unpool3dInferMeta',
    # fusion.h
    'FusedAttentionInferMeta',
    'BNActXPUInferMeta',
    'FusedFCElementwiseLayerNormInferMeta',
    'FusedConv2dAddActInferMeta',
    # multiary.h
    'AddNTensorArrayInferMeta',
    'AddNInferMeta',
    'AucInferMeta',
    'BatchNormInferMeta',
    'BatchNormInferInferMeta',
    'BilinearInferMeta',
    'CoalesceTensorInferMeta',
    'CheckMemoryContinueInferMeta',
    'ConcatInferMeta',
    'DeformableConvInferMeta',
    'FusedBiasActInferMeta',
    'InterpolateInferMeta',
    'NceInferMeta',
    'SigmoidCrossEntropyWithLogitsInferMeta',
    'StackInferMeta',
    'FusedConvInferMeta',
    # nullary.h
    'EyeInferMeta',
    # ternary.h
    'AccuracyInferMeta',
    'BoxCoderInferMeta',
    'InstanceNormInferMeta',
    'LayerNormInferMeta',
    'MatchMatrixTensorInferMeta',
    'MultiClassNMSInferMeta',
    'NllLossRawInferMeta',
    'RoiAlignInferMeta',
    'SpectralNormInferMeta',
    'ViterbiDecodeInferMeta',
    'TdmSamplerInferMeta',
    # unary.h
    'GetSplitAxisValue',
    'ArgMinMaxInferMeta',
    'ArrayToTensorInferMeta',
    'CropInferMeta',
    'EigvalsInferMeta',
    'FractionalMaxPoolInferMeta',
    'MaxPoolWithIndexInferMeta',
    'MaxPoolV2InferMeta',
    'MultinomialInferMeta',
    'OverlapAddInferMeta',
    'PadInferMeta',
    'Pad3dInferMeta',
    'PoolInferMeta',
    'Pool2DInferMeta',
    'ReduceIntArrayAxisInferMetaBase',
    'ReduceIntArrayAxisInferMeta',
    'ReshapeInferMeta',
    'ReshapeWithXShapeInferMeta',
    'ReverseInferMeta',
    'ReverseArrayInferMeta',
    'ShardIndexInferMeta',
    'SliceArrayInferMeta',
    'SliceArrayDenseInferMeta',
    'SliceRawInferMeta',
    'SplitInferMeta',
    'SplitWithNumInferMeta',
    'SqueezeInferMeta',
    'SqueezeWithXShapeInferMeta',
    'StridedSliceRawInferMeta',
    'StridedSliceInferMeta',
    'SumInferMeta',
    'SumRawInferMeta',
    'TemporalShiftInferMeta',
    'TileInferMeta',
    'TopKInferMeta',
    'UnfoldInferMeta',
    'UnsqueezeInferMeta',
    'UnsqueezeWithXShapeInferMeta',
    'ArrayPopInferMeta',
}

_PREPARE_DATA_WITH_VECTOR_INT64_MTTABLE_ATTRIBUTE = {'FrobeniusNormOp'}

OP_BUILD_TEMPLATE = """
void {op_name}::Build({build_args}) {{
{build_info}
{get_attributes}
{build_mutable_attributes}
{build_inputs}
{build_attributes}
{build_outputs}
}}
"""

OP_INFO_TEMPLATE = '  VLOG(4) << "Start build {op_name}";\n'


def GenBuildInputArgsStr(
    op_input_name_list,
    op_attribute_name_list,
    op_attribute_build_arg_type_list,
    op_attribute_default_value_list,
    op_mutable_attribute_name_list,
    op_non_mutable_attribute_name_list,
    op_non_mutable_attribute_build_arg_type_list,
    op_non_mutable_attribute_default_value_list,
    for_func_define=True,
    mutable_attr_is_input=False,
    attr_args_is_map=False,
):
    '''
    Example: pir::Builder &builder, pir::OperationArgument &argument, pir::Value x_, phi::DataType dtype=phi::DataType::UNDEFINED, phi::Place place={}
    '''
    # add inputs
    build_args_str = "pir::Builder &builder, pir::OperationArgument &argument"
    if len(op_input_name_list) > 0:
        for input_name in op_input_name_list:
            build_args_str += ", pir::Value " + input_name + "_"

    if mutable_attr_is_input:
        # add mutable attributes as inputs
        if len(op_mutable_attribute_name_list) > 0:
            for mutable_attr in op_mutable_attribute_name_list:
                build_args_str += ", pir::Value " + mutable_attr + "_"
        if attr_args_is_map:
            build_args_str += ", pir::AttributeMap attributes"
        else:
            # add non-mutable attributes
            for attr_idx in range(len(op_non_mutable_attribute_name_list)):
                build_args_str += (
                    ", "
                    + op_non_mutable_attribute_build_arg_type_list[attr_idx]
                    + " "
                    + op_non_mutable_attribute_name_list[attr_idx]
                )
                if for_func_define:
                    if (
                        op_non_mutable_attribute_default_value_list[attr_idx]
                        is not None
                    ):
                        default_value = (
                            op_non_mutable_attribute_default_value_list[
                                attr_idx
                            ]
                        )
                        if (
                            op_non_mutable_attribute_build_arg_type_list[
                                attr_idx
                            ]
                            != "const std::string&"
                        ):
                            if (
                                default_value[0] == "'"
                                or default_value[0] == '"'
                            ):
                                default_value = default_value[1:]
                            if (
                                default_value[-1] == "'"
                                or default_value[-1] == '"'
                            ):
                                default_value = default_value[0:-1]
                        build_args_str += "=" + default_value
    else:
        if attr_args_is_map:
            build_args_str += ", pir::AttributeMap attributes"
        else:
            # add attributes
            for attr_idx in range(len(op_attribute_name_list)):
                build_args_str += (
                    ", "
                    + op_attribute_build_arg_type_list[attr_idx]
                    + " "
                    + op_attribute_name_list[attr_idx]
                )
                if for_func_define:
                    if op_attribute_default_value_list[attr_idx] is not None:
                        default_value = op_attribute_default_value_list[
                            attr_idx
                        ]
                        if (
                            op_attribute_build_arg_type_list[attr_idx]
                            != "const std::string&"
                        ):
                            if (
                                default_value[0] == "'"
                                or default_value[0] == '"'
                            ):
                                default_value = default_value[1:]
                            if (
                                default_value[-1] == "'"
                                or default_value[-1] == '"'
                            ):
                                default_value = default_value[0:-1]
                        build_args_str += "=" + default_value

    return build_args_str


mutable_attribute_phi_type_maps = {
    'int': 'phi::DataType::INT32',
    'int64_t': 'phi::DataType::INT64',
    'float': 'phi::DataType::FLOAT32',
    'double': 'phi::DataType::FLOAT64',
    'std::vector<int64_t>': 'phi::DataType::INT64',
    'const std::vector<int64_t>&': 'phi::DataType::INT64',
    'bool': 'phi::DataType::BOOL',
}


def GenBuildInsertFullForMutableAttribute(
    args,
    op_info,
    op_attribute_name_list,
    op_attribute_build_arg_type_list,
    op_mutable_attribute_name_list,
    op_mutable_attribute_type_list,
):
    build_mutable_attribute = ""
    BUILD_INTARRAY_ATTRIBUTE_TEMPLATE = """  // Generate int_array mutable attribute: {attr_name}
  paddle::dialect::FullIntArrayOp full_{attr_name}_op = builder.Build<paddle::dialect::FullIntArrayOp>({attr_name}, {phi_dtype}, phi::CPUPlace());
  pir::Value {attr_name}_ = full_{attr_name}_op->result(0);
    """
    BUILD_SCALAR_ATTRIBUTE_TEMPLATE = """  // Generate scalar mutable attribute: {attr_name}
  paddle::dialect::FullOp full_{attr_name}_op = builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{{1}}, {attr_name}, {phi_dtype}, phi::CPUPlace());
  pir::Value {attr_name}_ = full_{attr_name}_op->result(0);
    """
    for idx in range(len(op_mutable_attribute_name_list)):
        attr_name = op_mutable_attribute_name_list[idx]
        attr_type = op_mutable_attribute_type_list[idx][0]
        if attr_name in op_attribute_name_list:
            phi_dtype = mutable_attribute_phi_type_maps[
                op_attribute_build_arg_type_list[
                    op_attribute_name_list.index(attr_name)
                ]
            ]
        else:
            phi_dtype = mutable_attribute_phi_type_maps[
                op_mutable_attribute_type_list[idx][1]
            ]

        if attr_type == "paddle::dialect::IntArrayAttribute":
            build_mutable_attribute += BUILD_INTARRAY_ATTRIBUTE_TEMPLATE.format(
                attr_name=attr_name, phi_dtype=phi_dtype
            )
        else:
            build_mutable_attribute += BUILD_SCALAR_ATTRIBUTE_TEMPLATE.format(
                attr_name=attr_name, phi_dtype=phi_dtype
            )
    return build_mutable_attribute


def GenBuildInputs(op_input_name_list, op_mutable_attribute_name_list):
    BUILD_INPUT_TEMPLATE = """  std::vector<pir::Value> argument_inputs = {{{inputs_args}}};
  argument.AddInputs(argument_inputs);
"""
    build_input_str = '  VLOG(4) << "Builder construction inputs";\n'
    input_name_list = op_input_name_list + op_mutable_attribute_name_list
    if len(input_name_list) > 0:
        inputs_args_str = ""
        inputs_args_str += "_, ".join(input_name_list) + "_"
        build_input_str += BUILD_INPUT_TEMPLATE.format(
            inputs_args=inputs_args_str
        )
    else:
        build_input_str += '  std::vector<pir::Value> argument_inputs = {};\n'
    return build_input_str


def GenBuildAttributes(
    op_non_mutable_attribute_name_list, op_non_mutable_attribute_type_list
):
    INTARRAY_STR_TEMPLATE = """  pir::Attribute attr_{attr_name} = {op_attribute_type}::get(pir::IrContext::Instance(), phi::IntArray({attr}));
"""
    SCALAR_STR_TEMPLATE = """  pir::Attribute attr_{attr_name} = paddle::dialect::TransToIrAttribute({attr}, pir::IrContext::Instance());
"""
    STR_TEMPLATE = """  pir::Attribute attr_{attr_name} = {op_attribute_type}::get(pir::IrContext::Instance(), {attr});
"""
    ARRAY_ATTRIBUTE_TEMPLATE = """  std::vector<pir::Attribute> vec_{attr_name};
  for (size_t i = 0; i < static_cast<size_t>({attr_size}); i++) {{
    {create_attribute}
    vec_{attr_name}.push_back(attr_{attr_name});
  }}
  pir::Attribute attr_{attr_name} = pir::ArrayAttribute::get(pir::IrContext::Instance(), vec_{attr_name});
"""
    attr_str = '  VLOG(4) << "Builder construction attributes";\n'
    attr_str += '  pir::AttributeMap argument_attributes = {};\n'
    array_attr_type = "pir::ArrayAttribute<"
    for idx in range(len(op_non_mutable_attribute_name_list)):
        if array_attr_type in op_non_mutable_attribute_type_list[idx]:
            inner_attribute_type = op_non_mutable_attribute_type_list[idx][
                len(array_attr_type) : -1
            ]
            if inner_attribute_type == "paddle::dialect::IntArrayAttribute":
                attr_str += ARRAY_ATTRIBUTE_TEMPLATE.format(
                    attr_name=op_non_mutable_attribute_name_list[idx],
                    attr_size=op_non_mutable_attribute_name_list[idx]
                    + ".size()",
                    create_attribute=INTARRAY_STR_TEMPLATE.format(
                        attr_name=op_non_mutable_attribute_name_list[idx],
                        op_attribute_type=inner_attribute_type,
                        attr=op_non_mutable_attribute_name_list[idx] + "[i]",
                    ),
                )
            elif inner_attribute_type == "paddle::dialect::ScalarAttribute":
                attr_str += ARRAY_ATTRIBUTE_TEMPLATE.format(
                    attr_name=op_non_mutable_attribute_name_list[idx],
                    attr_size=op_non_mutable_attribute_name_list[idx]
                    + ".size()",
                    create_attribute=SCALAR_STR_TEMPLATE.format(
                        attr_name=op_non_mutable_attribute_name_list[idx],
                        attr=op_non_mutable_attribute_name_list[idx] + "[i]",
                    ),
                )
            else:
                attr_str += ARRAY_ATTRIBUTE_TEMPLATE.format(
                    attr_name=op_non_mutable_attribute_name_list[idx],
                    attr_size=op_non_mutable_attribute_name_list[idx]
                    + ".size()",
                    create_attribute=STR_TEMPLATE.format(
                        attr_name=op_non_mutable_attribute_name_list[idx],
                        op_attribute_type=inner_attribute_type,
                        attr=op_non_mutable_attribute_name_list[idx] + "[i]",
                    ),
                )
        elif (
            op_non_mutable_attribute_type_list[idx]
            == "paddle::dialect::IntArrayAttribute"
        ):
            attr_str += INTARRAY_STR_TEMPLATE.format(
                attr_name=op_non_mutable_attribute_name_list[idx],
                op_attribute_type=op_non_mutable_attribute_type_list[idx],
                attr=op_non_mutable_attribute_name_list[idx],
            )

        elif (
            op_non_mutable_attribute_type_list[idx]
            == "paddle::dialect::ScalarAttribute"
        ):
            attr_str += SCALAR_STR_TEMPLATE.format(
                attr_name=op_non_mutable_attribute_name_list[idx],
                attr=op_non_mutable_attribute_name_list[idx],
            )
        else:
            attr_str += STR_TEMPLATE.format(
                attr_name=op_non_mutable_attribute_name_list[idx],
                op_attribute_type=op_non_mutable_attribute_type_list[idx],
                attr=op_non_mutable_attribute_name_list[idx],
            )
        attr_str += f"""  argument_attributes.insert({{"{op_non_mutable_attribute_name_list[idx]}", attr_{op_non_mutable_attribute_name_list[idx]}}});\n"""

    return attr_str


def GenBuildOutputs(
    op_class_name,
    op_input_name_list,
    op_input_type_list,
    op_input_optional_list,
    op_mutable_attribute_name_list,
    op_mutable_attribute_type_list,
    op_output_name_list,
    op_output_type_list,
    op_output_size_list,
    op_output_optional_list,
    op_infer_meta_map,
    op_inplace_map,
    mutable_attr_is_input=False,
):
    build_output_str = '  VLOG(4) << "Builder construction outputs";\n'
    CREATE_INPUT_METATENSOR_TEMPLATE = """
  VLOG(4) << "Builder construction  dense_{name}";
  paddle::dialect::IrTensor ir_tensor_{name}(paddle::dialect::TransToPhiDataType({name}.dtype()),
                                                      {name}.dims(),
                                                      {name}.data_layout(),
                                                      {name}.lod(),
                                                      {name}.offset());
  VLOG(4) << "Builder construction  meta_{name}";
  paddle::dialect::IrMetaTensor meta_{name}(&ir_tensor_{name});
"""

    CREATE_OPTIONAL_INPUT_METATENSOR_TEMPLATE = """
  paddle::dialect::IrMetaTensor meta_{name};
  paddle::dialect::IrTensor ir_tensor_{name};
  if ({name}_.impl() != nullptr) {{
    {type} {name} = {name}_.type().dyn_cast<{type}>();
    VLOG(4) << "Builder construction  dense_{name}";
    ir_tensor_{name} = paddle::dialect::IrTensor(paddle::dialect::TransToPhiDataType({name}.dtype()),
                                                        {name}.dims(),
                                                        {name}.data_layout(),
                                                        {name}.lod(),
                                                        {name}.offset());
    VLOG(4) << "Builder construction  meta_{name}";
    meta_{name} = paddle::dialect::IrMetaTensor(&ir_tensor_{name});
  }}

"""

    CREATE_INPUT_VEC_METATENSOR_TEMPLATE = """  std::vector<paddle::dialect::IrTensor> vec_ir_tensor_{name};
  for (size_t i=0; i < static_cast<size_t>({name}.size()); i++) {{
    vec_ir_tensor_{name}.push_back(paddle::dialect::IrTensor(paddle::dialect::TransToPhiDataType({name}[i].dyn_cast<paddle::dialect::DenseTensorType>().dtype()),
                                                                     {name}[i].dyn_cast<paddle::dialect::DenseTensorType>().dims(),
                                                                     {name}[i].dyn_cast<paddle::dialect::DenseTensorType>().data_layout(),
                                                                     {name}[i].dyn_cast<paddle::dialect::DenseTensorType>().lod(),
                                                                     {name}[i].dyn_cast<paddle::dialect::DenseTensorType>().offset()));
  }}
  std::vector<paddle::dialect::IrMetaTensor> vec_meta_{name};
  for (size_t i=0; i < vec_ir_tensor_{name}.size(); i++) {{
    vec_meta_{name}.push_back(paddle::dialect::IrMetaTensor(&vec_ir_tensor_{name}[i]));
  }}

  std::vector<const phi::MetaTensor*> meta_{name};
  for (size_t i=0; i < static_cast<size_t>(vec_meta_{name}.size()); i++) {{
    meta_{name}.push_back(&vec_meta_{name}[i]);
  }}
 """

    CREATE_OPTIONAL_INPUT_VEC_METATENSOR_TEMPLATE = """  std::vector<paddle::dialect::IrTensor> vec_ir_tensor_{name};
  if ({name}_.impl() != nullptr) {{
    pir::VectorType {name} = {name}_.type().dyn_cast<pir::VectorType>();
    for (size_t i=0; i < static_cast<size_t>({name}.size()); i++) {{
        vec_ir_tensor_{name}.push_back(paddle::dialect::IrTensor(paddle::dialect::TransToPhiDataType({name}[i].dyn_cast<paddle::dialect::DenseTensorType>().dtype()),
                                                                        {name}[i].dyn_cast<paddle::dialect::DenseTensorType>().dims(),
                                                                        {name}[i].dyn_cast<paddle::dialect::DenseTensorType>().data_layout(),
                                                                        {name}[i].dyn_cast<paddle::dialect::DenseTensorType>().lod(),
                                                                        {name}[i].dyn_cast<paddle::dialect::DenseTensorType>().offset()));
    }}
  }}

  std::vector<paddle::dialect::IrMetaTensor> vec_meta_{name};
  for (size_t i=0; i < vec_ir_tensor_{name}.size(); i++) {{
    vec_meta_{name}.push_back(paddle::dialect::IrMetaTensor(&vec_ir_tensor_{name}[i]));
  }}

  std::vector<const phi::MetaTensor*> meta_{name};
  for (size_t i=0; i < static_cast<size_t>(vec_meta_{name}.size()); i++) {{
    meta_{name}.push_back(&vec_meta_{name}[i]);
  }}

"""

    CREATE_INTARRAY_MUTABLE_ATTRIBUTE_WITH_UNKNOWN_DATA_TEMPLATE = """  phi::IntArray {name};
  if ({name}_.isa<pir::OpResult>() && {name}_.defining_op()->isa<paddle::dialect::FullIntArrayOp>()) {{
    {name} = phi::IntArray(paddle::dialect::GetInt64Vector(
                          {name}_.defining_op()
                          ->dyn_cast<paddle::dialect::FullIntArrayOp>()
                          .attribute("value")));
  }} else if ({name}_.type().isa<pir::VectorType>()) {{
    size_t {name}_size = {name}_.type().dyn_cast<pir::VectorType>().size();
    {name} = phi::IntArray(std::vector<int64_t>({name}_size, -1));
    {name}.SetFromTensor(true);
  }} else if ({name}_.type().isa<paddle::dialect::DenseTensorType>()) {{
    common::DDim {name}_dim = {name}_.type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
    size_t {name}_size = common::product({name}_dim);
    if (common::contain_unknown_dim({name}_dim)) {{
      {name}_size = 1;
    }}
    {name} = phi::IntArray(std::vector<int64_t>({name}_size, -1));
    {name}.SetFromTensor(true);
  }} else {{
    PADDLE_THROW(phi::errors::Unimplemented("Only support VectorType or DenseTensorType"));
  }}\n"""

    CREATE_VECTOR_INT_MUTABLE_ATTRIBUTE_WITH_UNKNOWN_DATA_TEMPLATE = """  std::vector<int64_t> {name};
  if ({name}_.isa<pir::OpResult>() && {name}_.defining_op()->isa<paddle::dialect::FullIntArrayOp>()) {{
    {name} = paddle::dialect::GetInt64Vector(
                    {name}_.defining_op()
                    ->dyn_cast<paddle::dialect::FullIntArrayOp>()
                    .attribute("value"));
  }} else if ({name}_.type().isa<pir::VectorType>()) {{
    size_t {name}_size = {name}_.type().dyn_cast<pir::VectorType>().size();
    {name} = std::vector<int64_t>({name}_size, -1);
  }} else if ({name}_.type().isa<paddle::dialect::DenseTensorType>()) {{
    common::DDim {name}_dim = {name}_.type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
    size_t {name}_size = common::product({name}_dim);
    if (common::contain_unknown_dim({name}_dim)) {{
      {name}_size = 1;
    }}
    {name} = std::vector<int64_t>({name}_size, -1);
  }} else {{
    PADDLE_THROW(phi::errors::Unimplemented("Only support VectorType or DenseTensorType"));
  }}\n"""

    CREATE_SCALAR_MUTABLE_ATTRIBUTE_WITH_UNKNOWN_DATA_TEMPLATE = """  phi::Scalar {name};
  if ({name}_.isa<pir::OpResult>() && {name}_.defining_op()->isa<paddle::dialect::FullOp>()) {{
    {name} = phi::Scalar({name}_.defining_op()
                                  ->dyn_cast<paddle::dialect::FullOp>()
                                  .attribute("value")
                                  .dyn_cast<paddle::dialect::ScalarAttribute>()
                                  .data()
                                  .to<int>());
  }}
  else {{
    {name} = phi::Scalar(-1);
    {name}.SetFromTensor(true);
  }}\n"""

    CREATE_OUTPUT_METATENSOR_TEMPLATE = """  paddle::dialect::IrTensor dense_{name};
  paddle::dialect::IrMetaTensor meta_{name}(&dense_{name});
"""
    CREATE_OUTPUT_METASELETEROWS_TEMPLATE = """  paddle::dialect::IrSelectedRows dense_{name};
  paddle::dialect::IrMetaTensor meta_{name}(&dense_{name});
"""
    CREATE_OUTPUT_VEC_METATENSOR_TEMPLATE = """  std::vector<paddle::dialect::IrTensor> vec_dense_{name}(({output_size}), paddle::dialect::IrTensor());
  std::vector<paddle::dialect::IrMetaTensor> vec_meta_{name};
  for (size_t i=0; i < static_cast<size_t>({output_size}); i++) {{
    vec_meta_{name}.push_back(paddle::dialect::IrMetaTensor(&vec_dense_{name}[i]));
  }}
  std::vector<phi::MetaTensor*> meta_{name};
  for (size_t i=0; i < static_cast<size_t>(vec_meta_{name}.size()); i++) {{
    meta_{name}.push_back(&vec_meta_{name}[i]);
  }}
"""
    # Prepar input type
    for idx in range(len(op_input_name_list)):
        # is a vector<Tensor>
        if 'pir::VectorType' in op_input_type_list[idx]:
            if op_input_optional_list[idx] == 'false':
                build_output_str += f"  pir::VectorType {op_input_name_list[idx]} = {op_input_name_list[idx]}_.type().dyn_cast<pir::VectorType>(); (void){op_input_name_list[idx]};\n"
        # is a Tensor
        else:
            if op_input_optional_list[idx] == 'false':
                build_output_str += f"  {op_input_type_list[idx]} {op_input_name_list[idx]} = {op_input_name_list[idx]}_.type().dyn_cast<{op_input_type_list[idx]}>(); (void){op_input_name_list[idx]};\n"

    # Prepare mutable attributes
    if mutable_attr_is_input:
        for idx in range(len(op_mutable_attribute_name_list)):
            attr_dtype = op_mutable_attribute_type_list[idx]
            # int_array
            if attr_dtype[0] == "paddle::dialect::IntArrayAttribute":
                if (
                    op_class_name
                    in _PREPARE_DATA_WITH_VECTOR_INT64_MTTABLE_ATTRIBUTE
                ):
                    build_output_str += CREATE_VECTOR_INT_MUTABLE_ATTRIBUTE_WITH_UNKNOWN_DATA_TEMPLATE.format(
                        name=op_mutable_attribute_name_list[idx]
                    )
                else:
                    build_output_str += CREATE_INTARRAY_MUTABLE_ATTRIBUTE_WITH_UNKNOWN_DATA_TEMPLATE.format(
                        name=op_mutable_attribute_name_list[idx]
                    )
            # scalar
            elif attr_dtype[0] == "paddle::dialect::ScalarAttribute":
                build_output_str += CREATE_SCALAR_MUTABLE_ATTRIBUTE_WITH_UNKNOWN_DATA_TEMPLATE.format(
                    name=op_mutable_attribute_name_list[idx],
                    dtype=attr_dtype[1],
                )
            # string
            elif attr_dtype[0] == "pir::StrAttribute":
                build_output_str += ""
            else:
                assert "mutable attribute type is not right."
        build_output_str += "\n"

    # Prepare inputs_meta_tensor & attributes for infer meta
    infer_meta_args = []
    for idx in range(len(op_infer_meta_map['param'])):
        # is input
        if op_infer_meta_map['param'][idx] in op_input_name_list:
            if (
                "meta_" + op_infer_meta_map['param'][idx]
            ) not in infer_meta_args:
                # is a vector<Tensor>
                if (
                    'pir::VectorType'
                    in op_input_type_list[
                        op_input_name_list.index(
                            op_infer_meta_map['param'][idx]
                        )
                    ]
                ):
                    input_index = op_input_name_list.index(
                        op_infer_meta_map['param'][idx]
                    )
                    if op_input_optional_list[input_index] == 'true':
                        build_output_str += CREATE_OPTIONAL_INPUT_VEC_METATENSOR_TEMPLATE.format(
                            name=op_infer_meta_map['param'][idx]
                        )
                    else:
                        build_output_str += (
                            CREATE_INPUT_VEC_METATENSOR_TEMPLATE.format(
                                name=op_infer_meta_map['param'][idx]
                            )
                        )
                # is a Tensor
                else:
                    input_index = op_input_name_list.index(
                        op_infer_meta_map['param'][idx]
                    )
                    if op_input_optional_list[input_index] == 'true':
                        build_output_str += (
                            CREATE_OPTIONAL_INPUT_METATENSOR_TEMPLATE.format(
                                name=op_infer_meta_map['param'][idx],
                                type=op_input_type_list[idx],
                            )
                        )
                    else:
                        build_output_str += (
                            CREATE_INPUT_METATENSOR_TEMPLATE.format(
                                name=op_infer_meta_map['param'][idx]
                            )
                        )

            infer_meta_args.append("meta_" + op_infer_meta_map['param'][idx])
        # is attribute
        else:
            infer_meta_args.append(op_infer_meta_map['param'][idx])

    # Prepare outputs_meta_tensor for infer meta
    for idx in range(len(op_output_name_list)):
        # is a vector<Tensor>
        if 'pir::VectorType' in op_output_type_list[idx]:
            build_output_str += CREATE_OUTPUT_VEC_METATENSOR_TEMPLATE.format(
                name=op_output_name_list[idx],
                output_size=op_output_size_list[idx],
            )
            infer_meta_args.append(f"meta_{op_output_name_list[idx]}")
        # is a Tensor
        else:
            if op_output_type_list[idx] == "paddle::dialect::DenseTensorType":
                build_output_str += CREATE_OUTPUT_METATENSOR_TEMPLATE.format(
                    name=op_output_name_list[idx]
                )
                infer_meta_args.append(f"&meta_{op_output_name_list[idx]}")
            else:
                build_output_str += (
                    CREATE_OUTPUT_METASELETEROWS_TEMPLATE.format(
                        name=op_output_name_list[idx]
                    )
                )
                infer_meta_args.append(f"&meta_{op_output_name_list[idx]}")

    # Execute infer meta function
    CREATE_INFER_META_FUNC_TEMPLATE = """
  phi::{func}({args});
"""
    CREATE_INFER_META_FUNC_WITH_META_CONFIG_TEMPLATE = """
  phi::{func}({args}, phi::MetaConfig(false, false));
"""
    if op_infer_meta_map['func'] in _INFERMETA_NEED_META_CONFIG:
        build_output_str += (
            CREATE_INFER_META_FUNC_WITH_META_CONFIG_TEMPLATE.format(
                func=op_infer_meta_map['func'], args=", ".join(infer_meta_args)
            )
        )
    else:
        build_output_str += CREATE_INFER_META_FUNC_TEMPLATE.format(
            func=op_infer_meta_map['func'], args=", ".join(infer_meta_args)
        )

    # use dense_{name} or vec_dense_{name} to create Outputs type
    build_output_str += "\n  std::vector<pir::Type> argument_outputs;"

    CREATE_OUTPUT_DENSE_TENSOR_TEMPLATE = """
  pir::Type {name}_dense_tensor_type = {type}::get(pir::IrContext::Instance(), paddle::dialect::TransToIrDataType(dense_{name}.dtype()), dense_{name}.dims(), dense_{name}.layout(), dense_{name}.lod(), dense_{name}.offset());
  argument_outputs.push_back({name}_dense_tensor_type);
"""

    CREATE_OUTPUT_INPLACE_OPTIONAL_DENSE_TENSOR_TEMPLATE = """
  if ({input_name}_.impl() != nullptr) {{
    pir::Type {output_name}_dense_tensor_type = {type}::get(pir::IrContext::Instance(), paddle::dialect::TransToIrDataType(dense_{output_name}.dtype()), dense_{output_name}.dims(), dense_{output_name}.layout(), dense_{output_name}.lod(), dense_{output_name}.offset());
    argument_outputs.push_back({output_name}_dense_tensor_type);
  }} else {{
    pir::Type {output_name}_type;
    argument_outputs.push_back({output_name}_type);
  }}

"""

    CREATE_OUTPUT_VEC_DENSE_TENSOR_TEMPLATE = """
  std::vector<pir::Type> {name}_types;
  for (size_t i=0; i < static_cast<size_t>({output_size}); i++) {{
    {name}_types.push_back(paddle::dialect::DenseTensorType::get(pir::IrContext::Instance(), paddle::dialect::TransToIrDataType(vec_dense_{name}[i].dtype()), vec_dense_{name}[i].dims(), vec_dense_{name}[i].layout(), vec_dense_{name}[i].lod(), vec_dense_{name}[i].offset()));
  }}
  pir::Type {name}_vector_type = pir::VectorType::get(pir::IrContext::Instance(), {name}_types);
  argument_outputs.push_back({name}_vector_type);
"""
    for idx in range(len(op_output_name_list)):
        # is a vector<Tensor>
        if 'pir::VectorType' in op_output_type_list[idx]:
            build_output_str += CREATE_OUTPUT_VEC_DENSE_TENSOR_TEMPLATE.format(
                name=op_output_name_list[idx],
                output_size=op_output_size_list[idx],
            )
        # is a Tensor
        else:
            output_name = op_output_name_list[idx]
            has_input_inplace = (
                op_inplace_map is not None
                and output_name in op_inplace_map.keys()
            )
            if op_output_optional_list[idx] == 'true' and has_input_inplace:
                # is a inplace optional output
                build_output_str += (
                    CREATE_OUTPUT_INPLACE_OPTIONAL_DENSE_TENSOR_TEMPLATE.format(
                        input_name=op_inplace_map[output_name],
                        output_name=output_name,
                        type=op_output_type_list[idx],
                    )
                )
            else:
                build_output_str += CREATE_OUTPUT_DENSE_TENSOR_TEMPLATE.format(
                    type=op_output_type_list[idx], name=output_name
                )

    build_output_str += "  argument.AddAttributes(argument_attributes);\n"
    build_output_str += "  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());\n"
    # NOTE(Aurelius84): PassStopGradients must be placed after argument.AddOutputs.
    build_output_str += "  ::pir::PassStopGradientsDefaultly(argument);\n"

    return build_output_str


def gen_build_func_str(
    args,
    op_info,
    op_attribute_name_list,
    op_attribute_type_list,
    op_attribute_build_arg_type_list,
    op_attribute_default_value_list,
    op_mutable_attribute_name_list,
    op_mutable_attribute_type_list,
    op_non_mutable_attribute_name_list,
    op_non_mutable_attribute_type_list,
    op_non_mutable_attribute_build_arg_type_list,
    op_non_mutable_attribute_default_value_list,
    muta_attr_is_input=False,
    attr_args_is_map=False,
):
    op_input_name_list = op_info.input_name_list
    build_args_for_declare = ""
    build_func = ""
    build_info_str = OP_INFO_TEMPLATE.format(op_name=op_info.class_name)

    build_args_for_declare = GenBuildInputArgsStr(
        op_input_name_list,
        op_attribute_name_list,
        op_attribute_build_arg_type_list,
        op_attribute_default_value_list,
        op_mutable_attribute_name_list,
        op_non_mutable_attribute_name_list,
        op_non_mutable_attribute_build_arg_type_list,
        op_non_mutable_attribute_default_value_list,
        True,
        muta_attr_is_input,
        attr_args_is_map,
    )

    build_args_for_define = GenBuildInputArgsStr(
        op_input_name_list,
        op_attribute_name_list,
        op_attribute_build_arg_type_list,
        op_attribute_default_value_list,
        op_mutable_attribute_name_list,
        op_non_mutable_attribute_name_list,
        op_non_mutable_attribute_build_arg_type_list,
        op_non_mutable_attribute_default_value_list,
        False,
        muta_attr_is_input,
        attr_args_is_map,
    )
    inset_full_for_mutable_attributes_str = ""
    if not muta_attr_is_input:
        inset_full_for_mutable_attributes_str = (
            GenBuildInsertFullForMutableAttribute(
                args,
                op_info,
                op_attribute_name_list,
                op_attribute_build_arg_type_list,
                op_mutable_attribute_name_list,
                op_mutable_attribute_type_list,
            )
        )

    build_inputs_str = GenBuildInputs(
        op_input_name_list, op_mutable_attribute_name_list
    )
    build_attributes_str = GenBuildAttributes(
        op_non_mutable_attribute_name_list,
        op_non_mutable_attribute_type_list,
    )
    str_is_forward_only = "true" if op_info.is_forward_only else "false"
    build_outputs_str = f"""
  std::vector<pir::Type> argument_outputs = {op_info.class_name}::InferMeta(argument_inputs, &argument_attributes);
  argument.AddAttributes(argument_attributes);
  argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
  ::pir::PassStopGradientsDefaultly(argument, {str_is_forward_only});"""

    GET_ATTRIBUTES_FROM_MAP_TEMPLATE = """
  PADDLE_ENFORCE_NE(
      attributes.find("{attribute_name}"),
      attributes.end(),
      phi::errors::InvalidArgument(
          "'{attribute_name}' Attribute is expected for {op_name}. "));
  {attr_type} {attribute_name} = attributes.at("{attribute_name}").dyn_cast<{attr_ir_type}>().data();
"""
    GET_STR_ATTRIBUTES_FROM_MAP_TEMPLATE = """
  PADDLE_ENFORCE_NE(
      attributes.find("{attribute_name}"),
      attributes.end(),
      phi::errors::InvalidArgument(
          "'{attribute_name}' Attribute is expected for {op_name}. "));
  {attr_type} {attribute_name} = attributes.at("{attribute_name}").dyn_cast<pir::StrAttribute>().AsString();
"""
    GET_ARRAY_ATTRIBUTE_FROM_MAP_TEMPLATE = """
  PADDLE_ENFORCE_NE(
      attributes.find("{attribute_name}"),
      attributes.end(),
      phi::errors::InvalidArgument(
          "'{attribute_name}' Attribute is expected for {op_name}. "));
  {attr_type} {attribute_name};
  for (size_t i = 0; i < attributes.at("{attribute_name}").dyn_cast<pir::ArrayAttribute>().size(); i++) {{
    {attribute_name}.push_back(attributes.at("{attribute_name}").dyn_cast<pir::ArrayAttribute>().at(i).dyn_cast<{inner_type}>().{data_name}());
  }}
"""
    GET_INTARRAY_ATTRIBUTE_FROM_MAP_TEMPLATE = """
  PADDLE_ENFORCE_NE(
      attributes.find("{attribute_name}"),
      attributes.end(),
      phi::errors::InvalidArgument(
          "'{attribute_name}' Attribute is expected for {op_name}. "));
  {attr_type} {attribute_name} = attributes.at("{attribute_name}").dyn_cast<paddle::dialect::IntArrayAttribute>().data().GetData();
"""
    GET_SCALAR_ATTRIBUTE_FROM_MAP_TEMPLATE = """
  PADDLE_ENFORCE_NE(
      attributes.find("{attribute_name}"),
      attributes.end(),
      phi::errors::InvalidArgument(
          "'{attribute_name}' Attribute is expected for {op_name}. "));
  {attr_type} {attribute_name} = attributes.at("{attribute_name}").dyn_cast<paddle::dialect::ScalarAttribute>().data().to<{attr_type}>();
"""

    get_attributes_str = ""
    array_attr_str = "pir::ArrayAttribute"

    attr_names = []
    attr_types = []
    attr_build_arg_types = []
    if not muta_attr_is_input:
        attr_names = op_attribute_name_list
        attr_types = op_attribute_type_list
        attr_build_arg_types = op_attribute_build_arg_type_list
    else:
        attr_names = op_non_mutable_attribute_name_list
        attr_types = op_non_mutable_attribute_type_list
        attr_build_arg_types = op_non_mutable_attribute_build_arg_type_list
    if attr_args_is_map:
        for idx in range(len(attr_names)):
            attr_type = attr_build_arg_types[idx]
            attr_type = attr_type.replace("const ", "")
            attr_type = attr_type.replace("&", "")
            # if attr_build_arg_types[idx] == "const std::vector<int>&":
            #     attr_type = "std::vector<int>"

            if array_attr_str in attr_types[idx]:
                inner_type = attr_types[idx][len(array_attr_str) + 1 : -1]
                data_name = "data"
                if inner_type == "pir::StrAttribute":
                    data_name = "AsString"
                get_attributes_str += (
                    GET_ARRAY_ATTRIBUTE_FROM_MAP_TEMPLATE.format(
                        op_name=op_info.class_name,
                        attr_type=attr_type,
                        attribute_name=attr_names[idx],
                        inner_type=inner_type,
                        data_name=data_name,
                    )
                )
            elif "paddle::dialect::IntArrayAttribute" in attr_types[idx]:
                get_attributes_str += (
                    GET_INTARRAY_ATTRIBUTE_FROM_MAP_TEMPLATE.format(
                        op_name=op_info.class_name,
                        attr_type=attr_type,
                        attribute_name=attr_names[idx],
                    )
                )
            elif "paddle::dialect::ScalarAttribute" in attr_types[idx]:
                get_attributes_str += (
                    GET_SCALAR_ATTRIBUTE_FROM_MAP_TEMPLATE.format(
                        op_name=op_info.class_name,
                        attr_type=attr_type,
                        attribute_name=attr_names[idx],
                    )
                )
            elif "pir::StrAttribute" in attr_types[idx]:
                get_attributes_str += (
                    GET_STR_ATTRIBUTES_FROM_MAP_TEMPLATE.format(
                        op_name=op_info.class_name,
                        attr_type=attr_type,
                        attribute_name=attr_names[idx],
                        attr_ir_type=attr_types[idx],
                    )
                )
            else:
                get_attributes_str += GET_ATTRIBUTES_FROM_MAP_TEMPLATE.format(
                    op_name=op_info.class_name,
                    attr_type=attr_type,
                    attribute_name=attr_names[idx],
                    attr_ir_type=attr_types[idx],
                )

    build_func = OP_BUILD_TEMPLATE.format(
        op_name=op_info.class_name,
        build_info=build_info_str,
        build_args=build_args_for_define,
        build_mutable_attributes=inset_full_for_mutable_attributes_str,
        get_attributes=get_attributes_str,
        build_inputs=build_inputs_str,
        build_attributes=build_attributes_str,
        build_outputs=build_outputs_str,
    )

    return (build_args_for_declare, build_func)


OP_BUILD_BY_INVOKE_TEMPLATE = """
void {op_name}::Build({build_args}) {{
  {invoke_class}::Build(builder, argument{invoke_args});
}}
"""


def gen_build_func_str_by_invoke(
    op_class_name,
    op_input_name_list,
    op_input_type_list,
    op_input_optional_list,
    op_attribute_name_list,
    op_attribute_type_list,
    op_attribute_build_arg_type_list,
    op_attribute_default_value_list,
    op_mutable_attribute_name_list,
    op_mutable_attribute_type_list,
    op_non_mutable_attribute_name_list,
    op_non_mutable_attribute_type_list,
    op_non_mutable_attribute_build_arg_type_list,
    op_non_mutable_attribute_default_value_list,
    op_invoke_class_name,
    op_invoke_map,
):
    build_args_for_declare = ""
    build_func = ""

    build_args_for_declare = GenBuildInputArgsStr(
        op_input_name_list,
        op_attribute_name_list,
        op_attribute_build_arg_type_list,
        op_attribute_default_value_list,
        op_mutable_attribute_name_list,
        op_non_mutable_attribute_name_list,
        op_non_mutable_attribute_build_arg_type_list,
        op_non_mutable_attribute_default_value_list,
        True,
        False,
        False,
    )

    build_args_for_define = GenBuildInputArgsStr(
        op_input_name_list,
        op_attribute_name_list,
        op_attribute_build_arg_type_list,
        op_attribute_default_value_list,
        op_mutable_attribute_name_list,
        op_non_mutable_attribute_name_list,
        op_non_mutable_attribute_build_arg_type_list,
        op_non_mutable_attribute_default_value_list,
        False,
        False,
        False,
    )

    invoke_args = op_invoke_map['args'].split(", ")
    invoke_args_str = ""
    for item in invoke_args:
        if item in op_input_name_list:
            invoke_args_str += ", " + item + "_"
        elif ".dtype()" in item:
            invoke_args_str += (
                ", paddle::dialect::TransToPhiDataType("
                + item[:-8]
                + "_"
                + ".type().dyn_cast<paddle::dialect::DenseTensorType>().dtype())"
            )
        else:
            invoke_args_str += ", " + item

    build_func = OP_BUILD_BY_INVOKE_TEMPLATE.format(
        op_name=op_class_name,
        build_args=build_args_for_define,
        invoke_class=op_invoke_class_name,
        invoke_args=invoke_args_str,
    )

    return (build_args_for_declare, build_func)
