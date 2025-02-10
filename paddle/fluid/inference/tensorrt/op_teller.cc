// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/inference/tensorrt/op_teller.h"

#include <bitset>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/inference/tensorrt/dynamic_shape_infermeta_factory.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/phi/core/compat/op_utils.h"
#include "paddle/phi/core/kernel_factory.h"

namespace paddle::framework {
class OpDesc;
}  // namespace paddle::framework

namespace paddle::inference::tensorrt {

// Check if it is a dynamic shape. If it is a dynamic shape, return true;
// otherwise, return false
bool IsDynamicShapeOp(const framework::OpDesc& desc) {
  VLOG(3) << "forbid_dynamic_op_enter_into_trt is open";
  auto* block = desc.Block();
  auto inputs = desc.Inputs();
  for (auto iter : inputs) {
    for (auto var_name : iter.second) {
      if (block) {
        auto* var_desc = block->FindVar(var_name);
        const auto shape = var_desc->GetShape();
        for (auto ele : shape) {
          if (ele < 0) {
            return true;
          }
        }
      }
    }
  }

  auto outputs = desc.Outputs();
  for (auto iter : outputs) {
    for (auto var_name : iter.second) {
      if (block) {
        auto* var_desc = block->FindVar(var_name);
        const auto shape = var_desc->GetShape();
        for (auto ele : shape) {
          if (ele < 0) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

// Just tell by the op_types.
struct SimpleOpTypeSetTeller : public Teller {
  SimpleOpTypeSetTeller() {  // NOLINT
#if IS_TRT_VERSION_GE(7130)
    // use TensorRT plugin
    teller_set.insert("group_norm");
    teller_set.insert("multiclass_nms3");
    teller_set.insert("multiclass_nms");
    int8_teller_set.insert("multiclass_nms3");
    int8_teller_set.insert("multiclass_nms");
#endif
#if IS_TRT_VERSION_GE(7000)
    teller_set.insert("tile");
    int8_teller_set.insert("tile");
    teller_set.insert("flatten_contiguous_range");
    int8_teller_set.insert("flatten_contiguous_range");
    teller_set.insert("rnn");
    int8_teller_set.insert("rnn");
    teller_set.insert("fill_constant_batch_size_like");
    int8_teller_set.insert("fill_constant_batch_size_like");
#endif
#if CUDA_VERSION >= 10020
    teller_set.insert("reshape");
    teller_set.insert("reshape2");
    int8_teller_set.insert("reshape");
    int8_teller_set.insert("reshape2");
#endif
#if IS_TRT_VERSION_GE(8000)
    teller_set.insert("sparse_fc");
    int8_teller_set.insert("sparse_fc");
    teller_set.insert("sparse_multihead_matmul");
    int8_teller_set.insert("sparse_multihead_matmul");
#endif
#if IS_TRT_VERSION_GE(8522)
    teller_set.insert("flash_multihead_matmul");
    int8_teller_set.insert("flash_multihead_matmul");
    teller_set.insert("cross_multihead_matmul");
    int8_teller_set.insert("cross_multihead_matmul");
    teller_set.insert("qk_multihead_matmul");
    int8_teller_set.insert("qk_multihead_matmul");
#endif
#if IS_TRT_VERSION_GE(8200)
    teller_set.insert("round");
    int8_teller_set.insert("round");
    teller_set.insert("set_value");
    teller_set.insert("index_select");
    int8_teller_set.insert("index_select");
    int8_teller_set.insert("einsum");
    teller_set.insert("einsum");
#endif
  }

  bool operator()(const framework::OpDesc& desc,
                  bool use_no_calib_int8 = false,
                  bool with_dynamic_shape = false,
                  bool forbid_dynamic_op_enter_into_trt = false,
                  bool use_explicit_quantization = false) override {
    const std::string op_type = desc.Type();

    std::unordered_set<std::string> control_set = {"conditional_block",
                                                   "while"};
    std::unordered_set<std::string> feed_fetch_set = {"feed", "fetch"};
    if (control_set.find(op_type) != control_set.end()) {
      return false;
    }

    if (feed_fetch_set.find(op_type) != feed_fetch_set.end()) {
      return false;
    }
    if (forbid_dynamic_op_enter_into_trt && IsDynamicShapeOp(desc)) {
      return false;
    }

    // do not support the op which is labeled the `skip_quant`
    if ((desc.HasAttr("namescope") &&
         PADDLE_GET_CONST(std::string, desc.GetAttr("op_namescope")) ==
             "/skip_quant_2/") ||
        desc.HasAttr("skip_quant"))
      return false;
    std::unordered_set<std::string> act_op_list = {
        "relu",       "relu6",       "sigmoid",
        "elu",        "selu",        "softsign",
        "softplus",   "stanh",       "thresholded_relu",
        "exp",        "log",         "sqrt",
        "abs",        "sin",         "cos",
        "tan",        "tanh",        "sinh",
        "cosh",       "asin",        "acos",
        "atan",       "asinh",       "acosh",
        "atanh",      "ceil",        "celu",
        "erf",        "floor",       "round",
        "sign",       "silu",        "logical_not",
        "reciprocal", "tanh_shrink", "logsigmoid",
        "rsqrt",      "swish",       "hard_sigmoid",
        "hard_swish", "leaky_relu"};
    std::unordered_set<std::string> unary_list = {
        "exp",   "log",         "sqrt",       "abs",         "sin",
        "cos",   "tan",         "tanh",       "sinh",        "cosh",
        "asin",  "acos",        "atan",       "asinh",       "acosh",
        "atanh", "ceil",        "celu",       "floor",       "round",
        "sign",  "logical_not", "reciprocal", "tanh_shrink", "logsigmoid",
        "erf",   "bitwise_not", "equal",      "not_equal",   "rsqrt"};

    // Static shape does not support 0 or 1 dim's input.
    if (!with_dynamic_shape) {
      auto inputs = desc.Inputs();
      for (auto iter : inputs) {
        for (auto var_name : iter.second) {
          auto* block = desc.Block();
          if (block) {
            auto* var_desc = block->FindVarRecursive(var_name);
            // Can't get feed op's TensorDesc
            if (op_type != "feed" && var_desc && !var_desc->Persistable()) {
              const auto shape = var_desc->GetShape();
              if (shape.size() == 1 || shape.empty()) return false;
            }
          }
        }
      }
    }

    if (act_op_list.find(op_type) != act_op_list.end()) {
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
#if !IS_TRT_VERSION_GE(7000)
      if (op_type == "erf") {
        VLOG(3) << op_type << " op does not support tensorrt.";
        return false;
      }
#endif
#if !IS_TRT_VERSION_GE(8600)
      auto x_var_name = desc.Input("X")[0];
      auto* x_var_desc = block->FindVarRecursive(x_var_name);
      const auto x_shape = x_var_desc->GetShape();
      if (x_shape.empty() && unary_list.find(op_type) != unary_list.end()) {
        VLOG(3) << op_type
                << " op does not support 0 dim input when TensorRT < 8.6.";
        return false;
      }
#endif
    }

    if (op_type == "dropout") {
      /*
       * Some OpDescs Attribute support both constant value and dynamic
       * runtime value (which is a Variable(s) type). But TensorRT maybe
       * only support constant value Attribute, so we shall distinguish
       * this case in time and return False in OpTeller.Tell().
       * If Attribute is Variable(s), HasAttr() will return False
       */
      if (!desc.HasAttr("dropout_prob", /*with_attr_var=*/false)) {
        VLOG(3)
            << "Skip to convert into TRT while found Attribute('dropout_prob') "
               "is Variable type in dropout.";
        return false;
      }
    }

    if (op_type == "pool2d") {
      // If Attribute is Variable(s), HasAttr() will return False
      if (!desc.HasAttr("ksize", /*with_attr_var=*/false)) {
        VLOG(3) << "Skip to convert into TRT while found Attribute('ksize') is "
                   "Variable type in pool2d.";
        return false;
      }

      std::vector<int> paddings =
          PADDLE_GET_CONST(std::vector<int>, desc.GetAttr("paddings"));
      if (paddings.size() > 2) {
        return false;
      }
      if (desc.Input("X").size() != 1) {
        VLOG(3) << "TRT Pool2d expect 1 input, but got "
                << desc.Input("X").size();
        return false;
      }
      if (desc.Output("Out").size() != 1) {
        VLOG(3) << "TRT Pool2d has only 1 output, but got "
                << desc.Output("Out").size();
        return false;
      }
      if (desc.HasAttr("data_format")) {
        std::string data_format =
            PADDLE_GET_CONST(std::string, desc.GetAttr("data_format"));
        if (data_format == "NHWC" || data_format == "NDHWC") {
          return false;
        }
      }
      if (!desc.HasAttr("pooling_type")) {
        return false;
      } else {
        std::string pool_type =
            PADDLE_GET_CONST(std::string, desc.GetAttr("pooling_type"));
        if (pool_type != "max" && pool_type != "avg") {
          VLOG(3) << "Wrong pool op type, the trt do not support the "
                  << pool_type << " pool type.";
          return false;
        }
        if (pool_type == "avg") {
          if (desc.HasAttr("global_pooling")) {
            if (!PADDLE_GET_CONST(bool, desc.GetAttr("global_pooling"))) {
              if (desc.HasAttr("exclusive")) {
                if (PADDLE_GET_CONST(bool, desc.GetAttr("exclusive"))) {
                  std::vector<int> ksize =
                      PADDLE_GET_CONST(std::vector<int>, desc.GetAttr("ksize"));
                  for (size_t i = 0; i < ksize.size(); i++) {
                    if (ksize[i] <= paddings[i]) {
                      VLOG(3) << "the padding size should be less than the "
                                 "filter size "
                                 "for exclusive-counting pooling.";
                      return false;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    if (op_type == "conv2d" || op_type == "conv2d_transpose" ||
        op_type == "fused_conv2d_add_act" || op_type == "depthwise_conv2d" ||
        op_type == "depthwise_conv2d_transpose") {
      if (desc.Input("Input").size() != 1) {
        VLOG(3) << "TRT Conv2d expect 1 input, but got "
                << desc.Input("Input").size() << " input.";
        return false;
      }

      if (desc.Input("Filter").size() != 1) {
        VLOG(3) << "TRT Conv2d expect 1 filter, but got "
                << desc.Input("Filter").size() << " filter.";
        return false;
      }

      if (desc.HasAttr("enable_int8")) {
        if (op_type == "conv2d" || op_type == "fused_conv2d_add_act") {
          if (!desc.HasAttr("Input_scale")) {
            VLOG(3) << "Input scale not found. TRT int8"
                       " requires conv/deconv to have "
                       "input quantization scales.";
            return false;
          }
        }
      }

      if (op_type == "conv2d_transpose" ||
          op_type == "depthwise_conv2d_transpose") {
        if (!desc.HasAttr("dilations")) {
          return false;
        } else {
          const std::vector<int> dilations =
              PADDLE_GET_CONST(std::vector<int>, desc.GetAttr("dilations"));
          if (dilations[0] != 1 || dilations[1] != 1) {
            VLOG(3) << "In conv2d_transpose, Dilations must be (1, 1) for "
                       "tensorRT, but given ("
                    << dilations[0] << ", " << dilations[1] << ")";
            return false;
          }
        }
      }

      if (desc.Output("Output").size() != 1) {
        VLOG(3) << "TRT Conv2d expect 1 output, but got "
                << desc.Output("Output").size() << " output.";
        return false;
      }

// strides > 1 and 'SAME' is only supported by trt7.0 above
#if !IS_TRT_VERSION_GE(7000)
      if (op_type == "conv2d" || op_type == "fused_conv2d_add_act" ||
          op_type == "depthwise_conv2d") {
        if (desc.HasAttr("padding_algorithm") && with_dynamic_shape) {
          auto padding_algorithm =
              PADDLE_GET_CONST(std::string, desc.GetAttr("padding_algorithm"));
          if (padding_algorithm == "SAME" && desc.HasAttr("strides")) {
            const std::vector<int> strides =
                PADDLE_GET_CONST(std::vector<int>, desc.GetAttr("strides"));
            // there is no issue if strides.size() less than 2
            if (strides.size() > 1) {
              for (size_t i = 0; i < strides.size(); i++) {
                if (strides[i] > 1) return false;
              }
            }
          }
        }
      }
#endif
      auto* block = desc.Block();
      if (block) {
        auto* filter_var_desc =
            block->FindVarRecursive(desc.Input("Filter")[0]);
        if (!filter_var_desc->Persistable()) {
#if IS_TRT_VERSION_GE(8600)
#else
          LOG(INFO)
              << "Trt below 8.6 not support conv2d's filter is a intermediate "
                 "tensor in conv2d op, please upgrade your TensorRT.";
          return false;
#endif
        }
      }
    }

    if (op_type == "deformable_conv") {
      if (!desc.HasAttr("groups") || !desc.HasAttr("strides") ||
          !desc.HasAttr("paddings"))
        return false;
      auto* block = desc.Block();
      auto input_name = desc.Input("Input")[0];
      auto* input_desc = block->FindVarRecursive(input_name);
      const auto input_shape = input_desc->GetShape();

      if (input_shape.size() != 4) {
        VLOG(3) << "Input of deformable conv should be 4-D Tensor, but got "
                << input_shape.size();
        return false;
      }

      auto filter_name = desc.Input("Filter")[0];
      auto* filter_desc = block->FindVarRecursive(filter_name);
      const auto filter_shape = filter_desc->GetShape();

      int groups = PADDLE_GET_CONST(int, desc.GetAttr("groups"));
      if (input_shape[1] != filter_shape[1] * groups) {
        VLOG(3) << "The number of input channels should be equal to filter "
                << "channels * groups. But got input channels "
                << input_shape[1] << "filter channels " << filter_shape[1];
        return false;
      }

      const std::vector<int> strides =
          PADDLE_GET_CONST(std::vector<int>, desc.GetAttr("strides"));
      if (strides.size() != 2) {
        VLOG(3) << "The size of strides should be 2, but got "
                << strides.size();
        return false;
      }

      const std::vector<int> paddings =
          PADDLE_GET_CONST(std::vector<int>, desc.GetAttr("paddings"));
      if (paddings.size() != 2) {
        VLOG(3) << "The size of paddings should be 2, but got "
                << paddings.size();
        return false;
      }
    }

    if (op_type == "bmm") {
      if (!with_dynamic_shape) {
        return false;
      }
    }

    if (op_type == "range") {
      if (!with_dynamic_shape) {
        return false;
      }
#if IS_TRT_VERSION_LT(8400)
      auto* block = desc.Block();
      auto start_var_name = desc.Input("Start")[0];
      auto* start_var_desc = block->FindVarRecursive(start_var_name);
      auto start_dtype = start_var_desc->GetDataType();
      if (start_dtype == framework::proto::VarType::FP32 ||
          start_dtype == framework::proto::VarType::FP64) {
        return false;
      }
#endif
    }

    if (op_type == "sign") {
#if IS_TRT_VERSION_GE(8200)
      if (!with_dynamic_shape) {
        return false;
      }
#else
      VLOG(3) << "sign op is only supported by trt8.2 above ";
      return false;
#endif
    }

    if (op_type == "logical_not") {
#if IS_TRT_VERSION_GE(8400)
      if (!with_dynamic_shape) {
        return false;
      }
#else
      VLOG(3) << "logical_not op is only supported by trt8.4 above because of "
                 "cast op";
      return false;
#endif
    }

    if (op_type == "softmax") {
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto x_var_name = desc.Input("X")[0];
      auto* x_var_desc = block->FindVarRecursive(x_var_name);
      const auto x_shape = x_var_desc->GetShape();

      if (with_dynamic_shape && (x_shape.size() == 1 || x_shape.empty())) {
        int axis = desc.HasAttr("axis")
                       ? PADDLE_GET_CONST(int, desc.GetAttr("axis"))
                       : -1;
        if (axis > 0) {
          return false;
        }
      }
    }

    if (op_type == "group_norm") {
      if (!desc.HasAttr("epsilon") || !desc.HasAttr("groups") ||
          !desc.HasAttr("data_layout"))
        return false;

      auto registry = GetPluginRegistry();
      if (registry == nullptr) return false;
      std::string layout_str =
          PADDLE_GET_CONST(std::string, desc.GetAttr("data_layout"));
      if (layout_str != "NCHW") {
        VLOG(3) << "Group norm trt plugin only support NCHW layout, but got "
                << layout_str;
        return false;
      }
    }
    if (op_type == "concat") {
      if (!desc.HasAttr("axis")) {
        return false;
      }
      int axis = PADDLE_GET_CONST(int, desc.GetAttr("axis"));
      if (!with_dynamic_shape) {
        if (axis == 0) return false;
      }
      auto concat_inputs = desc.Inputs();
      if (concat_inputs.find("AxisTensor") != concat_inputs.end()) {
        if (!desc.Input("AxisTensor").empty()) {
          return false;
        }
      }
    }
    if (op_type == "transpose2" || op_type == "transpose") {
      if (!desc.HasAttr("axis")) {
        return false;
      }
      std::vector<int> axis =
          PADDLE_GET_CONST(std::vector<int>, desc.GetAttr("axis"));
      if (!with_dynamic_shape && axis[0] != 0) return false;
      if (axis.size() >= nvinfer1::Dims::MAX_DIMS) return false;

      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto x_var_name = desc.Input("X")[0];
      auto* x_var_desc = block->FindVarRecursive(x_var_name);
      const auto x_shape = x_var_desc->GetShape();
      if (axis.size() != x_shape.size()) return false;
      int dims = x_shape.size();

      std::vector<int> perm(nvinfer1::Dims::MAX_DIMS);
      for (int i = 0; i < dims; i++) {
        perm[i] = axis[i];
      }
      auto is_valid_permutation = [&](int dims,
                                      const std::vector<int>& permutation) {
        std::bitset<nvinfer1::Dims::MAX_DIMS> found;
        for (int i = 0; i < dims; ++i) {
          const int x = permutation[i];
          if ((x < 0) || (x >= dims) || found[x])
            return false;  // Out of bounds or duplicate
          found.set(x);
        }
        return true;
      };
      if (!is_valid_permutation(dims, perm)) {
        VLOG(3) << "Invalid permutation dimensions for trt transpose op "
                   "converter: duplicate or out of bound.";
        return false;
      }
    }
    if (op_type == "flatten2" || op_type == "flatten") {
      if (!desc.HasAttr("axis")) {
        return false;
      } else {
#if IS_TRT_VERSION_GE(7130)
#else
        if (with_dynamic_shape) return false;
#endif
        int axis = PADDLE_GET_CONST(int, desc.GetAttr("axis"));
        if (axis != 1) return false;
      }
    }
    if (op_type == "flatten_contiguous_range") {
      if (!with_dynamic_shape) {
        if (!desc.HasAttr("start_axis") || !desc.HasAttr("stop_axis")) {
          return false;
        }
        int start_axis = PADDLE_GET_CONST(int, desc.GetAttr("start_axis"));
        int stop_axis = PADDLE_GET_CONST(int, desc.GetAttr("stop_axis"));
        auto x_var_name = desc.Input("X")[0];
        auto* block = desc.Block();
        if (block == nullptr) {
          VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                     "Developers need to check whether block_desc is passed in "
                     "the pass.";
          return false;
        }
        auto* x_var_desc = block->FindVarRecursive(x_var_name);
        const auto x_shape = x_var_desc->GetShape();
        int dims = x_shape.size();
        if (dims == 0) {
          VLOG(3) << op_type
                  << " op does not support input's dim is 0 in tensorrt "
                     "static shape mode.";
          return false;
        }
        if (start_axis < 0) start_axis += dims;
        if (start_axis == 0) {
          VLOG(3) << "TRT flatten_contiguous_range not support the "
                     "batch-dimension being changed";
          return false;
        }
        if (stop_axis < 0) stop_axis += dims;
        for (int i = start_axis; i <= stop_axis; ++i) {
          if (x_shape[i] < 0) {
            VLOG(3) << "On TRT static shape,flatten_contiguous_range input dim "
                       "should be > 0";
            return false;
          }
        }
      }
    }

    if (op_type == "gather") {
      auto gather_inputs = desc.Inputs();
      if (gather_inputs.find("Axis") != gather_inputs.end()) {
        if (!desc.Input("Axis").empty()) {
          return false;
        }
      }
      if (!with_dynamic_shape) {
        return false;
      } else {
        auto* block = desc.Block();
        if (block == nullptr) {
          VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                     "Developers need to check whether block_desc is passed in "
                     "the pass.";
          return false;
        }
#if !IS_TRT_VERSION_GE(7000)
        auto* x_var_desc = block->FindVarRecursive(desc.Input("X")[0]);
        const auto x_shape = x_var_desc->GetShape();
        if (x_shape.size() == 1) {
          VLOG(3) << "Gather does not support 1-dimensional input in tensorrt";
          return false;
        }
#endif
      }
    }

    if (op_type == "gather_nd") {
      if (!with_dynamic_shape) return false;

      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
#if IS_TRT_VERSION_LT(8200)
      auto index_var_name = desc.Input("Index")[0];
      auto* index_var_desc = block->FindVarRecursive(index_var_name);
      auto x_var_name = desc.Input("X")[0];
      auto* x_var_desc = block->FindVarRecursive(x_var_name);
      const auto index_shape = index_var_desc->GetShape();
      const auto x_shape = x_var_desc->GetShape();
      if (x_shape.size() <= 2) {
        VLOG(3) << "gather_nd op requires the input's dimension to be greater "
                   "than 2";
        return false;
      }

      if (x_shape.size() != index_shape.size()) {
        VLOG(3) << "gather_nd op Index input dims size [" << index_shape.size()
                << " ] not equal to x dims size [" << x_shape.size() << "]";
        return false;
      }
#endif
    }
    if (op_type == "index_select") {
#if !IS_TRT_VERSION_GE(8200)
      return false;
#endif
      auto gather_inputs = desc.Inputs();
      if (!with_dynamic_shape) {
        return false;
      } else {
        auto* block = desc.Block();
        if (block == nullptr) {
          VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                     "Developers need to check whether block_desc is passed in "
                     "the pass.";
          return false;
        }

        auto index_var_name = desc.Input("Index")[0];
        auto* index_var_desc = block->FindVarRecursive(index_var_name);

        // The index input must be int32 or int64 datatype.
        if (index_var_desc->GetDataType() !=
                paddle::framework::proto::VarType_Type::VarType_Type_INT32 &&
            index_var_desc->GetDataType() !=
                paddle::framework::proto::VarType_Type::VarType_Type_INT64) {
          VLOG(3)
              << "Index select op Index input data type must be int32 or int64";
          return false;
        }
      }
    }
    if (op_type == "take_along_axis") {
#if IS_TRT_VERSION_GE(8200)
      if (!with_dynamic_shape) return false;
      auto* block = desc.Block();
      auto input_var_name = desc.Input("Input")[0];
      auto index_var_name = desc.Input("Index")[0];
      auto* input_var_desc = block->FindVarRecursive(input_var_name);
      auto* index_var_desc = block->FindVarRecursive(index_var_name);

      const auto input_shape = input_var_desc->GetShape();
      const auto index_shape = index_var_desc->GetShape();
      if (input_shape.size() != index_shape.size()) {
        VLOG(3) << "take_along_axis op Index input dims size ["
                << index_shape.size() << " ] not equal to input dims size ["
                << input_shape.size() << "]";
        return false;
      }
#else
      VLOG(3) << "take_along_axis op is only supported by trt8.2 above ";
      return false;
#endif
    }

    if (op_type == "anchor_generator") {
      if (!with_dynamic_shape) return false;
    }

    if (op_type == "yolo_box") {
      return false;
    }

    if (op_type == "yolo_box_head") {
      return false;
    }

    if (op_type == "arg_max" || op_type == "arg_min") {
      if (!desc.HasAttr("axis", /*with_attr_var=*/false)) {
        VLOG(3) << "Skip to convert into TRT while found Attribute('axis') is "
                   "Variable type in arg_max.";
        return false;
      }

      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto x_var_name = desc.Input("X")[0];
      auto* x_var_desc = block->FindVarRecursive(x_var_name);
      auto x_dtype = x_var_desc->GetDataType();

      if (!(x_dtype == framework::proto::VarType::FP32 ||
            x_dtype == framework::proto::VarType::FP16 ||
            x_dtype == framework::proto::VarType::FP64)) {
        return false;
      }

      int axis = desc.HasAttr("axis")
                     ? PADDLE_GET_CONST(int64_t, desc.GetAttr("axis"))
                     : -1;
      bool flatten = desc.HasAttr("flatten")
                         ? PADDLE_GET_CONST(bool, desc.GetAttr("flatten"))
                         : false;
      int dtype = desc.HasAttr("dtype")
                      ? PADDLE_GET_CONST(int, desc.GetAttr("dtype"))
                      : 3;
      if (axis == 0 || flatten || (dtype != 2 && dtype != 3)) return false;
    }

    if (op_type == "affine_channel") {
      if (!desc.HasAttr("data_layout")) return false;
      auto data_layout = common::StringToDataLayout(
          PADDLE_GET_CONST(std::string, desc.GetAttr("data_layout")));
      if (data_layout != phi::DataLayout::kNCHW) return false;

      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto x_var_name = desc.Input("X")[0];
      auto* x_var_desc = block->FindVarRecursive(x_var_name);
      const auto x_shape = x_var_desc->GetShape();
      if (x_shape.size() == 2) {
        return false;
      }
    }

    if (op_type == "multiclass_nms" || op_type == "multiclass_nms3") {
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto multiclass_nms_inputs = desc.Inputs();
      if (multiclass_nms_inputs.find("RoisNum") !=
          multiclass_nms_inputs.end()) {
        if (!desc.Input("RoisNum").empty()) {
          return false;
        }
      }
      for (auto& param_name : multiclass_nms_inputs) {
        for (auto& var_name : param_name.second) {
          auto* var_desc = block->FindVarRecursive(var_name);
          const auto shape = var_desc->GetShape();
          if (shape.size() != 3) {
            VLOG(3) << "multiclass_nms op dims != 3 not supported in tensorrt, "
                       "but got dims "
                    << shape.size() << ", so jump it.";
            return false;
          }
        }
      }
      bool has_attrs =
          (desc.HasAttr("background_label") &&
           desc.HasAttr("score_threshold") && desc.HasAttr("nms_top_k") &&
           desc.HasAttr("keep_top_k") && desc.HasAttr("normalized"));
      if (has_attrs == false) return false;

      // TODO(wangxinxin08): tricky solution because the outputs of batchedNMS
      // plugin are not constient with those of multiclass_nms3
      if (desc.HasAttr("nms_eta") == false) return false;
      auto nms_eta = PADDLE_GET_CONST(float, desc.GetAttr("nms_eta"));
      if (nms_eta <= 1.0) return false;

      auto nms_top_k = PADDLE_GET_CONST(int, desc.GetAttr("nms_top_k"));
      if (nms_top_k < 0) return false;

      auto keep_top_k = PADDLE_GET_CONST(int, desc.GetAttr("keep_top_k"));
      if (keep_top_k < 0) return false;

      auto registry = GetPluginRegistry();
      if (registry == nullptr) return false;
    }

    if (op_type == "nearest_interp") {
      std::vector<std::string> attrs{
          "interp_method", "align_corners", "scale", "out_h", "out_w"};
      for (auto const& attr : attrs) {
        if (!desc.HasAttr(attr)) return false;
      }
      if (desc.HasAttr("data_layout")) {
        auto data_layout = common::StringToDataLayout(
            PADDLE_GET_CONST(std::string, desc.GetAttr("data_layout")));
        if (data_layout != phi::DataLayout::kNCHW &&
            data_layout != phi::DataLayout::kNHWC)
          return false;
      }
      auto interp_method =
          PADDLE_GET_CONST(std::string, desc.GetAttr("interp_method"));
      if (interp_method != "nearest") return false;
      auto scale = PADDLE_GET_CONST(float, desc.GetAttr("scale"));
      auto out_h = PADDLE_GET_CONST(int, desc.GetAttr("out_h"));
      auto out_w = PADDLE_GET_CONST(int, desc.GetAttr("out_w"));
      auto align_corners =
          PADDLE_GET_CONST(bool, desc.GetAttr("align_corners"));
      if (!(scale > 0.f && (out_h <= 0 && out_w <= 0))) {
        if (out_h <= 0) {
          VLOG(3) << "out_h must be greater than 0 if scale is not set.";
          return false;
        }
        if (out_w <= 0) {
          VLOG(3) << "out_w must be greater than 0 if scale is not set.";
          return false;
        }
      }
      if ((scale <= 0.f) && with_dynamic_shape) {
        VLOG(3) << "dynamic shape not support scale not set.";
        return false;
      }
      // When align_corners = true, the paddle's and trt_layer's results has
      // diff
      if (align_corners && scale != 1) {
        return false;
      }
    }

    if (op_type == "nearest_interp_v2") {
      std::vector<std::string> attrs{"data_layout",
                                     "interp_method",
                                     "align_corners",
                                     "scale",
                                     "out_h",
                                     "out_w"};
      for (auto const& attr : attrs) {
        if (!desc.HasAttr(attr)) return false;
      }
      auto data_layout = common::StringToDataLayout(
          PADDLE_GET_CONST(std::string, desc.GetAttr("data_layout")));
      if (data_layout != phi::DataLayout::kNCHW &&
          data_layout != phi::DataLayout::kNHWC)
        return false;
      auto interp_method =
          PADDLE_GET_CONST(std::string, desc.GetAttr("interp_method"));
      if (interp_method != "nearest") return false;

#if IS_TRT_VERSION_GE(8200)
      auto resize_inputs = desc.Inputs();
      if (with_dynamic_shape &&
          resize_inputs.find("SizeTensor") != resize_inputs.end() &&
          desc.Input("SizeTensor").size() == 2) {
        return true;
      }
#endif

      auto scale = PADDLE_GET_CONST(std::vector<float>, desc.GetAttr("scale"));
      auto out_h = PADDLE_GET_CONST(int, desc.GetAttr("out_h"));
      auto out_w = PADDLE_GET_CONST(int, desc.GetAttr("out_w"));
      if (!(out_h > 0 && out_w > 0)) {
        if (scale.size() < 2) return false;
        if (scale[0] <= 0.f || scale[1] <= 0.f) {
          VLOG(3) << "scale factor must be greater than 0 if out_h or out_w is "
                     "not set.";
          return false;
        }
      }
    }

    if (op_type == "bilinear_interp_v2") {
      // trt 7011 result in test_solov2_trt_fp32.py TRT fp32 diff
#if IS_TRT_VERSION_LT(7100)
      return false;
#endif
      std::vector<std::string> attrs{"data_layout",
                                     "interp_method",
                                     "align_corners",
                                     "scale",
                                     "out_h",
                                     "out_w"};
      for (auto const& attr : attrs) {
        if (!desc.HasAttr(attr)) {
          VLOG(3) << "The op_type " << op_type << " doesn't have the attr "
                  << attr << " and return false";
          return false;
        }
      }

      auto resize_inputs = desc.Inputs();
      if (resize_inputs.find("SizeTensor") != resize_inputs.end()) {
#if IS_TRT_VERSION_GE(8200)
        if (desc.Input("SizeTensor").size() == 2) {
          return true;
        }
#else
        if (!desc.Input("SizeTensor").empty()) {
          VLOG(3)
              << "The Paddle-TRT doesn't support the SizeTensor for op_type "
              << op_type;
          return false;
        }
#endif
      }
      if (resize_inputs.find("OutSize") != resize_inputs.end()) {
        if (!with_dynamic_shape) {
          VLOG(3) << "Static shape don't support the OutSize for op_type "
                  << op_type;
          return false;
        }
      }

      auto data_layout = common::StringToDataLayout(
          PADDLE_GET_CONST(std::string, desc.GetAttr("data_layout")));
      if (data_layout != phi::DataLayout::kNCHW &&
          data_layout != phi::DataLayout::kNHWC) {
        VLOG(3) << "The op_type " << op_type
                << " is not NCHW or NHWC return false";
        return false;
      }
      auto interp_method =
          PADDLE_GET_CONST(std::string, desc.GetAttr("interp_method"));
      if (interp_method != "bilinear") {
        VLOG(3) << "The interp_method of op_type " << op_type
                << " is not bilinear";
        return false;
      }

      bool has_scale_input_size =
          (resize_inputs.find("Scale") != resize_inputs.end());

      if (!has_scale_input_size ||
          (has_scale_input_size && desc.Input("Scale").size() != 1)) {
        const std::vector<float> scale =
            PADDLE_GET_CONST(std::vector<float>, desc.GetAttr("scale"));
        if (scale.size() <= 1) {
          if (!desc.HasAttr("out_h") || !desc.HasAttr("out_w")) {
            VLOG(3) << "The op_type " << op_type
                    << " doesn't have Scale and the scale size <=1 and without "
                       "out_h / out_w, it will return false";
            return false;
          }
          auto out_h = PADDLE_GET_CONST(int, desc.GetAttr("out_h"));
          auto out_w = PADDLE_GET_CONST(int, desc.GetAttr("out_w"));
          if (!(out_h <= 0 && out_w <= 0)) {
            if (out_h <= 0) {
              VLOG(3) << "The op_type " << op_type
                      << "'s out_h must be greater than 0 if scale is not set.";
              return false;
            }
            if (out_w <= 0) {
              VLOG(3) << "The op_type " << op_type
                      << "'s out_w must be greater than 0 if scale is not set.";
              return false;
            }
          }
        } else {
          for (size_t i = 0; i < scale.size(); i++) {
            if (scale[i] <= 0 && with_dynamic_shape) {
              VLOG(3) << "dynamic shape not support Attr(scale[" << i << "]) "
                      << scale[i]
                      << " less than 1 and Input(Scale) vector not set.";
              return false;
            }
          }
        }
      }
    }
    if (op_type == "linear_interp_v2") {
#if IS_TRT_VERSION_LT(7100)
      return false;
#endif
      std::vector<std::string> attrs{"data_layout",
                                     "interp_method",
                                     "align_corners",
                                     "scale",
                                     "out_h",
                                     "out_w"};
      for (auto const& attr : attrs) {
        if (!desc.HasAttr(attr)) {
          VLOG(3) << "The op_type " << op_type << " doesn't have the attr "
                  << attr << " and return false";
          return false;
        }
      }

      auto resize_inputs = desc.Inputs();
      if (resize_inputs.find("SizeTensor") != resize_inputs.end()) {
#if IS_TRT_VERSION_GE(8200)
        if (desc.Input("SizeTensor").size() == 1) {
          return true;
        }
#else
        if (!desc.Input("SizeTensor").empty()) {
          VLOG(3)
              << "The Paddle-TRT doesn't support the SizeTensor for op_type "
              << op_type;
          return false;
        }
#endif
      }
      if (resize_inputs.find("OutSize") != resize_inputs.end()) {
        if (!with_dynamic_shape) {
          VLOG(3) << "Static shape don't support the OutSize for op_type "
                  << op_type;
          return false;
        }
      }

      auto data_layout = common::StringToDataLayout(
          PADDLE_GET_CONST(std::string, desc.GetAttr("data_layout")));
      if (data_layout != phi::DataLayout::kNCHW &&
          data_layout != phi::DataLayout::kNHWC) {
        VLOG(3) << "The op_type " << op_type
                << " is not NCHW or NHWC return false";
        return false;
      }
      auto interp_method =
          PADDLE_GET_CONST(std::string, desc.GetAttr("interp_method"));
      if (interp_method != "linear") {
        VLOG(3) << "The interp_method of op_type " << op_type
                << " is not linear";
        return false;
      }
      bool has_scale_input_size =
          (resize_inputs.find("Scale") != resize_inputs.end());
      if (!has_scale_input_size ||
          (has_scale_input_size && desc.Input("Scale").size() != 1)) {
        const std::vector<float> scale =
            PADDLE_GET_CONST(std::vector<float>, desc.GetAttr("scale"));
        if (scale.size() == 0) {
          if (!desc.HasAttr("out_w")) {
            VLOG(3) << "The op_type " << op_type
                    << " doesn't have Scale and the scale size <=1 and without "
                       " out_w, it will return false";
            return false;
          }
          auto out_w = PADDLE_GET_CONST(int, desc.GetAttr("out_w"));
          if (out_w <= 0) {
            VLOG(3) << "The op_type " << op_type
                    << "'s out_w must be greater than 0 if scale is not set.";
            return false;
          }
        } else {
          for (size_t i = 0; i < scale.size(); i++) {
            if (scale[i] <= 0 && with_dynamic_shape) {
              VLOG(3) << "dynamic shape not support Attr(scale[" << i << "]) "
                      << scale[i]
                      << " less than 1 and Input(Scale) vector not set.";
              return false;
            }
          }
        }
      }
    }

    if (op_type == "unsqueeze2") {
      std::vector<int> axes;
      if (desc.HasAttr("axes")) {
        axes = PADDLE_GET_CONST(std::vector<int>, desc.GetAttr("axes"));
      }
      if (axes.empty()) {
        VLOG(3) << "The necessary attributes of the squeeze2 operator axes is "
                   "missing.";
        return false;
      }
      if (!with_dynamic_shape) {
        if (std::find(axes.begin(), axes.end(), 0) != axes.end()) {
          VLOG(3) << "Invalid squeeze axes. Axes having batch axis is not "
                     "supported in static shape";
          return false;
        }
      }
    }

    if (op_type == "batch_norm") {
      const std::vector<std::string> bn_inputs = {
          "X", "Bias", "Mean", "Scale", "Variance"};
      for (unsigned int i = 0; i < bn_inputs.size(); i++) {
        if (desc.Input(bn_inputs[i]).size() != 1) {
          VLOG(3) << "Invalid " << bn_inputs[i]
                  << "'s size of batch_norm TRT "
                     "converter. Expected 1, received "
                  << desc.Input(bn_inputs[i]).size() << ".";
          return false;
        }
      }
      auto batch_norm_inputs = desc.Inputs();
      if (batch_norm_inputs.find("MomentumTensor") != batch_norm_inputs.end()) {
        if (!desc.Input("MomentumTensor").empty()) {
          return false;
        }
      }
      if (desc.Output("Y").size() != 1) {
        VLOG(3) << "Invalid output Y's size of batch_norm TRT "
                   "converter. Expected 1, received "
                << desc.Output("Y").size() << ".";
        return false;
      }
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto x_var_name = desc.Input("X")[0];
      auto* x_var_desc = block->FindVarRecursive(x_var_name);
      const auto x_shape = x_var_desc->GetShape();
    }

    if (op_type == "split") {
      if (desc.Input("X").size() != 1) {
        VLOG(3) << "Invalid input X's size of split TRT converter. "
                   "Expected 1, received "
                << desc.Input("X").size() << ".";
        return false;
      }
      auto split_inputs = desc.Inputs();
      if (split_inputs.find("AxisTensor") != split_inputs.end()) {
        if (!desc.Input("AxisTensor").empty()) {
          return false;
        }
      }
      if (split_inputs.find("SectionsTensorList") != split_inputs.end()) {
        if (!desc.Input("SectionsTensorList").empty()) {
          if (!with_dynamic_shape) {
            return false;
          }
        }
      }
      if (!desc.HasAttr("axis")) {
        return false;
      }
      int axis = PADDLE_GET_CONST(int, desc.GetAttr("axis"));

      if (!with_dynamic_shape && axis == 0) {
        VLOG(3) << "Invalid split axis. Split on batch is not supported in "
                   "TensorRT with static shape";
        return false;
      }
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto x_var_name = desc.Input("X")[0];
      auto* x_var_desc = block->FindVarRecursive(x_var_name);
      const auto x_shape = x_var_desc->GetShape();
      size_t output_num = desc.Output("Out").size();
      std::vector<int> output_lengths;
      int num = 0;
      if (desc.HasAttr("num")) {
        num = PADDLE_GET_CONST(int, desc.GetAttr("num"));
      }
      if (desc.HasAttr("sections")) {
        output_lengths =
            PADDLE_GET_CONST(std::vector<int>, desc.GetAttr("sections"));
      }
      if (output_lengths.empty() && num == 0) {
        VLOG(3) << "sections and num cannot be equal to 0 at the same time";
        return false;
      }
      if (with_dynamic_shape) {
#if IS_TRT_VERSION_GE(6000)
#else
        VLOG(3) << "You are running the TRT Dynamic Shape mode, need to "
                   "confirm that "
                   "your TRT version is no less than 6.0";
        return false;
#endif
      }
      axis += (axis < 0) ? x_shape.size() : 0;
      if (x_shape[axis] == -1) {
        VLOG(3) << "The (" << axis << ") dim of input should not be -1";
        return false;
      }
      if (output_lengths.empty()) {
        if (num > 0) {
          int64_t in_axis_dim = x_shape[axis];
          if (in_axis_dim % num != 0) {
            VLOG(3) << "Invalid number to split. Tensor split does not result"
                       " in an equal division of dimensions. Axis dim = "
                    << in_axis_dim << " num = " << num << "!= 0";
            return false;
          }
          size_t out_axis_dim = in_axis_dim / num;
          for (int i = 0; i < num; ++i) {
            output_lengths.push_back(out_axis_dim);
          }
        }
      }
      if (output_lengths.size() != output_num) {
        VLOG(3) << "The output_length should be equal to the output size.";
        return false;
      }
    }

    if (op_type == "scale") {
      auto scale_inputs = desc.Inputs();
      if (scale_inputs.find("ScaleTensor") != scale_inputs.end()) {
        if (!desc.Input("ScaleTensor").empty()) {
          return false;
        }
      }
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto x_var_name = desc.Input("X")[0];
      auto* x_var_desc = block->FindVarRecursive(x_var_name);
      const auto x_shape = x_var_desc->GetShape();
      auto dtype = x_var_desc->GetDataType();
      if (!with_dynamic_shape) {
        // At present, only support float32 or float16 or float64 into trt.
        if (!(dtype == framework::proto::VarType::FP32 ||
              dtype == framework::proto::VarType::FP64 ||
              dtype == framework::proto::VarType::FP16)) {
          return false;
        }
      } else {
        // At present, only support float32 or float16 or float64 or int32 or
        // int64 into trt.
        if (!(dtype == framework::proto::VarType::FP32 ||
              dtype == framework::proto::VarType::FP16 ||
              dtype == framework::proto::VarType::FP64 ||
              dtype == framework::proto::VarType::INT32 ||
              dtype == framework::proto::VarType::INT64)) {
          return false;
        }
      }
    }

    if (op_type == "roll") {
#if !IS_TRT_VERSION_GE(7000)
      VLOG(3) << "roll converter does not support trt versions below 7.0";
      return false;
#endif
      if (!with_dynamic_shape) {
        return false;
      }
    }

    if (op_type == "strided_slice") {
#if !IS_TRT_VERSION_GE(7000)
      VLOG(3)
          << "strided_slice converter does not support trt versions below 7.0";
      return false;
#endif
      if (!desc.HasAttr("axes") || !desc.HasAttr("starts") ||
          !desc.HasAttr("ends") || !desc.HasAttr("strides")) {
        VLOG(3)
            << "The necessary attributes of the strided_slice operator miss ";
        return false;
      }
    }

    if (op_type == "rnn") {
      if (!with_dynamic_shape) {
        return false;
      }
      if (desc.HasAttr("mode")) {
        std::string mode = PADDLE_GET_CONST(std::string, desc.GetAttr("mode"));
        if (mode != "LSTM") return false;
      }
      if (desc.HasAttr("dropout_prob")) {
        float dropout_prob =
            PADDLE_GET_CONST(float, desc.GetAttr("dropout_prob"));
        if (dropout_prob > 1e-5) return false;
      }
      // not support following four inputs for rnn in paddle-trt
      auto rnn_inputs = desc.Inputs();
      if (rnn_inputs.find("SequenceLength") != rnn_inputs.end()) {
        if (!desc.Input("SequenceLength").empty()) {
          return false;
        }
      }
    }

    if (op_type == "fill_constant_batch_size_like") {
      if (!with_dynamic_shape) {
        return false;
      }
      if (!desc.HasAttr("input_dim_idx")) {
        return false;
      }
      if (!desc.HasAttr("output_dim_idx")) {
        return false;
      }
      if (!desc.HasAttr("shape")) {
        return false;
      }
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto x_var_name = desc.Input("Input")[0];
      auto* x_var_desc = block->FindVarRecursive(x_var_name);
      auto dtype = x_var_desc->GetDataType();
      // At present, only support float32 into trt.
      if (dtype != 5) {
        return false;
      }
    }

    if (op_type == "fill_any_like") {
      if (!with_dynamic_shape) {
        VLOG(3) << "the fill_any_like does not support static shape yet";
        return false;
      }
      int dtype = desc.HasAttr("dtype")
                      ? PADDLE_GET_CONST(int, desc.GetAttr("dtype"))
                      : -1;
      auto* block = desc.Block();
      auto* x_var_desc = block->FindVarRecursive(desc.Input("X")[0]);
      auto input_type = x_var_desc->GetDataType();
#if IS_TRT_VERSION_GE(8400)
      if (dtype == 0 ||
          (dtype == -1 && input_type == framework::proto::VarType::BOOL)) {
        VLOG(3) << "the fill_any_like supports input of BOOL by trt8.4 above";
        return true;
      }
#endif
      if (dtype != -1 && dtype != 2 && dtype != 3 && dtype != 5 && dtype != 6) {
        VLOG(3)
            << "the fill_any_like only supports int32/int64/float32/float64 by"
               "trt8.4 below";
        return false;
      }
      if (dtype == -1) {
        if (input_type != framework::proto::VarType::INT32 &&
            input_type != framework::proto::VarType::INT64 &&
            input_type != framework::proto::VarType::FP32 &&
            input_type != framework::proto::VarType::FP64) {
          VLOG(3) << "the fill_any_like only supports "
                     "int32/int64/float32/float64 by"
                     "trt8.4 below";
          return false;
        }
      }
    }

    if (op_type == "slice") {
      if (desc.HasAttr("decrease_axis")) {
        std::vector<int> decrease_axis =
            PADDLE_GET_CONST(std::vector<int>, desc.GetAttr("decrease_axis"));
        if (!with_dynamic_shape) {
          if (decrease_axis.end() !=
              std::find(decrease_axis.begin(), decrease_axis.end(), 0)) {
            return false;
          }
        }
      }
      std::vector<int> axes;
      if (!desc.HasAttr("axes")) {
        VLOG(3) << "The necessary attributes of the slice operator axes "
                   " are missing.";
        return false;
      } else {
        axes = PADDLE_GET_CONST(std::vector<int>, desc.GetAttr("axes"));
        if (!with_dynamic_shape) {
          for (size_t i = 0; i < axes.size(); i++) {
            if (axes[i] == 0) {
              VLOG(3) << "Invalid slice axis. Slice on batch axis is not "
                         "supported in TensorRT";
              return false;
            }
          }
        }
      }
      // not support following four inputs for slice in paddle-trt
      auto slice_inputs = desc.Inputs();  // its size == 5
      if (slice_inputs.find("StartsTensor") != slice_inputs.end() &&
          !desc.Input("StartsTensor").empty()) {
        VLOG(3) << "The Slice has StartsTensor input.";
      } else {
        if (!desc.HasAttr("starts")) {
          VLOG(3) << "The necessary attributes of the slice operator starts or "
                     "StartsTensor"
                     " are missing.";
          return false;
        } else {
          std::vector<int> starts =
              PADDLE_GET_CONST(std::vector<int>, desc.GetAttr("starts"));
          if (axes.size() != starts.size()) {
            VLOG(3) << "The shape of attributes of the slice operator axes "
                       "and starts are not equal.";
            return false;
          }
        }
      }
      if (slice_inputs.find("EndsTensor") != slice_inputs.end() &&
          !desc.Input("EndsTensor").empty()) {
        VLOG(3) << "The Slice has EndsTensor input.";
      } else {
        if (!desc.HasAttr("ends")) {
          VLOG(3) << "The necessary attributes of the slice operator ends or "
                     "EndsTensor"
                     " are missing.";
          return false;
        } else {
          std::vector<int> ends =
              PADDLE_GET_CONST(std::vector<int>, desc.GetAttr("ends"));
          if (axes.size() != ends.size()) {
            VLOG(3) << "The shape of attributes of the slice operator axes "
                       "and ends are not equal.";
            return false;
          }
        }
      }
      if (slice_inputs.find("StartsTensorList") != slice_inputs.end()) {
        VLOG(3) << "The Slice has StartsTensorList input.";
      }
      if (slice_inputs.find("EndsTensorList") != slice_inputs.end()) {
        VLOG(3) << "The Slice has EndsTensorList input.";
      }
    }

    if (op_type == "less_than" || op_type == "greater_than" ||
        op_type == "logical_or" || op_type == "logical_xor" ||
        op_type == "logical_and" || op_type == "less_equal" ||
        op_type == "greater_equal") {
#if IS_TRT_VERSION_GE(8400)
      // TRT does not support kEQUAL/kGREATER/kLESS work with implicit batch
      if (!with_dynamic_shape) {
        VLOG(3) << "Ops(" << op_type << ") do not support static shape yet.";
        return false;
      }
      auto* block = desc.Block();
      auto* x_var_desc = block->FindVarRecursive(desc.Input("X")[0]);
      auto* y_var_desc = block->FindVarRecursive(desc.Input("Y")[0]);
      auto x_dtype = x_var_desc->GetDataType();
      auto y_dtype = y_var_desc->GetDataType();
      if (op_type == "logical_or" || op_type == "logical_xor" ||
          op_type == "logical_and") {
        if (x_dtype != framework::proto::VarType::BOOL ||
            y_dtype != framework::proto::VarType::BOOL) {
          VLOG(3) << "the op (" << op_type << ") only support input of BOOL.";
          return false;
        }
      }
      if (op_type == "less_than" || op_type == "greater_than" ||
          op_type == "less_equal" || op_type == "greater_equal") {
        if (x_dtype == framework::proto::VarType::BOOL ||
            y_dtype == framework::proto::VarType::BOOL) {
          VLOG(3)
              << "ElementWiseOperation::kLESS/ElementWiseOperation::kGREATER "
                 "do not support boolean datatype.";
          return false;
        }
      }
#else
      VLOG(3) << "these are not supported when TensorRT < 8.4";
      return false;
#endif
    }
    if (op_type == "elementwise_add" || op_type == "elementwise_mul" ||
        op_type == "elementwise_sub" || op_type == "elementwise_div" ||
        op_type == "elementwise_pow" || op_type == "elementwise_min" ||
        op_type == "elementwise_max" || op_type == "elementwise_floordiv" ||
        op_type == "elementwise_mod") {
      if (desc.Input("X").size() != 1) {
        VLOG(3) << "The input op's Input(\"X\").size() "
                   "should equal to 1, but received Input(\"X\").size() = "
                << desc.Input("X").size() << ".";
        return false;
      }
      if (desc.Input("Y").size() != 1) {
        VLOG(3) << "The input op's Input(\"Y\").size() "
                   "should equal to 1, but received Input(\"Y\").size() = "
                << desc.Input("Y").size() << ".";
        return false;
      }
      if (desc.Output("Out").size() != 1) {
        VLOG(3) << "The input op's Output(\"Out\").size() "
                   "should equal to 1, but received Output(\"Out\").size() = "
                << desc.Output("Out").size() << ".";
        return false;
      }
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto* x_var_desc = block->FindVarRecursive(desc.Input("X")[0]);
      auto* y_var_desc = block->FindVarRecursive(desc.Input("Y")[0]);
      const auto x_shape = x_var_desc->GetShape();
      const auto y_shape = y_var_desc->GetShape();

      // These operations do not support boolean datatype.
      if (op_type == "elementwise_add" || op_type == "elementwise_mul" ||
          op_type == "elementwise_sub" || op_type == "elementwise_div" ||
          op_type == "elementwise_pow" || op_type == "elementwise_min" ||
          op_type == "elementwise_max" || op_type == "elementwise_floordiv" ||
          op_type == "elementwise_mod") {
        if (x_var_desc->GetDataType() ==
            paddle::framework::proto::VarType_Type::VarType_Type_BOOL) {
          VLOG(3)
              << "These operations "
                 "(elementwise_add/mul/sub/div/pow/min/max/floordiv/mod) do "
                 "not support boolean datatype.";
          return false;
        }
      }
      // These operations input do not support int32 datatype.
      if (op_type == "elementwise_pow") {
        if (x_var_desc->GetDataType() ==
            paddle::framework::proto::VarType_Type::VarType_Type_INT32) {
          VLOG(3) << "These operations (elementwise_pow) do not support int32 "
                     "datatype.";
          return false;
        }
      }

      // The case when x_shape.size() == 1 is dealt with in common case
      if (!with_dynamic_shape && (!y_var_desc->Persistable()) &&
          y_shape.size() == 1) {
        VLOG(3) << "Static shape in trt not support y is  a 1D intermediate "
                   "tensor in "
                   "elementwise op.";
        return false;
      }

      if (x_var_desc->Persistable() && !with_dynamic_shape) {
        VLOG(3)
            << "Input X is a parameter which is not supported for "
               "elementwise in tensorrt's static shape, swap x and y will work";
        return false;
      }
    }

    if (op_type == "pow") {
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto* x_var_desc = block->FindVarRecursive(desc.Input("X")[0]);

      // the same as `elementwise_pow`.
      if (x_var_desc->GetDataType() ==
          paddle::framework::proto::VarType_Type::VarType_Type_INT32) {
        VLOG(3) << "These operations (pow) do not support int32 "
                   "datatype.";
        return false;
      }
    }

    if (op_type == "stack") {
      if (!with_dynamic_shape) {
        VLOG(3)
            << "static shape mode is not supported for TRT stack.\n"
               "You can use the config.SetTRTDynamicShapeInfo(...) interface"
               " to set the shape information to run the dynamic shape "
               "mode.";
        return false;
      }

      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto x_var_name = desc.Input("X")[0];
      auto* x_var_desc = block->FindVarRecursive(x_var_name);
      const auto x_shape = x_var_desc->GetShape();
      int rank = x_shape.size();
      int axis = desc.HasAttr("axis")
                     ? PADDLE_GET_CONST(int, desc.GetAttr("axis"))
                     : -1;
      if (axis > rank || axis < -(rank + 1)) {
        return false;
      }
    }

    if (op_type == "shape" && !with_dynamic_shape) {
      return false;
    }

    if (op_type == "fused_embedding_eltwise_layernorm") {
      if (!with_dynamic_shape) {
        VLOG(3) << "fused_embedding_eltwise_layernorm should run on dynamic "
                   "shape mode.";
        return false;
      }
      if (desc.Input("Ids").size() != desc.Input("Embs").size()) {
        return false;
      }
    }
    if (op_type == "fused_bias_dropout_residual_layer_norm") {
      if (!with_dynamic_shape) {
        VLOG(3) << "fused_bias_dropout_residual_layer_norm should run on "
                   "dynamic shape mode.";
        return false;
      }
      float dropout_rate =
          PADDLE_GET_CONST(float, desc.GetAttr("dropout_rate"));
      if (dropout_rate != 0.0f) {
        VLOG(4) << "preln_residual_bias trt layer can not work with "
                   "fused_bias_dropout_residual_layer_norm op in which the "
                   "dropout_rate != 0, stop convert";
        return false;
      }
    }
    if (op_type == "fused_preln_embedding_eltwise_layernorm") {
      if (!with_dynamic_shape) {
        VLOG(3) << "fused_preln_embedding_eltwise_layernorm should run on "
                   "dynamic "
                   "shape mode.";
        return false;
      }
      if (desc.Input("Ids").size() != desc.Input("Embs").size()) {
        VLOG(3) << "The id and emb size of fused PrelnEmbEltwiseLayerNormOp "
                   "should be same ";
        return false;
      }
      if (!desc.HasAttr("enable_int8")) {
        VLOG(3) << "PrelnEmbEltwiseLayerNormOp must use int8 mode.";
        return false;
      }
    }

    if (op_type == "gelu") {
      if (desc.Input("X").size() != 1) {
        VLOG(3) << "gelu op has only 1 input, but got "
                << desc.Input("X").size();
        return false;
      }
      if (desc.Output("Out").size() != 1) {
        VLOG(3) << "gelu op has only 1 output, but got "
                << desc.Output("Out").size();
        return false;
      }

#if IS_TRT_VERSION_LT(7000)
      if (desc.HasAttr("approximate")) {
        VLOG(3) << "approximate gelu op needs TensorRT 7.0 and after";
        if (PADDLE_GET_CONST(bool, desc.GetAttr("approximate"))) return false;
      }
#endif
    }

    if (op_type == "layer_norm") {
      if (desc.Input("X").size() != 1) {
        VLOG(3) << "input of layer_norm op converter should be 1, got "
                << desc.Input("X").size();
        return false;
      }
      if (desc.Input("Bias").size() != 1) {
        VLOG(3) << "Bias of layer_norm op converter should be 1, got "
                << desc.Input("Bias").size();
        return false;
      }
      if (desc.Input("Scale").size() != 1) {
        VLOG(3) << "Scale of layer_norm op converter should be 1, got "
                << desc.Input("Scale").size();
        return false;
      }
      if (desc.Output("Y").size() != 1) {
        VLOG(3) << "output of layer_norm op converter should be 1, got "
                << desc.Output("Y").size();
        return false;
      }
    }

    if (op_type == "fill_constant") {
      auto fill_constant_inputs = desc.Inputs();
      if (fill_constant_inputs.find("ValueTensor") !=
          fill_constant_inputs.end()) {
        if (!desc.Input("ValueTensor").empty()) return false;
      }

      if (desc.HasInput("ShapeTensor")) {
        if (desc.Input("ShapeTensor").size() > 1) return false;
        if (desc.Input("ShapeTensor").size() == 1) {
#if IS_TRT_VERSION_LT(8500)
          VLOG(3) << "fill_constant ShapeTensor is not supported when TensorRT "
                     "< 8.5.0";
          return false;
#endif
        }
      }

#if IS_TRT_VERSION_LT(8500)
      if (desc.HasInput("ShapeTensorList")) {
        if (desc.Input("ShapeTensorList").size() >= 1) {
          VLOG(3) << "fill_constant ShapeTensorList is not supported when "
                     "TensorRT < 8.5.0";
          return false;
        }
      }
#endif

      int dtype = desc.HasAttr("dtype")
                      ? PADDLE_GET_CONST(int, desc.GetAttr("dtype"))
                      : 5;
      // only support int32, int64, float32
      if (!(dtype == 2 || dtype == 3 || dtype == 5)) {
        return false;
      }
    }

    if (op_type == "instance_norm") {
      if (desc.Input("X").size() != 1) {
        VLOG(3) << "input of instance_norm op converter should be 1, got "
                << desc.Input("X").size();
        return false;
      }
      if (desc.Input("Bias").size() != 1) {
        VLOG(3) << "Bias of instance_norm op converter should be 1, got "
                << desc.Input("Bias").size();
        return false;
      }
      if (desc.Input("Scale").size() != 1) {
        VLOG(3) << "Scale of instance_norm op converter should be 1, got "
                << desc.Input("Scale").size();
        return false;
      }
      if (desc.Output("Y").size() != 1) {
        VLOG(3) << "output of layer_norm op converter should be 1, got "
                << desc.Output("Y").size();
        return false;
      }

      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto x_var_name = desc.Input("X")[0];
      auto* x_var_desc = block->FindVarRecursive(x_var_name);
      const auto x_shape = x_var_desc->GetShape();
      if (x_shape.size() != 4) {
        VLOG(3) << "The instance_norm op only support 4-dimensional input in "
                   "tensorrt.";
        return false;
      }
    }

    if (op_type == "pad") {
      if (!desc.HasAttr("pad_value") || !desc.HasAttr("paddings")) return false;
      const float pad_value =
          PADDLE_GET_CONST(float, desc.GetAttr("pad_value"));
      if (pad_value != 0.0f) {
        VLOG(3) << "The pad layer of TRT only support zero.";
        return false;
      }
      std::vector<int64_t> shape;
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      for (auto& param_name : desc.Inputs()) {
        for (auto& var_name : param_name.second) {
          auto* var_desc = block->FindVarRecursive(var_name);
          shape = var_desc->GetShape();
        }
      }
      int nbDims = shape.size();
      std::vector<int> paddings =
          PADDLE_GET_CONST(std::vector<int>, desc.GetAttr("paddings"));
      int pad_size = paddings.size();
      if (nbDims < 2) {
        return false;
      }
      if (nbDims * 2 != pad_size) {
        return false;
      }
      for (int i = 0; i < pad_size - 4; i++) {
        if (paddings[i] != 0) {
          return false;
        }
      }
    }

    if (op_type == "bitwise_and") {
#if IS_TRT_VERSION_LT(8400)
      VLOG(3) << "bitwise_and is not supported when TensorRT < 8.4";
      return false;
#endif
      if (!with_dynamic_shape) {
        VLOG(3) << "Ops(" << op_type << ") do not support static shape yet.";
        return false;
      }
      auto x_var_name = desc.Input("X")[0];
      auto y_var_name = desc.Input("Y")[0];
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto* x_var_desc = block->FindVarRecursive(x_var_name);
      auto* y_var_desc = block->FindVarRecursive(y_var_name);
      auto x_dtype = x_var_desc->GetDataType();
      auto y_dtype = y_var_desc->GetDataType();
      if (x_dtype != framework::proto::VarType::BOOL ||
          y_dtype != framework::proto::VarType::BOOL) {
        VLOG(3) << "the bitwise_and only support input of BOOL.";
        return false;
      }
    }

    if (op_type == "bitwise_or") {
#if IS_TRT_VERSION_LT(8400)
      VLOG(3) << "bitwise_or is not supported when TensorRT < 8.4";
      return false;
#endif
      if (!with_dynamic_shape) {
        VLOG(3) << "Ops(" << op_type << ") do not support static shape yet.";
        return false;
      }
      auto x_var_name = desc.Input("X")[0];
      auto y_var_name = desc.Input("Y")[0];
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto* x_var_desc = block->FindVarRecursive(x_var_name);
      auto* y_var_desc = block->FindVarRecursive(y_var_name);
      auto x_dtype = x_var_desc->GetDataType();
      auto y_dtype = y_var_desc->GetDataType();
      if (x_dtype != framework::proto::VarType::BOOL ||
          y_dtype != framework::proto::VarType::BOOL) {
        VLOG(3) << "the bitwise_or only support input of BOOL.";
        return false;
      }
    }
    if (op_type == "size") {
      if (!with_dynamic_shape) {
        VLOG(3) << "Ops(" << op_type << ") do not support static shape yet.";
        return false;
      }
    }

    if (op_type == "pad3d") {
#if !IS_TRT_VERSION_GE(8200)
      VLOG(3) << "pad3d is not supported when TensorRT < 8.2";
      return false;
#endif
      if (!with_dynamic_shape) {
        VLOG(3) << "pad3d is not supported static shape";
        return false;
      }
      if (!desc.HasAttr("paddings") && !desc.HasInput("Paddings")) {
        return false;
      }
      if (desc.HasAttr("mode")) {
        std::string mode = PADDLE_GET_CONST(std::string, desc.GetAttr("mode"));
        if (mode != "constant" && mode != "reflect" && mode != "replicate") {
          VLOG(3) << "The pad3d layer of TRT only support "
                     "constant/reflect/replicate mode.";
          return false;
        }
      }
      if (desc.HasAttr("data_format")) {
        std::string data_format =
            PADDLE_GET_CONST(std::string, desc.GetAttr("data_format"));
        if (data_format != "NCDHW") {
          VLOG(3) << "The pad3d layer of TRT only support NCDHW data format.";
          return false;
        }
      }
    }

    if (op_type == "prelu") {
      if (desc.Input("X").size() != 1) {
        VLOG(3) << "Invalid input X's size of prelu TRT converter. "
                   "Expected 1, received "
                << desc.Input("X").size() << ".";
        return false;
      }
      if (desc.Output("Out").size() != 1) {
        VLOG(3) << "Invalid output Out's size of prelu TRT converter. "
                   "Expected 1, received "
                << desc.Output("Out").size() << ".";
        return false;
      }

      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto* alpha_var = block->FindVarRecursive(desc.Input("Alpha")[0]);
      if (!alpha_var) {
        VLOG(3) << "Variable Alpha of prelu TRT converter not found.";
        return false;
      }
      auto alpha_shape = alpha_var->GetShape();
      if (!with_dynamic_shape && alpha_shape.empty()) {
        VLOG(3) << op_type
                << " op does not support alpha's dim is 0 in tensorrt "
                   "static shape mode.";
        return false;
      }
    }

    if (op_type == "mish") {
      if (desc.Input("X").size() != 1) {
        VLOG(3) << "Invalid input X's size of mish TRT converter. "
                   "Expected 1, received "
                << desc.Input("X").size() << ".";
        return false;
      }
      if (desc.Output("Out").size() != 1) {
        VLOG(3) << "Invalid output Out's size of mish TRT converter. "
                   "Expected 1, received "
                << desc.Output("Out").size() << ".";
        return false;
      }
    }

    if (op_type == "roi_align") {
      if (!with_dynamic_shape) {
        VLOG(3) << "TRT roi align plugin only accept the dynamic shape, "
                   "because that "
                   "the roi_align will change the batch size.";
        return false;
      }
      std::vector<std::string> attrs{"pooled_height",
                                     "pooled_width",
                                     "spatial_scale",
                                     "sampling_ratio",
                                     "aligned"};
      for (auto const& attr : attrs) {
        if (!desc.HasAttr(attr)) return false;
      }

      const auto pooled_height =
          PADDLE_GET_CONST(int, desc.GetAttr("pooled_height"));
      if (pooled_height <= 0) return false;

      const auto pooled_width =
          PADDLE_GET_CONST(int, desc.GetAttr("pooled_width"));
      if (pooled_width <= 0) return false;

      const auto spatial_scale =
          PADDLE_GET_CONST(float, desc.GetAttr("spatial_scale"));
      if (spatial_scale <= 0.f) return false;

      auto roi_align_inputs = desc.Inputs();
      if (roi_align_inputs.find("RoisNum") != roi_align_inputs.end()) {
        if (!desc.Input("RoisNum").empty()) {
          return false;
        }
      }
    }

    if (op_type == "shuffle_channel") {
#if !IS_TRT_VERSION_GE(8000)
      if (with_dynamic_shape) {
        VLOG(3) << "You are running the TRT Dynamic Shape mode, "
                   "the shuffle_channel op does not support dynamic shape "
                   "trt versions below 8.0 yet";
        return false;
      }
#endif
    }

    if (op_type == "where") {
#if !IS_TRT_VERSION_GE(8400)
      VLOG(3) << "where is not supported when TensorRT < 8.4";
      return false;
#endif
      if (!with_dynamic_shape) {
        VLOG(3) << "the where op does not support static shape yet";
        return false;
      }
    }

    if (op_type == "bitwise_not") {
      auto* block = desc.Block();
      auto x_var_name = desc.Input("X")[0];
      auto* x_var_desc = block->FindVarRecursive(x_var_name);
      auto dtype = x_var_desc->GetDataType();
      if (dtype == framework::proto::VarType::INT8 ||
          dtype == framework::proto::VarType::UINT8) {
        VLOG(3) << "INT8 / UINT8 type convert to trt is not supported";
        return false;
      }
      if (dtype == framework::proto::VarType::BOOL) {
#if !IS_TRT_VERSION_GE(8400)
        VLOG(3) << "BOOL type support requires TensorRT 8.4";
        return false;
#elif !IS_TRT_VERSION_GE(8600)
        const auto x_shape = x_var_desc->GetShape();
        if (x_shape.empty()) {
          VLOG(3) << "BOOL type does not support 0 dim input when TensorRT < "
                     "8.6.";
          return false;
        }
#endif
      }
    }

    if (op_type == "one_hot" || op_type == "one_hot_v2") {
#if IS_TRT_VERSION_LT(8510)
      VLOG(3) << "one_hot/one_hot_v2 is not supported when TensorRT < 8.5.1";
      return false;
#endif
      if (!with_dynamic_shape) {
        VLOG(3)
            << "the one_hot/one_hot_v2 op does not support static shape yet";
        return false;
      }
      if (desc.HasAttr("allow_out_of_range")) {
        VLOG(3) << "allow_out_of_range one_hot/one_hot_v2 op is not "
                   "supported now.";
        if (PADDLE_GET_CONST(bool, desc.GetAttr("allow_out_of_range")))
          return false;
      }
      if (desc.HasAttr("dtype")) {
        const int dtype = PADDLE_GET_CONST(int, desc.GetAttr("dtype"));
        if (dtype != 2 && dtype != 3 && dtype != 5) {
          VLOG(3) << "one_hot/one_hot_v2 op only support int32, int64, float.";
          return false;
        }
      }
      auto one_hot_inputs = desc.Inputs();
      if (one_hot_inputs.find("depth_tensor") != one_hot_inputs.end()) {
        if (!desc.Input("depth_tensor").empty()) {
          return true;
        }
      }

      if (desc.HasAttr("depth")) {
        const int depth = PADDLE_GET_CONST(int, desc.GetAttr("depth"));
        if (depth <= 0) {
          VLOG(3) << "depth only support positive in one_hot/one_hot_v2 op.";
          return false;
        }
      }
    }

    if (op_type == "skip_layernorm") {
      if (!with_dynamic_shape) {
        VLOG(3) << "the skip_layernorm does not support static shape yet";
        return false;
      }
    }

    if (op_type == "preln_skip_layernorm") {
      if (!with_dynamic_shape) {
        VLOG(3) << "the preln_skip_layernorm does not support static shape yet";
        return false;
      }
      if (!desc.HasAttr("enable_int8")) {
        VLOG(3) << "PrelnEmbEltwiseLayerNormOp must use int8 mode.";
        return false;
      }
    }

    if (op_type == "multihead_matmul") {
      if (!with_dynamic_shape) {
        VLOG(3) << "the multihead_matmul does not support static shape yet";
        return false;
      }

      if (desc.HasAttr("enable_int8") && !desc.HasAttr("Input_scale")) {
        VLOG(3) << "Multihead layers must have input scale in int8 mode.";
        return false;
      }

      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto* input_desc = block->FindVarRecursive(desc.Input("Input").front());
      const auto input_shape = input_desc->GetShape();
      const auto head_number =
          PADDLE_GET_CONST(int, desc.GetAttr("head_number"));
      auto inputs = desc.Inputs();
      bool has_bias_qk = (inputs.find("BiasQK") == inputs.end()) ? false : true;
      if (has_bias_qk) {
        auto* bias_qk_desc =
            block->FindVarRecursive(desc.Input("BiasQK").front());
        const auto bias_qk_shape = bias_qk_desc->GetShape();
        // The BiasQK's shape requires to be
        // [batch, 1, 1, length] or [batch, head, length, length].
        bool has_same_shape = head_number == bias_qk_shape[1] &&
                              input_shape[1] == bias_qk_shape[2] &&
                              input_shape[1] == bias_qk_shape[3];
        bool is_broadcastable = bias_qk_shape[1] == 1 &&
                                bias_qk_shape[2] == 1 &&
                                input_shape[1] == bias_qk_shape[3];
        is_broadcastable = is_broadcastable ||
                           (bias_qk_shape[0] == 1 && bias_qk_shape[1] == 1 &&
                            input_shape[1] == bias_qk_shape[2] &&
                            input_shape[1] == bias_qk_shape[3]);
        if (!(has_same_shape || is_broadcastable)) {
          VLOG(3) << "The BiasQK's shape is invalid, expect [" << input_shape[0]
                  << ", 1, 1, " << input_shape[1] << "] "
                  << "or [" << input_shape[0] << ", " << head_number << ", "
                  << input_shape[1] << ", " << input_shape[1] << "] "
                  << "or [" << input_shape[0] << "/1, " << 1 << ", "
                  << input_shape[1] << ", " << input_shape[1] << "] "
                  << "but got [" << bias_qk_shape[0] << ", " << bias_qk_shape[1]
                  << ", " << bias_qk_shape[2] << ", " << bias_qk_shape[3]
                  << "].";
          return false;
        }
      } else {
#if (IS_TRT_VERSION_GE(8000) && IS_TRT_VERSION_LT(8100)) || \
    (IS_TRT_VERSION_LT(7200))
        VLOG(3) << "There are some bugs with trt 8.0";
        return false;
#endif
      }
    }

    if (op_type == "multihead_matmul_roformer") {
      if (!with_dynamic_shape) {
        VLOG(3) << "the multihead_matmul_roformer does not support static "
                   "shape yet";
        return false;
      }

      if (desc.HasAttr("enable_int8") && !desc.HasAttr("Input_scale")) {
        VLOG(3) << "Multihead layers must have input scale in int8 mode.";
        return false;
      }

      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto* input_desc = block->FindVarRecursive(desc.Input("Input").front());
      const auto input_shape = input_desc->GetShape();
      const auto head_number =
          PADDLE_GET_CONST(int, desc.GetAttr("head_number"));
      auto inputs = desc.Inputs();
      bool has_bias_qk = (inputs.find("BiasQK") == inputs.end()) ? false : true;
      if (has_bias_qk) {
        auto* bias_qk_desc =
            block->FindVarRecursive(desc.Input("BiasQK").front());
        const auto bias_qk_shape = bias_qk_desc->GetShape();
        // The BiasQK's shape requires to be
        // [batch, 1, 1, length] or [batch, head, length, length].
        bool has_same_shape = head_number == bias_qk_shape[1] &&
                              input_shape[1] == bias_qk_shape[2] &&
                              input_shape[1] == bias_qk_shape[3];
        bool is_broadcastable = bias_qk_shape[1] == 1 &&
                                bias_qk_shape[2] == 1 &&
                                input_shape[1] == bias_qk_shape[3];
        if (!(has_same_shape || is_broadcastable)) {
          VLOG(3) << "The BiasQK's shape is invalid, expect [" << input_shape[0]
                  << ", 1, 1, " << input_shape[1] << "] or [" << input_shape[0]
                  << ", " << head_number << ", " << input_shape[1] << ", "
                  << input_shape[1] << "] but [" << bias_qk_shape[0] << ", "
                  << bias_qk_shape[1] << ", " << bias_qk_shape[2] << ", "
                  << bias_qk_shape[3] << "].";
          return false;
        }
      } else {
#if !IS_TRT_VERSION_GE(8000)
        VLOG(3) << "The version of TRT must be greater than 8000";
        return false;
#endif
      }
    }

    if (op_type == "reshape" || op_type == "reshape2") {
      if (!desc.HasAttr("shape")) {
        return false;
      }
      if (with_dynamic_shape) {
        return true;
      }
      // Static shape does not support the input tensors: Shape and
      // ShapeTensor
      auto reshape_inputs = desc.Inputs();
      if (reshape_inputs.find("Shape") != reshape_inputs.end()) {
        if (!desc.Input("Shape").empty()) {
          return false;
        }
      }
      if (reshape_inputs.find("ShapeTensor") != reshape_inputs.end()) {
        if (!desc.Input("ShapeTensor").empty()) {
          return false;
        }
      }
      std::vector<int> shape =
          PADDLE_GET_CONST(std::vector<int>, desc.GetAttr("shape"));
      if (shape.size() >= nvinfer1::Dims::MAX_DIMS) return false;
      if (!with_dynamic_shape) {
        if (shape.size() == 1) {
          return false;
        }
        if (shape[0] == 0) {
          return true;
        } else {
          auto* block = desc.Block();
          auto x_var_name = desc.Input("X")[0];
          auto* x_var_desc = block->FindVarRecursive(x_var_name);
          const auto x_shape = x_var_desc->GetShape();
          int input_num = std::accumulate(
              x_shape.begin() + 1, x_shape.end(), 1, std::multiplies<int>());
          int shape_num = std::accumulate(
              shape.begin() + 1, shape.end(), 1, std::multiplies<int>());
          if (input_num == shape_num) {
            return true;
          }
        }
        return false;
      }
    }

    if (op_type == "clip") {
      if (!with_dynamic_shape) {
        VLOG(3) << "the clip does not support static "
                   "shape yet";
        return false;
      }
      // Paddle-TRT does not support the input tensors: Min and Max
      auto clip_inputs = desc.Inputs();
      if (clip_inputs.find("Min") != clip_inputs.end()) {
        if (!desc.Input("Min").empty()) {
          return false;
        }
      }
      if (clip_inputs.find("Max") != clip_inputs.end()) {
        if (!desc.Input("Max").empty()) {
          return false;
        }
      }

      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto x_var_name = desc.Input("X")[0];
      auto* x_var_desc = block->FindVarRecursive(x_var_name);
      const auto x_shape = x_var_desc->GetShape();
      if (x_shape.empty()) {
        VLOG(3) << op_type
                << " op does not support input's dim is 0 in tensorrt.";
        return false;
      }
    }

    if (op_type == "reduce_sum" || op_type == "reduce_mean" ||
        op_type == "reduce_max" || op_type == "reduce_min" ||
        op_type == "reduce_prod" || op_type == "reduce_any" ||
        op_type == "reduce_all") {
      if (!desc.HasAttr("dim", /*with_attr_var=*/false)) {
        VLOG(3) << "Skip to convert into TRT while found Attribute('dim') is "
                   "Variable type in "
                << desc.Type();
        return false;
      }

      if (!(desc.HasAttr("keep_dim") && desc.HasAttr("dim") &&
            desc.HasAttr("reduce_all"))) {
        VLOG(3) << "the " << op_type
                << " does not have attr (keep_dim or dim or "
                   "reduce_all)";
        return false;
      }

      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }

      // The batch size dimension cannot be reduced if it's not dynamic shape.
      auto* x_var_desc = block->FindVarRecursive(desc.Input("X")[0]);
      if (!with_dynamic_shape) {
        if (PADDLE_GET_CONST(bool, desc.GetAttr("reduce_all"))) return false;
        std::vector<int32_t> dim =
            PADDLE_GET_CONST(std::vector<int32_t>, desc.GetAttr("dim"));
        const auto input_shape = x_var_desc->GetShape();
        for (auto x : dim) {
          if (x == 0 || (x + input_shape.size() == 0)) return false;
        }
      }

      auto dtype = x_var_desc->GetDataType();
      if (op_type == "reduce_all" || op_type == "reduce_any") {
        if (dtype != framework::proto::VarType::BOOL) {
          VLOG(3)
              << "reduce_all and reduce_any op input data type must be bool";
          return false;
        }
      } else {
#if IS_TRT_VERSION_GE(7000)
        if (dtype != framework::proto::VarType::INT32 &&
            dtype != framework::proto::VarType::INT64 &&
            dtype != framework::proto::VarType::FP32 &&
            dtype != framework::proto::VarType::FP64) {
          VLOG(3) << "reduce op input data type must be int32 or int64 or "
                     "float32 or "
                     "float64";
          return false;
        }
#else
        if (dtype != framework::proto::VarType::FP32 &&
            dtype != framework::proto::VarType::FP64) {
          VLOG(3) << "reduce op input data type must be float32 or float64 "
                     "using TensorRT "
                     "< 7.0";
          return false;
        }
#endif
      }
    }
#if IS_TRT_VERSION_GE(7000)
    if (op_type == "tile") {
      // Paddle-TRT does not support the input tensors.
      auto tile_inputs = desc.Inputs();
      if (!with_dynamic_shape) {
        if (tile_inputs.find("repeat_times_tensor") != tile_inputs.end()) {
          if (!desc.Input("repeat_times_tensor").empty()) {
            VLOG(3) << "Tile op: repeat_times_tensor is not empty.";
            return false;
          }
        }
        if (tile_inputs.find("RepeatTimes") != tile_inputs.end()) {
          if (!desc.Input("RepeatTimes").empty()) {
            VLOG(3) << "Tile op: RepeatTimes is not empty.";
            return false;
          }
        }
        if (!desc.HasAttr("repeat_times")) {
          VLOG(3) << "Tile op:`repeat_times` is not set.";
          return false;
        }
      }
    }
#endif

    // conv3d_transpose
    if (op_type == "conv3d_transpose") {
      // trt doesn't support output_padding when < 8406
      // output_padding is usually set when stride > 1
#if !IS_TRT_VERSION_GE(8400)
      if (desc.HasAttr("output_padding")) {
        const std::vector<int> output_padding =
            PADDLE_GET_CONST(std::vector<int>, desc.GetAttr("output_padding"));
        if (output_padding.size() > 0) {
          int max_padding =
              *std::max_element(output_padding.begin(), output_padding.end());
          if (max_padding > 0) return false;
        }
      }
#endif
    }

    if (op_type == "conv3d" || op_type == "conv3d_transpose") {
      if (desc.HasAttr("padding_algorithm")) {
        std::string padding_algorithm =
            PADDLE_GET_CONST(std::string, desc.GetAttr("padding_algorithm"));

        // trt error is raised if conv3d_transpose and SAME
        if (op_type == "conv3d_transpose" && padding_algorithm == "SAME" &&
            !with_dynamic_shape) {
          return false;
        }
      }

#if !IS_TRT_VERSION_GE(7000)
      // looks like some issues with trt6.0
      if (with_dynamic_shape) {
        return false;
      }
#endif

      std::vector<int> paddings =
          PADDLE_GET_CONST(std::vector<int>, desc.GetAttr("paddings"));

      // conv3d and conv3d_transpose need padding check
      if (paddings.size() > 3) return false;

      if (desc.Input("Input").size() != 1) {
        VLOG(3) << "TRT Conv3d expect 1 input, but got "
                << desc.Input("Input").size() << " input.";
        return false;
      }

      if (desc.Input("Filter").size() != 1) {
        VLOG(3) << "TRT Conv3d expect 1 filter, but got "
                << desc.Input("Filter").size() << " filter.";
        return false;
      }

      if (op_type == "conv3d_transpose") {
        if (!desc.HasAttr("dilations")) {
          return false;
        } else {
          const std::vector<int> dilations =
              PADDLE_GET_CONST(std::vector<int>, desc.GetAttr("dilations"));
          if (dilations[0] != 1 || dilations[1] != 1 || dilations[2] != 1) {
            VLOG(3) << "In conv3d_transpose, Dilations must be (1, 1, 1) for "
                       "tensorRT, but given ("
                    << dilations[0] << ", " << dilations[1] << ", "
                    << dilations[2] << ")";
            return false;
          }
        }
      }

      if (desc.Output("Output").size() != 1) {
        VLOG(3) << "TRT Conv3d expect 1 output, but got "
                << desc.Output("Output").size() << " output.";
        return false;
      }
    }

    if (op_type == "cast") {
// trt 6015 result in Windows ppyolo_mbv3 TRT fp32 diff
#if !IS_TRT_VERSION_GE(7000)
      return false;
#endif
      if (!(desc.HasAttr("in_dtype") && desc.HasAttr("out_dtype"))) {
        VLOG(3) << "the " << op_type
                << " does not have attr (in_dtype or "
                   "out_dtype)";
        return false;
      }
      int in_dtype = PADDLE_GET_CONST(int, desc.GetAttr("in_dtype"));
      int out_dtype = PADDLE_GET_CONST(int, desc.GetAttr("out_dtype"));

      if (in_dtype == 0 || out_dtype == 0) {
#if IS_TRT_VERSION_GE(8400)
        if (with_dynamic_shape) {
          VLOG(3) << "the cast op supports inputs and outputs of BOOL by "
                     "trt8.4 above ";
          return true;
        }
#endif
        return false;
      }
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto x_var_name = desc.Input("X")[0];
      auto* x_var_desc = block->FindVarRecursive(x_var_name);
      const auto x_shape = x_var_desc->GetShape();
      if (!with_dynamic_shape && (x_shape.size() == 1 || x_shape.empty())) {
        VLOG(3) << op_type
                << " op does not support input's dim is 1 or 0 in tensorrt "
                   "static shape mode.";
        return false;
      }
    }

    if (op_type == "set_value") {
#if !IS_TRT_VERSION_GE(8200)
      return false;
#endif
      auto inputs = desc.Inputs();
      if (inputs.find("StartsTensorList") != inputs.end()) {
        if (!desc.Input("StartsTensorList").empty()) {
          return false;
        }
      }
      if (inputs.find("EndsTensorList") != inputs.end()) {
        if (!desc.Input("EndsTensorList").empty()) {
          return false;
        }
      }
      if (inputs.find("StepsTensorList") != inputs.end()) {
        if (!desc.Input("StepsTensorList").empty()) {
          return false;
        }
      }
      if (!(desc.HasAttr("axes") && desc.HasAttr("starts") &&
            desc.HasAttr("steps"))) {
        VLOG(3) << "the " << op_type
                << " does not have attr (axes or "
                   "starts or steps)";
        return false;
      }
      if (desc.HasAttr("axes")) {
        auto axes =
            PADDLE_GET_CONST(std::vector<int64_t>, desc.GetAttr("axes"));
        if (axes.size() != 1UL) {
          VLOG(3) << "the set_value op"
                  << "has more than one element in attribute axes, it can not "
                     "enter into trt.";
          return false;
        }
      }
    }

    if (op_type == "top_k_v2" || op_type == "top_k") {
      if (desc.HasAttr("axis")) {
        int axis = PADDLE_GET_CONST(int, desc.GetAttr("axis"));
        if (!with_dynamic_shape && axis == 0) {
          VLOG(3) << "top_k_v2 does not support axis == 0 in "
                     "tensorrt static shape.";
          return false;
        }
      }
      if (desc.HasAttr("sorted")) {
        bool sorted = PADDLE_GET_CONST(bool, desc.GetAttr("sorted"));
        if (!sorted) {
          VLOG(3) << op_type
                  << " does not support results not sorted in "
                     "tensorrt";
          return false;
        }
      }
    }

#if IS_TRT_VERSION_GE(8000)
    if (op_type == "sparse_fc" || op_type == "sparse_multihead_matmul") {
      if (!with_dynamic_shape) {
        VLOG(3) << "the sparse_fc and sparse_multihead_matmul does not support "
                   "static shape yet";
        return false;
      }
    }
#endif

    if (op_type == "equal" || op_type == "not_equal") {
#if !IS_TRT_VERSION_GE(8000)
      VLOG(3) << "equal is not supported when TensorRT < 8.0";
      return false;
#else
      // TRT does not support kEQUAL/kGREATER/kLESS work with implicit batch
      if (!with_dynamic_shape) {
        VLOG(3) << "the equal does not support "
                   "static shape yet";
        return false;
      }
      if (!desc.HasAttr("axis")) {
        return false;
      }
      int axis = PADDLE_GET_CONST(int, desc.GetAttr("axis"));
      if (axis == 0) {
        return false;
      }
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
#endif
    }

    if (op_type == "layernorm_shift_partition") {
      if (!with_dynamic_shape) {
        VLOG(3) << "the layernorm_shift_partition does not support "
                   "static shape yet";
        return false;
      }
    }

    if (op_type == "preln_layernorm_shift_partition") {
      if (!with_dynamic_shape) {
        VLOG(3) << "the layernorm_shift_partition does not support "
                   "static shape yet";
        return false;
      }
    }

    if (op_type == "merge_layernorm") {
      if (!with_dynamic_shape) {
        VLOG(3) << "The merge_layernorm op does not support "
                   "static shape yet";
        return false;
      }
    }

    if (op_type == "reverse_roll") {
      if (!with_dynamic_shape) {
        VLOG(3) << "The reverse roll fused op does not support static shape "
                   "mode yet.";
        return false;
      }
    }
    if (op_type == "skip_merge_layernorm") {
      if (!with_dynamic_shape) {
        VLOG(3) << "The merge_layernorm op does not support "
                   "static shape yet";
        return false;
      }
    }

    if (op_type == "skip_groupnorm_act") {
      if (!with_dynamic_shape) {
        VLOG(3) << "The skip_groupnorm_act op does not support "
                   "static shape yet";
        return false;
      }
    }

    if (op_type == "preln_groupnorm_act") {
      if (!with_dynamic_shape) {
        VLOG(3) << "The preln_groupnorm_act op does not support "
                   "static shape yet";
        return false;
      }
    }
    if (op_type == "trans_layernorm") {
      if (!with_dynamic_shape) {
        VLOG(3) << "The trans_layernorm op does not support "
                   "static shape yet";
        return false;
      }
    }
    if (op_type == "fuse_eleadd_transpose") {
      if (!with_dynamic_shape) {
        VLOG(3) << "The fuse_eleadd_transpose op does not support "
                   "static shape yet";
        return false;
      }
    }
    if (op_type == "lookup_table" || op_type == "lookup_table_v2") {
      if (!with_dynamic_shape) {
        VLOG(3) << "the lookup_table does not support "
                   "static shape yet";
        return false;
      }
    }

    if (op_type == "expand_as_v2" || op_type == "expand_v2") {
      if (!with_dynamic_shape) {
        VLOG(3) << "the " << op_type
                << "does not support "
                   "static shape yet";
        return false;
      }

      auto inputs = desc.Inputs();
      if (op_type == "expand_as_v2") {
        if (!desc.HasAttr("target_shape") && inputs.find("Y") == inputs.end()) {
          VLOG(3)
              << "expand_as_v2 op need have input(Y) or attr(target_shape). ";
          return false;
        }
      } else if (op_type == "expand_v2") {
        if (!desc.HasAttr("shape") && inputs.find("Shape") == inputs.end() &&
            inputs.find("expand_shapes_tensor") == inputs.end()) {
          VLOG(3) << "expand_v2 op need have input(Shape) or "
                     "input(expand_shapes_tensor) or attr(shape) . ";
          return false;
        }
      }

      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }

#if IS_TRT_VERSION_LT(8000)
      auto x_var_name = desc.Input("X")[0];
      auto* x_var_desc = block->FindVarRecursive(x_var_name);
      const auto x_shape = x_var_desc->GetShape();
      if (x_shape.size() == 0) {
        return false;  // not supported 0 dim.
      }
#endif
    }

    if (op_type == "grid_sampler") {
#if !IS_TRT_VERSION_GE(8510)
      VLOG(3) << "grid_sampler is not supported when TensorRT < 8.5.1";
      return false;
#else
      if (!with_dynamic_shape) {
        VLOG(3) << "the grid_sampler does not support "
                   "static shape yet";
        return false;
      }

      if (!desc.HasAttr("mode") || !desc.HasAttr("padding_mode") ||
          !desc.HasAttr("align_corners")) {
        VLOG(3) << "grid_sampler need attributes : mode, padding_mode, "
                   "align_corners";
        return false;
      }

      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto input_name = desc.Input("X")[0];
      auto* input_desc = block->FindVarRecursive(input_name);
      const auto input_shape = input_desc->GetShape();

      auto grid_name = desc.Input("Grid")[0];
      auto* grid_desc = block->FindVarRecursive(grid_name);
      const auto grid_shape = grid_desc->GetShape();

      if (input_shape.size() != 4 || grid_shape.size() != 4) {
        VLOG(3) << "The input and grid tensors must be shape tensors of rank 4 "
                   "using TRT GridSample layer.";
        return false;
      }

#endif
    }

    if (op_type == "cumsum") {
#if !IS_TRT_VERSION_GE(7220)
      VLOG(3) << "cumsum is not supported when TensorRT < 7.2.2";
      return false;
#endif
      if (!with_dynamic_shape) {
        VLOG(3) << "the cumsum does not support "
                   "static shape yet";
        return false;
      }
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
    }

    if (op_type == "argsort") {
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      if (!desc.HasAttr("descending") || !desc.HasAttr("axis")) {
        VLOG(3) << op_type << " needs attributes: descending and axis.";
      }
      auto x_var_name = desc.Input("X")[0];
      auto* x_var_desc = block->FindVarRecursive(x_var_name);
      std::vector<int64_t> shape = x_var_desc->GetShape();
      int axis = PADDLE_GET_CONST(int, desc.GetAttr("axis"));
      if (axis < 0) {
        axis += shape.size();
      }
      if (shape[axis] > 3840 || shape[axis] < 0) {
        VLOG(3) << op_type << " shape[" << axis << "] = " << shape[axis]
                << " is invalid, it should less than 3840 and greater than "
                   "zero in TensorRT.";
        return false;
      }
    }

    if (op_type == "unbind") {
      if (!with_dynamic_shape) {
        VLOG(3) << "the unbind does not support "
                   "static shape yet";
        return false;
      }
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
    }

    if (op_type == "isnan_v2") {
      if (!with_dynamic_shape) {
        VLOG(3) << "the isnan_v2 does not support "
                   "static shape yet";
        return false;
      }
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
    }

    if (op_type == "p_norm") {
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      if (!(desc.HasAttr("asvector") && desc.HasAttr("axis") &&
            desc.HasAttr("porder") && desc.HasAttr("keepdim"))) {
        VLOG(3) << op_type << " op need attrs asvector, porder, axis, keepdim.";
        return false;
      }
      bool asvector = PADDLE_GET_CONST(bool, desc.GetAttr("asvector"));
      int axis = PADDLE_GET_CONST(int, desc.GetAttr("axis"));
      float porder = PADDLE_GET_CONST(float, desc.GetAttr("porder"));
      if (asvector || porder != 2.0f || axis != -1) {
        VLOG(3) << op_type
                << " op only support asvector=False, porder=2, axis = -1.";
        return false;
      }
    }

    if (op_type == "index_put") {
#if IS_TRT_VERSION_LT(8510)
      VLOG(3) << "index_put is not supported when TensorRT < 8.5.1";
      return false;
#endif
      if (!with_dynamic_shape) {
        VLOG(3) << "the index_put does not support "
                   "static shape yet";
        return false;
      }
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto value_var_name = desc.Input("value")[0];
      auto* value_var_desc = block->FindVarRecursive(value_var_name);
      const auto value_shape = value_var_desc->GetShape();
      int value_num = std::accumulate(
          value_shape.begin(), value_shape.end(), 1, std::multiplies<int>());
      if (value_num != 1) {
        VLOG(3) << op_type << " op only support value_num = 1 in tensorrt.";
        return false;
      }
      auto indices_var_name = desc.Input("indices")[0];
      auto* indices_var_desc = block->FindVarRecursive(indices_var_name);
      auto dtype = indices_var_desc->GetDataType();
      if (dtype != framework::proto::VarType::BOOL) {
        VLOG(3) << op_type << " op only support bool indices in tensorrt.";
        return false;
      }
    }

    if (op_type == "temporal_shift") {
#if !IS_TRT_VERSION_GE(8200)
      VLOG(3) << "temporal_shift is not supported when TensorRT < 8.2";
      return false;
#endif

      if (!with_dynamic_shape) {
        VLOG(3) << "the temporal shift does not support "
                   "static shape yet";
        return false;
      }

      if (!desc.HasAttr("shift_ratio") || !desc.HasAttr("seg_num")) {
        VLOG(3) << "temporal shift need attributes : shift_ratio and seg_num";
        return false;
      }

      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }

      auto input_name = desc.Input("X")[0];
      auto* input_desc = block->FindVarRecursive(input_name);
      const auto input_shape = input_desc->GetShape();

      if (input_shape.size() != 4) {
        VLOG(3) << "The input and grid tensors must be shape tensors of rank 4 "
                   "using TRT TemporalShift layer.";
        return false;
      }
    }

    if (op_type == "einsum") {
#if !IS_TRT_VERSION_GE(8200)
      VLOG(3) << "einsum is not supported when TensorRT < 8.2";
      return false;
#else
      if (!with_dynamic_shape) {
        VLOG(3) << "the einsum does not support "
                   "static shape yet";
        return false;
      }
      auto operand_inputs = desc.Input("Operands");
      if (operand_inputs.size() > 2) {
        VLOG(3) << "TensorRT currently supports up to 2 input tensors"
                << "to einsum but operation had" << operand_inputs.size()
                << "input tensors !";
        return false;
      }

      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto equation = PADDLE_GET_CONST(std::string, desc.GetAttr("equation"));
      if (equation.find("...") != std::string::npos) {
        VLOG(3) << "TensorRT currently does not support ellipses !";
        return false;
      }
#endif
    }
    if (op_type == "quantize_linear" || op_type == "dequantize_linear") {
#if !IS_TRT_VERSION_GE(8000)
      VLOG(3) << "quantize / dequantize linear is not supported when TensorRT "
                 "< 8.0";
      return false;
#else
      return true;
#endif
    }

    if (op_type == "flip") {
      if (!with_dynamic_shape) {
        VLOG(3) << "the flip does not support "
                   "static shape yet";
        return false;
      }
#if !IS_TRT_VERSION_GE(7220)
      VLOG(3) << "flip is not supported when TensorRT below 7.2.2";
      return false;
#endif
    }

    if (use_no_calib_int8) {
      return int8_teller_set.count(op_type);
    } else {
      return teller_set.count(op_type);
    }
  }

 private:
  // use this set for no calib int8.
  std::unordered_set<std::string> int8_teller_set{
      "matrix_multiply",
      "bmm",
      "range",
      "conv2d",
      "fused_conv2d_add_act",
      "pool2d",
      "relu",
      "elu",
      "selu",
      "softsign",
      "softplus",
      "stanh",
      "thresholded_relu",
      "exp",
      "log",
      "sqrt",
      "abs",
      "sin",
      "cos",
      "tan",
      "sinh",
      "cosh",
      "asin",
      "acos",
      "atan",
      "asinh",
      "acosh",
      "atanh",
      "ceil",
      "floor",
      "rsqrt",
      "sign",
      "reciprocal",
      "logical_not",
      "erf",
      "square",
      "softmax",
      "sigmoid",
      "hard_swish",
      "depthwise_conv2d",
      "batch_norm",
      "concat",
      "tanh",
      "pad3d",
      "pad",
      "elementwise_add",
      "elementwise_sub",
      "elementwise_mul",
      "elementwise_div",
      "elementwise_pow",
      "elementwise_min",
      "elementwise_max",
      "elementwise_floordiv",
      "elementwise_mod",
      "equal",
      "not_equal",
      "less_than",
      "greater_than",
      "logical_or",
      "logical_xor",
      "logical_and",
      "less_equal",
      "greater_equal",
      "dropout",
      "fill_any_like",
      "prelu",
      "conv2d_transpose",
      "depthwise_conv2d_transpose",
      "leaky_relu",
      "shuffle_channel",
      "where",
      "bitwise_not",
      "one_hot",
      "one_hot_v2",
      "swish",
      "silu",
      "celu",
      "split",
      "instance_norm",
      "gelu",
      "layer_norm",
      "scale",
      "stack",
      "transpose2",
      "transpose",
      "top_k",
      "top_k_v2",
      "flatten2",
      "flatten",
      "gather",
      "gather_nd",
      "group_norm",
      "yolo_box",
      "yolo_box_head",
      "arg_max",
      "arg_min",
      "roi_align",
      "affine_channel",
      "nearest_interp",
      "anchor_generator",
      "reduce_max",
      "reduce_min",
      "reduce_mean",
      "reduce_sum",
      "reduce_prod",
      "reduce_any",
      "reduce_all",
      "conv3d",
      "conv3d_transpose",
      "mish",
      "nearest_interp_v2",
      "bilinear_interp_v2",
      "linear_interp_v2",
      "pool3d",
      "deformable_conv",
      "relu6",
      "hard_sigmoid",
      "clip",
      "prompt_tuning_emb_eltwise_layernorm",
      "fused_embedding_eltwise_layernorm",
      "multihead_matmul",
      "multihead_matmul_roformer",
      "skip_layernorm",
      "slice",
      "strided_slice",
      "fused_preln_embedding_eltwise_layernorm",
      "fused_bias_dropout_residual_layer_norm",
      "c_allreduce_sum",
      "c_allreduce_min",
      "c_allreduce_max",
      "c_allreduce_prod",
      "roll",
      "cast",
      "preln_skip_layernorm",
      "transformer_input_convert",
      "recover_padding",
      "remove_padding",
      "fill_constant",
      "sum",
      "shape",
      "squeeze2",
      "unsqueeze2",
      "index_put",
      "layernorm_shift_partition",
      "reverse_roll",
      "take_along_axis",
      "tanh_shrink",
      "logsigmoid",
      "preln_layernorm_shift_partition",
      "lookup_table",
      "lookup_table_v2",
      "trans_layernorm",
      "merge_layernorm",
      "skip_merge_layernorm",
      "expand_v2",
      "expand_as_v2",
      "fuse_eleadd_transpose",
      "skip_groupnorm_act",
      "preln_groupnorm_act",
      "temporal_shift",
      "grid_sampler",
      "cumsum",
      "unbind",
      "isnan_v2",
      "p_norm",
      "assign",
      "flip",
      "quantize_linear",
      "dequantize_linear",
      "share_data",
      "argsort",
      "bitwise_and",
      "bitwise_or",
      "size"};

  std::unordered_set<std::string> teller_set{
      "matrix_multiply",
      "bmm",
      "range",
      "conv2d",
      "fused_conv2d_add_act",
      "pool2d",
      "relu",
      "elu",
      "selu",
      "softsign",
      "softplus",
      "stanh",
      "thresholded_relu",
      "exp",
      "log",
      "sqrt",
      "abs",
      "sin",
      "cos",
      "tan",
      "sinh",
      "cosh",
      "asin",
      "acos",
      "atan",
      "asinh",
      "acosh",
      "atanh",
      "ceil",
      "floor",
      "rsqrt",
      "sign",
      "reciprocal",
      "logical_not",
      "erf",
      "square",
      "softmax",
      "sigmoid",
      "hard_swish",
      "depthwise_conv2d",
      "batch_norm",
      "concat",
      "tanh",
      "pad3d",
      "pad",
      "elementwise_add",
      "elementwise_sub",
      "elementwise_mul",
      "elementwise_div",
      "elementwise_pow",
      "pow",
      "elementwise_min",
      "elementwise_max",
      "elementwise_floordiv",
      "elementwise_mod",
      "equal",
      "not_equal",
      "less_than",
      "greater_than",
      "logical_or",
      "logical_xor",
      "logical_and",
      "less_equal",
      "greater_equal",
      "dropout",
      "fill_any_like",
      "prelu",
      "conv2d_transpose",
      "depthwise_conv2d_transpose",
      "leaky_relu",
      "shuffle_channel",
      "where",
      "bitwise_not",
      "one_hot",
      "one_hot_v2",
      "swish",
      "silu",
      "celu",
      "split",
      "instance_norm",
      "gelu",
      "layer_norm",
      "scale",
      "stack",
      "transpose2",
      "transpose",
      "top_k",
      "top_k_v2",
      "flatten2",
      "flatten",
      "gather",
      "gather_nd",
      "yolo_box",
      "yolo_box_head",
      "arg_max",
      "arg_min",
      "roi_align",
      "affine_channel",
      "nearest_interp",
      "anchor_generator",
      "reduce_max",
      "reduce_min",
      "reduce_mean",
      "reduce_sum",
      "reduce_prod",
      "reduce_any",
      "reduce_all",
      "conv3d",
      "conv3d_transpose",
      "mish",
      "bilinear_interp_v2",
      "linear_interp_v2",
      "nearest_interp_v2",
      "pool3d",
      "deformable_conv",
      "relu6",
      "hard_sigmoid",
      "clip",
      "prompt_tuning_emb_eltwise_layernorm",
      "fused_embedding_eltwise_layernorm",
      "multihead_matmul",
      "multihead_matmul_roformer",
      "skip_layernorm",
      "slice",
      "strided_slice",
      "fused_preln_embedding_eltwise_layernorm",
      "preln_skip_layernorm",
      "fused_bias_dropout_residual_layer_norm",
      "c_allreduce_sum",
      "c_allreduce_min",
      "c_allreduce_max",
      "c_allreduce_prod",
      "roll",
      "cast",
      "transformer_input_convert",
      "recover_padding",
      "remove_padding",
      "fill_constant",
      "sum",
      "shape",
      "squeeze2",
      "unsqueeze2",
      "fused_token_prune",
      "layernorm_shift_partition",
      "reverse_roll",
      "tanh_shrink",
      "index_put",
      "take_along_axis",
      "logsigmoid",
      "preln_layernorm_shift_partition",
      "trans_layernorm",
      "merge_layernorm",
      "skip_merge_layernorm",
      "lookup_table",
      "lookup_table_v2",
      "expand_v2",
      "expand_as_v2",
      "fuse_eleadd_transpose",
      "skip_groupnorm_act",
      "preln_groupnorm_act",
      "temporal_shift",
      "grid_sampler",
      "cumsum",
      "unbind",
      "isnan_v2",
      "p_norm",
      "assign",
      "flip",
      "quantize_linear",
      "dequantize_linear",
      "share_data",
      "argsort",
      "bitwise_and",
      "bitwise_or",
      "size"};
};

struct GenericPluginTeller : public Teller {
 public:
  GenericPluginTeller() = default;
  bool operator()(const framework::OpDesc& desc,
                  bool use_no_calib_int8 = false,
                  bool with_dynamic_shape = false,
                  bool forbid_dynamic_op_enter_into_trt = false,
                  bool use_explicit_quantization = false) override {
    const std::string op_type = desc.Type();

    // only consider dynamic_shape mode
    if (!with_dynamic_shape) {
      return false;
    }
    if (op_type == "yolo_box") {
      if (!desc.HasAttr("iou_aware") && !desc.HasAttr("iou_aware_factor"))
        return false;
    } else if (op_type == "solve") {
      auto x_var_name = desc.Input("X")[0];
      auto y_var_name = desc.Input("Y")[0];
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto* x_var_desc = block->FindVar(x_var_name);
      auto* y_var_desc = block->FindVar(y_var_name);
      auto x_dtype = x_var_desc->GetDataType();
      auto y_dtype = y_var_desc->GetDataType();
      if (x_dtype == framework::proto::VarType::FP64 ||
          y_dtype == framework::proto::VarType::FP64) {
        VLOG(3) << op_type << " not support input of FP64.";
        return false;
      }
    }
    // TODO(lizexu123): the tensorrt version on Windows supports low-level
    // and inconsistent supportformation
    if (op_type == "argsort") {
      if (!with_dynamic_shape) {
        VLOG(3) << "Ops(" << op_type << ") do not support static shape yet.";
        return false;
      }
    }
    if (use_no_calib_int8) {
      return false;
    } else {
      framework::InitDefaultKernelSignatureMap();
      bool res = phi::OpUtilsMap::Instance().HasArgumentMappingFn(op_type) ||
                 phi::DefaultKernelSignatureMap::Instance().Has(op_type);
      if (!res) {
        VLOG(3) << op_type << " has no KernelSignature";
        return false;
      }
      res = phi::KernelFactory::Instance().HasCompatiblePhiKernel(op_type);
      if (!res) {
        VLOG(3) << op_type << " has no CompatiblePhiKernel in phi.";
        return false;
      }
      auto& dynamic_infermeta_factory =
          tensorrt::DynamicMetaFnFactory::Instance();
      res = dynamic_infermeta_factory.Contains(op_type);
      if (!res) {
        VLOG(3) << op_type << " has no DynamicMetaFn.";
        return false;
      }
      if (forbid_dynamic_op_enter_into_trt && IsDynamicShapeOp(desc)) {
        return false;
      }
      return true;
    }
  }
};

struct CustomPluginTeller : public Teller {
 public:
  CustomPluginTeller() = default;
  bool operator()(const framework::OpDesc& desc,
                  bool use_no_calib_int8 = false,
                  bool with_dynamic_shape = false,
                  bool forbid_dynamic_op_enter_into_trt = false,
                  bool use_explicit_quantization = false) override {
    const std::string op_type = desc.Type();
    std::string expect_plugin_name;

    if (with_dynamic_shape) {
      expect_plugin_name = op_type + "_paddle_trt_dynamic_plugin";
    } else {
      expect_plugin_name = op_type + "_paddle_trt_plugin";
    }

    int num = 0;
    auto creators = GetPluginRegistry()->getPluginCreatorList(&num);

    for (int i = 0; i < num; i++) {
      if (std::string(creators[i]->getPluginName()) == expect_plugin_name)
        return true;
    }
    return false;
    if (forbid_dynamic_op_enter_into_trt && IsDynamicShapeOp(desc)) {
      return false;
    }
  }
};

struct CustomGenericPluginTeller : public Teller {
  CustomGenericPluginTeller() = default;
  bool operator()(const framework::OpDesc& desc,
                  bool use_no_calib_int8 = false,
                  bool with_dynamic_shape = false,
                  bool forbid_dynamic_op_enter_into_trt = false,
                  bool use_explicit_quantization = false) override {
    const std::string op_type = desc.Type();

    auto& op_meta_info_map = OpMetaInfoMap::Instance();
    const auto& meta_info_map = op_meta_info_map.GetMap();
    if (meta_info_map.count(op_type) > 0) {
      auto& op_info = meta_info_map.at(op_type).front();
      auto& trt_infer_shape_fn = OpMetaInfoHelper::GetTrtInferShapeFn(op_info);
      if (trt_infer_shape_fn == nullptr) {
        VLOG(3) << op_type
                << " has no trt getOutputDimensions function. Please set by "
                   "SetTrtInferShapeFn.";
        return false;
      }
      auto& trt_supports_format_config =
          OpMetaInfoHelper::GetTrtSupportsFormatConfig(op_info);
      if (trt_supports_format_config.empty()) {
        VLOG(3)
            << op_type
            << " has no trt supportsFormatCombination config. Please set by "
               "SetTrtSupportsFormatConfig.";
        return false;
      }
      return true;
    }
    VLOG(3) << op_type << " has no meta info";
    return false;
    if (forbid_dynamic_op_enter_into_trt && IsDynamicShapeOp(desc)) {
      return false;
    }
  }
};

bool OpTeller::Tell(const framework::ir::Node* node,
                    bool use_no_calib_int8,
                    bool with_dynamic_shape,
                    bool forbid_dynamic_op_enter_into_trt,
                    bool use_explicit_quantization) {
  const std::string op_type = node->Op()->Type();
  const framework::OpDesc desc = *node->Op();

  // do not support the op which is labeled the `skip_quant`
  if ((desc.HasAttr("namescope") &&
       PADDLE_GET_CONST(std::string, desc.GetAttr("op_namescope")) ==
           "/skip_quant_2/") ||
      desc.HasAttr("skip_quant"))
    return false;
  auto& default_teller = GetDefaultTeller();
  if ((*default_teller)(desc,
                        use_no_calib_int8,
                        with_dynamic_shape,
                        forbid_dynamic_op_enter_into_trt,
                        use_explicit_quantization)) {
    SetOpConverterType(node->Op(), OpConverterType::Default);
    return true;
  }
  auto& generic_plugin_teller = GetGenericPluginTeller();
  if ((*generic_plugin_teller)(desc,
                               use_no_calib_int8,
                               with_dynamic_shape,
                               forbid_dynamic_op_enter_into_trt,
                               use_explicit_quantization)) {
    SetOpConverterType(node->Op(), OpConverterType::GenericPluginCreator);
    return true;
  }
  auto& custom_plugin_teller = GetCustomPluginTeller();
  if ((*custom_plugin_teller)(desc,
                              use_no_calib_int8,
                              with_dynamic_shape,
                              forbid_dynamic_op_enter_into_trt,
                              use_explicit_quantization)) {
    SetOpConverterType(
        node->Op(),
        OpConverterType::CustomPluginCreater);  // typos: disable-line
    return true;
  }
  auto& custom_generic_plugin_teller = GetCustomGenericPluginTeller();
  if ((*custom_generic_plugin_teller)(desc,
                                      use_no_calib_int8,
                                      with_dynamic_shape,
                                      forbid_dynamic_op_enter_into_trt,
                                      use_explicit_quantization)) {
    SetOpConverterType(node->Op(), OpConverterType::CustomGenericPluginCreator);
    return true;
  }
  return false;
}

OpTeller::OpTeller() {  // NOLINT
  tellers_.emplace_back(new tensorrt::SimpleOpTypeSetTeller);
  tellers_.emplace_back(new tensorrt::GenericPluginTeller);
  tellers_.emplace_back(new tensorrt::CustomPluginTeller);
  tellers_.emplace_back(new tensorrt::CustomGenericPluginTeller);
}

}  // namespace paddle::inference::tensorrt
