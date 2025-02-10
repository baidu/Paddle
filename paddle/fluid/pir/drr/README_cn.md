# DRR( Declarative Rewrite Rule) PASS用户使用手册
---
## 1. 相关背景

PASS 是对 IR 进行优化的关键组件，而 DAG-to-DAG 的变换（将原图中的一个 DAG 子图替换成另一个 DAG 子图）是最常见的Pass类型。DAG-to-DAG 的变换可以划分为匹配和重写两个步骤：匹配是根据已知子图在 Program 中完全匹配到对应的目标子图，重写是将匹配到的图结构替换为新的子图。

DRR ( Declarative Rewrite Rule ) 是来处理这种 DAG-to-DAG 类型的一套 PASS 组件。DRR 能降低 PASS 的开发成本，让开发者集中在对优化逻辑的处理上，而不需要关心底层 IR 的数据结构。开发者通过一套简洁易用的接口对目标子图和需要替换成的新子图进行模式声明后，DRR 就能自动的在 Program 中对原图进行匹配，并替换成新子图。

以消除冗余 CastOp 的 PASS 为例，使用 DRR 的代码开发示例如下：
~~~ c++
// 1. 继承 DrrPatternBase 类
class RemoveRedundantCastPattern : public paddle::drr::DrrPatternBase {
public:
	std::string name() const override { return "RemoveRedundantCastPattern"; }

  // 2. 重载 operator()
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    // 3. 使用 Op、Tensor 和 Attribute 定义一个包含两个连续 CastOp 的 SourcePattern
    auto pat = ctx->SourcePattern();

    pat.Tensor("tmp") =                          // CastOp 输出 Tensor 命名为"tmp"
        pat.Op(paddle::dialect::CastOp::name(),  // 传入 CastOp 的 name
               {{"dtype", pat.Attr("dtype1")}})  // CastOp 的"dtype"属性的对应的全局唯一ID为"dtype1"
               (pat.Tensor("arg0"));             // CastOp 输入 Tensor 为"arg0"
    pat.Tensor("ret") =
        pat.Op(paddle::dialect::CastOp::name(),
               {{"dtype", pat.Attr("dtype2")}})(pat.Tensor("tmp"));
    // 4. 定义 Constraint
    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto ret_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("ret"));
      auto arg0_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("tmp"));
      return ret_dtype == arg0_dtype;
    });

    // 5. 定义 ResultPattern
    auto res = pat.ResultPattern();
    res.Tensor("ret") =
        res.Op(paddle::dialect::CastOp::name(),
               {{"dtype", pat.Attr("dtype2")}})(res.Tensor("arg0"));
  }
};
~~~

DRR PASS 包含以下三个部分：
+ `SourcePattern`：用于描述在 Program 中待匹配的目标子图
+ `Constraints`：用于指定`SourcePattern`匹配的限制条件（非必需）
+ `ResultPattern`：用于描述需要替换为的模式子图
开发者只需要定义出`SourcePattern`, `Constraints`和`ResultPattern`即可实现一个完整的 PASS。

**注意：**
1. **DRR 仅支持对闭包（除 Pattern 输入输出 Tensor 以外，所有的内部 Tensor 不能被 Pattern 外部 Op 使用）的 SourcePattern 和 ResultPattern 进行匹配替换，若定义的 Pattern 在 Program 中不闭包则匹配失败**
2. **ResultPattern 的输入输出需要满足是 SourcePattern 的输入输出的子集**
## 2. 接口列表

<table>
	 <tr>
		<th> 类 </th>
		<th> 函数 </th>
		<th> 功能描述 </th>
		<th> 参数解释 </th>
	 </tr>
	<tr>
		<td rowspan="1">DrrPatternBase</td>
		<td> <pre> virtual void operator()(
        paddle::drr::DrrPatternContext* ctx) const </pre></td>
		<td> 实现 DRR PASS 的入口函数 </td>
		<td> ctx: 创建 Pattern 所需要的 Context 参数</td>
	</tr>
	<tr>
		<td rowspan="6"> SourcePattern</td>
		<td><pre> const drr::Op& Op(
    const std::string& op_type,
    const std::unordered_map&lt;std::string, Attribute&gt;& attributes)</pre></td>
		<td> 在 SourcePattern 中定义一个 Op</td>
		<td> op_type: 定义的 Op 名称，可以通过 paddle::dialect::xxOp
	::name() 接口获取 <br> attributes : 所创建的 Op 的属性信息 </td>
	</tr>
	<tr>
		<td><pre> const drr::Tensor& Tensor(
        const std::string& tensor_name) </pre></td>
		<td> 在 SourcePattern 中定义一个名为 tensor_name 的 tensor</td>
		<td>  tensor_name: 定义的 Tensor 的名称，需要满足 SourcePattern 内唯一 </td>
	</tr>
	<tr>
		<td> <pre> Attribute Attr(
        const std::string& attr_name) const </pre></td>
		<td> 在 SourcePattern 中定义一个名为 attr_name 的属性 </td>
		<td> attr_name: 属性的名称，需要满足 SourcePattern 内唯一 </td>
	</tr>
	<tr>
		<tr>
	<tr>
		<td> <pre>void AddConstraint(
        const std::function&lt;bool(const MatchContext&)&gt;& constraint_fn)</pre></td>
		<td> 在 SourcePattern 中定义一个约束，可以利用此接口和 lambda 表达式实现对 SourcePattern 的自定义约束</td>
		<td> constraint_fn: 自定义的约束函数</td>
	</tr>
	<tr>
		<td rowspan="5"> ResultPattern</td>
				<td><pre> const drr::Op& Op(
    const std::string& op_type,
    const std::unordered_map&lt;std::string, Attribute&gt;&  attributes) </pre></td>
		<td> 在ResultPattern中定义一个Op </td>
		<td> op_type: 定义的 Op 名称，可以通过 paddle::dialect::xxOp
	::name() 接口获取<br> attributes : 所创建的 Op 的属性信息 </td>
	</tr>
	<tr>
		<td> <pre>const drr::Tensor& Tensor(
        const std::string& tensor_name)</pre></td>
		<td> 在 ResultPattern 中定义一个名为 tensor_name 的 tensor</td>
		<td> tensor_name: 定义的 Tensor 的名称，需要满足 ResultPattern 内唯一 </td>
	</tr>
	<tr>
		<td><pre>Attribute Attr(
        const std::string& attr_name) const </pre></td>
		<td> 在 ResultPattern 中定义一个名为 attr_name 的属性 </td>
		<td> attr_name: 属性的名称，需要满足 ResultPattern 内唯一 </td>
	</tr>
<tr>
		<td><pre>using AttrComputeFunc = std::function&lt;std::any(const MatchContext&)&gt;;
Attribute Attr(const AttrComputeFunc& attr_compute_func) const</pre></td>
		<td> 通过自定义的计算逻辑 AttrComputeFunc，创建出一个 Attribute</td>
		<td>attr_compute_func: 自定义的计算逻辑</td>
	</tr>
	<tr>
		<td> <pre>drr::Tensor& InputNoneTensor()</pre></td>
		<td> 当一个 Op 的输入 Tensor 是一个可选项并且不需要时，需要使用 InputNoneTensor 来占位</td>
		<td> / </td>
	</tr>
	<tr>
		<td rowspan="1"> DrrPatternContext</td>
		<td><pre>drr::SourcePattern DrrPatternContext::SourcePattern()</pre> </td>
		<td> 创建一个 SourcePattern 对象，并返回 </td>
		<td> / </td>
	</tr>
</table>

## 3 使用示例
Example 1: Matmul + Add -> FusedGemmEpilogue
~~~ c++
class FusedLinearPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "FusedLinearPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    // 定义 Source Pattern
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul = pat.Op(paddle::dialect::MatmulOp::name(),
                                {{"transpose_x", pat.Attr("trans_x")},
                                 {"transpose_y", pat.Attr("trans_y")}});
    const auto &add = pat.Op(paddle::dialect::AddOp::name());

    pat.Tensor("tmp") = matmul(pat.Tensor("x"), pat.Tensor("w"));
    pat.Tensor("out") = add(pat.Tensor("tmp"), pat.Tensor("bias"));

    // 定义 Result Pattern
    paddle::drr::ResultPattern res = pat.ResultPattern();
    // 定义 Constraint
    const auto &fused_gemm_epilogue = res.Op(paddle::dialect::FusedGemmEpilogueOp::name(),
                                             {{{"trans_x", pat.Attr("trans_x")},
                                               {"trans_y", pat.Attr("trans_y")},
                                               {"activation", res.StrAttr("none")}}});
    fused_gemm_epilogue(
        {&res.Tensor("x"), &res.Tensor("w"), &res.Tensor("bias")},
        {&res.Tensor("out")});
  }
};
~~~

Example 2: Full + Expand -> Full
~~~ c++
class FoldExpandToConstantPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "FoldExpandToConstantPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    // 定义 Source Pattern
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &full1 = pat.Op(paddle::dialect::FullOp::name(),
                               {{"shape", pat.Attr("shape_1")},
                                {"value", pat.Attr("value_1")},
                                {"dtype", pat.Attr("dtype_1")},
                                {"place", pat.Attr("place_1")}});
    const auto &full_int_array1 =
        pat.Op(paddle::dialect::FullIntArrayOp::name(),
               {{"value", pat.Attr("expand_shape_value")},
                {"dtype", pat.Attr("dtype_2")},
                {"place", pat.Attr("place_2")}});
    const auto &expand = pat.Op(paddle::dialect::ExpandOp::name());
    pat.Tensor("ret") = expand(full1(), full_int_array1());

    // 定义 Result Pattern      Constraints: 本 Pass 无额外约束规则
    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &full2 = res.Op(paddle::dialect::FullOp::name(),
                               {{"shape", pat.Attr("expand_shape_value")},
                                {"value", pat.Attr("value_1")},
                                {"dtype", pat.Attr("dtype_1")},
                                {"place", pat.Attr("place_1")}});
    res.Tensor("ret") = full2();
  }
};
~~~
