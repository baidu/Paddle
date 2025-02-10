# DRR (Declarative Rewrite Rule) Tool User Manual
---
## 1. Related Background

PASS is a crucial component for optimizing intermediate representations (IR), and the transformation of DAG-to-DAG (Replace a subgraph of the directed acyclic graph (DAG) type in the original graph with another subgraph) is the most common type of Pass. The transformation of DAG-to-DAG can be divided into two steps: matching and rewriting. Matching refers to the complete matching of a known subgraph to the corresponding target subgraph in the Program, while rewriting refers to replacing the matched graph with a new subgraph.

DRR can reduce the development cost of PASS, allowing developers to focus on processing optimization logic without caring about the data structure of the underlying IR. After the developer declares the pattern of the target subgraph and the new subgraph to be replaced through a set of simple and easy-to-use interfaces, DRR can automatically match the original subgraph in the Program and replace it with the new subgraph.

Taking PASS to eliminate redundant CastOp as an example, the code example developed using DRR is as follows:
~~~ c++
// 1. Inherit class from DrPatternBase
class RemoveRedundantCastPattern : public paddle::drr::DrrPatternBase {
public:
  std::string name() const override { return "RemoveRedundantCastPattern"; }

  // 2. Overload operator()
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    // 3. Define a SourcePattern containing two consecutive CastOps using Op, Tensor, and Attribute
    auto pat = ctx->SourcePattern();

    pat.Tensor("tmp") =                          // CastOp output Tensor named "tmp"
        pat.Op(paddle::dialect::CastOp::name(),  // Pass in the name of the CastOp
               {{"dtype", pat.Attr("dtype1")}})  // The corresponding globally unique ID of the "dtype" attribute of CastOp is "dtype1"
               (pat.Tensor("arg0"));             // The input Tensor of CastOp is "arg0"
    pat.Tensor("ret") =
        pat.Op(paddle::dialect::CastOp::name(),
               {{"dtype", pat.Attr("dtype2")}})(pat.Tensor("tmp"));
    // 4. Define Constrain
    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto ret_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("ret"));
      auto arg0_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("tmp"));
      return ret_dtype == arg0_dtype;
    });

    // 5. Define ResultPattern
    auto res = pat.ResultPattern();
    res.Tensor("ret") =
        res.Op(paddle::dialect::CastOp::name(),
               {{"dtype", pat.Attr("dtype2")}})(res.Tensor("arg0"));
  }
};
~~~

DRR PASS contains the following three parts:
+ `Source Pattern`：used to describe the target subgraph to be matched in Program
+  `Constraints`：used to specify constraints for SourcePattern matching(nonessential)
+ `Result Pattern`：Used to describe the subgraph that needs to be replaced by
Developers only need to define `SourcePattern`, `Constraints` and `ResultPattern` to implement a complete PASS.

**Note:**
1. **DRR only supports matching and replacing the closed SourcePattern and ResultPattern (except for the Pattern input and output Tensor, all internal Tensors cannot be used by the Pattern external Op). If the defined Pattern is not closed in the Program, the matching will fail.**
2. **The input and output of ResultPattern need to be a subset of the input and output of SourcePattern.**
## 2. Interface List
<table>
	 <tr>
		<th> Class </th>
		<th> Function </th>
		<th> Function Description </th>
		<th> Parameter Interpretation </th>
	 </tr>
	<tr>
		<td rowspan="1">DrrPatternBase</td>
		<td> <pre> virtual void operator()(
        paddle::drr::DrrPatternContext* ctx) const </pre></td>
		<td> Implement the entry function of DRR PASS </td>
		<td> ctx: Context parameters required to create Pattern</td>
	</tr>
	<tr>
		<td rowspan="6"> SourcePattern</td>
		<td><pre> const drr::Op& Op(
    const std::string& op_type,
    const std::unordered_map&lt;std::string, Attribute&gt;& attributes)</pre></td>
		<td> Define an Op in the SourcePattern</td>
		<td> op_type: The defined Op name. Can be obtained through paddle::dialect::xxOp::name() interface <br> attributes : Attribute information of the created Op </td>
	</tr>
	<tr>
		<td><pre> const drr::Tensor& Tensor(
        const std::string& tensor_name) </pre></td>
		<td> Define a tensor named tensor_name in SourcePattern</td>
		<td>  tensor_name: The name of the defined Tensor must be unique within the SourcePattern </td>
	</tr>
	<tr>
		<td> <pre> Attribute Attr(
        const std::string& attr_name) const </pre></td>
		<td> Define an attribute named attr_name in SourcePattern</td>
		<td> attr_name: The name of the attribute, which needs to be unique within SourcePattern </td>
	</tr>
	<tr>
		<tr>
	<tr>
		<td> <pre>void AddConstraint(
        const std::function&lt;bool(const MatchContext&)&gt;& constraint_fn)</pre></td>
		<td> Define a constraint in SourcePattern. You can use this interface and lambda expressions to implement custom constraints on SourcePattern.</td>
		<td> constraint_fn: Customized constraint functions</td>
	</tr>
	<tr>
		<td rowspan="5"> ResultPattern</td>
				<td><pre> const drr::Op& Op(
    const std::string& op_type,
    const std::unordered_map&lt;std::string, Attribute&gt;&  attributes) </pre></td>
		<td> Define an Op in ResultPattern </td>
		<td> op_type: The defined Op name. Can be obtained through paddle::dialect::xxOp::name() interface<br> attributes : Attribute information of the created Op </td>
	</tr>
	<tr>
		<td> <pre>const drr::Tensor& Tensor(
        const std::string& tensor_name)</pre></td>
		<td> Define a tensor named tensor_name in ResultPattern</td>
		<td> tensor_name: The name of the defined Tensor must be unique within the ResultPattern </td>
	</tr>
	<tr>
		<td><pre>Attribute Attr(
        const std::string& attr_name) const </pre></td>
		<td> Define an attribute named attr_name in ResultPattern </td>
		<td> attr_name: The name of the attribute must be unique within the ResultPattern </td>
	</tr>
<tr>
		<td><pre>using AttrComputeFunc = std::function&lt;std::any(const MatchContext&)&gt;;
Attribute Attr(const AttrComputeFunc& attr_compute_func) const</pre></td>
		<td> Create an Attribute through a custom calculation logic AttrComputeFunc</td>
		<td>attr_compute_func: Customized calculation logic</td>
	</tr>
	<tr>
		<td> <pre>drr::Tensor& InputNoneTensor()</pre></td>
		<td> When the input Tensor of an Op is optional and not needed, InputNoneTensor needs to be used to occupy the place.</td>
		<td> / </td>
	</tr>
	<tr>
		<td rowspan="1"> DrrPatternContext</td>
		<td><pre>drr::SourcePattern DrrPatternContext::SourcePattern()</pre> </td>
		<td> Create a SourcePattern object and return </td>
		<td> / </td>
	</tr>
</table>

## 3 Example
Example 1: Matmul + Add -> FusedGemmEpilogue
~~~ c++
class FusedLinearPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "FusedLinearPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
	// Define SourcePattern
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul = pat.Op(paddle::dialect::MatmulOp::name(),
                                {{"transpose_x", pat.Attr("trans_x")},
                                 {"transpose_y", pat.Attr("trans_y")}});
    const auto &add = pat.Op(paddle::dialect::AddOp::name());

    pat.Tensor("tmp") = matmul(pat.Tensor("x"), pat.Tensor("w"));
    pat.Tensor("out") = add(pat.Tensor("tmp"), pat.Tensor("bias"));

    // Define ResultPattern
    paddle::drr::ResultPattern res = pat.ResultPattern();
    // Define Constrain
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
    // Define SourcePattern
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

    // Define ResultPattern
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
