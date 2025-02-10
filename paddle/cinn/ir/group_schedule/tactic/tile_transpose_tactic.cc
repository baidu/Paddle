// Copyright (c) 2025 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/group_schedule/tactic/tile_transpose_tactic.h"
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"

PD_DECLARE_bool(cinn_enable_tile_transpose);

namespace cinn {
namespace ir {
namespace {

/**
 * Tiling template for Elementwise+Transpose fusion graph.
 *
 * This tactic accelerates transposes by doing inner-block transpose in shared
 * memory, making both reads and writes coaleased, therefore achieving nearly
 * copy-like throughput.
 *
 * This tactic literally supports any permutation, as long as the last dimension
 * is permuted. If the last dimension is consistent, the general tactic has been
 * good. However, this tactic has limitations in the fusion:
 * 1) Reduce is not supported, because reduce usually requires a larger inner
 *    loop (~32) for better performance, while transpose prefers a smaller inner
 *    loop (~4) to restrict the shared memory size.
 * 2) All transposes in the graph must have the same permutation, because the
 *    size of the shared memory we need is 32^(n+1), where n is the number of
 *    different permutations. More than one permutation makes it impossible to
 *    allocate such a huge space.
 *
 *
 * How does this tactic work:
 *
 * First, we generalize a transpose `src => dst` as:
 *    src [ ..., dst_low_axis, ..., src_low_axis ]
 *    dst [ ..., src_low_axis, ..., dst_low_axis ]
 * The rest `...` are called high_axis, which may contain any permutation, and
 * can be transposed by simple index mapping without impacting performance.
 *
 * Second, we split both src_low_axis and dst_low_axis into {-1, 32}:
 *    src [ ..., d_h, d32, ... s_h, s32 ]
 *    dst [ ..., s_h, s32, ... d_h, d32 ]
 *
 * Third, we create a shared memory of shape [32, 32], and bind (s32, d32) to
 * (thread.y, thread.x) to transpose them in the shared memory. The (s_h, d_h)
 * also become high_axis. We transpose high_axis using (block.y, block.x).
 *    src [ block.x, thread.y, block.y, thread.x ]
 *                        \              /
 *                          \          /
 *                 shm [ thread.y, thread.x ]  (write cache)
 *                            |      |
 *                 shm [ thread.x, thread.y ]  (read cache)
 *                               \ /
 *                          _____/ \_____
 *                         /             \
 *    dst [ block.y, thread.y, block.x, thread.x ]
 *
 * Finally, the IR is like:
 *    shm[thread.y][thread.x] = src[block.x, thread.y, block.y, thread.x]
 *    __syncthreads()
 *    dst[block.y, thread.y, block.x, thread.x] = shm[thread.x][thread.y]
 *
 * Notes:
 * 1) For simplicity, the high_axis are actually all bound to block.x.
 * 2) For performance, thread.y is actually composed of 4 loops * 8 threads.
 * 3) To support multiple transpose inputs, we actually store the transposed
 *    value to a local buffer before later computation, so that all inputs can
 *    reuse the same shared memory.
 */
class TileTransposeTactic final : public ScheduleTactic {
 public:
  void Init(ScheduleContext* context, ir::IRSchedule* sch) override;

  void Apply(ir::IRSchedule* sch, const std::string& block_id) override;

  std::string TacticName() const override { return "TileTransposeTactic"; }

 private:
  struct Candidate {
    // The target load to do CacheRead.
    ir::Expr load;

    // The block where this load first appears. We will do CacheRead on this
    // block, and later blocks will simply reuse the first load's value.
    std::string first_appear_block_id;

    // The buffer index of this load in the first block it appears.
    int buffer_index;
  };

  void InitCandidates(ir::IRSchedule* sch);
  void InitUnconditionalLoads(ir::IRSchedule* sch);
  void InitAxisInfo();

  std::vector<int> GetSrcLowAxis(const std::vector<int>& perm);
  std::vector<int> GetDstLowAxis(const std::vector<int>& perm);

  std::string CreateCacheBlock(ir::IRSchedule* sch,
                               const std::string& block_id,
                               int buffer_index,
                               const std::string& memory_type);
  void TileCacheBlock(ir::IRSchedule* sch,
                      const std::string& block_id,
                      int buffer_index);
  void TileBlock(ir::IRSchedule* sch, const std::string& block_id);
  void CanonicalizeLayout(ir::IRSchedule* sch, const std::string& block_id);
  void FuseAndBind(ir::IRSchedule* sch,
                   const std::string& block_id,
                   bool need_sync = false);

 private:
  ScheduleContext* context_;
  bool can_apply_;

  // The common permutation of all transposes in the graph.
  std::vector<int> common_perm_;

  // Groups of axis as illustrated in the above graph.
  std::vector<int> high_axis_;
  std::vector<int> src_low_axis_;
  std::vector<int> dst_low_axis_;

  struct LoadHash {
    size_t operator()(const ir::Expr& load) const {
      auto& tensor_name = load.As<ir::Load>()->tensor.as_tensor()->name;
      return std::hash<std::string>()(tensor_name);
    }
  };

  // Map from each candidate load to the corresponding Candidate struct.
  // Note: the same tensor name doesn't necessarily refers to the same load,
  // because the load indices may differ. Therefore, we hash loads by their
  // tensor names but check equality by their indices.
  std::unordered_map<ir::Expr, Candidate, LoadHash> load2candidates_;

  // Map from each block's id to the candidates in the block.
  // Note: this map records all possible candidates for a block, including
  // candidates whose first appearing block are not the block.
  std::unordered_map<std::string, std::vector<Candidate>> block2candidates_;

  // Candidate loads that have been cache-read and tiled.
  std::unordered_set<ir::Expr, LoadHash> processed_loads_;

  // Loads that are executed unconditionally (not inside Select).
  std::unordered_set<ir::Expr, LoadHash> unconditional_loads_;
};

std::vector<int> GetTransposePerm(const std::vector<ir::Expr>& indices,
                                  int data_rank) {
  if (indices.size() != data_rank) return {};
  std::vector<int> perm(data_rank);
  for (int i = 0; i < data_rank; ++i) {
    if (!indices[i].is_var()) return {};
    auto* loop_var = indices[i].as_var();
    // Strip the prefix "loop_var_" to get the loop_index.
    int loop_index =
        std::stoi(loop_var->name.substr(strlen(ir::analyzer::kLoopVar)));
    perm[loop_index] = i;
  }
  return perm;
}

std::vector<int> OffsetVec(const std::vector<int>& vec, int offset) {
  std::vector<int> new_vec = vec;
  for (auto& e : new_vec) e += offset;
  return new_vec;
}

std::vector<int> ArangeVec(int count, int begin = 0) {
  std::vector<int> vec(count);
  std::iota(vec.begin(), vec.end(), begin);
  return vec;
}

int64_t GetLoopRangeProduct(const std::vector<ir::Expr>& loops,
                            const std::vector<int>& loops_index) {
  int64_t prod = 1;
  for (int i : loops_index) {
    auto* node = loops[i].As<ir::For>();
    if (!node->extent.is_constant()) return -1;
    prod *= node->extent.as_int64();
  }
  return prod;
}

void TileTransposeTactic::Init(ScheduleContext* context, ir::IRSchedule* sch) {
  context_ = context;
  can_apply_ = false;
  if (!FLAGS_cinn_enable_tile_transpose) return;

  ir::Expr module_root = sch->GetModule().GetExprs().front();
  ir::Expr root_block = ir::analyzer::GetRootSBlock(module_root);
  auto* root_node = root_block.As<ir::ScheduleBlockRealize>()
                        ->schedule_block.As<ir::ScheduleBlock>();

  if (root_node->attrs.count(kTileMethod) > 0) return;
  if (!context->config.base_info->reduce_axis.empty()) return;

  // There must be at least 8 warps (256 threads) to perform this tactic.
  if (context->config.tile_config.warp_num < 8) return;

  InitUnconditionalLoads(sch);
  InitCandidates(sch);

  VLOG(4) << "Common permutation: " << utils::Join(common_perm_, ", ");
  if (common_perm_.empty()) return;

  can_apply_ = true;
  root_node->attrs[kTileMethod] = TacticName();

  InitAxisInfo();
}

void TileTransposeTactic::InitUnconditionalLoads(ir::IRSchedule* sch) {
  struct Collector : public ir::IRMutator<> {
    void operator()(ir::Expr* expr) { IRMutator<>::Visit(expr, expr); }

    void Visit(const ir::Select* op, ir::Expr* expr) override {
      auto* select = expr->As<ir::Select>();
      IRMutator<>::Visit(&select->condition, &select->condition);
    }

    void Visit(const ir::Load* op, ir::Expr* expr) override {
      results_.insert(*expr);
      IRMutator<>::Visit(op, expr);
    }

    std::unordered_set<ir::Expr, LoadHash> results_;
  };

  Collector collector;
  for (auto& block : sch->GetAllBlocks()) {
    std::vector<ir::Expr> loops = sch->GetLoops(block);
    ir::Expr store = ir::analyzer::GetStoreOfSBlock(block);
    store = ir::analyzer::ExpandIterVar(store, block);
    store = ir::analyzer::CanonicalizeLoopVar(store, loops);
    collector(&store.As<ir::Store>()->value);
  }
  unconditional_loads_ = std::move(collector.results_);
}

void TileTransposeTactic::InitCandidates(ir::IRSchedule* sch) {
  common_perm_.clear();
  load2candidates_.clear();
  block2candidates_.clear();
  processed_loads_.clear();

  for (auto& block : sch->GetAllBlocks()) {
    std::vector<ir::Expr> loops = sch->GetLoops(block);
    std::string block_id = ir::analyzer::GetBlockName(block);

    ir::Expr store = ir::analyzer::GetStoreOfSBlock(block);
    store = ir::analyzer::ExpandIterVar(store, block);
    store = ir::analyzer::CanonicalizeLoopVar(store, loops);

    std::vector<ir::Expr> loads = ir::ir_utils::CollectIRNodesInOrder(
        store.As<ir::Store>()->value,
        [](const ir::Expr* x) { return x->As<ir::Load>(); });

    // Find candidate loads in this schedule block using the following rules.
    for (int i = 0; i < loads.size(); ++i) {
      ir::Expr load = loads[i];

      // 1. Skip loads that have been added.
      auto candidate_it = load2candidates_.find(load);
      if (candidate_it != load2candidates_.end()) {
        block2candidates_[block_id].push_back(candidate_it->second);
        continue;
      }

      // 2. Skip conditional loads (loads inside Select). As we are currently
      //    unable to analyze the Select's condition, these loads may lead to
      //    out-of-bound accesses.
      if (unconditional_loads_.count(load) == 0) continue;

      // 3. The load tensor should not be defined by a previous schedule block,
      //    otherwise we should do CacheRead on that block rather than here.
      auto* tensor = load.As<ir::Load>()->tensor.as_tensor();
      if (sch->HasBlock(tensor->name)) continue;

      std::vector<int> perm =
          GetTransposePerm(load.As<ir::Load>()->indices, loops.size());

      // 4. This is a critical transpose, including:
      // 1) its dim size equals to the loop size (not a broadcast).
      // 2) its last dim is changed in permutation (incurs discrete access).
      // 3) both the src/dst_low_axis are non-unit (not a squeeze/unsqueeze).
      if (perm.size() != loops.size()) continue;
      if (perm.back() == perm.size() - 1) continue;
      if (GetLoopRangeProduct(loops, GetSrcLowAxis(perm)) == 1) continue;
      if (GetLoopRangeProduct(loops, GetDstLowAxis(perm)) == 1) continue;

      // 5. All transposes in this graph should have the same permutation.
      //    Otherwise, it would be too complex to ensure the correctness and
      //    performance. The violating cases should be rare.
      if (common_perm_.empty()) {
        common_perm_ = perm;
      } else if (common_perm_ != perm) {
        common_perm_.clear();
        return;
      }

      Candidate candidate{load, block_id, i};
      load2candidates_.emplace(load, candidate);
      block2candidates_[block_id].push_back(candidate);
    }
  }
}

void TileTransposeTactic::InitAxisInfo() {
  src_low_axis_ = GetSrcLowAxis(common_perm_);
  dst_low_axis_ = GetDstLowAxis(common_perm_);

  std::set<int> high_axis;
  for (int i = 0; i < common_perm_.size(); ++i) high_axis.insert(i);
  for (auto i : src_low_axis_) high_axis.erase(i);
  for (auto i : dst_low_axis_) high_axis.erase(i);
  high_axis_.assign(high_axis.begin(), high_axis.end());
}

std::vector<int> TileTransposeTactic::GetSrcLowAxis(
    const std::vector<int>& perm) {
  std::set<int> src_low_axis;
  for (int i = 0; i < perm.size(); ++i) {
    if (perm[i] == perm.size() - 1) {
      src_low_axis.insert(i);
      for (int j = i - 1; j >= 0; j--) {
        if (perm[j] + 1 != perm[j + 1]) break;
        src_low_axis.insert(j);
      }
    }
  }
  return {src_low_axis.begin(), src_low_axis.end()};
}

std::vector<int> TileTransposeTactic::GetDstLowAxis(
    const std::vector<int>& perm) {
  std::set<int> dst_low_axis{perm.size() - 1};
  for (int i = perm.size() - 2; i >= 0; --i) {
    if (perm[i] + 1 != perm[i + 1]) break;
    dst_low_axis.insert(i);
  }
  return {dst_low_axis.begin(), dst_low_axis.end()};
}

void TileTransposeTactic::Apply(ir::IRSchedule* sch,
                                const std::string& block_id) {
  if (!can_apply_) return;

  // Handle all candidate loads in the block before tiling the block itself,
  // otherwise we will lose track of some occurrences of the loads due to
  // indices change.
  for (auto& candidate : block2candidates_[block_id]) {
    if (processed_loads_.count(candidate.load) == 0) {
      TileCacheBlock(
          sch, candidate.first_appear_block_id, candidate.buffer_index);
      processed_loads_.insert(candidate.load);
    }
  }

  // Tile the block itself.
  TileBlock(sch, block_id);

  VLOG(4) << "After TileTransposeTactic on [" << block_id
          << "]: " << sch->GetModule().GetExprs().front();
}

std::string TileTransposeTactic::CreateCacheBlock(
    ir::IRSchedule* sch,
    const std::string& block_id,
    int buffer_index,
    const std::string& memory_type) {
  ir::Expr block = sch->GetBlock(block_id);
  ir::Expr cache_block = sch->CacheRead(block, buffer_index, memory_type);

  std::string transpose_stage = (memory_type == "shared") ? "write" : "read";
  sch->Annotate(cache_block, "transpose_stage", transpose_stage);

  // Mark the cache block as a virtual output to prevent inlining. This doesn't
  // affect the actual outputs of the graph.
  std::string cache_block_id = ir::analyzer::GetBlockName(cache_block);
  context_->output_names.insert(cache_block_id);

  // Note: the CacheRead primitive de-transposes the input, so we need to apply
  // the transpose permutation again on the cache block.
  sch->Reorder(cache_block_id, common_perm_);
  return cache_block_id;
}

void TileTransposeTactic::TileCacheBlock(ir::IRSchedule* sch,
                                         const std::string& block_id,
                                         int buffer_index) {
  // Step 1. Create the shared and local buffers.
  std::string shared_cache_block_id =
      CreateCacheBlock(sch, block_id, buffer_index, "shared");
  std::string local_cache_block_id =
      CreateCacheBlock(sch, block_id, buffer_index, "local");

  // Step 2. Convert the layout to [ high_axis, src_low_axis, dst_low_axis ].
  CanonicalizeLayout(sch, shared_cache_block_id);
  CanonicalizeLayout(sch, local_cache_block_id);

  // Step 3. Do inner-block transpose.
  int offset = high_axis_.size();
  sch->Split(shared_cache_block_id, offset + 1, {-1, 4, 8});
  sch->Split(shared_cache_block_id, offset, {-1, 32});

  sch->Split(local_cache_block_id, offset + 1, {-1, 32});
  sch->Split(local_cache_block_id, offset, {-1, 4, 8});

  sch->Reorder(shared_cache_block_id, OffsetVec({0, 2, 3, 4, 1}, offset));
  sch->Reorder(local_cache_block_id, OffsetVec({0, 3, 1, 2, 4}, offset));

  // Step 4. Fuse and bind as [ block_x, inner_loop, thread_y, thread_x ].
  FuseAndBind(sch, shared_cache_block_id, /* need_sync = */ true);
  FuseAndBind(sch, local_cache_block_id, /* need_sync = */ true);
}

void TileTransposeTactic::TileBlock(ir::IRSchedule* sch,
                                    const std::string& block_id) {
  CanonicalizeLayout(sch, block_id);

  int offset = high_axis_.size();
  sch->Split(block_id, offset + 1, {-1, 32});
  sch->Split(block_id, offset, {-1, 4, 8});

  sch->Reorder(block_id, OffsetVec({0, 3, 1, 2, 4}, offset));

  FuseAndBind(sch, block_id);

  if (context_->output_names.count(block_id) == 0) {
    ir::Expr block = sch->GetBlock(block_id);
    sch->SetBuffer(block, "local");
  }
}

void TileTransposeTactic::CanonicalizeLayout(ir::IRSchedule* sch,
                                             const std::string& block_id) {
  std::vector<int> order = high_axis_;
  order.insert(order.end(), src_low_axis_.begin(), src_low_axis_.end());
  order.insert(order.end(), dst_low_axis_.begin(), dst_low_axis_.end());

  sch->Reorder(block_id, order);

  std::vector<int> src_low_axis =
      ArangeVec(src_low_axis_.size(), high_axis_.size());
  std::vector<int> dst_low_axis =
      ArangeVec(dst_low_axis_.size(), high_axis_.size() + src_low_axis_.size());

  sch->Fuse(block_id, dst_low_axis);
  sch->Fuse(block_id, src_low_axis);
}

void TileTransposeTactic::FuseAndBind(ir::IRSchedule* sch,
                                      const std::string& block_id,
                                      bool need_sync) {
  int offset = high_axis_.size();
  sch->Fuse(block_id, ArangeVec(offset + 2));

  std::vector<ir::Expr> loops = sch->GetLoops(block_id);
  sch->Bind(loops[0], "blockIdx.x");
  sch->Bind(loops[2], "threadIdx.y");
  sch->Bind(loops[3], "threadIdx.x");

  if (need_sync) {
    sch->SyncThreads(sch->GetLoops(block_id)[0], /* after_node = */ false);
  }
}

}  // namespace

std::unique_ptr<ScheduleTactic> CreateTileTransposeTactic() {
  return std::make_unique<TileTransposeTactic>();
}

}  // namespace ir
}  // namespace cinn
