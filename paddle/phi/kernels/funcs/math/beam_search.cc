// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/funcs/math/beam_search.h"
#include "glog/logging.h"
#include "paddle/phi/backends/cpu/cpu_context.h"

namespace phi {
namespace math {

template <typename T>
class BeamSearchFunctor<phi::CPUContext, T> {
 public:
  void operator()(const phi::CPUContext &context UNUSED,
                  const phi::DenseTensor *pre_ids,
                  const phi::DenseTensor *pre_scores,
                  const phi::DenseTensor *ids,
                  const phi::DenseTensor *scores,
                  phi::DenseTensor *selected_ids,
                  phi::DenseTensor *selected_scores,
                  phi::DenseTensor *parent_idx,
                  size_t level,
                  size_t beam_size,
                  int end_id,
                  bool is_accumulated) {
    auto abs_lod = phi::ToAbsOffset(scores->lod());
    auto &high_level = abs_lod[level];

    auto items = SelectTopBeamSizeItems(pre_ids,
                                        pre_scores,
                                        ids,
                                        scores,
                                        level,
                                        beam_size,
                                        end_id,
                                        is_accumulated);
    auto selected_items = ToMap(items, high_level.back());
    if (FLAGS_v == 3) {
      VLOG(3) << "selected_items:";
      for (size_t i = 0; i < selected_items.size(); ++i) {
        VLOG(3) << "offset: " << i;
        for (auto &item : selected_items[i]) {
          VLOG(3) << item.ToString();
        }
      }
    }

    PruneEndBeams(pre_ids, abs_lod, &selected_items, level, end_id);
    // calculate the output tensor's height
    size_t num_instances = std::accumulate(
        std::begin(selected_items),
        std::end(selected_items),
        0,
        [](size_t a, std::vector<Item> &b) { return a + b.size(); });
    // the output tensor shape should be [num_instances, 1]
    auto dims = common::make_ddim(
        std::vector<int64_t>({static_cast<int>(num_instances), 1}));
    selected_ids->Resize(dims);
    auto *selected_ids_data = context.template Alloc<int64_t>(selected_ids);
    selected_scores->Resize(dims);
    auto *selected_scores_data = context.template Alloc<float>(selected_scores);
    if (parent_idx != nullptr) {
      parent_idx->Resize({static_cast<int64_t>(num_instances)});
    }
    auto *parent_idx_data =
        parent_idx ? context.template Alloc<int>(parent_idx) : nullptr;

    // fill in data
    std::vector<size_t> low_level;
    size_t low_offset = 0;
    for (auto &items : selected_items) {
      low_level.push_back(low_offset);
      for (auto &item : items) {
        if (parent_idx) {
          parent_idx_data[low_offset] = static_cast<int>(low_level.size() - 1);
        }
        selected_ids_data[low_offset] = item.id;
        selected_scores_data[low_offset] = item.score;
        low_offset++;
      }
    }
    low_level.push_back(low_offset);

    // fill lod
    phi::LegacyLoD lod(2);
    lod[0].assign(high_level.begin(), high_level.end());
    lod[1].assign(low_level.begin(), low_level.end());
    if (!CheckLegacyLoD(lod)) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "lod %s is not right in"
          " beam_search, please check your code.",
          LoDToString(lod)));
    }
    selected_ids->set_lod(lod);
    selected_scores->set_lod(lod);
  }

  /*
   * The basic items help to sort.
   */
  struct Item {
    Item() = default;
    Item(size_t offset, size_t id, float score)
        : offset(offset), id(id), score(score) {}
    // offset in the higher lod level.
    size_t offset;
    // prefix id in the lower lod level.
    // size_t prefix;
    // the candidate id
    size_t id;
    // the corresponding score
    float score;

    inline bool operator<(const Item &in) const {
      return (score < in.score) ||
             ((score == in.score) && (offset < in.offset));
    }

    inline Item &operator=(const Item &in) {
      if (this != &in) {
        this->offset = in.offset;
        this->id = in.id;
        this->score = in.score;
        return *this;
      }
      return *this;
    }

    std::string ToString() {
      std::ostringstream os;
      os << "{";
      os << "offset: " << offset << ", ";
      os << "id: " << id << ", ";
      os << "score: " << score << "";
      os << "}";
      return os.str();
    }
  };

 protected:
  /*
   * Prune the source sentences all branches finished, and it is optional.
   * Pruning must one step later than finishing (thus pre_ids is needed here),
   * since the end tokens must be written out.
   */
  void PruneEndBeams(const phi::DenseTensor *pre_ids,
                     const phi::LegacyLoD &abs_lod,
                     std::vector<std::vector<Item>> *items,
                     size_t lod_level,
                     int end_id) {
    auto *pre_ids_data = pre_ids->data<int64_t>();
    auto &high_level = abs_lod[lod_level];
    for (size_t src_idx = 0; src_idx < high_level.size() - 1; ++src_idx) {
      size_t src_prefix_start = high_level[src_idx];
      size_t src_prefix_end = high_level[src_idx + 1];
      bool finish_flag = true;
      for (size_t offset = src_prefix_start; offset < src_prefix_end;
           offset++) {
        for (auto &item : items->at(offset)) {
          if (item.id != static_cast<size_t>(end_id) ||
              pre_ids_data[offset] != end_id) {
            finish_flag = false;
            break;
          }
        }
        if (!finish_flag) break;
      }
      if (finish_flag) {  // all branches of the beam (source sentence) end and
                          // prune this beam
        for (size_t offset = src_prefix_start; offset < src_prefix_end;
             offset++)
          items->at(offset).clear();
      }
    }
  }

  /*
   * Transform the items into a map whose key is offset, value is the items.
   * NOTE low performance.
   */
  std::vector<std::vector<Item>> ToMap(
      const std::vector<std::vector<Item>> &items, size_t element_num) {
    std::vector<std::vector<Item>> result;
    result.resize(element_num);
    for (auto &entries : items) {
      for (const auto &item : entries) {
        result[item.offset].push_back(item);
      }
    }
    return result;
  }

  void Insert(std::vector<Item> *top_beam_ptr,
              const Item &item,
              size_t beam_size) {
    std::vector<Item> &top_beam = *top_beam_ptr;

    size_t num_beams = top_beam.size();
    if (num_beams < beam_size) {
      top_beam.resize(num_beams + 1);
      num_beams++;
    } else {
      if (item < top_beam[beam_size - 1]) {
        return;
      }
    }

    for (int k = static_cast<int>(num_beams) - 2; k >= 0; --k) {
      if (top_beam[k] < item) {
        top_beam[k + 1] = top_beam[k];
      } else {
        top_beam[k + 1] = item;
        return;
      }
    }
    top_beam[0] = item;
  }

  /*
   * For each source, select top beam_size records.
   */
  std::vector<std::vector<Item>> SelectTopBeamSizeItems(
      const phi::DenseTensor *pre_ids,
      const phi::DenseTensor *pre_scores,
      const phi::DenseTensor *ids,
      const phi::DenseTensor *scores,
      size_t lod_level,
      size_t beam_size,
      int end_id,
      bool is_accumulated) {
    std::vector<std::vector<Item>> result;

    // find the current candidates
    auto abs_lod = phi::ToAbsOffset(scores->lod());

    auto *pre_ids_data = pre_ids->data<int64_t>();
    auto *pre_scores_data = pre_scores->data<float>();

    auto *ids_data = ids ? ids->data<int64_t>() : nullptr;
    auto *scores_data = scores->data<float>();

    size_t num_seqs = scores->NumElements(lod_level);
    size_t seq_width = 1;
    for (int i = 1; i < scores->dims().size(); i++) {
      seq_width *= scores->dims()[i];
    }

    for (size_t seq_id = 0; seq_id < num_seqs; ++seq_id) {
      size_t seq_offset_start = abs_lod[lod_level][seq_id];
      size_t seq_offset_end = abs_lod[lod_level][seq_id + 1];

      std::vector<Item> top_beam;
      top_beam.reserve(beam_size);

      for (size_t offset = seq_offset_start; offset < seq_offset_end;
           ++offset) {
        auto pre_id = pre_ids_data[offset];
        auto pre_score = pre_scores_data[offset];
        if (pre_id == end_id) {
          // Allocate all probability mass to end_id for finished branches and
          // the other candidate ids can be ignored.
          Item item(offset, end_id, pre_score);
          Insert(&top_beam, item, beam_size);
        } else {
          size_t index = offset * seq_width;
          for (size_t d = 0; d < seq_width; d++, index++) {
            int64_t id = ids_data ? ids_data[index] : static_cast<int64_t>(d);
            float score = is_accumulated
                              ? scores_data[index]
                              : pre_score + std::log(scores_data[index]);
            Item item(offset, id, score);
            Insert(&top_beam, item, beam_size);
          }
        }
      }

      result.emplace_back(top_beam);
    }

    if (FLAGS_v == 3) {
      VLOG(3) << "SelectTopBeamSizeItems result size " << result.size();
      for (auto &items : result) {
        VLOG(3) << "item set:";
        for (auto &item : items) {
          VLOG(3) << item.ToString();
        }
      }
    }

    return result;
  }
};

template class BeamSearchFunctor<phi::CPUContext, int>;
template class BeamSearchFunctor<phi::CPUContext, int64_t>;
template class BeamSearchFunctor<phi::CPUContext, float>;
template class BeamSearchFunctor<phi::CPUContext, double>;

}  // namespace math
}  // namespace phi
