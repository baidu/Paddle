// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <sstream>
#include <unordered_map>

#include <nlohmann/json.hpp>
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/ft_gemm_configs.h"

namespace phi {

using json = nlohmann::json;

enum GemmDataType {
  _FLOAT,
  _HALF,
  _NVBFLOAT16,
  _INT8,
  _INT4,
};

enum GemmType {
  FPAINTBGEMM,
  MOEGEMM,
};

template <typename T>
constexpr GemmDataType getGemmDataType() {
  if constexpr (std::is_same<T, float>::value) {
    return GemmDataType::_FLOAT;
  } else if constexpr (std::is_same<T, half>::value) {
    return GemmDataType::_HALF;
  } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    return GemmDataType::_NVBFLOAT16;
  } else if constexpr (std::is_same<T, uint8_t>::value) {
    return GemmDataType::_INT8;
  } else if constexpr (std::is_same<T, cutlass::uint4b_t>::value) {
    return GemmDataType::_INT4;
  } else {
    static_assert(!std::is_same<T, T>::value,
                  "Unsupported data type combination for GemmDataType.");
  }
}

struct GemmIDType {
  int gemm_n{};
  int gemm_k{};
  GemmType compute_type;
  GemmDataType dtype{};
  GemmDataType wdtype{};
  int num_experts{};

  // Equality operator
  bool operator==(GemmIDType const& id) const {
    return id.gemm_n == gemm_n && id.gemm_k == gemm_k &&
           id.compute_type == compute_type && id.dtype == dtype &&
           id.wdtype == wdtype && id.num_experts == num_experts;
  }

  // Stream output operator
  friend std::ostream& operator<<(std::ostream& out, GemmIDType const& id) {
    out << "gemm_n, gemm_k, compute_type, dtype, wdtype, num_experts="
        << id.gemm_n << "," << id.gemm_k << ","
        << static_cast<int>(id.compute_type) << ","
        << static_cast<int>(id.dtype) << "," << static_cast<int>(id.wdtype)
        << "," << id.num_experts;
    return out;
  }

  friend void to_json(json& j, const GemmIDType& id) {  // NOLINT
    j = json{{"gemm_n", id.gemm_n},
             {"gemm_k", id.gemm_k},
             {"compute_type", id.compute_type},
             {"dtype", id.dtype},
             {"wdtype", id.wdtype},
             {"num_experts", id.num_experts}};
  }

  friend void from_json(const json& j, GemmIDType& id) {  // NOLINT
    j.at("gemm_n").get_to(id.gemm_n);
    j.at("gemm_k").get_to(id.gemm_k);
    j.at("compute_type").get_to(id.compute_type);
    j.at("dtype").get_to(id.dtype);
    j.at("wdtype").get_to(id.wdtype);
    j.at("num_experts").get_to(id.num_experts);
  }
};

// Hash of GemmIDType
struct GemmIDTypeHash {
  std::size_t operator()(GemmIDType const& id) const {
    size_t hash = std::hash<int>{}(id.gemm_n);
    hash ^= std::hash<int>{}(id.gemm_k) << 1;
    hash ^= std::hash<int>{}(static_cast<int>(id.compute_type)) << 2;
    hash ^= std::hash<int>{}(static_cast<int>(id.dtype)) << 3;
    hash ^= std::hash<int>{}(static_cast<int>(id.wdtype)) << 4;
    hash ^= std::hash<int>{}(id.num_experts) << 5;
    return hash;
  }
};

class GemmConfigManager {
 public:
  using MProfileMap = std::unordered_map<int, std::optional<CutlassGemmConfig>>;
  using MProfileMapPtr = std::shared_ptr<MProfileMap>;

  // requires exclusive ownership to write to *this
  using writer_lock = std::unique_lock<std::shared_timed_mutex>;
  // requires shared ownership to read from other
  using reader_lock = std::shared_lock<std::shared_timed_mutex>;

  // Struct of continuing map if GEMMs to the best profiles for different Ms
  struct GemmProfileMap {
    // Mutex guarding map
    std::shared_timed_mutex mutex;
    // Map from GEMM type to profile for particular GEMM
    std::unordered_map<GemmIDType, MProfileMapPtr, GemmIDTypeHash> profileMap;

    bool existsMProfileMap(GemmIDType const& id) {
      auto const iter = profileMap.find(id);
      return iter != profileMap.end();
    }

    void createMProfileMap(GemmIDType const& id) {
      profileMap[id] = std::make_shared<MProfileMap>();
    }

    MProfileMapPtr getMProfileMap(GemmIDType const& id) {
      auto const iter = profileMap.find(id);
      if (iter == profileMap.end()) {
        std::ostringstream msg;
        msg << "[GemmConfigManager]Cannot find ID (" << id
            << ") in the profile map. Abort.";
        PADDLE_FATAL(msg.str());
      }
      return iter->second;
    }

    json serialize() {
      json j;
      for (const auto& pair : profileMap) {
        json mProfileJson;
        for (const auto& mPair : *(pair.second)) {
          if (mPair.second.has_value()) {
            const auto config = mPair.second.value();
            mProfileJson[std::to_string(mPair.first)] =
                json{{"tile_config", config.tile_config},
                     {"split_k_style", config.split_k_style},
                     {"split_k_factor", config.split_k_factor},
                     {"stages", config.stages}};
          } else {
            PADDLE_FATAL("[GemmConfigManager] Serialize Empty Config");
          }
        }
        j.push_back({{"gemm_id", pair.first}, {"m_profile", mProfileJson}});
      }
      return j;
    }

    void deserialize(const json& j) {
      for (const auto& elem : j) {
        GemmIDType gemmId = elem["gemm_id"];
        if (!existsMProfileMap(gemmId)) {
          createMProfileMap(gemmId);
        }
        auto mProfileMap = getMProfileMap(gemmId);
        for (const auto& mElem : elem["m_profile"].items()) {
          int m = std::stoi(mElem.key());
          CutlassGemmConfig config;
          config.tile_config = mElem.value()["tile_config"];
          config.split_k_style = mElem.value()["split_k_style"];
          config.split_k_factor = mElem.value()["split_k_factor"];
          config.stages = mElem.value()["stages"];
          mProfileMap->insert({m, config});
        }
      }
    }
  };

  using GemmProfileMapPtr = std::shared_ptr<GemmProfileMap>;

  static GemmConfigManager& Instance() {
    static GemmConfigManager gemm_config_manager;
    return gemm_config_manager;
  }

  std::optional<CutlassGemmConfig> getBestConfig(GemmIDType gemmId, int m) {
    reader_lock lock(mGemmProfileMap->mutex);
    int mRounded = std::min(nextPowerOfTwo(m), getMaxProfileM());
    if (mGemmProfileMap->existsMProfileMap(gemmId)) {
      auto profileMap = mGemmProfileMap->getMProfileMap(gemmId);
      auto iter = profileMap->find(mRounded);
      if (iter != profileMap->end()) {
        return iter->second;
      } else {
        return std::nullopt;
      }
    } else {
      return std::nullopt;
    }
  }

  bool addBestConfig(GemmIDType gemmId, int m, CutlassGemmConfig config) {
    writer_lock lock(mGemmProfileMap->mutex);
    int mRounded = std::min(nextPowerOfTwo(m), getMaxProfileM());
    if (!mGemmProfileMap->existsMProfileMap(gemmId)) {
      // create map for Gemm ID
      mGemmProfileMap->createMProfileMap(gemmId);
    }
    auto mProfileMap = mGemmProfileMap->getMProfileMap(gemmId);
    if (mProfileMap->find(mRounded) == mProfileMap->end()) {
      mProfileMap->insert({m, config});
    } else {
      return false;
    }
    return true;
  }

  int nextPowerOfTwo(int v) {
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return ++v;
  }

  int getMaxProfileM() const { return 1024; }

  bool loadFromJson(const std::string& filename) {
    std::ifstream inFile(filename);
    if (!inFile.is_open()) {
      // no gemm profile json file
      VLOG(4) << "No gemm profile json file";
      return false;
    }
    try {
      json j;
      inFile >> j;
      mGemmProfileMap->deserialize(j);
      VLOG(4) << "Parse gemm profile json file successfully";
      return true;
    } catch (const std::exception& e) {
      PADDLE_FATAL("[GemmConfigManager] Failed to parse gemm profiles json");
      return false;
    }
  }

 private:
  GemmConfigManager() {
    mGemmProfileMap = std::make_shared<GemmProfileMap>();
    loadFromJson("gemm_profiles.json");
  }

  ~GemmConfigManager() {
    json j = mGemmProfileMap->serialize();
    std::ofstream outFile("gemm_profiles.json");
    outFile << j.dump(4);
    outFile.close();
  }

  GemmConfigManager(const GemmConfigManager&) = delete;
  void operator=(const GemmConfigManager&) = delete;

 private:
  GemmProfileMapPtr mGemmProfileMap{};
};

}  // namespace phi
