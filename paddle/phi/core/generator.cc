/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/generator.h"

#include <glog/logging.h>

#include <cstdint>
#include <memory>
#include <utility>

#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/xpu/xpu_info.h"
#include "paddle/phi/core/enforce.h"

static uint64_t GetRandomSeed() {
  std::random_device rd;
  // double has 53 bit significant, so limit uint64 to 53 bits
  return ((((uint64_t)rd()) << 32) + rd()) & 0x1FFFFFFFFFFFFF;
}

namespace phi {

const std::shared_ptr<Generator>& DefaultXPUGenerator(int64_t device_id) {
#if defined(PADDLE_WITH_XPU)

  static int64_t num_xpu_devices = -1;
  static std::once_flag num_devices_init_flag;
  static std::deque<std::once_flag> xpu_device_flags;
  static std::vector<std::shared_ptr<Generator>> default_xpu_generators;

  std::call_once(num_devices_init_flag, []() {
    num_xpu_devices = phi::backends::xpu::GetXPUDeviceCount();
    xpu_device_flags.resize(num_xpu_devices);
    default_xpu_generators.resize(num_xpu_devices);
  });
  if (device_id < 0) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "xpu device id should be greater than 0"));
  }

  std::call_once(xpu_device_flags[device_id], [device_id]() {
    default_xpu_generators[device_id] =
        std::make_shared<Generator>(GetRandomSeed(), device_id);
    VLOG(4) << "initial seed: "
            << default_xpu_generators[device_id]->GetCurrentSeed();
  });
  return default_xpu_generators[device_id];
#else
  PADDLE_THROW(common::errors::PermissionDenied(
      "getDefaultXPUGenerator only support in XPU place"));
#endif
}

const std::shared_ptr<Generator>& DefaultCUDAGenerator(int64_t device_id) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

  static int64_t num_cuda_devices = -1;
  static std::once_flag num_devices_init_flag;
  static std::deque<std::once_flag> cuda_device_flags;
  static std::vector<std::shared_ptr<Generator>> default_cuda_generators;

  std::call_once(num_devices_init_flag, []() {
    num_cuda_devices = phi::backends::gpu::GetGPUDeviceCount();
    cuda_device_flags.resize(num_cuda_devices);
    default_cuda_generators.resize(num_cuda_devices);
  });
  if (device_id < 0) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "cuda device id should be greater than 0"));
  }

  std::call_once(cuda_device_flags[device_id], [device_id]() {
    default_cuda_generators[device_id] =
        std::make_shared<Generator>(GetRandomSeed(), device_id);
    VLOG(4) << "initial seed: "
            << default_cuda_generators[device_id]->GetCurrentSeed();
  });
  return default_cuda_generators[device_id];
#else
  PADDLE_THROW(common::errors::PermissionDenied(
      "getDefaultCUDAGenerator only support in CUDA place"));
#endif
}

const std::shared_ptr<Generator>& DefaultCPUGenerator() {
  static auto default_cpu_generator =
      std::make_shared<Generator>(GetRandomSeed());
  return default_cpu_generator;
}

const std::shared_ptr<Generator>& DefaultCustomDeviceGenerator(
    const phi::CustomPlace& place) {
  static std::
      unordered_map<phi::Place, std::shared_ptr<Generator>, phi::Place::Hash>
          generators;
  if (generators.find(place) == generators.end()) {
    generators.insert({place, std::make_shared<Generator>(GetRandomSeed())});
  }
  return generators[place];
}

using RNGMap = std::unordered_map<std::string, std::shared_ptr<Generator>>;

static RNGMap& GetRandomSeedGeneratorMap() {
  static auto random_seed_generator_map = RNGMap();
  return random_seed_generator_map;
}

const std::shared_ptr<Generator>& SetRandomSeedGenerator(
    const std::string& name, uint64_t seed) {
  auto& rng_map = GetRandomSeedGeneratorMap();
  auto iter = rng_map.find(name);
  PADDLE_ENFORCE_EQ(iter == rng_map.end(),
                    true,
                    common::errors::AlreadyExists(
                        "%s RandomSeedGenerator is already exist", name));

  auto generator = std::make_shared<Generator>(seed);
  bool emplace_success = rng_map.emplace(name, generator).second;
  PADDLE_ENFORCE_EQ(
      emplace_success,
      true,
      common::errors::PermissionDenied(
          "SetRandomSeedGenerator cannot emplace %s RandomSeedGenerator",
          name));
  return rng_map[name];
}

const std::shared_ptr<Generator>& GetRandomSeedGenerator(
    const std::string& name) {
  auto& rng_map = GetRandomSeedGeneratorMap();
  auto iter = rng_map.find(name);
  PADDLE_ENFORCE_EQ(iter != rng_map.end(),
                    true,
                    common::errors::NotFound(
                        "%s RandomSeedGenerator is not found, please "
                        "use `set_random_seed_generator` to set rng first",
                        name));
  return iter->second;
}

// There are 3 conditions:
// (1) op seed is set, use op seed.
// (2) op seed is not set, global seed is set, use global seed.
// (3) op seed is not set, global seed is not set too, use random seed from
// RandomGenerator.
std::shared_ptr<std::mt19937_64> GetCPURandomEngine(uint64_t seed) {
  if (seed == 0) {
    VLOG(4) << "Use random cpu_engine from generator";
    return DefaultCPUGenerator()->GetCPUEngine();
  } else {
    // NOTE(zhiqiu): creating an cpu_engine instance everytime instead of using
    // OpDefaultCPUEngine(), this is the legacy behavior of random operators.
    // The benefit is that when running PE with fixed-seed in multiple threads,
    // each thread has their own cpu_engine, and doesn't affect each other.
    //
    // And we need to measure the determinacy of Generator in PE.
    auto cpu_engine = std::make_shared<std::mt19937_64>();
    static std::mutex mu_;
    {
      std::lock_guard<std::mutex> lock(mu_);
      cpu_engine->seed(seed);
    }
    return cpu_engine;
  }
}

inline void Generator::print_state_info() {
  VLOG(4) << "Generator Random state "
          << "device id: " << state().device << ", seed: " << state().seed
          << ", offset: " << state().offset << ", cpu_engine: " << cpu_engine();
}

Generator::Generator() {
  auto seed = GetRandomSeed();
  current_index = states_.size();
  states_.emplace_back(-1, seed);
  print_state_info();
}

Generator::Generator(uint64_t seed) {
  current_index = states_.size();
  states_.emplace_back(-1, seed);
  print_state_info();
}

Generator::Generator(uint64_t seed, int64_t device_id) {
  current_index = states_.size();
  // device id first, then seed
  states_.emplace_back(device_id, seed);
  print_state_info();
}

phi::Generator::GeneratorState Generator::GetState() { return state(); }

void Generator::SetState(const phi::Generator::GeneratorState& state) {
  std::lock_guard<std::mutex> lock(mu_);
  if (current_index < states_.size())
    states_[current_index] = state;
  else
    PADDLE_THROW(common::errors::NotFound("Generator index is not found"));
  print_state_info();
}

uint64_t Generator::GetStateIndex() { return current_index; }

void Generator::SetStateIndex(uint64_t StateIndex) {
  std::lock_guard<std::mutex> lock(mu_);
  if (current_index < states_.size())
    current_index = StateIndex;
  else
    PADDLE_THROW(common::errors::NotFound("Generator index is not found"));
}

uint64_t Generator::RegisterStateIndex(const GeneratorState& state) {
  std::lock_guard<std::mutex> lock(mu_);
  auto new_index = states_.size();
  states_.push_back(state);
  current_index = new_index;
  return new_index;
}

inline Generator::GeneratorState& Generator::state() {
  if (current_index < states_.size())
    return states_[current_index];
  else
    PADDLE_THROW(common::errors::NotFound("Generator index is not found"));
}

inline std::shared_ptr<std::mt19937_64> Generator::cpu_engine() {
  return state().cpu_engine;
}

uint64_t Generator::GetCurrentSeed() {
  std::lock_guard<std::mutex> lock(mu_);
  return state().seed;
}

uint64_t Generator::Seed() {
  std::lock_guard<std::mutex> lock(mu_);
  uint64_t seed = GetRandomSeed();
  state().reset(seed);
  return seed;
}

void Generator::SetCurrentSeed(uint64_t seed) {
  std::lock_guard<std::mutex> lock(mu_);
  state().reset(seed);
}

std::shared_ptr<std::mt19937_64> Generator::GetCPUEngine() {
  return cpu_engine();
}

uint64_t Generator::Random64() {
  std::lock_guard<std::mutex> lock(mu_);
  auto current_engine = cpu_engine();
  return (*current_engine)();
}

std::pair<uint64_t, uint64_t> Generator::IncrementOffset(uint64_t increment) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_CUSTOM_DEVICE) || defined(PADDLE_WITH_XPU)
  std::lock_guard<std::mutex> lock(mu_);
  uint64_t offset = state().offset;
  state().offset = offset + increment;
  print_state_info();
  return std::make_pair(state().seed, offset);
#else
  PADDLE_THROW(common::errors::PermissionDenied(
      "Increment Offset only support in CUDA place"));
#endif
}

}  // namespace phi
