// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#pragma once
#include <memory>

#include "glog/logging.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/platform/device_defs.h"
#include "paddle/utils/test_macros.h"

namespace paddle {
namespace platform {

class DeviceOption;
class DeviceEvent;

constexpr int MaxDeviceTypes =
    static_cast<int>(platform::DeviceType::MAX_DEVICE_TYPES);

typedef void (*EventCreateFunction)(DeviceEvent*,
                                    const phi::Place&,
                                    unsigned int flag);
typedef void (*EventRecordFunction)(DeviceEvent*, const DeviceContext*);
typedef bool (*EventQueryFunction)(const DeviceEvent*);
typedef void (*EventFinishFunction)(const DeviceEvent*);
typedef void (*EventSetFinishedFunction)(const DeviceEvent*);
typedef void (*EventWaitFunction)(const DeviceEvent*, const DeviceContext*);
typedef void (*EventResetFunction)(const DeviceEvent*);

inline int DeviceTypeToId(const DeviceType& device_type) {
  return static_cast<int>(device_type);
}

unsigned int GenerateDeviceEventFlag(bool enable_timing = false,
                                     bool blocking = false,
                                     bool interprocess = false);

enum EventStatus {
  INITIALIZED = 0,
  SCHEDULED = 1,
  SUCCESS = 2,
  FAILED = 3,
};

class DeviceEvent {
 public:
  explicit DeviceEvent(const phi::Place& place, unsigned int flag)
      : event_(), place_(place), flag_(flag) {
    type_id_ = DeviceTypeToId(platform::Place2DeviceType(place));
    PADDLE_ENFORCE_LT(type_id_,
                      MaxDeviceTypes,
                      common::errors::PreconditionNotMet(
                          "Required type < %d, but received type = %d",
                          MaxDeviceTypes,
                          type_id_));
#ifndef PADDLE_WITH_CUSTOM_DEVICE
    // TODO(Aurelius84): only support CPU/CUDA.
    PADDLE_ENFORCE_LT(type_id_,
                      3,
                      common::errors::Unavailable(
                          "Currently DeviceEvent do not support %s", place));
#endif
    PADDLE_ENFORCE_NOT_NULL(
        event_creator_[type_id_],
        common::errors::Unavailable("event_creator_[%d] shall not be nullptr.",
                                    type_id_));
    event_creator_[type_id_](this, place, flag);
  }

  ~DeviceEvent() {}

  void Record(const DeviceContext* dev_ctx) {
    PADDLE_ENFORCE_NOT_NULL(
        event_recorder_[type_id_],
        common::errors::Unavailable("event_recorder_[%d] shall not be nullptr.",
                                    type_id_));
    if (!recorded_) {
      recorded_ = true;
    }
    event_recorder_[type_id_](this, dev_ctx);
  }

  bool Query() {
    PADDLE_ENFORCE_NOT_NULL(
        event_querier_[type_id_],
        common::errors::Unavailable("event_querier_[%d] shall not be nullptr.",
                                    type_id_));
    if (!recorded_) {
      VLOG(4) << "Event " << this << " is not recorded yet, and skip query!";
      return true;
    }
    return event_querier_[type_id_](this);
  }

  void Finish() const {
    PADDLE_ENFORCE_NOT_NULL(
        event_finisher_[type_id_],
        common::errors::Unavailable("event_finisher_[%d] shall not be nullptr.",
                                    type_id_));
    event_finisher_[type_id_](this);
  }

  void SetFinished() {
    PADDLE_ENFORCE_NOT_NULL(
        event_finished_setter_[type_id_],
        common::errors::Unavailable(
            "event_finished_setter_[%d] shall not be nullptr.", type_id_));
    event_finished_setter_[type_id_](this);
  }

  void Reset() {
    PADDLE_ENFORCE_NOT_NULL(
        event_resetter_[type_id_],
        common::errors::Unavailable("event_resetter_[%d] shall not be nullptr.",
                                    type_id_));
    event_resetter_[type_id_](this);
  }

  void Wait(const DeviceType& waiter_type, const DeviceContext* context) const {
    auto waiter_idx = DeviceTypeToId(waiter_type);
    PADDLE_ENFORCE_NOT_NULL(event_waiter_[waiter_idx][type_id_],
                            common::errors::Unavailable(
                                "event_waiter_[%d][%d] shall not be nullptr.",
                                waiter_idx,
                                type_id_));
    if (!recorded_) {
      VLOG(4) << "Event " << this << " is not recorded yet, and skip wait!";
      return;
    }
    event_waiter_[waiter_idx][type_id_](this, context);
  }

  void InitEvent(std::shared_ptr<void> event) { event_ = event; }

  std::shared_ptr<void> GetEvent() const { return event_; }

 private:
  std::shared_ptr<void> event_;
  phi::Place place_;
  int type_id_;
  unsigned int flag_;

  // NOTE(chenruibiao): In cross-step stream synchronization, an event may be
  // recorded in the first step and waited in the second step. So, in the first
  // step, the WaitEvent may be called without RecordEvent.
  // On cuda device, it is ok to wait event that is not recorded yet;
  // while on npu device, it results in error.
  // So, we add flag recorded_ to handle this case uniformly.
  bool recorded_{false};

  static EventCreateFunction event_creator_[MaxDeviceTypes];
  static EventRecordFunction event_recorder_[MaxDeviceTypes];
  static EventQueryFunction event_querier_[MaxDeviceTypes];
  static EventFinishFunction event_finisher_[MaxDeviceTypes];
  static EventSetFinishedFunction event_finished_setter_[MaxDeviceTypes];
  static EventWaitFunction event_waiter_[MaxDeviceTypes][MaxDeviceTypes];
  static EventResetFunction event_resetter_[MaxDeviceTypes];

  template <DeviceType device_typ>
  friend struct EventCreateFunctionRegisterer;

  template <DeviceType device_typ>
  friend struct EventRecordFunctionRegisterer;

  template <DeviceType device_typ>
  friend struct EventQueryFunctionRegisterer;

  template <DeviceType device_typ>
  friend struct EventFinishFunctionRegisterer;

  template <DeviceType device_typ>
  friend struct EventSetFinishedFunctionRegisterer;

  template <DeviceType waiter_typ, DeviceType event_type>
  friend struct EventWaitFunctionRegisterer;

  template <DeviceType device_typ>
  friend struct EventResetFunctionRegisterer;
};

}  // namespace platform
}  // namespace paddle
