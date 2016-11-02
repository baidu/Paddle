# Copyright (c) 2016 Baidu, Inc. All Rights Reserved
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

#!/bin/sh

PATH_TO_LOCAL_WORKSPACE=/home/sparse_test/workspace


mv cluster_config.conf conf.py

python paddle.py \
  --job_dispatch_package="${PATH_TO_LOCAL_WORKSPACE}" \
  --use_gpu=0 \
  --config=./sparse_trainer_config.py \
  --saving_period=1 \
  --test_period=0 \
  --num_passes=4 \
  --dot_period=2 \
  --log_period=20 \
  --trainer_count=10 \
  --saving_period_by_batches=5000 \
  --ports_num_for_sparse=4 \
  --local=0 \
