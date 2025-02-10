# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
ARG PYTHON_VERSION=3.7

FROM python:${PYTHON_VERSION}-alpine3.11

USER root

WORKDIR /root

VOLUME /root/.ccache

VOLUME /root/.cache

RUN apk update

RUN apk add --no-cache \
    g++ gfortran make cmake patchelf git ccache

ARG package

RUN if [ "$package" ]; then \
        set -e; \
        pkgs=$(echo "$package" | base64 -d -); \
        echo ">>> decode package:"; \
        echo "$pkgs"; \
        for nm in $pkgs; do \
            echo ">>> install package: $nm"; \
            apk add --no-cache --force-overwrite "$nm"; \
        done; \
    fi

ARG requirement
ARG requirement_ut
ARG pip_index

RUN if [ "$requirement" ]; then \
        set -e; \
        echo "$requirement" | base64 -d - > "requirement.txt"; \
        echo ">>> decode requirement:"; \
        cat "requirement.txt"; \
        echo ">>> install python requirement:"; \
        PIP_ARGS="--timeout 300"; \
        if [ "$pip_index" ]; then \
            PIP_DOMAIN=$(echo "$pip_index" | awk -F/ '{print $3}'); \
            PIP_ARGS="$PIP_ARGS -i $pip_index --trusted-host $PIP_DOMAIN"; \
            echo ">>> pip index: $pip_index"; \
        fi; \
        pip3 install $PIP_ARGS -r "requirement.txt"; \
        rm -f "requirement.txt"; \
        if [ "$requirement_ut" ]; then \
            echo "$requirement_ut" | base64 -d - > "requirement_ut.txt"; \
            cat "requirement_ut.txt"; \
            pip3 install $PIP_ARGS -r "requirement_ut.txt"; \
            rm -f "requirement_ut.txt"; \
        fi; \
    fi


ENTRYPOINT [ "/bin/sh" ]
