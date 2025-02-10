ARG CUDA_VERSION=11.8
ARG BASE_TARGET=cuda${CUDA_VERSION}
FROM amd64/almalinux:8 as base

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

ARG DEVTOOLSET_VERSION=11

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

RUN yum -y update
RUN yum -y install epel-release
RUN yum install -y sudo wget curl perl util-linux xz bzip2 git patch which perl zlib-devel openssl-devel yum-utils autoconf automake make gcc-toolset-${DEVTOOLSET_VERSION}-toolchain unzip bison diffutils automake which file vim libffi-devel ncurses-devel sqlite-devel readline-devel tk-devel gdbm-devel glibc-devel libstdc++-devel glib2-devel libX11-devel libXext-devel libXrender-devel  mesa-libGL-devel libICE-devel libSM-devel ncurses-devel freetype-devel libpng-devel kernel-devel kmod

# Just add everything as a safe.directory for git since these will be used in multiple places with git
RUN git config --global --add safe.directory '*'
RUN ln -s /opt/rh/gcc-toolset-${DEVTOOLSET_VERSION}/root/usr/lib/gcc/x86_64-redhat-linux/${DEVTOOLSET_VERSION} /usr/lib/gcc/x86_64-redhat-linux/${DEVTOOLSET_VERSION}
ENV PATH=/opt/rh/gcc-toolset-${DEVTOOLSET_VERSION}/root/usr/bin:$PATH

# cmake-3.18.4
WORKDIR /home
RUN wget -q https://cmake.org/files/v3.18/cmake-3.18.0-Linux-x86_64.tar.gz && tar -zxvf cmake-3.18.0-Linux-x86_64.tar.gz && rm cmake-3.18.0-Linux-x86_64.tar.gz
ENV PATH=/home/cmake-3.18.0-Linux-x86_64/bin:$PATH

# go-1.15.12
WORKDIR /home
RUN wget --no-check-certificate -qO- https://paddle-ci.gz.bcebos.com/go1.15.12.linux-amd64.tar.gz | \
    tar -xz -C /usr/local && \
    mkdir /root/gopath && \
    mkdir /root/gopath/bin && \
    mkdir /root/gopath/src
ENV GOROOT=/usr/local/go GOPATH=/root/gopath
ENV PATH=/usr/local/ssl:${GOROOT}/bin:${GOPATH}/bin:${PATH}

FROM base as python
# Install python
ADD ./common/install_python.sh install_python.sh
RUN bash ./install_python.sh && rm install_python.sh

FROM base as openssl
# Install openssl
ADD ./common/install_openssl.sh install_openssl.sh
RUN bash ./install_openssl.sh && rm install_openssl.sh

FROM base as patchelf
# Install patchelf
ADD ./common/install_patchelf.sh install_patchelf.sh
RUN bash ./install_patchelf.sh && rm install_patchelf.sh

FROM base as ccache
# Install ccache
ADD ./common/install_ccache.sh install_ccache.sh
RUN bash ./install_ccache.sh && rm install_ccache.sh


# Install CUDA
FROM base as cuda
ARG CUDA_VERSION=12.4
RUN rm -rf /usr/local/cuda-*
ENV CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
ENV CUDA_VERSION=${CUDA_VERSION}
ENV PATH=/usr/local/cuda-${CUDA_VERSION}/bin:$PATH
ADD ./common/install_cuda.sh install_cuda.sh

FROM cuda as cuda11.8
RUN bash ./install_cuda.sh 11.8
ENV DESIRED_CUDA=11.8

FROM cuda as cuda12.3
RUN bash ./install_cuda.sh 12.3
ENV DESIRED_CUDA=12.3

FROM cuda as cuda12.4
RUN bash ./install_cuda.sh 12.4
ENV DESIRED_CUDA=12.4
ENV DESIRED_CUDA=12.6

FROM cuda as cuda12.6
RUN bash ./install_cuda.sh 12.6
ENV DESIRED_CUDA=12.6


# Install paddle
FROM python as paddle
RUN wget https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/python/requirements.txt -O /root/requirements.txt

RUN LD_LIBRARY_PATH=/opt/_internal/cpython-3.8.0/lib/:${LD_LIBRARY_PATH} /opt/_internal/cpython-3.8.0/bin/pip3 install setuptools pyyaml wheel -U && \
    LD_LIBRARY_PATH=/opt/_internal/cpython-3.9.0/lib/:${LD_LIBRARY_PATH} /opt/_internal/cpython-3.9.0/bin/pip3 install setuptools pyyaml wheel -U && \
    LD_LIBRARY_PATH=/opt/_internal/cpython-3.10.0/lib/:${LD_LIBRARY_PATH} /opt/_internal/cpython-3.10.0/bin/pip3 install setuptools pyyaml wheel -U && \
    LD_LIBRARY_PATH=/opt/_internal/cpython-3.11.0/lib/:${LD_LIBRARY_PATH} /opt/_internal/cpython-3.11.0/bin/pip3 install setuptools pyyaml wheel -U && \
    LD_LIBRARY_PATH=/opt/_internal/cpython-3.12.0/lib/:${LD_LIBRARY_PATH} /opt/_internal/cpython-3.12.0/bin/pip3 install setuptools pyyaml wheel -U && \
    LD_LIBRARY_PATH=/opt/_internal/cpython-3.13.0/lib/:${LD_LIBRARY_PATH} /opt/_internal/cpython-3.13.0/bin/pip3 install setuptools pyyaml wheel -U

RUN LD_LIBRARY_PATH=/opt/_internal/cpython-3.8.0/lib/:${LD_LIBRARY_PATH} /opt/_internal/cpython-3.8.0/bin/pip3 install -r /root/requirements.txt && \
    LD_LIBRARY_PATH=/opt/_internal/cpython-3.9.0/lib/:${LD_LIBRARY_PATH} /opt/_internal/cpython-3.9.0/bin/pip3 install -r /root/requirements.txt && \
    LD_LIBRARY_PATH=/opt/_internal/cpython-3.10.0/lib/:${LD_LIBRARY_PATH} /opt/_internal/cpython-3.10.0/bin/pip3 install -r /root/requirements.txt && \
    LD_LIBRARY_PATH=/opt/_internal/cpython-3.11.0/lib/:${LD_LIBRARY_PATH} /opt/_internal/cpython-3.11.0/bin/pip3 install -r /root/requirements.txt && \
    LD_LIBRARY_PATH=/opt/_internal/cpython-3.12.0/lib/:${LD_LIBRARY_PATH} /opt/_internal/cpython-3.12.0/bin/pip3 install -r /root/requirements.txt && \
    LD_LIBRARY_PATH=/opt/_internal/cpython-3.13.0/lib/:${LD_LIBRARY_PATH} /opt/_internal/cpython-3.13.0/bin/pip3 install -r /root/requirements.txt
RUN go get github.com/Masterminds/glide
RUN rm -rf /root/requirements.txt

ENV LD_LIBRARY_PATH=/opt/rh/gcc-toolset-${DEVTOOLSET_VERSION}/root/usr/lib64:/opt/rh/gcc-toolset-${DEVTOOLSET_VERSION}/root/usr/lib:$LD_LIBRARY_PATH
RUN rm -rf /usr/local/cuda
RUN chmod o+rw /usr/local


# Final step
FROM ${BASE_TARGET} as final
COPY --from=openssl          /usr/local/ssl          /usr/local/ssl
COPY --from=patchelf         /usr/local/bin/patchelf /usr/local/bin/patchelf
COPY --from=ccache           /usr/local/bin/ccache   /usr/local/bin/ccache
COPY --from=paddle           /opt/_internal          /opt/_internal
