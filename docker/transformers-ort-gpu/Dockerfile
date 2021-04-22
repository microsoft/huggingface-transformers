# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# docker build --network=host --file Dockerfile --tag wezhan:ortmodule-dev .

# multi-stage arguments (repeat ARG NAME below)
ARG UCX_VERSION=1.10.0
ARG OPENMPI_VERSION=4.0.4
ARG CONDA_VERSION=latest
ARG NUMPY_VERSION=1.18.5
ARG ONNX_VERSION=1.7.0
ARG PYTORCH_VERSION=1.8.0

ARG BUILD_CONFIG=Release
ARG OPENMPI_PATH=/opt/openmpi-${OPENMPI_VERSION}
ARG ORT_COMMIT=master

# cuda development image for building sources
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04 as builder

# set location for builds
WORKDIR /stage

# install curl, git, ssh (required by MPI when running ORT tests)
RUN apt-get -y update &&\
    apt-get -y --no-install-recommends install \
        curl wget \
        git \
        vim \
        emacs \
        gcc \
        g++ \
        ninja-build \
        language-pack-en \
        openssh-client \
        unattended-upgrades

# update existing packages to minimize security vulnerabilities
RUN unattended-upgrade

RUN locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8

# install miniconda (comes with python 3.8 default)
ARG CONDA_VERSION
ARG CONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh
RUN cd /stage && curl -fSsL --insecure ${CONDA_URL} -o install-conda.sh &&\
    /bin/bash ./install-conda.sh -b -p /opt/conda &&\
    /opt/conda/bin/conda clean -ya
ENV PATH=/opt/conda/bin:${PATH}

RUN pip install --upgrade pip

# install cmake, setuptools, numpy, and onnx
ARG NUMPY_VERSION
ARG ONNX_VERSION
RUN conda install -y \
        setuptools \
        cmake \
        numpy=${NUMPY_VERSION} &&\
    pip install \
        onnx=="${ONNX_VERSION}"

# install cerberus for the new pytorch front-end
RUN pip install cerberus

# build ucx suite
# note: openmpi will not select ucx without multithreading enabled
ARG UCX_VERSION
ARG UCX_TARNAME=ucx-$UCX_VERSION
ARG UCX_URL=https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/${UCX_TARNAME}.tar.gz
RUN apt-get -y update && apt-get -y --no-install-recommends install \
        libibverbs-dev \
        libnuma-dev &&\
    cd /stage && curl -fSsL ${UCX_URL} | tar xzf - &&\
    cd ${UCX_TARNAME} &&\
    ./configure \
        --prefix=/opt/ucx \
        --with-cuda=/usr/local/cuda \
        --with-verbs=/usr/lib/x86_64-linux-gnu \
        --enable-mt &&\
    make -j"$(nproc)" &&\
    make install

# build openmpi (use --prefix /opt/openmpi-xxx to move to runtime image)
# note: require --enable-orterun-prefix-by-default for Azure machine learning compute
# note: disable verbs as we use ucx middleware and don't want btl openib warnings
ARG OPENMPI_VERSION
ARG OPENMPI_PATH
ARG OPENMPI_TARNAME=openmpi-${OPENMPI_VERSION}
ARG OPENMPI_URL=https://download.open-mpi.org/release/open-mpi/v%OMPI_BASE%/${OPENMPI_TARNAME}.tar.gz
RUN export OMPI_BASE=${OPENMPI_VERSION%.*} &&\
    cd /stage && curl -fSsL `echo ${OPENMPI_URL} | sed s/%OMPI_BASE%/$OMPI_BASE/` | tar xzf - &&\
    cd ${OPENMPI_TARNAME} &&\
    ./configure \
        --prefix=${OPENMPI_PATH} \
        --with-ucx=/opt/ucx \
        --without-verbs \
        --with-cuda=/usr/local/cuda \
        --enable-mpirun-prefix-by-default \
        --enable-orterun-prefix-by-default \
        --enable-mca-no-build=btl-uct &&\
    make -j"$(nproc)" install &&\
    ldconfig
ENV PATH=${OPENMPI_PATH}/bin:$PATH
ENV LD_LIBRARY_PATH=${OPENMPI_PATH}/lib:$LD_LIBRARY_PATH

# install mpi4py (be sure to link existing /opt/openmpi-xxx)
RUN CC=mpicc MPICC=mpicc pip install mpi4py --no-binary mpi4py

# build onnxruntime wheel with cuda and mpi support
ARG BUILD_CONFIG
RUN cd /stage && git clone https://github.com/microsoft/onnxruntime.git &&\
   cd onnxruntime &&\
   git checkout ${ORT_COMMIT} &&\
   cp ThirdPartyNotices.txt /stage/ThirdPartyNotices.txt &&\
   cp dockerfiles/LICENSE-IMAGE.txt /stage/LICENSE-IMAGE.txt &&\
   export CUDACXX=/usr/local/cuda-11.1/bin/nvcc &&\
   python tools/ci_build/build.py \
       --cmake_extra_defines \
           ONNXRUNTIME_VERSION=`cat ./VERSION_NUMBER` \
       --config ${BUILD_CONFIG} \
       --enable_training \
       --mpi_home ${OPENMPI_PATH} \
       --use_cuda \
       --cuda_home /usr/local/cuda-11.1/ \
       --cudnn_home /usr/local/cuda-11.1/ \
       --nccl_home /usr/lib/x86_64-linux-gnu/ \
       --update \
       --parallel \
       --build_dir build \
       --build \
       --build_wheel \
       --skip_tests &&\
   pip install build/${BUILD_CONFIG}/dist/*.whl

# install pytorch
ARG PYTORCH_VERSION
# RUN pip install torch==${PYTORCH_VERSION}
# RUN pip install --pre torch torchvision torchtext -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html
RUN cd /stage && wget -q https://download.pytorch.org/whl/nightly/cu111/torch-1.9.0.dev20210414%2Bcu111-cp38-cp38-linux_x86_64.whl && \ 
    pip install torch-1.9.0.dev20210414+cu111-cp38-cp38-linux_x86_64.whl
# RUN pip install --pre --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ ort-gpu-nightly-training
# RUN pip install torch-ort
# RUN conda install -c anaconda cudatoolkit=11.1

#Install huggingface transformers
# RUN cd /stage && git clone https://github.com/microsoft/huggingface-transformers.git &&\
#     cd huggingface-transformers &&\
#     git checkout master &&\
#     pip install -e .
RUN cd /stage && git clone https://github.com/ashbhandare/transformers &&\
    cd transformers &&\
    git checkout aibhanda/ortmodule_t5

# Install AzureML support and commonly used packages.
RUN pip install azureml-defaults sentencepiece==0.1.92 msgpack==1.0.0 tensorboardX==1.8 tensorboard==2.3.0
RUN pip install wget sklearn h5py sympy nltk sacrebleu
RUN pip install boto3 requests onnx datasets fairscale sacremoses
RUN pip install git+https://github.com/NVIDIA/dllogger.git
# RUN pip install deepspeed

#AzureNL dependencies
RUN conda install ruamel
RUN conda install ruamel.yaml

ENV PATH=/opt/conda/bin:${PATH}

ADD run_translation.sh /stage/transformers/examples/seq2seq

#WORKDIR /workspace
WORKDIR /stage/transformers/examples/seq2seq
