FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04
RUN echo > /etc/apt/sources.list.d/cuda.list # Ignore CUDA GPG keys
WORKDIR /opt/
ENV PYTHON_VERSION='3.10.8'
RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends wget curl unzip \
   build-essential openssh-client gcc libdb-dev libleveldb-dev libsodium-dev zlib1g-dev libtinfo-dev \
   libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev liblzma-dev
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz \
    && tar xvf Python-${PYTHON_VERSION}.tar.xz && cd Python-${PYTHON_VERSION} && ./configure && make altinstall
RUN cp Python-${PYTHON_VERSION}/python /usr/bin
RUN python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install torch_geometric pyg_lib torch_scatter torch_sparse \
    torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
RUN python -m pip install ogb