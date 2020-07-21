FROM tensorflow/tensorflow:latest-gpu-jupyter
ARG PYTHON_VERSION=3.7

ENV DEBIAN_FRONTEND=noninteractive
ENV HOME="/workspace"
ENV PYTHONPATH="$HOME/source:$PYTHONPATH"
ENV NOTEBOOKS="$HOME/notebook"

# Install base dependencies
RUN apt-get update                                            && \
    apt-get -y --no-install-recommends install                   \
        apt-utils                                                \
        python3-opencv                                           \
        python3-pip                                              \
        python3-setuptools                                       \
        git vim curl                                             \
        ffmpeg                                                   \
        build-essential                                          \
        libsm6 libxrender1 libfontconfig1                        \
        wget unzip tig                                           \
                                                              && \
    python3 -m pip install --upgrade pip                         \
                                                              && \
    pip install matplotlib                                       \
                jupyterlab                                       \
                ipympl                                           \
                tensorflow-probability                           \
                                                              && \
   rm -rf /var/lib/apt/lists/*

# Install the required packages
COPY ./requirements.txt  requirements.txt
RUN pip install -r requirements.txt

## Jupyter directory
RUN mkdir -p /.jupyter && chmod -R a+rwx /.jupyter

# Install Anaconda
ARG ANACONDA="https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh"
RUN curl -o ~/miniconda.sh -O ${ANACONDA}            && \
     chmod +x ~/miniconda.sh                         && \
     ~/miniconda.sh -b -p /opt/conda                 && \
     rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH

WORKDIR /opt/pytorch
COPY . .


EXPOSE 8888
EXPOSE 6060
CMD ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]


WORKDIR /workspace
RUN chmod -R a+rwx /workspace
