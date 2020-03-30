FROM pytorch/pytorch:latest

ENV TZ=Europe/Rome
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
                                build-essential \
                                cmake \
                                git \
                                libavcodec-dev \
                                libavformat-dev \
                                libswscale-dev \
                                libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
                                libgtk2.0-dev \
                                libgtk-3-dev \
                                libpng-dev \
                                libjpeg-dev \
                                libopenexr-dev \
                                libtiff-dev \
                                libtbb2 \
                                libtbb-dev \
                                libwebp-dev \
                                qtbase5-dev \
                                qtdeclarative5-dev \
                                qttools5-dev \
                                python3-setuptools \
                                python3-pip \
                                git \
                                wget \
                                unzip \
                                yasm \
                                cython \
                                && rm -rf /var/lib/apt/lists/*

RUN conda install Cython numpy scipy matplotlib scikit-learn

WORKDIR /
ENV OPENCV_VERSION="4.1.1"
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
    && unzip ${OPENCV_VERSION}.zip \
    && mkdir /opencv-${OPENCV_VERSION}/cmake_binary \
    && cd /opencv-${OPENCV_VERSION}/cmake_binary \
    && cmake -DBUILD_TIFF=ON \
    -DBUILD_opencv_java=OFF \
    -DWITH_CUDA=OFF \
    -DWITH_OPENGL=ON \
    -DWITH_OPENCL=ON \
    -DWITH_IPP=ON \
    -DWITH_TBB=ON \
    -DWITH_EIGEN=ON \
    -DWITH_V4L=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=$(python3.7 -c "import sys; print(sys.prefix)") \
    -DPYTHON_EXECUTABLE=$(which python3.7) \
    -DPYTHON_INCLUDE_DIR=$(python3.7 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -DPYTHON_PACKAGES_PATH=$(python3.7 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
    .. \
    && make -j install \
    && rm /${OPENCV_VERSION}.zip \
    && rm -r /opencv-${OPENCV_VERSION}

RUN conda install pandas

RUN mkdir -p /root/.cache/torch/checkpoints/ && \
    wget https://download.pytorch.org/models/resnet34-333f7ec4.pth -P /root/.cache/torch/checkpoints/