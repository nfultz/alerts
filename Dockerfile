FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu18.04

RUN apt-get update && apt-get  -y install python3.8 python3.8-dev python3-pip
#RUN apt-get install -y  wget gfortran libcublas-11-6 libcublas-dev-11-6 libatlas-base-dev


RUN python3.8 -m pip install --upgrade pip setuptools wheel 

#RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.0-rc2/cmake-3.26.0-rc2-linux-x86_64.tar.gz -O - | tar xz -C /opt && ln -s /opt/cmake*/bin/cmake /usr/local/bin/cmake

RUN python3.8 -m pip install numpy==1.22.4 pandas==1.4.4

RUN python3.8 -m pip install mxnet-cu116==1.9.1  gluonnlp --no-deps


#RUN python3.8 -m pip install Cython pybind11 xgboost pythran

RUN python3.8 -m pip install xgboost
RUN python3.8 -m pip install hyperopt
RUN python3.8 -m pip install bs4 boto3
RUN python3.8 -m pip install scikit-learn


RUN python3.8 -m pip install requests packaging


COPY *.py /usr/local/bin/
COPY entry.sh /usr/local/bin
#docker tag sagemaker-mxnet:1.6.0-gpu-py3 887983324737.dkr.ecr.us-east-1.amazonaws.com/sagemaker-mxnet:1.6.0-gpu-py3

RUN python3.8 /usr/local/bin/download-model.py
#
ENV AWS_DEFAULT_REGION=us-east-1
ENTRYPOINT ["bash", "/usr/local/bin/entry.sh"]
