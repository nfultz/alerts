FROM sagemaker-mxnet:1.6.0-gpu-py3


RUN pip install  bert-embedding --no-deps
RUN pip install xgboost
RUN pip install hyperopt
RUN pip install bs4 boto3

COPY *.py /usr/local/bin/
#docker tag sagemaker-mxnet:1.6.0-gpu-py3 887983324737.dkr.ecr.us-east-1.amazonaws.com/sagemaker-mxnet:1.6.0-gpu-py3

ENV AWS_DEFAULT_REGION=us-east-1
ENTRYPOINT ["python3", "/usr/local/bin/analysis.py"]