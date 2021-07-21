FROM python:3.8-slim-buster

RUN apt-get update --fix-missing && apt-get install -y wget \
    sndfile-tools \
    sox

RUN python -m pip --no-cache-dir install git+https://github.com/osanseviero/s3prl.git@huggingface2#egg=s3prl
RUN python -m pip --no-cache-dir install git+https://github.com/pytorch/fairseq.git@f2146bd#egg=fairseq

COPY s3prl/ /app
WORKDIR /app
# CMD ["python", "run_downstream.py", "-n", "asr-test", "-m", "train", "-u", "fbank", "-d", "asr"]