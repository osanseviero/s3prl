FROM python:3.8-slim-buster

RUN apt-get update --fix-missing && apt-get install -y wget \
    sndfile-tools \
    sox \
    git

RUN python -m pip install --upgrade pip
# TODO: Replace with correct branch git+https://github.com/osanseviero/s3prl.git@huggingface2#egg=s3prl
RUN python -m pip --no-cache-dir install s3prl
RUN python -m pip --no-cache-dir install git+https://github.com/pytorch/fairseq.git@f2146bd#egg=fairseq

COPY s3prl/ /app
WORKDIR /app

# Setup filesystem
RUN mkdir data

# Fine-tune!
# CMD ["python", "run_downstream.py", "-n", "asr-test", "-m", "train", "-u", "fbank", "-d", "asr"]