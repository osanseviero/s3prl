# We need this to use GPUs inside the container
FROM nvidia/cuda:10.2-base
# Using a multi-stage build simplifies the s3prl installation
# TODO: find a slimmer base image that also "just works"
FROM tiangolo/uvicorn-gunicorn:python3.8


RUN apt-get update --fix-missing && apt-get install -y wget \
    libsndfile1 \
    sox \
    git

RUN python -m pip install --upgrade pip
RUN python -m pip --no-cache-dir install fairseq@git+https://github.com//pytorch/fairseq.git@f2146bdc7abf293186de9449bfa2272775e39e1d#egg=fairseq
RUN python -m pip --no-cache-dir install git+https://github.com/osanseviero/s3prl.git@huggingface2#egg=s3prl

COPY s3prl/ /app/s3prl
COPY src/ /app/src
WORKDIR /app

# Setup filesystem
RUN mkdir data

# Fine-tune!
# python run_downstream.py -n asr-test -m train -u fbank -d asr --device cpu -o "config.downstream_expert.datarc.dict_path='./downstream/asr/char.dict',,config.downstream_expert.datarc.libri_root='/app/data/LibriSpeech',,config.downstream_expert.datarc.batch_size=1,,config.downstream_expert.datarc.bucket_file='/app/data/LibriSpeech/len_for_bucket'"
# CMD ["python", "run_downstream.py", "-n", "asr-test", "-m", "train", "-u", "fbank", "-d", "asr"]