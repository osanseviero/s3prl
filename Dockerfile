FROM tiangolo/uvicorn-gunicorn:python3.8

RUN apt-get update --fix-missing && apt-get install -y wget \
    libsndfile1 \
    sox \
    git

RUN python -m pip install --upgrade pip
RUN python -m pip --no-cache-dir install fairseq@git+https://github.com//pytorch/fairseq.git@f2146bdc7abf293186de9449bfa2272775e39e1d#egg=fairseq
RUN python -m pip --no-cache-dir install git+https://github.com/osanseviero/s3prl.git@huggingface2#egg=s3prl

COPY s3prl/ /app
WORKDIR /app

# Setup filesystem
RUN mkdir data

# Fine-tune!
# CMD ["python", "run_downstream.py", "-n", "asr-test", "-m", "train", "-u", "fbank", "-d", "asr"]