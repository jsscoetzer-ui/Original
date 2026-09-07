FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    MODEL_CACHE=/models/pixrestore \
    PIXRESTORE_ROOT=/opt/PixRestore \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/csslc/PixRestore.git /opt/PixRestore \
    && cd /opt/PixRestore \
    && git checkout 909ca06e614cf72e218d52b4fa41077c603d872b

COPY requirements-worker.txt /tmp/requirements-worker.txt
RUN python -m pip install --upgrade pip \
    && pip install -r /tmp/requirements-worker.txt \
    && pip install --no-deps torchvision==0.19.0

WORKDIR /app
COPY server.py /app/server.py

EXPOSE 8000
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
