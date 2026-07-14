FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# Dipendenze di sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Dipendenze Python (installate durante il build, con rete)
COPY pyproject.toml .
RUN pip install --no-cache-dir timm pandas tqdm Pillow torchvision

# Codice sorgente
COPY src/ src/
COPY scripts/infer.py scripts/infer.py

# Pesi DINOv2 (copiati dal filesystem locale durante il build)
# Vedi docker_build.sh per come prepararli
COPY docker_weights/dinov2/ /model/dinov2/

# Checkpoint del modello migliore (tiled ViT, epoch-7)
COPY checkpoints/epoch-7.pt /model/checkpoint.pt

ENV PYTHONPATH=/app/src
ENV HF_HOME=/model/dinov2

ENTRYPOINT ["python", "scripts/infer.py"]
