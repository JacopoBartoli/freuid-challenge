#!/usr/bin/env bash
# Prepara i pesi DINOv2 e costruisce l'immagine Docker.
# Eseguire dalla root del progetto.
set -e

HF_CACHE="$HOME/.cache/huggingface/hub/models--timm--vit_base_patch14_dinov2.lvd142m"
DOCKER_WEIGHTS="docker_weights/dinov2/hub/models--timm--vit_base_patch14_dinov2.lvd142m"

echo "Copio pesi DINOv2 in docker_weights/..."
mkdir -p "$DOCKER_WEIGHTS"
cp -r "$HF_CACHE/." "$DOCKER_WEIGHTS/"
echo "Pesi copiati: $(du -sh docker_weights/ | cut -f1)"

echo "Build immagine Docker..."
docker build -t freuid-inference:latest .

echo ""
echo "Build completata. Test con:"
echo "  docker run --gpus all --network none \\"
echo "    -v /path/to/test/images:/data:ro \\"
echo "    -v /path/to/output:/submissions \\"
echo "    freuid-inference:latest"
