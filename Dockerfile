# 1. Lightweight Python image (pinned to bookworm for stable system packages; matches the local 3.12 venv)
FROM python:3.12-slim-bookworm

# Unbuffered logs (model-loading messages show up in `docker logs`); a fixed Hugging Face
# cache path that docker-compose.yml persists in a volume, so weights download once.
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/code/.cache/huggingface

# 2. Set the working directory
WORKDIR /code

# 3. Install system dependencies (glib for OpenCV; git for the MobileSAM package)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. CPU-only torch first, from the PyTorch CPU index. On linux/aarch64 PyPI's default torch
#    wheel pulls in several GB of CUDA libraries, which a CPU server can't use.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 5. Remaining requirements. No --upgrade, so the CPU torch above counts as satisfied.
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# 6. Create necessary directories to prevent path errors
RUN mkdir -p /code/catalog_tiles /code/uploads /code/outputs /code/.cache/huggingface

# 7. Copy the project into the container
COPY . /code

# 8. Expose the port Hugging Face Spaces requires (docker-compose maps it to the host)
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/catalog', timeout=4)" || exit 1

# 9. Start the FastAPI server on port 7860. One worker: models live in process memory
#    and inference is serialised by a lock, so extra workers would only duplicate RAM.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
