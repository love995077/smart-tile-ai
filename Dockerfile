# 1. Use a lightweight Python image
FROM python:3.11-slim

# 2. Set the working directory
WORKDIR /code

# 3. Install system dependencies (OpenCV runtime libs; git for the MobileSAM package)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy and install Python requirements
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 5. Create necessary directories to prevent path errors
RUN mkdir -p /code/catalog_tiles /code/uploads /code/outputs

# 6. Copy your entire project into the container
COPY . /code

# 7. Expose the port Hugging Face requires
EXPOSE 7860

# 8. Start the FastAPI server on port 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]