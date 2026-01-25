FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OpenCV and SAM-2
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install SAM-2 from source (placeholder until pypi release or manual wheel)
# RUN pip install git+https://github.com/facebookresearch/sam2.git

COPY . .

CMD ["celery", "-A", "worker.celery_app", "worker", "--loglevel=info", "-Q", "sam3_queue"]
