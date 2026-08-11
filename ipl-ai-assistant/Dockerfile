FROM python:3.12-slim

WORKDIR /app

# Torch CPU wheels keep the image ~2GB smaller than default CUDA builds.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV DATA_DIR=/data
VOLUME /data

EXPOSE 8000 8765

CMD ["python", "app.py", "api"]
