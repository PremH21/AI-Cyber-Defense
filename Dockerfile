FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install CPU-only torch first (avoids pulling in ~2GB of unnecessary CUDA/GPU packages)
RUN pip install --no-cache-dir --timeout 120 --retries 5 \
    torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 --retries 5 -r requirements.txt

# Copy only what the API needs at runtime
COPY api/ ./api/
COPY ml-engine/models/ ./ml-engine/models/
COPY response-engine/models/ ./response-engine/models/
COPY data/processed/unsw-nb15_test.parquet ./data/processed/unsw-nb15_test.parquet

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
