FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY training ./training
COPY data ./data
RUN pip install --no-cache-dir -e ".[training]"

ENTRYPOINT ["python", "training/verifier/train_jax_tpu.py"]

