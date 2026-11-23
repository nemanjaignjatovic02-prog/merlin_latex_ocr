FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# CPU-only PyTorch (smaller image; free hosting has no GPU)
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user
EXPOSE 7860
# Some hosts (e.g. Google Cloud Run) set PORT; the default 7860 matches Hugging Face Spaces
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
