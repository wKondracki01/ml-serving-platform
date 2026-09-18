FROM python:3.11-slim AS builder
WORKDIR /install
COPY v1/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY common ./common
COPY v1 ./v1
EXPOSE 8000
CMD ["uvicorn", "v1.app:app", "--host", "0.0.0.0", "--port", "8000"]
