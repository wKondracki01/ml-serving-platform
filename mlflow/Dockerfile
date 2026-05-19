FROM python:3.11-slim
RUN pip install mlflow psycopg2-binary
EXPOSE 5000
CMD mlflow server --backend-store-uri ${BACKEND_STORE_URI} --host 0.0.0.0 --port 5000 --allowed-hosts "*mlflow:5000*,*localhost:5001*"