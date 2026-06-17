FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

COPY risk.py .

EXPOSE 8001

CMD ["uvicorn", "risk:app", "--host", "0.0.0.0", "--port", "8001"]
