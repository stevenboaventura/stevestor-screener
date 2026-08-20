FROM python:3.11-slim

WORKDIR /app

COPY screener_requirements.txt .
RUN pip install --no-cache-dir -r screener_requirements.txt

COPY screener.py .

CMD ["python", "screener.py"]
