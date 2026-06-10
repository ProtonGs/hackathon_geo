FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python manage.py migrate --no-input

EXPOSE 8000

CMD ["gunicorn", "geo_project.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120"]
