FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so Docker layer-caches them.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application.
COPY app ./app
COPY alembic.ini .
COPY alembic ./alembic
COPY entrypoint.sh .

# Migrate (alembic upgrade head) then serve — see entrypoint.sh.
RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
