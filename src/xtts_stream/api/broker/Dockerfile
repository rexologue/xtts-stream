FROM python:3.10-slim

WORKDIR /app

COPY src/xtts_stream/api/broker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src /app/src
COPY broker.config.example.yaml /app/broker.config.example.yaml

ENV PYTHONPATH=/app/src
ENV XTTS_BROKER_CONFIG_FILE=/app/broker.config.example.yaml

EXPOSE 9000

CMD ["python", "-m", "xtts_stream.api.broker.server"]
