FROM python:3.12-slim

LABEL org.opencontainers.image.title="freetpu-tpu-exporter" \
      org.opencontainers.image.description="Clean-room Prometheus telemetry exporter for Google Cloud TPUs (0x1ae0 / VFIO)" \
      org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /opt/freetpu
COPY tpu_exporter.py /opt/freetpu/tpu_exporter.py

EXPOSE 9400

ENTRYPOINT ["python3", "/opt/freetpu/tpu_exporter.py"]
CMD ["--port", "9400"]
