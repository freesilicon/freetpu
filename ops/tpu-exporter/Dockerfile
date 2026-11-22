FROM python:3.12-slim

ARG INSTALL_TPU_INFO=0

LABEL org.opencontainers.image.title="freetpu-tpu-exporter" \
      org.opencontainers.image.description="Clean-room Prometheus telemetry exporter for Google Cloud TPUs (0x1ae0 / VFIO)" \
      org.opencontainers.image.licenses="Apache-2.0"

# Set --build-arg INSTALL_TPU_INFO=1 when building the workload-sidecar image
# that queries a co-located libtpu process on localhost:8431 for live HBM and
# MXU duty-cycle metrics.
RUN if [ "$INSTALL_TPU_INFO" = "1" ]; then \
      pip install --no-cache-dir tpu-info; \
    fi

WORKDIR /opt/freetpu
COPY tpu_exporter.py /opt/freetpu/tpu_exporter.py

EXPOSE 9400

# Default bind and loopback health flags live in ENTRYPOINT so passing extra
# runtime flags (e.g. `--no-pid-scan --runtime-metrics-addr 127.0.0.1:8431`)
# preserves `--host 0.0.0.0` and `--health-port 9401` while still allowing
# explicit flag overrides via CMD/args.
ENTRYPOINT ["python3", "/opt/freetpu/tpu_exporter.py", "--host", "0.0.0.0", "--port", "9400", "--health-host", "127.0.0.1", "--health-port", "9401"]
CMD []
