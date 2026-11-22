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

# Default bind, loopback health, and `--no-pid-scan` flags live in ENTRYPOINT so
# containers in Docker's default private PID namespace never report misleading
# `freetpu_tpu_pid_telemetry_available=1` with zero active PIDs, while extra
# runtime flags (e.g. `--runtime-metrics-addr 127.0.0.1:8431`) preserve
# `--host 0.0.0.0` and `--health-port 9401`. To opt into host PID scanning, run
# with `--pid=host --cap-add=SYS_PTRACE` (and an AppArmor profile permitting
# cross-profile `/proc/<pid>/fd` reads) and pass `--pid-scan`.
ENTRYPOINT ["python3", "/opt/freetpu/tpu_exporter.py", "--host", "0.0.0.0", "--port", "9400", "--health-host", "127.0.0.1", "--health-port", "9401", "--no-pid-scan"]
CMD []
