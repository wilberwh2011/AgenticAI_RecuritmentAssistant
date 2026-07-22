import base64
import os

from opentelemetry import metrics, trace
from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def init_tracer():
    # public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    # secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    # auth_header = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()

    # provider = TracerProvider(
    #     resource=Resource.create({"service.name": "recruitment-assistant"})
    # )
    # exporter = OTLPSpanExporter(
    #     endpoint="http://localhost:3000/api/public/otel/v1/traces",
    #     headers={"Authorization": f"Basic {auth_header}"},
    # )
    # provider.add_span_processor(BatchSpanProcessor(exporter))
    # trace.set_tracer_provider(provider)

    """Attach our own OTLP exporter onto whatever global TracerProvider already
    exists (Langfuse's CallbackHandler registers one on import) — do NOT call
    trace.set_tracer_provider() here, that would conflict with Langfuse's own.
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    auth_header = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()

    provider = trace.get_tracer_provider()  # the existing global one, set by Langfuse

    # manual OTel path → Langfuse
    langfuse_exporter = OTLPSpanExporter(
        endpoint="http://localhost:3000/api/public/otel/v1/traces",
        headers={"Authorization": f"Basic {auth_header}"},
    )
    provider.add_span_processor(BatchSpanProcessor(langfuse_exporter))  # type: ignore

    # same spans, also → Cloud Trace
    gcp_project = os.getenv("GOOGLE_CLOUD_PROJECT")
    gcp_exporter = CloudTraceSpanExporter(project_id=gcp_project)
    provider.add_span_processor(BatchSpanProcessor(gcp_exporter))  # type: ignore


from opentelemetry.sdk.resources import Resource

_meter_initialized = False


def init_meter():
    """Independent pipeline: metrics (counters/histograms), not spans."""

    global _meter_initialized
    if _meter_initialized:
        return

    resource = Resource.create({"service.name": "recruitment-assistant"})
    gcp_project = os.getenv("GOOGLE_CLOUD_PROJECT")

    reader = PeriodicExportingMetricReader(
        CloudMonitoringMetricsExporter(project_id=gcp_project),
        export_interval_millis=15000,  # push every 15s; Cloud Monitoring has ingestion rate limits, don't go much lower
    )

    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(meter_provider)

    _meter_initialized = True


def shutdown_tracer():
    trace.get_tracer_provider().shutdown()  # trace.get_tracer_provider().force_flush()  # type: ignore


import logging


# Suppress Cloud Monitoring export errors specifically at shutdown, since we've established they're non-actionable noise
def shutdown_meter():
    logging.getLogger("opentelemetry.exporter.cloud_monitoring").setLevel(
        logging.CRITICAL
    )
    metrics.get_meter_provider().shutdown()  # type: ignore
