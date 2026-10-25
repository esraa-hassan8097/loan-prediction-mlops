import os

from azure.monitor.opentelemetry import configure_azure_monitor
from dotenv import load_dotenv
from opentelemetry import trace

load_dotenv()


def configure_monitoring():
    """
    Configure Azure Monitor OpenTelemetry.

    The connection string is read from:
    APPLICATIONINSIGHTS_CONNECTION_STRING
    """

    connection_string = os.getenv(
        "APPLICATIONINSIGHTS_CONNECTION_STRING"
    )

    if not connection_string:
        raise RuntimeError(
            "APPLICATIONINSIGHTS_CONNECTION_STRING is not configured."
        )

    configure_azure_monitor(
        connection_string=connection_string
    )


tracer = trace.get_tracer("loan-prediction-agent")


