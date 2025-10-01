import time
from typing import Callable

from fastapi import Request, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

REQUEST_TIME = Histogram("http_request_latency_ms","Request latency",
                         buckets=(5,10,25,50,100,250,500,1000,2500,5000))
REQUESTS = Counter("http_requests_total", "HTTP requests", ["method","endpoint","status"])
ERRORS = Counter("errors_total", "Error count", ["endpoint"])
EXPLORE_RATE = Histogram("explore_rate", "Exploration rate (0-1)",
                         buckets=(0.0,0.01,0.02,0.05,0.1,0.2,0.5,1.0))
CTR = Histogram("ctr_observed", "Observed click/reward rate", buckets=(0,1))

def metrics_endpoint() -> Response:
    from starlette.responses import Response
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

async def metrics_middleware(request: Request, call_next: Callable) -> Response:
    start = time.time()
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception as e:
        status = 500
        ERRORS.labels(endpoint=request.url.path).inc()
        raise
    finally:
        duration_ms = (time.time() - start) * 1000.0
        REQUEST_TIME.observe(duration_ms)
        REQUESTS.labels(method=request.method, endpoint=request.url.path,
                        status=str(status)).inc()
    return response
