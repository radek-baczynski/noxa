from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from noxa.model_stats import reset_model_infer_stats
from noxa.request_context import log_prefix, request_id_var

logger = logging.getLogger("noxa.request")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        token = request_id_var.set(request_id)
        reset_model_infer_stats()
        start = time.perf_counter()
        path = request.url.path
        method = request.method
        logger.info("%s→ %s %s", log_prefix(), method, path)
        try:
            response = await call_next(request)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.info(
                "%s← %s %s status=%d latency=%dms",
                log_prefix(),
                method,
                path,
                response.status_code,
                elapsed_ms,
            )
            response.headers["x-request-id"] = request_id
            return response
        except Exception:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.exception(
                "%s← %s %s failed latency=%dms",
                log_prefix(),
                method,
                path,
                elapsed_ms,
            )
            raise
        finally:
            request_id_var.reset(token)
