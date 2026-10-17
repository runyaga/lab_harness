"""An ASGI app factory for soliplex that joins each request to its trace.

soliplex instruments FastAPI inside its lifespan, after uvicorn has the
app, so a server started by ``soliplex-cli serve`` opens no request span
and never reads an incoming ``traceparent``: every "AG-UI event stream"
span starts its own trace. :class:`TraceparentMiddleware` extracts the
header and attaches it as the current OpenTelemetry context for the whole
request, so spans opened while handling it, including tasks it starts,
join the client's trace.

``agui.serve`` copies this file into a directory on the server's
``PYTHONPATH`` and passes ``--workers 1 --app-factory-name <module>:app``.
It runs in the server's environment, so it imports nothing from this
package, and imports soliplex and opentelemetry only when called.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class TraceparentMiddleware:
    """Run each HTTP request inside the context its headers carry."""

    def __init__(
        self,
        app: Any,
        *,
        extract: Callable[[dict[str, str]], Any] | None = None,
        attach: Callable[[Any], Any] | None = None,
        detach: Callable[[Any], None] | None = None,
    ):
        if extract is None or attach is None or detach is None:
            from opentelemetry import context
            from opentelemetry import propagate

            extract = extract or propagate.extract
            attach = attach or context.attach
            detach = detach or context.detach
        self.app = app
        self.extract = extract
        self.attach = attach
        self.detach = detach

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        carrier = {
            name.decode("latin-1").lower(): value.decode("latin-1")
            for name, value in scope.get("headers") or ()
        }
        token = self.attach(self.extract(carrier))
        try:
            await self.app(scope, receive, send)
        finally:
            self.detach(token)


def app() -> Any:
    """``soliplex.main.create_app_from_environment``, wrapped."""
    from soliplex import main

    return TraceparentMiddleware(main.create_app_from_environment())
