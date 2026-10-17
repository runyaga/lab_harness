import asyncio

from soliplex_lab_harness import asgi_trace


class FakePropagation:
    """A fake propagator and context stack."""

    def __init__(self):
        self.current = "root"
        self.events = []

    def extract(self, carrier):
        self.events.append(("extract", dict(carrier)))
        return f"ctx<{carrier.get('traceparent')}>"

    def attach(self, context):
        previous, self.current = self.current, context
        self.events.append(("attach", context))
        return previous

    def detach(self, token):
        self.events.append(("detach", token))
        self.current = token


def make_middleware(fake, seen):
    async def inner(scope, receive, send):
        seen.append((scope["type"], fake.current))

    return asgi_trace.TraceparentMiddleware(
        inner, extract=fake.extract, attach=fake.attach, detach=fake.detach
    )


def test_http_request_runs_inside_the_extracted_context():
    fake = FakePropagation()
    seen = []
    app = make_middleware(fake, seen)
    scope = {
        "type": "http",
        "headers": [
            (b"traceparent", b"00-abc-def-01"),
            (b"content-type", b"application/json"),
        ],
    }

    asyncio.run(app(scope, None, None))

    assert seen == [("http", "ctx<00-abc-def-01>")]
    assert fake.events[0] == (
        "extract",
        {"traceparent": "00-abc-def-01", "content-type": "application/json"},
    )
    assert fake.current == "root"


def test_context_is_detached_when_the_app_raises():
    fake = FakePropagation()

    async def inner(scope, receive, send):
        raise KeyError

    app = asgi_trace.TraceparentMiddleware(
        inner, extract=fake.extract, attach=fake.attach, detach=fake.detach
    )

    try:
        asyncio.run(app({"type": "http", "headers": []}, None, None))
    except KeyError:
        pass

    assert fake.current == "root"
    assert [e[0] for e in fake.events] == ["extract", "attach", "detach"]


def test_lifespan_passes_through():
    fake = FakePropagation()
    seen = []
    app = make_middleware(fake, seen)

    asyncio.run(app({"type": "lifespan"}, None, None))

    assert seen == [("lifespan", "root")]
    assert fake.events == []


def test_module_imports_nothing_from_the_server_at_import_time():
    source = asgi_trace.__file__
    text = open(source).read().split("\nclass ", 1)[0]

    assert "import soliplex" not in text
    assert "import opentelemetry" not in text
    assert "from opentelemetry" not in text
