"""Drive a soliplex room over AG-UI HTTP, as an application client does.

``drive`` runs one turn in process through ``soliplex ask``, which sends no
client tools, context, state or forwarded props. This module talks to a
running ``soliplex-cli serve`` instead: it declares client tools, executes
the model's calls to them itself, and sends the results back in the next
run of the same thread, carrying the message history and state across runs
and turns.

``httpx`` and ``logfire`` are imported lazily and are deliberately *not*
dependencies of this package. Both arrive with soliplex.

What a record built here can see is set by the AG-UI event stream, not by
``pydantic_ai.all_messages``. pydantic-ai's AG-UI adapter streams every tool
call part under the name the model used, before ``_resolve_tool`` runs, and
streams every ``RetryPromptPart`` that answers a tool call as a
``TOOL_CALL_RESULT`` whose content is ``model_response()``. So invented tool
names and tool retries reach ``TrialRecord.tool_calls`` and ``tool_results``
with the same text ``collect`` reads. A retry that answers no tool call (an
output validator's ``ModelRetry``) appears in neither: ``collect`` reads it
as a plain text part, and the event stream does not carry it at all.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import pathlib
import re
import socket
import subprocess
import time
import uuid
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

from . import records
from .collect import ToolCall
from .collect import ToolResult

_MISSING = (
    "{name} is not importable. It arrives with soliplex. Install the "
    "software under test, or {name} itself, into this environment."
)

#: The turn error when the model still calls client tools after the cap.
MAX_ROUNDS = "max client rounds reached"

#: ``metadata["outcomes"]`` on a record from :func:`run_trial`.
OUTCOMES_SOURCE = "ag-ui events"

#: Parsed tool-call arguments in, a result string for the model out.
Handler = Callable[[dict[str, Any]], str]

_SESSION: Any = object()  # "use the session's value"


def _httpx() -> Any:
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover -- environment-dependent
        raise RuntimeError(_MISSING.format(name="httpx")) from exc
    return httpx


@contextlib.contextmanager
def _span(name: str, **attributes: Any) -> Iterator[Any]:
    """A logfire span when logfire is importable, else ``None``."""
    try:
        import logfire
    except ImportError:
        yield None
        return
    with logfire.span(name, **attributes) as span:
        yield span


def _error_json(exc: Exception) -> str:
    return json.dumps(
        {"error": {"type": type(exc).__name__, "message": str(exc)}}
    )


# -- protocol ----------------------------------------------------------


class Server:
    """AG-UI endpoints of one soliplex server.

    ``transport`` is handed to ``httpx.Client``, and tests pass an
    ``httpx.MockTransport``. One client is shared by every session on this
    server, and ``httpx.Client`` is safe to share between threads.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 600.0,
        transport: Any = None,
    ):
        httpx = _httpx()
        self.base_url = base_url
        self.http = httpx.Client(
            base_url=base_url, timeout=timeout, transport=transport
        )

    def close(self) -> None:
        self.http.close()

    def new_thread(
        self, room: str, metadata: Mapping[str, Any] | None = None
    ) -> Any:
        return self.http.post(
            f"/api/v1/rooms/{room}/agui",
            json={"metadata": dict(metadata or {})},
        )

    def new_run(
        self, room: str, thread: str, parent_run_id: str | None = None
    ) -> Any:
        body = (
            {} if parent_run_id is None else {"parent_run_id": parent_run_id}
        )
        return self.http.post(f"/api/v1/rooms/{room}/agui/{thread}", json=body)

    def get_thread(self, room: str, thread: str) -> Any:
        return self.http.get(f"/api/v1/rooms/{room}/agui/{thread}")

    def get_run(self, room: str, thread: str, run: str) -> Any:
        return self.http.get(f"/api/v1/rooms/{room}/agui/{thread}/{run}")

    def post_run(
        self, room: str, thread: str, run: str, body: Mapping[str, Any]
    ) -> dict[str, Any]:
        """POST a run input and read its event stream to the end.

        Returns ``status``, the raw SSE ``sse`` lines, the parsed
        ``events``, and ``error_body`` when the status is not 200.
        """
        out: dict[str, Any] = {
            "status": None,
            "sse": [],
            "events": [],
            "error_body": None,
        }
        with self.http.stream(
            "POST",
            f"/api/v1/rooms/{room}/agui/{thread}/{run}",
            json=body,
            headers={"accept": "text/event-stream"},
        ) as response:
            out["status"] = response.status_code
            if response.status_code != 200:
                response.read()
                out["error_body"] = response.text
                return out
            for line in response.iter_lines():
                out["sse"].append(line)
                if line.startswith("data: "):
                    out["events"].append(json.loads(line[6:]))
        return out


def run_input(
    thread: str,
    run: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]] = (),
    context: Sequence[Mapping[str, Any]] = (),
    state: Any = None,
    forwarded_props: Any = None,
) -> dict[str, Any]:
    """An AG-UI ``RunAgentInput`` body."""
    return {
        "threadId": thread,
        "runId": run,
        "state": state,
        "messages": list(messages),
        "tools": list(tools),
        "context": list(context),
        "forwardedProps": forwarded_props,
    }


def user_message(text: str) -> dict[str, Any]:
    return {"id": str(uuid.uuid4()), "role": "user", "content": text}


class History:
    """The AG-UI message list, extended from each run's events."""

    def __init__(self, messages: list[dict[str, Any]]):
        self.messages = messages

    def absorb(self, events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Apply one run's events to the history.

        Returns ``final_text`` (the last assistant text in the whole
        history), ``run_text`` (assistant text started in this run),
        ``error``, ``tool_calls`` as ``{id, name, args}`` in stream order,
        and ``tool_results`` as ``{id: content}``.
        """
        by_id = {m["id"]: m for m in self.messages}
        calls: dict[str, dict[str, Any]] = {}
        results: dict[str, Any] = {}
        started: list[dict[str, Any]] = []
        final_text = None
        error = None
        for event in events:
            kind = event.get("type")
            if kind == "TEXT_MESSAGE_START":
                message = by_id.get(event["messageId"])
                if message is None:
                    message = {
                        "id": event["messageId"],
                        "role": "assistant",
                        "content": "",
                    }
                    self.messages.append(message)
                    by_id[message["id"]] = message
                started.append(message)
            elif kind == "TEXT_MESSAGE_CONTENT":
                message = by_id[event["messageId"]]
                message["content"] = (message.get("content") or "") + (
                    event.get("delta", "")
                )
                final_text = message["content"]
            elif kind == "TOOL_CALL_START":
                parent = event.get("parentMessageId") or str(uuid.uuid4())
                message = by_id.get(parent)
                if message is None:
                    message = {
                        "id": parent,
                        "role": "assistant",
                        "content": None,
                    }
                    self.messages.append(message)
                    by_id[parent] = message
                tool_call = {
                    "id": event["toolCallId"],
                    "type": "function",
                    "function": {
                        "name": event["toolCallName"],
                        "arguments": "",
                    },
                }
                message.setdefault("toolCalls", []).append(tool_call)
                calls[event["toolCallId"]] = tool_call
            elif kind == "TOOL_CALL_ARGS":
                function = calls[event["toolCallId"]]["function"]
                function["arguments"] += event.get("delta", "")
            elif kind == "TOOL_CALL_RESULT":
                self.messages.append(
                    {
                        "id": event.get("messageId") or str(uuid.uuid4()),
                        "role": "tool",
                        "toolCallId": event["toolCallId"],
                        "content": event.get("content", ""),
                    }
                )
                results[event["toolCallId"]] = event.get("content", "")
            elif kind == "RUN_ERROR":
                error = event.get("message") or json.dumps(event)
        for message in reversed(self.messages):
            if message["role"] == "assistant" and message.get("content"):
                final_text = message["content"]
                break
        return {
            "final_text": final_text,
            "run_text": "\n".join(m["content"] for m in started),
            "error": error,
            "tool_calls": [
                {
                    "id": call_id,
                    "name": call["function"]["name"],
                    "args": call["function"]["arguments"],
                }
                for call_id, call in calls.items()
            ],
            "tool_results": results,
        }

    def add_tool_result(self, call_id: str, content: str) -> None:
        self.messages.append(
            {
                "id": str(uuid.uuid4()),
                "role": "tool",
                "toolCallId": call_id,
                "content": content,
            }
        )


# -- fenced blocks -----------------------------------------------------

_FENCE = re.compile(r"^```[ \t]*([\w+-]*)[ \t]*\n(.*?)^```", re.M | re.S)


def fenced_blocks(text: str, tag: str) -> list[str]:
    """The bodies of the ```` ```tag ```` blocks in ``text``, in order."""
    return [body for found, body in _FENCE.findall(text) if found == tag]


def fence_tags(text: str) -> list[str]:
    """The tag of every fenced block in ``text``, ``""`` when untagged."""
    return [found for found, _ in _FENCE.findall(text)]


def _default_reply(tag: str) -> Callable[[list[str]], str]:
    def reply(results: list[str]) -> str:
        return "\n\n".join(f"{tag} result:\n{r}" for r in results)

    return reply


@dataclasses.dataclass(frozen=True, slots=True)
class Fence:
    """A protocol without client tools: programs travel in fenced blocks.

    After a run in which the model called no client tool, every
    ```` ```tag ```` block in that run's assistant text goes to ``handler``
    in order, and ``reply(results)`` is sent as the next user message. The
    exchange counts as a round toward ``ClientProfile.max_rounds``.
    ``reply`` defaults to ``"<tag> result:\\n<result>"`` per block.
    """

    tag: str
    handler: Callable[[str], str]
    reply: Callable[[list[str]], str] | None = None

    def format(self, results: list[str]) -> str:
        reply = self.reply or _default_reply(self.tag)
        return reply(results)


# -- sessions ----------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ClientTool:
    """A tool the client declares in ``RunAgentInput.tools`` and runs."""

    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: Handler
    metadata: Mapping[str, Any] | None = None

    def definition(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }
        if self.metadata is not None:
            data["metadata"] = dict(self.metadata)
        return data


@dataclasses.dataclass(frozen=True, slots=True)
class ClientProfile:
    """What a client sends with every run, and how it runs the loop.

    ``state`` is the starting state: ``None`` takes the state the server
    gives a new thread. With ``adopt_snapshots`` the session replaces its
    state with each ``STATE_SNAPSHOT``, as a client mirroring server state
    does. ``max_rounds`` caps the client-tool and fence rounds per turn.
    """

    tools: Sequence[ClientTool] = ()
    context: Sequence[Mapping[str, Any]] = ()
    state: Any = None
    forwarded_props: Any = None
    max_rounds: int = 6
    adopt_snapshots: bool = True
    fence: Fence | None = None


@dataclasses.dataclass(slots=True)
class TurnResult:
    """One user message and every run it took.

    ``client_calls`` and ``server_calls`` are ``{id, name, args, result}``
    (client calls add ``round``). ``runs`` holds each run's request body,
    status and raw SSE lines. ``tool_calls`` and ``tool_results`` are the
    same calls in stream order, as :mod:`collect` types.
    """

    user: str
    final_text: str | None = None
    error: str | None = None
    rounds: int = 0
    client_calls: list[dict[str, Any]] = dataclasses.field(
        default_factory=list
    )
    server_calls: list[dict[str, Any]] = dataclasses.field(
        default_factory=list
    )
    runs: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    state: Any = None
    fence_blocks: list[dict[str, Any]] = dataclasses.field(
        default_factory=list
    )
    tool_calls: list[ToolCall] = dataclasses.field(default_factory=list)
    tool_results: list[ToolResult] = dataclasses.field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        data = {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name not in ("tool_calls", "tool_results")
        }
        data["tool_calls"] = [
            {"name": c.name, "arguments": c.arguments, "id": c.id}
            for c in self.tool_calls
        ]
        data["tool_results"] = [
            {"name": r.name, "id": r.id, "result": r.result}
            for r in self.tool_results
        ]
        return data


class Session:
    """One thread in one room, carrying history and state across turns."""

    def __init__(
        self,
        server: Server,
        room: str,
        profile: ClientProfile | None = None,
        *,
        thread_metadata: Mapping[str, Any] | None = None,
    ):
        self.server = server
        self.room = room
        self.profile = profile if profile is not None else ClientProfile()
        self.thread_metadata = dict(thread_metadata or {"name": "lab"})
        self.state = self.profile.state
        self.thread_id: str | None = None
        self.messages: list[dict[str, Any]] = []
        self.runs: list[dict[str, Any]] = []

    def _next_run_id(self) -> str:
        if self.thread_id is None:
            response = self.server.new_thread(self.room, self.thread_metadata)
            response.raise_for_status()
            body = response.json()
            self.thread_id = body["thread_id"]
            (run,) = body["runs"]
            if self.state is None:
                self.state = (
                    body["runs"][run].get("run_input", {}).get("state")
                )
            return run
        response = self.server.new_run(self.room, self.thread_id)
        response.raise_for_status()
        return response.json()["run_id"]

    def turn(
        self,
        text: str,
        *,
        forwarded_props: Any = _SESSION,
        context: Sequence[Mapping[str, Any]] | None = None,
    ) -> TurnResult:
        """Send ``text``, then run rounds until the model stops calling.

        ``forwarded_props`` and ``context`` replace the profile's values
        for every run of this turn only. An HTTP refusal or a ``RUN_ERROR``
        ends the turn with ``error`` set. A failure to create the thread or
        a run raises.
        """
        profile = self.profile
        props = (
            profile.forwarded_props
            if forwarded_props is _SESSION
            else forwarded_props
        )
        ctx = list(profile.context if context is None else context)
        tools = {tool.name: tool for tool in profile.tools}
        definitions = [tool.definition() for tool in profile.tools]
        self.messages.append(user_message(text))
        history = History(self.messages)
        turn = TurnResult(user=text, state=self.state)
        for round_index in range(profile.max_rounds + 1):
            with _span(
                "client round {round}", round=round_index, room=self.room
            ) as span:
                run_id = self._next_run_id()
                if span is not None:
                    span.set_attribute("thread_id", self.thread_id)
                    span.set_attribute("run_id", run_id)
                body = run_input(
                    self.thread_id,
                    run_id,
                    self.messages,
                    tools=definitions,
                    context=ctx,
                    state=self.state,
                    forwarded_props=props,
                )
                sent = json.loads(json.dumps(body))
                started = time.time()
                out = self.server.post_run(
                    self.room, self.thread_id, run_id, sent
                )
                run_record = {
                    "run_id": run_id,
                    "request": sent,
                    "status": out["status"],
                    "error_body": out["error_body"],
                    "sse": out["sse"],
                    "elapsed_s": round(time.time() - started, 3),
                }
                turn.runs.append(run_record)
                self.runs.append(run_record)
                if out["status"] != 200:
                    turn.error = f"HTTP {out['status']}: {out['error_body']}"
                    return turn
                seen = history.absorb(out["events"])
                if profile.adopt_snapshots:
                    for event in out["events"]:
                        if event.get("type") == "STATE_SNAPSHOT":
                            self.state = event.get("snapshot")
                turn.state = self.state
                turn.final_text = seen["final_text"]
                self._collect(turn, seen)
                if seen["error"]:
                    turn.error = seen["error"]
                    return turn
                pending = [
                    c
                    for c in seen["tool_calls"]
                    if c["id"] not in seen["tool_results"]
                ]
                if any(c["name"] in tools for c in pending):
                    if round_index == profile.max_rounds:
                        turn.error = MAX_ROUNDS
                        return turn
                    turn.rounds += 1
                    self._run_client_calls(
                        turn, history, tools, pending, round_index + 1
                    )
                    continue
                blocks = (
                    fenced_blocks(seen["run_text"], profile.fence.tag)
                    if profile.fence is not None
                    else []
                )
                if not blocks:
                    return turn
                if round_index == profile.max_rounds:
                    turn.error = MAX_ROUNDS
                    return turn
                turn.rounds += 1
                self._run_fence(turn, blocks, round_index + 1)
        return turn  # pragma: no cover -- every round returns or continues

    def _collect(self, turn: TurnResult, seen: Mapping[str, Any]) -> None:
        results = seen["tool_results"]
        for call in seen["tool_calls"]:
            turn.tool_calls.append(
                ToolCall(
                    name=call["name"], arguments=call["args"], id=call["id"]
                )
            )
        for call in seen["tool_calls"]:
            if call["id"] in results:
                turn.server_calls.append(
                    call | {"result": results[call["id"]]}
                )
                turn.tool_results.append(
                    ToolResult(
                        name=call["name"],
                        id=call["id"],
                        result=results[call["id"]],
                    )
                )

    def _run_client_calls(
        self,
        turn: TurnResult,
        history: History,
        tools: Mapping[str, ClientTool],
        pending: Sequence[Mapping[str, Any]],
        round_number: int,
    ) -> None:
        for call in pending:
            tool = tools.get(call["name"])
            try:
                args = json.loads(call["args"] or "{}")
                if tool is None:
                    result = json.dumps(
                        {"error": f"unknown tool {call['name']}"}
                    )
                else:
                    result = tool.handler(args)
            except Exception as exc:  # noqa: BLE001 -- goes to the model
                result = _error_json(exc)
            turn.client_calls.append(
                dict(call) | {"round": round_number, "result": result}
            )
            turn.tool_results.append(
                ToolResult(name=call["name"], id=call["id"], result=result)
            )
            history.add_tool_result(call["id"], result)

    def _run_fence(
        self, turn: TurnResult, blocks: Sequence[str], round_number: int
    ) -> None:
        fence = self.profile.fence
        results = []
        for code in blocks:
            try:
                result = fence.handler(code)
            except Exception as exc:  # noqa: BLE001 -- goes to the model
                result = _error_json(exc)
            results.append(result)
            turn.fence_blocks.append(
                {
                    "round": round_number,
                    "tag": fence.tag,
                    "code": code,
                    "result": result,
                }
            )
        self.messages.append(user_message(fence.format(results)))


# -- trials ------------------------------------------------------------


def run_trial(
    server: Server,
    room: str,
    prompts: str | Sequence[str],
    *,
    cell: str,
    trial: int,
    profile: ClientProfile | None = None,
    metadata: Mapping[str, Any] | None = None,
    thread_metadata: Mapping[str, Any] | None = None,
) -> records.TrialRecord:
    """Run ``prompts`` as consecutive turns of one new thread.

    Thread-safe: nothing process-global changes, so trials may run
    concurrently (see ``records.top_up_async``). The trial is one span in
    a new trace when logfire is importable. An exception is recorded as
    the error, as :func:`to_record` describes.
    """
    prompts = [prompts] if isinstance(prompts, str) else list(prompts)
    session = Session(server, room, profile, thread_metadata=thread_metadata)
    turns: list[TurnResult] = []
    failure: str | None = None
    started = time.monotonic()
    with (
        new_trace(),
        _span("trial {cell} t{trial}", cell=cell, trial=trial) as span,
    ):
        try:
            for prompt in prompts:
                turns.append(session.turn(prompt))
                if turns[-1].error is not None:
                    break
        except Exception as exc:  # noqa: BLE001 -- recorded, not handled
            failure = f"{type(exc).__name__}: {exc}"
        if span is not None:
            span.set_attribute("thread_id", session.thread_id)
    return to_record(
        session,
        turns,
        cell=cell,
        trial=trial,
        elapsed_s=round(time.monotonic() - started, 2),
        trace_id=trace_id(span),
        failure=failure,
        metadata=metadata,
    )


def to_record(
    session: Session,
    turns: Sequence[TurnResult],
    *,
    cell: str,
    trial: int,
    elapsed_s: float,
    trace_id: str | None = None,
    failure: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> records.TrialRecord:
    """The record of ``turns``, all run on ``session``.

    ``metadata`` gains ``thread_id``, ``run_ids``, ``rounds`` (summed over
    turns), ``trace_id``, the final ``state``, any ``fence_blocks`` and
    ``outcomes``, which names where the tool outcomes came from.
    ``failure`` is an exception that stopped the trial. It becomes
    ``error``, and ``tool_results`` stays ``None`` because the outcomes
    were not all captured.
    """
    errors = [t.error for t in turns if t.error is not None]
    error = failure or (errors[0] if errors else None)
    data = dict(metadata or {})
    data.update(
        {
            "thread_id": session.thread_id,
            "run_ids": [run["run_id"] for run in session.runs],
            "rounds": sum(t.rounds for t in turns),
            "trace_id": trace_id,
            "state": session.state,
            "outcomes": OUTCOMES_SOURCE,
        }
    )
    fences = [block for t in turns for block in t.fence_blocks]
    if fences:
        data["fence_blocks"] = fences
    return records.TrialRecord(
        cell=cell,
        trial=trial,
        elapsed_s=elapsed_s,
        ok=error is None,
        error=error,
        response=turns[-1].final_text if turns else None,
        tool_calls=[c for t in turns for c in t.tool_calls],
        tool_results=(
            None
            if failure is not None
            else [r for t in turns for r in t.tool_results]
        ),
        metadata=data,
    )


# -- servers -----------------------------------------------------------


class ServeError(RuntimeError):
    """A soliplex server could not be started."""

    def __init__(self, message: str):
        super().__init__(message)


class ConflictingLauncher(ServeError):
    def __init__(self) -> None:
        super().__init__("pass either 'command' or 'environment', not both")


class ServerExited(ServeError):
    def __init__(self, returncode: int, log: pathlib.Path):
        self.returncode = returncode
        self.log = log
        super().__init__(f"server exited with {returncode}, see {log}")


class ServerNotReady(ServeError):
    def __init__(self, url: str, timeout_s: float, log: pathlib.Path):
        self.url = url
        self.timeout_s = timeout_s
        self.log = log
        super().__init__(
            f"server at {url} not ready after {timeout_s}s, see {log}"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Served:
    url: str
    log: pathlib.Path
    port: int


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def rooms_ready(url: str) -> bool:
    """Does ``GET /api/v1/rooms`` answer 200?"""
    httpx = _httpx()
    try:
        return httpx.get(f"{url}/api/v1/rooms", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


@contextlib.contextmanager
def serve(
    installation: pathlib.Path,
    *,
    log: pathlib.Path,
    command: Sequence[str] | None = None,
    environment: Any = None,
    port: int | None = None,
    env: Mapping[str, str] | None = None,
    timeout_s: float = 180.0,
    spawn: Callable[..., Any] = subprocess.Popen,
    ready: Callable[[str], bool] = rooms_ready,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Iterator[Served]:
    """Run ``soliplex-cli serve --no-auth-mode`` until the block exits.

    The CLI is ``command`` (an argv prefix, such as ``["uv", "run",
    "--directory", <checkout>, "soliplex-cli"]``), else the ``soliplex-cli``
    beside an ``environs.Environment``'s python, else ``soliplex-cli`` on
    ``PATH``. ``env`` is added to this process's environment. Output goes
    to ``log``. Waits until ``ready(url)``, and raises
    :class:`ServerExited` or :class:`ServerNotReady` when that fails.
    """
    if command is not None and environment is not None:
        raise ConflictingLauncher
    if environment is not None:
        argv = [str(environment.python.parent / "soliplex-cli")]
    else:
        argv = list(command or ["soliplex-cli"])
    port = port or free_port()
    argv += ["serve", str(installation), "--no-auth-mode", "--port", str(port)]
    url = f"http://127.0.0.1:{port}"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        process = spawn(
            argv,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, **(env or {})),
        )
    try:
        deadline = clock() + timeout_s
        while not ready(url):
            if process.poll() is not None:
                raise ServerExited(process.returncode, log)
            if clock() > deadline:
                raise ServerNotReady(url, timeout_s, log)
            sleep(1)
        yield Served(url=url, log=log, port=port)
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:  # pragma: no cover
            process.kill()


# -- telemetry ---------------------------------------------------------

_configured = False


def configure_telemetry(
    service_name: str,
    *,
    environment: str | None = None,
    send_to_logfire: Any = "if-token-present",
    scrubbing: bool = False,
) -> None:
    """Configure logfire once per process, for traces that cross HTTP.

    Instruments httpx, which adds a W3C ``traceparent`` to every request.
    Sends only when ``LOGFIRE_TOKEN`` is set, by default. Scrubbing is off
    by default because it redacts tool arguments that mention words such
    as "token" or "secret", which are the arguments a trial inspects.
    """
    global _configured
    if _configured:
        return
    try:
        import logfire
    except ImportError as exc:  # pragma: no cover -- environment-dependent
        raise RuntimeError(_MISSING.format(name="logfire")) from exc
    logfire.configure(
        service_name=service_name,
        environment=environment,
        send_to_logfire=send_to_logfire,
        console=False,
        scrubbing=scrubbing,
        distributed_tracing=True,
    )
    logfire.instrument_httpx(capture_headers=False)
    _configured = True


def trace_id(span: Any) -> str | None:
    """The span's trace id as 32 hex digits, or ``None``."""
    context = span.get_span_context() if span is not None else None
    if context is None or not context.trace_id:
        return None
    return format(context.trace_id, "032x")


@contextlib.contextmanager
def new_trace() -> Iterator[None]:
    """Spans opened inside start a new trace, detached from any parent."""
    try:
        from opentelemetry import context
    except ImportError:
        yield
        return
    token = context.attach(context.Context())
    try:
        yield
    finally:
        context.detach(token)
