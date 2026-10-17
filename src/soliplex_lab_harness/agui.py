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
output validator's ``ModelRetry``) is counted by neither. pydantic-ai
serializes a ``RetryPromptPart`` without a tool name as a text part, which
``collect`` skips, and the event stream carries only the rejected text.
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
import tempfile
import time
import uuid
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

from . import preconditions
from . import records
from .collect import ToolCall
from .collect import ToolResult

_MISSING = (
    "{name} is not importable. It arrives with soliplex. Install the "
    "software under test, or {name} itself, into this environment."
)

#: The turn error when the model still calls client tools after the cap.
MAX_ROUNDS = "max client rounds reached"

#: The turn error when a run ends waiting on calls that are not client
#: tools, such as a server tool that needs approval.
UNANSWERED = "unanswered tool calls"

#: The turn error when one run starts two tool calls with the same id, so
#: results cannot be matched to calls.
DUPLICATE_CALL_ID = "duplicate tool call id"

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
    """A logfire span when logfire is importable, else ``None``.

    ``name`` reaches ``logfire.span`` through this wrapper, where logfire's
    ``inspect_arguments`` cannot find the call site and logs a warning per
    span. :func:`configure_telemetry` turns that inspection off. A caller
    that configures logfire itself passes ``inspect_arguments=False``.
    """
    try:
        import logfire
    except ImportError:
        yield None
        return
    with logfire.span(name, **attributes) as span:
        yield span


def _as_json(text: str) -> Any:
    """``text`` decoded as JSON, or ``text`` itself when it is not JSON."""
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


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

    Every POST carries the current trace context in its headers, written
    by ``inject`` (``opentelemetry.propagate.inject`` when importable), so
    the server's spans join the caller's trace. No span is opened for the
    HTTP call itself.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 600.0,
        transport: Any = None,
        inject: Callable[[dict[str, str]], None] | None = None,
    ):
        httpx = _httpx()
        self.base_url = base_url
        self.inject = inject
        self.http = httpx.Client(
            base_url=base_url, timeout=timeout, transport=transport
        )

    def _trace_headers(self) -> dict[str, str]:
        inject = self.inject
        if inject is None:
            try:
                from opentelemetry import propagate
            except ImportError:
                return {}
            inject = propagate.inject
        carrier: dict[str, str] = {}
        inject(carrier)
        return carrier

    def close(self) -> None:
        self.http.close()

    def new_thread(
        self, room: str, metadata: Mapping[str, Any] | None = None
    ) -> Any:
        return self.http.post(
            f"/api/v1/rooms/{room}/agui",
            json={"metadata": dict(metadata or {})},
            headers=self._trace_headers(),
        )

    def new_run(
        self, room: str, thread: str, parent_run_id: str | None = None
    ) -> Any:
        body = (
            {} if parent_run_id is None else {"parent_run_id": parent_run_id}
        )
        return self.http.post(
            f"/api/v1/rooms/{room}/agui/{thread}",
            json=body,
            headers=self._trace_headers(),
        )

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
            headers={"accept": "text/event-stream"} | self._trace_headers(),
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

    def absorb(
        self, events: Sequence[Mapping[str, Any]], since: int = 0
    ) -> dict[str, Any]:
        """Apply one run's events to the history.

        Message ids are looked up among this run's messages only, and tool
        calls without a ``parentMessageId`` share one assistant message.
        Returns a dict of:

        - ``final_text``: the last assistant text in ``messages[since:]``
        - ``run_text``: assistant text started in this run
        - ``error``: the ``RUN_ERROR`` message, if any
        - ``duplicate_ids``: tool call ids started twice in this run
        - ``tool_calls``: ``{id, name, args}`` in stream order
        - ``tool_results``: ``{id: content}``
        - ``result_order``: ``(id, content)`` per result, in stream order
        """
        by_id: dict[str, dict[str, Any]] = {}
        duplicates: list[str] = []
        ordered: list[dict[str, Any]] = []
        calls: dict[str, dict[str, Any]] = {}
        results: dict[str, Any] = {}
        result_order: list[tuple[str, Any]] = []
        started: dict[str, dict[str, Any]] = {}
        orphan_parent: str | None = None
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
                started[message["id"]] = message
                orphan_parent = None
            elif kind == "TEXT_MESSAGE_CONTENT":
                message = by_id[event["messageId"]]
                message["content"] = (message.get("content") or "") + (
                    event.get("delta", "")
                )
                final_text = message["content"]
            elif kind == "TOOL_CALL_START":
                parent = event.get("parentMessageId")
                if not parent:
                    orphan_parent = orphan_parent or str(uuid.uuid4())
                    parent = orphan_parent
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
                if event["toolCallId"] in calls:
                    duplicates.append(event["toolCallId"])
                calls[event["toolCallId"]] = tool_call
                ordered.append(tool_call)
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
                orphan_parent = None
                result_order.append(
                    (event["toolCallId"], event.get("content", ""))
                )
            elif kind == "RUN_ERROR":
                error = event.get("message") or json.dumps(event)
        final_text = None
        for message in reversed(self.messages[since:]):
            if message["role"] == "assistant" and message.get("content"):
                final_text = message["content"]
                break
        return {
            "final_text": final_text,
            "run_text": "\n".join(
                m.get("content") or "" for m in started.values()
            ),
            "error": error,
            "duplicate_ids": duplicates,
            "tool_calls": [
                {
                    "id": call["id"],
                    "name": call["function"]["name"],
                    "args": call["function"]["arguments"],
                }
                for call in ordered
            ],
            "tool_results": results,
            "result_order": result_order,
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


class BadMaxRounds(ValueError):
    def __init__(self, max_rounds: int):
        self.max_rounds = max_rounds
        super().__init__(f"max_rounds must be 0 or more, not {max_rounds}")


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

    def __post_init__(self) -> None:
        if self.max_rounds < 0:
            raise BadMaxRounds(self.max_rounds)


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
    outcomes_complete: bool = True

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
        self.failed: str | None = None

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
        for every run of this turn only. An HTTP refusal, a ``RUN_ERROR``
        or a run that ends waiting on non-client calls ends the turn with
        ``error`` set. A failure to create the thread or a run raises.
        After either, the history can hold a partial run, and a further
        turn raises :class:`SessionFailed`.
        """
        if self.failed is not None:
            raise SessionFailed(self.failed)
        try:
            result = self._turn(text, forwarded_props, context)
        except Exception as exc:
            self.failed = f"{type(exc).__name__}: {exc}"
            raise
        if result.error is not None:
            self.failed = result.error
        return result

    def _turn(
        self,
        text: str,
        forwarded_props: Any,
        context: Sequence[Mapping[str, Any]] | None,
    ) -> TurnResult:
        profile = self.profile
        props = (
            profile.forwarded_props
            if forwarded_props is _SESSION
            else forwarded_props
        )
        ctx = list(profile.context if context is None else context)
        tools = {tool.name: tool for tool in profile.tools}
        definitions = [tool.definition() for tool in profile.tools]
        start = len(self.messages)
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
                seen = history.absorb(out["events"], since=start)
                if profile.adopt_snapshots:
                    for event in out["events"]:
                        if event.get("type") == "STATE_SNAPSHOT":
                            self.state = event.get("snapshot")
                turn.state = self.state
                turn.final_text = seen["final_text"]
                self._collect(turn, seen)
                if seen["error"]:
                    turn.error = seen["error"]
                    turn.outcomes_complete = False
                    return turn
                if seen["duplicate_ids"]:
                    ids = ", ".join(seen["duplicate_ids"])
                    turn.error = f"{DUPLICATE_CALL_ID}: {ids}"
                    turn.outcomes_complete = False
                    return turn
                pending = [
                    c
                    for c in seen["tool_calls"]
                    if c["id"] not in seen["tool_results"]
                ]
                if any(c["name"] in tools for c in pending):
                    if round_index == profile.max_rounds:
                        turn.error = MAX_ROUNDS
                        turn.outcomes_complete = False
                        return turn
                    turn.rounds += 1
                    self._run_client_calls(
                        turn, history, tools, pending, round_index + 1
                    )
                    continue
                if pending:
                    names = ", ".join(c["name"] for c in pending)
                    turn.error = f"{UNANSWERED}: {names}"
                    turn.outcomes_complete = False
                    return turn
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
        names = {
            tool_call["id"]: tool_call["function"]["name"]
            for message in self.messages
            for tool_call in message.get("toolCalls") or ()
        }
        names.update({c["id"]: c["name"] for c in seen["tool_calls"]})
        for call_id, content in seen["result_order"]:
            turn.tool_results.append(
                ToolResult(name=names.get(call_id), id=call_id, result=content)
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
            with _span(
                f"execute_tool {call['name']}",
                **{
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": call["name"],
                    "gen_ai.tool.call.id": call["id"],
                    "gen_ai.tool.call.arguments": _as_json(
                        call["args"] or "{}"
                    ),
                    "gen_ai.conversation.id": self.thread_id,
                },
            ) as span:
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
                if span is not None:
                    parsed = _as_json(result)
                    span.set_attribute("gen_ai.tool.call.result", parsed)
                    if isinstance(parsed, dict) and "error" in parsed:
                        span.set_level("error")
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
            for key, value in conversation_attributes(session, turns).items():
                span.set_attribute(key, value)
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


def otel_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """AG-UI messages as pydantic-ai's OpenTelemetry message parts.

    The shape of ``pydantic_ai.all_messages``: text and ``tool_call``
    parts per message, and each run of tool messages folded into one
    ``user`` message of ``tool_call_response`` parts.
    """
    names: dict[str, str] = {}  # the latest call with each id so far
    out: list[dict[str, Any]] = []
    for message in messages:
        for tool_call in message.get("toolCalls") or ():
            names[tool_call["id"]] = tool_call["function"]["name"]
        if message["role"] == "tool":
            part = {
                "type": "tool_call_response",
                "id": message["toolCallId"],
                "name": names.get(message["toolCallId"]),
                "result": _as_json(message.get("content")),
            }
            last = out[-1] if out else None
            if last is not None and last.get("_tool_responses"):
                last["parts"].append(part)
            else:
                out.append(
                    {"role": "user", "parts": [part], "_tool_responses": True}
                )
            continue
        parts: list[dict[str, Any]] = []
        if message.get("content"):
            parts.append({"type": "text", "content": message["content"]})
        for tool_call in message.get("toolCalls") or ():
            parts.append(
                {
                    "type": "tool_call",
                    "id": tool_call["id"],
                    "name": tool_call["function"]["name"],
                    "arguments": tool_call["function"]["arguments"],
                }
            )
        out.append({"role": message["role"], "parts": parts})
    for message in out:
        message.pop("_tool_responses", None)
    return out


def conversation_attributes(
    session: Session,
    turns: Sequence[TurnResult],
    *,
    system_instructions: str | None = None,
    agent_name: str | None = None,
    model: str | None = None,
    server_tools: Sequence[str] = (),
) -> dict[str, Any]:
    """The trial as pydantic-ai's ``invoke_agent`` span attributes.

    Set on a trial span, these make Logfire show every round, tool call,
    result and the final answer as one chat history. Values are lists and
    dicts, so logfire writes ``logfire.json_schema`` itself.

    The client cannot see the room's system prompt, so
    ``system_instructions`` defaults to the context entries sent in the
    latest run.
    ``server_tools`` names tools that ran on the server, which join the
    client tools in ``gen_ai.tool.definitions``.
    """
    messages = otel_messages(session.messages)
    finished = bool(turns) and turns[-1].error is None
    finished = finished and session.failed is None
    final = (
        [messages[-1] | {"finish_reason": "stop"}]
        if finished and messages and messages[-1]["role"] == "assistant"
        else []
    )
    attrs: dict[str, Any] = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.conversation.id": session.thread_id or "",
        "logfire.agent_run_candidate.v1": True,
        "pydantic_ai.all_messages": messages,
        "gen_ai.input.messages": messages[: len(messages) - len(final)],
        "gen_ai.output.messages": final,
    }
    if agent_name is not None:
        attrs["gen_ai.agent.name"] = agent_name
    if model is not None:
        attrs["gen_ai.request.model"] = model
        attrs["gen_ai.response.model"] = model
    profile = session.profile
    definitions = [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": dict(tool.parameters),
        }
        for tool in profile.tools
    ] + [{"type": "function", "name": name} for name in server_tools]
    if definitions:
        attrs["gen_ai.tool.definitions"] = definitions
    context = (
        session.runs[-1]["request"]["context"]
        if session.runs
        else profile.context
    )
    if system_instructions is None and context:
        system_instructions = "\n".join(
            f"{c.get('description')}: {c.get('value')}" for c in context
        )
    if system_instructions is not None:
        attrs["gen_ai.system_instructions"] = [
            {"type": "text", "content": system_instructions}
        ]
    final_text = turns[-1].final_text if finished else None
    if final_text is not None:
        attrs["final_result"] = final_text
    return attrs


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
    ``error``. ``tool_results`` is ``None`` after a failure, or when any
    recorded call has no captured result (a turn stopped at the round
    cap, a ``RUN_ERROR`` mid-call): those outcomes are unknown.
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
    calls = [c for t in turns for c in t.tool_calls]
    results = [r for t in turns for r in t.tool_results]
    answered = {r.id for r in results}
    unknown = (
        failure is not None
        or any(not t.outcomes_complete for t in turns)
        or any(c.id not in answered for c in calls)
    )
    return records.TrialRecord(
        cell=cell,
        trial=trial,
        elapsed_s=elapsed_s,
        ok=error is None,
        error=error,
        response=turns[-1].final_text if turns else None,
        tool_calls=calls,
        tool_results=None if unknown else results,
        metadata=data,
    )


# -- servers -----------------------------------------------------------


class SessionFailed(RuntimeError):
    """A turn after an earlier turn on the session ended in an error."""

    def __init__(self, error: str):
        self.error = error
        super().__init__(f"session unusable after an earlier error: {error}")


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
    preconditions: list[preconditions.Result] = dataclasses.field(
        default_factory=list
    )


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


#: The module name of the copied :mod:`asgi_trace` in a server.
ASGI_TRACE_MODULE = "lab_harness_asgi_trace"

#: ``soliplex-cli audit`` sections run before a server starts. ``audit
#: all`` also audits Ollama, which raises on an installation without a
#: ``haiku.rag.yaml``.
AUDIT_SECTIONS = ("installation", "rooms", "skills")

#: ``(argv, env) -> (returncode, stdout, stderr)``.
AuditRunner = Callable[
    [Sequence[str], Mapping[str, str]], tuple[int, str, str]
]


def run_audit_command(
    argv: Sequence[str], env: Mapping[str, str]
) -> tuple[int, str, str]:
    done = subprocess.run(
        list(argv), env=dict(env), capture_output=True, text=True, check=False
    )
    return done.returncode, done.stdout, done.stderr


def _launcher(command: Sequence[str] | None, environment: Any) -> list[str]:
    if command is not None and environment is not None:
        raise ConflictingLauncher
    if environment is not None:
        return [str(environment.python.parent / "soliplex-cli")]
    return list(command or ["soliplex-cli"])


def _audit_detail(code: int, out: str, err: str) -> str:
    """The exit code, the head of the JSON report, the tail of stderr."""
    out, err = out.strip(), err.strip()
    if len(out) > 1500:
        out = out[:1500] + "..."
    if len(err) > 1000:
        err = "..." + err[-1000:]
    return f"exit {code}: {out}\n{err}".strip()


def audit(
    installation: pathlib.Path,
    *,
    command: Sequence[str] | None = None,
    environment: Any = None,
    env: Mapping[str, str] | None = None,
    sections: Sequence[str] = AUDIT_SECTIONS,
    cell: str | None = None,
    runner: AuditRunner = run_audit_command,
) -> list[preconditions.Result]:
    """Run ``soliplex-cli audit -q <section>`` on ``installation``.

    The launcher and ``env`` are those of :func:`serve`. One result per
    section, in order, stopping after the first non-zero exit. A failed
    result's detail holds the JSON error report and stderr.
    """
    argv = _launcher(command, environment)
    installation = installation.resolve()
    full_env = dict(os.environ, **(env or {}))
    name = cell if cell is not None else str(installation)
    results = []
    for section in sections:
        code, out, err = runner(
            [*argv, "audit", "-q", section, str(installation)], full_env
        )
        detail = "" if code == 0 else _audit_detail(code, out, err)
        results.append(
            preconditions.Result(
                name, f"soliplex-cli audit {section}", code == 0, detail
            )
        )
        if code != 0:
            break
    return results


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
    audit_sections: Sequence[str] = AUDIT_SECTIONS,
    link_traces: bool | None = None,
    audit_runner: AuditRunner = run_audit_command,
    spawn: Callable[..., Any] = subprocess.Popen,
    ready: Callable[[str], bool] = rooms_ready,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Iterator[Served]:
    """Audit, then run ``soliplex-cli serve --no-auth-mode`` in the block.

    The CLI is ``command`` (an argv prefix, such as ``["uv", "run",
    "--directory", <checkout>, "soliplex-cli"]``), else the ``soliplex-cli``
    beside an ``environs.Environment``'s python, else ``soliplex-cli`` on
    ``PATH``. ``env`` is added to this process's environment.

    :func:`audit` runs ``audit_sections`` first with the same launcher and
    environment, and a failed section raises ``preconditions.Failed``
    before the server starts. The passing results are
    ``Served.preconditions``. Server output goes to ``log``. Waits until
    ``ready(url)``, and raises :class:`ServerExited` or
    :class:`ServerNotReady` when that fails.

    ``link_traces`` serves the app through :mod:`asgi_trace`, so each run's
    server spans join the client's trace. It defaults to whether
    :func:`configure_telemetry` has run in this process.
    """
    prefix = _launcher(command, environment)
    installation = installation.resolve()
    log = log.resolve()
    checks = audit(
        installation,
        command=prefix,
        env=env,
        sections=audit_sections,
        runner=audit_runner,
    )
    preconditions.assert_ok(checks)
    port = port or free_port()
    argv = [
        *prefix,
        "serve",
        str(installation),
        "--no-auth-mode",
        "--port",
        str(port),
    ]
    url = f"http://127.0.0.1:{port}"
    log.parent.mkdir(parents=True, exist_ok=True)
    server_env = dict(os.environ, **(env or {}))
    if _configured if link_traces is None else link_traces:
        shim = pathlib.Path(
            tempfile.mkdtemp(prefix="lab_harness_asgi-", dir=log.parent)
        )
        source = pathlib.Path(__file__).with_name("asgi_trace.py")
        (shim / f"{ASGI_TRACE_MODULE}.py").write_text(source.read_text())
        paths = [str(shim), server_env.get("PYTHONPATH", "")]
        server_env["PYTHONPATH"] = os.pathsep.join(p for p in paths if p)
        argv += [
            "--workers",
            "1",
            "--app-factory-name",
            f"{ASGI_TRACE_MODULE}:app",
        ]
    with log.open("w") as handle:
        process = spawn(
            argv,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=server_env,
        )
    try:
        deadline = clock() + timeout_s
        while not ready(url):
            if process.poll() is not None:
                raise ServerExited(process.returncode, log)
            if clock() > deadline:
                raise ServerNotReady(url, timeout_s, log)
            sleep(1)
        if process.poll() is not None:
            raise ServerExited(process.returncode, log)
        yield Served(url=url, log=log, port=port, preconditions=checks)
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:  # pragma: no cover
            process.kill()
            process.wait(timeout=20)


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

    httpx is not instrumented: :class:`Server` writes ``traceparent`` into
    its POSTs itself, so no span is opened per HTTP call. Sends only when
    ``LOGFIRE_TOKEN`` is set, by default. Scrubbing is off
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
        inspect_arguments=False,  # see _span
    )
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
