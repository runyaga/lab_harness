"""Build and review compact evidence packets for failed trials."""

from __future__ import annotations

import dataclasses
import json
import re
import time
from collections import Counter
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from typing import Literal

GUARDRAIL = (
    "A model failing a question is never by itself evidence that the "
    "question or check is wrong. Change a question or a check only for a "
    "demonstrated flaw. Question changes go through the question panel. "
    "Scorer and harness changes need a failing (red) test first. Proposals "
    "are never applied automatically."
)

CAUSES = (
    "model_failure",
    "question_flaw",
    "scorer_flaw",
    "harness_bug",
    "doc_gap",
    "infra",
)
Cause = Literal[
    "model_failure",
    "question_flaw",
    "scorer_flaw",
    "harness_bug",
    "doc_gap",
    "infra",
]

CAUSE_GUIDE = """Discriminating tests:
model_failure: The question and check are sound, the supplied docs suffice,
the harness and serving stack worked, and the model's program or answer is
wrong.
question_flaw: The question is ambiguous or unanswerable, exposes its answer,
or the check does not test the requested result.
scorer_flaw: The answer or program is right but the scorer or gate marks it
wrong, or a wrong result is accepted.
harness_bug: The driver, executor, trace plumbing, or record is broken.
doc_gap: The model followed supplied documentation that is wrong or incomplete.
infra: The serving stack or network failed, or an output was dropped."""

_BASE_URL = "https://logfire-us.pydantic.dev/v1/query"
_SPACE = re.compile(r"\s+")
_TOKEN_KEY = re.compile(r"authorization|api[_-]?key|token", re.I)
_TRACE_ID = re.compile(r"[0-9a-f]{32}")
_THREAD_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")
_TRIAL_KEYS = {
    "question",
    "expected",
    "check",
    "final_text",
    "correct",
    "correct_lenient",
    "check_details",
    "failure_shapes",
    "shapes",
    "gate",
    "stitching",
    "outcome",
    "turn_checks",
    "turn_errors",
    "unit",
    "cell",
    "arm",
    "model",
    "model_id",
    "rounds",
    "program_count",
    "first_program_works",
    "thread_id",
    "gen_ai.conversation.id",
}


class QueryLimitError(Exception):
    """A query reached its row bound before proving completion."""

    def __init__(self, max_rows: int):
        self.max_rows = max_rows
        super().__init__(f"Logfire query reached the {max_rows}-row bound")


class QueryFetchError(Exception):
    """A Logfire transport failed without exposing request credentials."""

    def __init__(self, status_code: int | None = None):
        message = "Logfire query request failed"
        if status_code is not None:
            message += f" (HTTP {status_code})"
        super().__init__(message)


class InvalidTraceIdError(ValueError):
    """A trace identifier is not a 32-character lowercase hex value."""

    def __init__(self):
        super().__init__(
            "Trace IDs must be 32 lowercase hexadecimal characters"
        )


def columnar_to_rows(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a Logfire columnar response to row mappings."""
    columns = body.get("columns") or []
    width = max(
        (len(column.get("values") or []) for column in columns), default=0
    )
    return [
        {
            column.get("name"): (column.get("values") or [None] * width)[index]
            for column in columns
        }
        for index in range(width)
    ]


class LogfireQuery:
    """Run bounded, paged queries against Logfire."""

    def __init__(
        self,
        token: str,
        *,
        base_url: str = _BASE_URL,
        fetch: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        self._token = token
        self.base_url = base_url
        self._fetch = fetch or self._http_fetch

    def _http_fetch(self, params: dict[str, Any]) -> dict[str, Any]:
        import httpx

        response = httpx.get(
            self.base_url,
            params=params,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/json",
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def rows(
        self,
        sql: str,
        *,
        min_timestamp: str,
        max_timestamp: str | None = None,
        max_rows: int = 10_000,
        page_size: int = 1_000,
    ) -> list[dict[str, Any]]:
        """Return all rows or raise when the bounded query stays full."""
        page_size = min(page_size, max_rows, 10_000)
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            request_size = min(page_size, max_rows - len(rows) + 1)
            paged = (
                f"{sql.rstrip()} ORDER BY start_timestamp, span_id "
                f"LIMIT {request_size} OFFSET {offset}"
            )
            params: dict[str, Any] = {
                "sql": paged,
                "min_timestamp": min_timestamp,
                "limit": request_size,
            }
            if max_timestamp is not None:
                params["max_timestamp"] = max_timestamp
            try:
                body = self._fetch(params)
            except Exception as error:
                response = getattr(error, "response", None)
                status_code = getattr(response, "status_code", None)
                if not isinstance(status_code, int):
                    status_code = None
                raise QueryFetchError(status_code) from None
            page = columnar_to_rows(body)
            rows.extend(page)
            if len(rows) > max_rows:
                raise QueryLimitError(max_rows)
            if len(page) < request_size:
                return rows
            offset += len(page)


@dataclass(frozen=True, slots=True)
class Verdict:
    """A classified cause with confidence from zero through one."""

    cause: Cause
    evidence: str
    explanation: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass(frozen=True, slots=True)
class Review:
    """One review attempt and its accounting."""

    trace_id: str
    reviewer: str
    verdict: Verdict | None
    error: str | None
    latency_s: float
    input_tokens: int | None
    output_tokens: int | None
    evidence_found: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass(frozen=True, slots=True)
class Cluster:
    """Repeated non-model cause within one evaluation scope."""

    cause: Cause
    scope: Literal["cell_model", "unit"]
    key: tuple[str, ...]
    trace_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass(frozen=True, slots=True)
class Proposal:
    """A lesson proposal that requires its named gate."""

    cause: Cause
    scope: Literal["cell_model", "unit"]
    key: tuple[str, ...]
    trace_ids: tuple[str, ...]
    lesson: str
    gate: str
    guardrail: str = GUARDRAIL

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass(frozen=True, slots=True)
class Escalation:
    """A trial that needs a frontier reviewer."""

    trace_id: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


def _decode(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def _attrs(row: dict[str, Any]) -> dict[str, Any]:
    raw = _decode(row.get("attributes"))
    attrs = raw if isinstance(raw, dict) else {}
    for key, value in row.items():
        if key.startswith("attr_") and value is not None:
            attrs[key[5:].replace("__", ".")] = _decode(value)
    return attrs


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _safe(item)
            for key, item in value.items()
            if not (
                _TOKEN_KEY.search(str(key))
                and str(key)
                not in {"input_tokens", "output_tokens", "reasoning_tokens"}
            )
            and str(key) not in {"audit.dir", "server_url"}
        }
    if isinstance(value, list):
        return [_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_safe(item) for item in value]
    return value


def _redact(value: Any, secret: str) -> Any:
    if isinstance(value, dict):
        return {key: _redact(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    if isinstance(value, str) and secret:
        return value.replace(secret, "[redacted]")
    return value


def _clip(
    value: Any,
    size: int,
    path: str,
    truncations: list[dict[str, Any]],
) -> Any:
    if not isinstance(value, str) or len(value) <= size:
        return value
    marker = f"...[clipped {len(value) - size} chars]..."
    room = max(0, size - len(marker))
    head = (room + 1) // 2
    clipped = value[:head] + marker + value[-(room - head) :]
    truncations.append(
        {
            "path": path,
            "original_chars": len(value),
            "kept_chars": len(clipped),
        }
    )
    return clipped


def _messages(raw: Any) -> list[dict[str, Any]]:
    decoded = _decode(raw)
    return decoded if isinstance(decoded, list) else []


def _render_transcript(
    raw: Any, truncations: list[dict[str, Any]], limit: int = 2_000
) -> list[str]:
    rendered: list[str] = []
    for message in _messages(raw):
        role = message.get("role", "unknown")
        parts = message.get("parts") or []
        if role == "assistant" and not parts:
            rendered.append("assistant: <empty>")
        for part in parts:
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            if kind == "text":
                content = part.get("content", "")
                text = str(content) if content else "<empty>"
                line = f"{role}: {text}"
            elif kind == "thinking":
                line = f"thinking: {part.get('content', '')}"
            elif kind == "tool_call":
                args = part.get("arguments", "")
                if not isinstance(args, str):
                    args = json.dumps(args, sort_keys=True)
                line = f"{role} -> {part.get('name')}({args})"
            elif kind == "tool_call_response":
                result = part.get("result", "")
                if not isinstance(result, str):
                    result = json.dumps(result, sort_keys=True)
                line = f"tool {part.get('name')} <- {result}"
            else:
                continue
            rendered.append(
                _clip(line, limit, f"transcript[{len(rendered)}]", truncations)
            )
        if (
            role == "assistant"
            and parts
            and not any(
                isinstance(part, dict)
                and part.get("type") in {"text", "thinking", "tool_call"}
                for part in parts
            )
        ):
            rendered.append("assistant: <empty>")
    return rendered


def _text_from_messages(raw: Any) -> tuple[str, bool, bool]:
    texts: list[str] = []
    has_tool = False
    for message in _messages(raw):
        for part in message.get("parts") or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and part.get("content"):
                texts.append(str(part["content"]))
            if part.get("type") == "tool_call":
                has_tool = True
    return "\n".join(texts), not texts and not has_tool, has_tool


def _standard_sql(where: str) -> str:
    return f"""SELECT trace_id, span_id, parent_span_id, span_name, message,
service_name, start_timestamp, end_timestamp, duration, otel_status_code,
otel_status_message, is_exception, exception_type, exception_message,
CASE WHEN span_name LIKE 'chat %' OR span_name = 'invoke_agent agent'
 THEN NULL ELSE attributes END AS attributes,
attributes->>'gen_ai.usage.input_tokens' AS attr_gen_ai__usage__input_tokens,
attributes->>'gen_ai.usage.output_tokens' AS attr_gen_ai__usage__output_tokens,
attributes->>'gen_ai.usage.details.reasoning_tokens'
 AS attr_gen_ai__usage__details__reasoning_tokens,
attributes->'gen_ai.response.finish_reasons'
 AS attr_gen_ai__response__finish_reasons,
attributes->'gen_ai.output.messages' AS attr_gen_ai__output__messages,
attributes->>'gen_ai.response.model' AS attr_gen_ai__response__model,
attributes->>'gen_ai.client.operation.time_to_first_chunk'
 AS attr_gen_ai__client__operation__time_to_first_chunk
FROM records WHERE {where}"""


def _trace_rows(query: LogfireQuery, trace_id: str) -> list[dict[str, Any]]:
    escaped = trace_id.replace("'", "''")
    return query.rows(
        _standard_sql(f"trace_id = '{escaped}'"),
        min_timestamp="1970-01-01T00:00:00Z",
    )


def _add_invoke_history(
    query: LogfireQuery, rows: list[dict[str, Any]]
) -> None:
    invokes = sorted(
        (row for row in rows if row.get("span_name") == "invoke_agent agent"),
        key=lambda row: str(row.get("start_timestamp", "")),
    )
    if not invokes or _attrs(invokes[-1]).get("pydantic_ai.all_messages"):
        return
    span_id = str(invokes[-1].get("span_id", "")).replace("'", "''")
    sql = (
        "SELECT trace_id, span_id, start_timestamp, "
        "attributes->'pydantic_ai.all_messages' "
        "AS attr_pydantic_ai__all_messages FROM records "
        f"WHERE span_id = '{span_id}'"
    )
    projected = query.rows(
        sql,
        min_timestamp="1970-01-01T00:00:00Z",
        max_rows=10_000,
        page_size=1_000,
    )
    if projected:
        invokes[-1].update(projected[-1])


def _identity(attrs: dict[str, Any]) -> dict[str, Any]:
    trial = {key: _safe(attrs[key]) for key in _TRIAL_KEYS if key in attrs}
    if "cell" not in trial and "arm" in trial:
        trial["cell"] = trial["arm"]
    if "model" not in trial and "model_id" in trial:
        trial["model"] = trial["model_id"]
    return trial


def _record_scoring(record: dict[str, Any] | None) -> dict[str, Any]:
    if not record:
        return {}
    return _safe(
        {
            key: value
            for key, value in record.items()
            if key == "correct"
            or key.startswith("correct_")
            or key in {"scorer_version", "check_details"}
            or key.startswith("failure")
        }
    )


def _packet_from_rows(
    trace_id: str,
    rows: list[dict[str, Any]],
    *,
    record: dict[str, Any] | None,
    review_reason: str,
    runaway_tokens: int,
) -> dict[str, Any]:
    truncations: list[dict[str, Any]] = []
    roots = [
        row
        for row in rows
        if row.get("parent_span_id") is None
        and " trial " in str(row.get("span_name", ""))
    ]
    root = roots[0] if roots else None
    root_attrs = _attrs(root) if root else {}
    trial = _identity(root_attrs)
    trial_sources: dict[str, str] = {}
    if record:
        for key in (
            "question",
            "expected",
            "check",
            "final_text",
            "unit",
            "task",
            "cell",
            "arm",
            "model",
        ):
            if key not in root_attrs and record.get(key) is not None:
                trial[key] = _safe(record[key])
                trial_sources[key] = "record"
    scoring = _record_scoring(record)
    trial["record_scoring"] = scoring

    histories = []
    if root_attrs.get("pydantic_ai.all_messages") is not None:
        histories.append(root_attrs["pydantic_ai.all_messages"])
    invokes = sorted(
        (row for row in rows if row.get("span_name") == "invoke_agent agent"),
        key=lambda row: str(row.get("start_timestamp", "")),
    )
    if not histories and invokes:
        histories.append(_attrs(invokes[-1]).get("pydantic_ai.all_messages"))
    transcript = _render_transcript(
        histories[-1] if histories else [], truncations
    )
    instructions = root_attrs.get("gen_ai.system_instructions")
    if instructions:
        if not isinstance(instructions, str):
            instructions = json.dumps(instructions, sort_keys=True)
        transcript.insert(
            0,
            _clip(
                f"system: {instructions}",
                2_000,
                "transcript[0]",
                truncations,
            ),
        )

    programs: list[dict[str, Any]] = []
    tool_rows = sorted(
        (
            row
            for row in rows
            if str(row.get("span_name", "")).startswith(
                ("execute_tool ", "executor call ")
            )
        ),
        key=lambda row: str(row.get("start_timestamp", "")),
    )
    for row in tool_rows:
        attrs = _attrs(row)
        rendered_tool = re.sub(
            r"^(?:executor call|execute_tool)\s+",
            "",
            str(row.get("message") or ""),
        )
        arguments = _decode(attrs.get("gen_ai.tool.call.arguments"))
        code = (
            arguments.get("code") if isinstance(arguments, dict) else arguments
        )
        code = code if code is not None else attrs.get("code")
        result = _decode(attrs.get("gen_ai.tool.call.result"))
        if result is None:
            result = next(
                (
                    attrs[key]
                    for key in ("output", "value", "error")
                    if attrs.get(key) is not None
                ),
                None,
            )
        details = {
            key: _safe(value)
            for key, value in attrs.items()
            if key
            not in {
                "gen_ai.input.messages",
                "gen_ai.tool.name",
                "gen_ai.tool.call.arguments",
                "gen_ai.tool.call.result",
                "tool",
                "code",
                "output",
                "value",
                "error",
            }
        }
        result_text = (
            json.dumps(result, sort_keys=True)
            if not isinstance(result, str)
            else result
        )
        output = result.get("output") if isinstance(result, dict) else result
        error = (
            result.get("error")
            if isinstance(result, dict)
            else attrs.get("error")
        )
        programs.append(
            {
                "tool": attrs.get("gen_ai.tool.name")
                or attrs.get("tool")
                or rendered_tool
                or str(row.get("span_name", "")).split()[-1],
                "code": _clip(
                    code, 4_000, f"programs[{len(programs)}].code", truncations
                ),
                "result": _clip(
                    result_text,
                    4_000,
                    f"programs[{len(programs)}].result",
                    truncations,
                ),
                "output": _clip(
                    output,
                    4_000,
                    f"programs[{len(programs)}].output",
                    truncations,
                ),
                "error": _clip(
                    error,
                    2_000,
                    f"programs[{len(programs)}].error",
                    truncations,
                ),
                "status": row.get("otel_status_code"),
                "duration": row.get("duration"),
                "attributes": details,
            }
        )
    if not programs and record:
        fallback = record.get("programs") or record.get("executor_log") or []
        programs = _safe(
            fallback if isinstance(fallback, list) else [fallback]
        )

    model_calls: list[dict[str, Any]] = []
    anomaly_names: dict[str, str] = {}
    for row in rows:
        if not str(row.get("span_name", "")).startswith("chat "):
            continue
        attrs = _attrs(row)
        output = attrs.get("gen_ai.output.messages")
        text, empty, _has_tool = _text_from_messages(output)
        output_tokens = _number(attrs.get("gen_ai.usage.output_tokens"))
        reasons = _decode(attrs.get("gen_ai.response.finish_reasons")) or []
        if isinstance(reasons, str):
            reasons = [reasons]
        call = {
            "input_tokens": _number(attrs.get("gen_ai.usage.input_tokens")),
            "output_tokens": output_tokens,
            "reasoning_tokens": _number(
                attrs.get("gen_ai.usage.details.reasoning_tokens")
            ),
            "finish_reasons": reasons,
            "duration": row.get("duration"),
            "time_to_first_chunk": _number(
                attrs.get("gen_ai.client.operation.time_to_first_chunk")
            ),
            "empty_output": empty,
            "output_text": _clip(
                text,
                4_000,
                f"model_calls[{len(model_calls)}].output_text",
                truncations,
            ),
            "trace_id": row.get("trace_id"),
        }
        model_calls.append(call)
        if empty and output_tokens > 0:
            anomaly_names["empty_reply_with_output_tokens"] = (
                "chat returned no text or tool call with "
                f"{output_tokens} output tokens"
            )
        if "length" in reasons:
            anomaly_names["length_finish"] = (
                "a model call finished because of length"
            )
        if output_tokens >= runaway_tokens:
            anomaly_names["runaway"] = (
                f"a model call produced {output_tokens} output tokens"
            )

    errors: list[dict[str, Any]] = []
    for row in rows:
        name = str(row.get("span_name", ""))
        if (
            row.get("is_exception")
            or str(row.get("otel_status_code", "")).upper() == "ERROR"
            or name.startswith("Stream error:")
        ):
            message = (
                row.get("exception_message")
                or row.get("otel_status_message")
                or _attrs(row).get("error_message")
                or row.get("message")
                or name
            )
            errors.append(
                {
                    "span_name": name,
                    "type": row.get("exception_type"),
                    "message": _clip(
                        str(message),
                        2_000,
                        f"errors[{len(errors)}].message",
                        truncations,
                    ),
                    "is_exception": bool(
                        row.get("is_exception") or row.get("exception_type")
                    ),
                }
            )
        if name.startswith("Stream error:"):
            anomaly_names["stream_error"] = str(message)
    errors.sort(key=lambda item: not item["is_exception"])

    driver_service = root.get("service_name") if root else None
    server_rows = [
        row for row in rows if row.get("service_name") != driver_service
    ]
    if not server_rows:
        anomaly_names["no_server_spans"] = (
            "the driver trace has no server spans"
        )
    if root is None:
        anomaly_names["trial_span_missing"] = (
            "the trace has no trial root span"
        )
    runtime_correct = trial.get("correct")
    record_correct = scoring.get("correct")
    if runtime_correct is not None and record_correct is not None:
        if runtime_correct != record_correct:
            anomaly_names["rescored"] = (
                f"run-time correct={runtime_correct!r}, "
                f"record correct={record_correct!r}"
            )
    final_messages = _messages(histories[-1] if histories else [])
    if final_messages:
        turn_start = 0
        for index, message in enumerate(final_messages):
            if message.get("role") != "user":
                continue
            text, _empty, _has_tool = _text_from_messages([message])
            first = text.splitlines()[0] if text else ""
            lowered = first.casefold()
            # Bracketed and labeled result texts continue the program turn.
            is_result = (
                first.startswith("[")
                or lowered.endswith(("result:", "results:"))
                or lowered.startswith("monty result")
            )
            if text and not is_result:
                turn_start = index
        turn = final_messages[turn_start:]
        has_tool = False
        fenced = False
        has_answer = False
        for message in turn:
            if message.get("role") != "assistant":
                continue
            text, _empty, message_has_tool = _text_from_messages([message])
            has_tool = has_tool or message_has_tool
            fenced = fenced or "```" in text
            has_answer = has_answer or bool(text.strip())
        if has_answer and not has_tool and not fenced:
            anomaly_names["answered_without_program"] = (
                "the final turn has no tool call or fenced program"
            )

    inventory = Counter(
        (str(row.get("service_name", "")), str(row.get("span_name", "")))
        for row in rows
    )
    spans = [
        {"service": service, "span_name": name, "count": count}
        for (service, name), count in sorted(inventory.items())
    ]
    return {
        "trace_id": trace_id,
        "review_reason": review_reason,
        "linked_traces": [],
        "trial": trial,
        "trial_sources": trial_sources,
        "transcript": transcript,
        "programs": programs,
        "model_calls": model_calls,
        "errors": errors,
        "spans": spans,
        "anomalies": [
            {
                "flag": flag,
                "detail": _clip(
                    detail,
                    300,
                    f"anomalies[{flag}].detail",
                    truncations,
                ),
            }
            for flag, detail in sorted(anomaly_names.items())
        ],
        "truncations": truncations,
    }


def _number(value: Any) -> int | float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    return int(number) if number.is_integer() else number


def _time_bounds(root: dict[str, Any], window: timedelta) -> tuple[str, str]:
    from datetime import datetime

    raw = str(root.get("start_timestamp"))
    stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return (stamp - window).isoformat(), (stamp + window).isoformat()


def _inventory(rows: list[dict[str, Any]], trace_id: str) -> dict[str, Any]:
    counts = Counter(
        (str(row.get("service_name", "")), str(row.get("span_name", "")))
        for row in rows
    )
    return {
        "trace_id": trace_id,
        "spans": [
            {"service": service, "span_name": name, "count": count}
            for (service, name), count in sorted(counts.items())
        ],
    }


def _fit_packet(packet: dict[str, Any], cap: int) -> None:
    def size() -> int:
        return len(json.dumps(packet, sort_keys=True, separators=(",", ":")))

    def cap_list(
        path: str, head: int, tail: int = 0, *, marker: bool = False
    ) -> None:
        entries = packet.get(path)
        if not isinstance(entries, list) or len(entries) <= head + tail:
            return
        original_count = len(entries)
        removed = original_count - head - tail
        kept = entries[:head]
        if marker:
            kept.append({"removed": removed})
        if tail:
            kept.extend(entries[-tail:])
        packet[path] = kept
        packet["truncations"].append(
            {"path": path, "original_count": original_count}
        )

    cap_list("programs", 6, 6, marker=True)
    cap_list("model_calls", 8, 8, marker=True)
    cap_list("errors", 10)

    def strings(
        value: Any, path: str
    ) -> Iterator[tuple[Any, int | str, str, str]]:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {
                    "trace_id",
                    "correct",
                    "input_tokens",
                    "output_tokens",
                    "reasoning_tokens",
                    "finish_reason",
                    "finish_reasons",
                }:
                    continue
                child_path = f"{path}.{key}" if path else str(key)
                if isinstance(child, str):
                    yield value, key, child_path, child
                else:
                    yield from strings(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                child_path = f"{path}[{index}]"
                if isinstance(child, str):
                    yield value, index, child_path, child
                else:
                    yield from strings(child, child_path)

    def transcript(middle: bool) -> Iterator[tuple[Any, int, str, str]]:
        entries = packet.get("transcript", [])
        middle_stop = max(2, len(entries) - 6)
        for index, value in enumerate(entries):
            is_middle = 2 <= index < middle_stop
            if is_middle == middle and isinstance(value, str):
                yield entries, index, f"transcript[{index}]", value

    def candidates() -> list[list[tuple[Any, int | str, str, str]]]:
        errors = []
        for parent, key, path, value in strings(
            packet.get("errors", []), "errors"
        ):
            if path.endswith(".message") and len(value) > 500:
                errors.append((parent, key, path, value))
        rest = errors
        rest.extend(strings(packet.get("linked_traces", []), "linked_traces"))
        rest.extend(strings(packet.get("spans", []), "spans"))
        rest.extend(transcript(False))
        model_texts = []
        for index, call in enumerate(packet.get("model_calls", [])):
            value = call.get("output_text")
            if isinstance(value, str):
                model_texts.append(
                    (
                        call,
                        "output_text",
                        f"model_calls[{index}].output_text",
                        value,
                    )
                )
        return [
            list(transcript(True)),
            list(strings(packet.get("programs", []), "programs")),
            model_texts,
            rest,
        ]

    while size() > cap:
        groups = candidates()
        group = next(
            (
                [item for item in items if len(item[3]) > 200]
                for items in groups
                if any(len(item[3]) > 200 for item in items)
            ),
            [],
        )
        if not group:
            break
        parent, key, path, value = max(group, key=lambda item: len(item[3]))
        parent[key] = _clip(
            value,
            max(200, len(value) // 2),
            path,
            packet["truncations"],
        )

    if size() > cap:
        for path, section in (
            ("trial", packet.get("trial")),
            ("record_scoring", packet.get("record_scoring")),
            (
                "trial.record_scoring",
                packet.get("trial", {}).get("record_scoring"),
            ),
        ):
            for parent, key, item_path, value in list(strings(section, path)):
                if len(value) > 2_000:
                    parent[key] = _clip(
                        value, 2_000, item_path, packet["truncations"]
                    )

    if size() > cap:
        entries = packet.get("transcript", [])
        removed = max(0, len(entries) - 8)
        if removed:
            entries[2:-6] = [f"...[{removed} entries removed]..."]

    if size() > cap:
        packet["over_cap"] = True


def extract_packet(
    query: LogfireQuery,
    trace_id: str,
    *,
    record: dict[str, Any] | None = None,
    related_trace_ids: Iterable[str] = (),
    window: timedelta = timedelta(minutes=15),
    cap: int = 40_000,
    review_reason: str = "failed",
    runaway_tokens: int = 30_000,
) -> dict[str, Any]:
    """Build one compact evidence packet from Logfire and a trial record."""
    related = list(dict.fromkeys(related_trace_ids))
    if not _TRACE_ID.fullmatch(trace_id) or any(
        not _TRACE_ID.fullmatch(related_id) for related_id in related
    ):
        raise InvalidTraceIdError
    rows = _trace_rows(query, trace_id)
    roots = [
        row
        for row in rows
        if row.get("parent_span_id") is None
        and " trial " in str(row.get("span_name", ""))
    ]
    root_has_history = roots and _attrs(roots[0]).get(
        "pydantic_ai.all_messages"
    )
    if not root_has_history:
        _add_invoke_history(query, rows)
    packet = _packet_from_rows(
        trace_id,
        rows,
        record=record,
        review_reason=review_reason,
        runaway_tokens=runaway_tokens,
    )
    root = next(
        (
            row
            for row in rows
            if row.get("parent_span_id") is None
            and " trial " in str(row.get("span_name", ""))
        ),
        None,
    )
    flags = {item["flag"] for item in packet["anomalies"]}
    driver_has_no_server_spans = "no_server_spans" in flags
    if "no_server_spans" in flags and root:
        attrs = _attrs(root)
        thread_id = attrs.get("thread_id") or attrs.get(
            "gen_ai.conversation.id"
        )
        thread_text = str(thread_id)
        if _THREAD_ID.fullmatch(thread_text):
            low, high = _time_bounds(root, window)
            escaped = (
                thread_text.replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            sql = _standard_sql(
                "(attributes->>'gen_ai.conversation.id' = "
                f"'{thread_text}' OR "
                f"span_name LIKE '%{escaped}%' ESCAPE '\\' OR "
                f"attributes::text LIKE '%{escaped}%' ESCAPE '\\')"
            )
            linked_rows = query.rows(
                sql, min_timestamp=low, max_timestamp=high, max_rows=10_000
            )
            related.extend(
                str(item["trace_id"])
                for item in linked_rows
                if item.get("trace_id") and item.get("trace_id") != trace_id
            )
    if any(not _TRACE_ID.fullmatch(related_id) for related_id in related):
        raise InvalidTraceIdError
    driver_service = root.get("service_name") if root else None
    accepted: list[tuple[str, list[dict[str, Any]]]] = []
    for related_id in dict.fromkeys(related):
        linked = _trace_rows(query, related_id)
        if driver_service and not any(
            row.get("service_name") != driver_service for row in linked
        ):
            continue
        accepted.append((related_id, linked))
        packet["linked_traces"].append(_inventory(linked, related_id))
    if not packet["transcript"] and accepted:
        evidence_rows = list(rows)
        for _related_id, linked in accepted:
            _add_invoke_history(query, linked)
            evidence_rows.extend(linked)
        linked_inventory = packet["linked_traces"]
        packet = _packet_from_rows(
            trace_id,
            evidence_rows,
            record=record,
            review_reason=review_reason,
            runaway_tokens=runaway_tokens,
        )
        packet["linked_traces"] = linked_inventory
        if driver_has_no_server_spans:
            packet["anomalies"].append(
                {
                    "flag": "no_server_spans",
                    "detail": "the driver trace has no server spans",
                }
            )
    if packet["linked_traces"]:
        packet["anomalies"].append(
            {
                "flag": "server_spans_in_other_traces",
                "detail": "server spans are linked from other traces",
            }
        )
    packet = _safe(packet)
    packet = _redact(packet, query._token)
    _fit_packet(packet, cap)
    return packet


def packet_from_record(
    record: dict[str, Any], *, review_reason: str = "failed"
) -> dict[str, Any]:
    """Build an offline evidence packet from one trial record."""
    trace_id = str(record.get("trace_id", ""))
    attrs = {key: value for key, value in record.items() if key in _TRIAL_KEYS}
    row = {
        "trace_id": trace_id,
        "parent_span_id": None,
        "span_name": "offline trial record",
        "service_name": "record",
        "start_timestamp": "",
        "attributes": attrs,
    }
    packet = _packet_from_rows(
        trace_id,
        [row],
        record=record,
        review_reason=review_reason,
        runaway_tokens=30_000,
    )
    packet["transcript"] = _render_transcript(
        record.get("pydantic_ai.all_messages")
        or record.get("all_messages")
        or [],
        packet["truncations"],
    )
    packet["programs"] = _safe(
        record.get("programs") or record.get("executor_log") or []
    )
    return _safe(packet)


def iter_failed(records: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Yield trial records whose scored result is false."""
    return (record for record in records if record.get("correct") is False)


def build_agent(
    *,
    base_url: str,
    model_name: str,
    api_key: str = "none",
    model_settings: dict[str, Any] | None = None,
    output_mode: Literal["tool", "native", "prompted"] = "tool",
    retries: int = 2,
) -> Any:
    """Build the structured pydantic-ai reviewer."""
    from pydantic_ai import Agent
    from pydantic_ai import NativeOutput
    from pydantic_ai import PromptedOutput
    from pydantic_ai import ToolOutput
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    model = OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(base_url=base_url, api_key=api_key),
    )
    outputs = {
        "tool": ToolOutput(Verdict),
        "native": NativeOutput(Verdict),
        "prompted": PromptedOutput(Verdict),
    }
    prompt = (
        f"{CAUSE_GUIDE}\n\n{GUARDRAIL}\n\nDecide from the packet only. "
        "Quote one packet line verbatim as evidence. A wrong answer alone is "
        "model_failure unless the packet shows a non-model cause. Use "
        "confidence below 0.5 when the packet lacks the evidence."
    )
    return Agent(
        model,
        output_type=outputs[output_mode],
        system_prompt=prompt,
        model_settings=model_settings,
        retries=retries,
    )


def evidence_in_packet(evidence: str, packet: dict[str, Any]) -> bool:
    """Test whether normalised evidence occurs in the packet."""
    needle = _SPACE.sub(" ", evidence).strip()
    if len(needle) < 12:
        return False
    texts = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            texts.append(value)
            decoded = _decode(value)
            if decoded is not value and decoded != value:
                walk(decoded)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(packet)
    return any(needle in _SPACE.sub(" ", text) for text in texts)


async def classify(
    agent: Any,
    packet: dict[str, Any],
    *,
    clock: Callable[[], float] = time.monotonic,
) -> Review:
    """Classify one packet while containing reviewer failures."""
    started = clock()
    reviewer = str(getattr(agent, "name", getattr(agent, "model", "reviewer")))
    try:
        result = await agent.run(json.dumps(packet, sort_keys=True))
        verdict = result.output
        try:
            usage = result.usage
            if callable(usage):
                usage = usage()
            input_tokens = getattr(
                usage, "input_tokens", getattr(usage, "request_tokens", None)
            )
            output_tokens = getattr(
                usage,
                "output_tokens",
                getattr(usage, "response_tokens", None),
            )
        except Exception:  # usage APIs and objects vary across versions
            input_tokens = None
            output_tokens = None
        return Review(
            trace_id=str(packet.get("trace_id", "")),
            reviewer=reviewer,
            verdict=verdict,
            error=None,
            latency_s=clock() - started,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            evidence_found=evidence_in_packet(verdict.evidence, packet),
        )
    except Exception as error:  # pydantic-ai and transports vary
        return Review(
            trace_id=str(packet.get("trace_id", "")),
            reviewer=reviewer,
            verdict=None,
            error=str(error),
            latency_s=clock() - started,
            input_tokens=None,
            output_tokens=None,
            evidence_found=False,
        )


def aggregate(
    reviews: Iterable[Review],
    trials: dict[str, dict[str, Any]],
    *,
    min_cluster: int = 2,
    threshold: float = 0.6,
) -> list[Cluster]:
    """Group repeated non-model causes by evaluation scope."""
    groups: defaultdict[tuple[Cause, str, tuple[str, ...]], set[str]] = (
        defaultdict(set)
    )
    for review in reviews:
        if (
            not review.verdict
            or review.verdict.cause == "model_failure"
            or not review.evidence_found
            or review.verdict.confidence < threshold
        ):
            continue
        trial = trials.get(review.trace_id, {})
        cell = str(trial.get("cell", trial.get("arm", "")))
        model = str(trial.get("model", trial.get("model_id", "")))
        unit = str(trial.get("unit", ""))
        if cell and model:
            groups[(review.verdict.cause, "cell_model", (cell, model))].add(
                review.trace_id
            )
        if unit:
            groups[(review.verdict.cause, "unit", (unit,))].add(
                review.trace_id
            )
    return [
        Cluster(cause, scope, key, tuple(sorted(trace_ids)))
        for (cause, scope, key), trace_ids in sorted(groups.items())
        if len(trace_ids) >= min_cluster
    ]


_LESSONS = {
    "question_flaw": (
        "Review the question and its check for the demonstrated flaw."
    ),
    "scorer_flaw": "Correct the scorer behavior demonstrated by these trials.",
    "harness_bug": (
        "Correct the harness behavior demonstrated by these trials."
    ),
    "doc_gap": "Correct the documented contract demonstrated by these trials.",
    "infra": "Correct the serving failure demonstrated by these trials.",
}
_GATES = {
    "question_flaw": "question panel review",
    "scorer_flaw": "failing test first",
    "harness_bug": "failing test first",
    "doc_gap": "doc change, then rerun the affected units",
    "infra": "serving-stack check, then rerun",
}


def propose(
    clusters: Iterable[Cluster], reviews: Iterable[Review]
) -> list[Proposal]:
    """Build gated lesson proposals for flagged clusters."""
    review_list = list(reviews)
    proposals = []
    for cluster in clusters:
        explanations = list(
            dict.fromkeys(
                review.verdict.explanation
                for review in review_list
                if review.trace_id in cluster.trace_ids and review.verdict
            )
        )[:3]
        lesson = _LESSONS[cluster.cause]
        if explanations:
            lesson += " " + " ".join(explanations)
        proposals.append(
            Proposal(
                cluster.cause,
                cluster.scope,
                cluster.key,
                cluster.trace_ids,
                lesson,
                _GATES[cluster.cause],
            )
        )
    return proposals


def escalations(
    reviews_by_reviewer: dict[str, list[Review]], threshold: float = 0.6
) -> list[Escalation]:
    """Find reviews that require a frontier reviewer."""
    by_trace: defaultdict[str, list[Review]] = defaultdict(list)
    for reviews in reviews_by_reviewer.values():
        for review in reviews:
            by_trace[review.trace_id].append(review)
    output = []
    for trace_id, reviews in sorted(by_trace.items()):
        reasons = []
        causes = {review.verdict.cause for review in reviews if review.verdict}
        for review in reviews:
            if review.error:
                reasons.append(f"{review.reviewer}: reviewer error")
            elif review.verdict and review.verdict.confidence < threshold:
                reasons.append(f"{review.reviewer}: low confidence")
            if review.verdict and not review.evidence_found:
                reasons.append(f"{review.reviewer}: evidence not found")
        if len(causes) > 1:
            reasons.append("reviewers disagree on cause")
        if reasons:
            output.append(Escalation(trace_id, tuple(dict.fromkeys(reasons))))
    return output


def render_markdown(
    proposals: Iterable[Proposal], escalations_: Iterable[Escalation]
) -> str:
    """Render proposals and escalations as Markdown."""
    lines = [GUARDRAIL, "", "## Proposals", ""]
    proposal_list = list(proposals)
    if not proposal_list:
        lines.append("None.")
    for proposal in proposal_list:
        lines.extend(
            [
                f"### {proposal.cause}: {proposal.scope} {proposal.key}",
                "",
                proposal.lesson,
                "",
                f"Gate: {proposal.gate}",
                "",
                "Trace IDs: " + ", ".join(proposal.trace_ids),
                "",
                proposal.guardrail,
            ]
        )
    lines.extend(["", "## Needs a frontier reviewer", ""])
    escalation_list = list(escalations_)
    if not escalation_list:
        lines.append("None.")
    for escalation in escalation_list:
        lines.append(
            f"- `{escalation.trace_id}`: " + ", ".join(escalation.reasons)
        )
    return "\n".join(lines).rstrip() + "\n"
