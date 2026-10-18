from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from soliplex_lab_harness import review

TRACE_1 = "1" * 32
TRACE_2 = "2" * 32


def columnar(rows):
    keys = list(rows[0]) if rows else []
    return {
        "columns": [
            {
                "name": key,
                "datatype": "Utf8",
                "values": [row.get(key) for row in rows],
            }
            for key in keys
        ]
    }


def trial_row(name="h7 trial cell unit t1", **attrs):
    base = {
        "trace_id": "trace-1",
        "span_id": "root",
        "parent_span_id": None,
        "span_name": name,
        "service_name": "x40-h7-driver",
        "start_timestamp": "2026-01-01T00:00:00Z",
        "duration": 1,
        "otel_status_code": "OK",
        "is_exception": False,
        "attributes": json.dumps(
            {"cell": "cell", "unit": "unit", "model": "model", **attrs}
        ),
    }
    return base


def flags(packet):
    return {item["flag"] for item in packet["anomalies"]}


def test_columnar_to_rows():
    body = {
        "columns": [
            {"name": "trace_id", "values": ["a", "b"]},
            {"name": "span_name", "values": ["one", "two"]},
        ]
    }
    assert review.columnar_to_rows(body) == [
        {"trace_id": "a", "span_name": "one"},
        {"trace_id": "b", "span_name": "two"},
    ]


def test_query_pages_and_hides_token():
    calls = []

    def fetch(params):
        calls.append(params)
        if "OFFSET 0" in params["sql"]:
            return columnar(
                [{"start_timestamp": "1"}, {"start_timestamp": "2"}]
            )
        return columnar([{"start_timestamp": "3"}])

    query = review.LogfireQuery("secret-token", fetch=fetch)
    rows = query.rows(
        "SELECT * FROM records", min_timestamp="now", page_size=2
    )
    assert len(rows) == 3
    assert "OFFSET 2" in calls[1]["sql"]
    assert "secret-token" not in repr(calls)

    with pytest.raises(review.QueryLimitError) as caught:
        review.LogfireQuery(
            "secret-token",
            fetch=lambda params: columnar(
                [{"start_timestamp": "1"}, {"start_timestamp": "2"}]
            ),
        ).rows(
            "SELECT * FROM records",
            min_timestamp="now",
            max_rows=2,
            page_size=2,
        )
    assert "secret-token" not in str(caught.value)

    class LeakyError(Exception):
        def __init__(self):
            super().__init__("request exposed secret-token")

    def fail(params):
        raise LeakyError

    with pytest.raises(review.QueryFetchError) as caught:
        review.LogfireQuery("secret-token", fetch=fail).rows(
            "SELECT * FROM records", min_timestamp="now"
        )
    assert "secret-token" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_query_fetch_error_exposes_only_response_status():
    class Response:
        status_code = 503

    class LeakyError(ValueError):
        response = Response()

        def __init__(self):
            super().__init__("Authorization: Bearer secret-token")

    def fetch(_params):
        raise LeakyError

    with pytest.raises(review.QueryFetchError) as caught:
        review.LogfireQuery("secret-token", fetch=fetch).rows(
            "SELECT * FROM records", min_timestamp="now"
        )

    assert str(caught.value) == "Logfire query request failed (HTTP 503)"
    assert caught.value.__cause__ is None


def test_query_pages_use_stable_order_and_bound_the_final_request():
    calls = []

    def fetch(params):
        calls.append(params)
        return columnar(
            [
                {"start_timestamp": "1", "span_id": str(index)}
                for index in range(params["limit"])
            ]
        )

    with pytest.raises(review.QueryLimitError):
        review.LogfireQuery("token", fetch=fetch).rows(
            "SELECT * FROM records",
            min_timestamp="now",
            max_rows=1_500,
            page_size=1_000,
        )

    assert [call["limit"] for call in calls] == [1_000, 501]
    assert all(
        "ORDER BY start_timestamp, span_id" in call["sql"] for call in calls
    )


@pytest.mark.parametrize(
    "name",
    [
        "x40 trial hypothesis model arm task t1",
        "h7 trial cell unit t1",
        "compose trial arm unit t1",
    ],
)
def test_trial_root_names(name):
    packet = review._packet_from_rows(
        "trace-1",
        [trial_row(name, question="q", correct=False)],
        record=None,
        review_reason="failed",
        runaway_tokens=30_000,
    )
    assert packet["trial"]["question"] == "q"
    assert "trial_span_missing" not in flags(packet)


def test_last_invoke_history_and_empty_assistant():
    first = trial_row(correct=False)
    invoke = {
        **trial_row(),
        "parent_span_id": "root",
        "span_name": "invoke_agent agent",
        "start_timestamp": "2026-01-01T00:00:02Z",
        "service_name": "soliplex-server",
        "attributes": json.dumps(
            {
                "pydantic_ai.all_messages": [
                    {
                        "role": "user",
                        "parts": [{"type": "text", "content": "q"}],
                    },
                    {"role": "assistant", "parts": []},
                ]
            }
        ),
    }
    packet = review._packet_from_rows(
        "trace-1",
        [first, invoke],
        record=None,
        review_reason="failed",
        runaway_tokens=30_000,
    )
    assert packet["transcript"] == ["user: q", "assistant: <empty>"]


def test_anomalies_and_rescoring():
    root = trial_row(
        correct=False,
        **{
            "pydantic_ai.all_messages": [
                {
                    "role": "assistant",
                    "parts": [{"type": "text", "content": "42"}],
                }
            ]
        },
    )
    chat = {
        **trial_row(),
        "parent_span_id": "root",
        "span_name": "chat model",
        "service_name": "soliplex-server",
        "otel_status_code": "ERROR",
        "otel_status_message": "bad",
        "attr_gen_ai__usage__output_tokens": "30000",
        "attr_gen_ai__response__finish_reasons": ["length"],
        "attr_gen_ai__output__messages": [{"role": "assistant", "parts": []}],
    }
    stream = {
        **chat,
        "span_name": "Stream error: disconnected",
        "attr_gen_ai__usage__output_tokens": None,
    }
    packet = review._packet_from_rows(
        "trace-1",
        [root, chat, stream],
        record={"correct": True},
        review_reason="failed",
        runaway_tokens=30_000,
    )
    assert {
        "empty_reply_with_output_tokens",
        "length_finish",
        "runaway",
        "stream_error",
        "rescored",
        "answered_without_program",
    } <= flags(packet)

    missing = review._packet_from_rows(
        "trace-1",
        [],
        record=None,
        review_reason="failed",
        runaway_tokens=30_000,
    )
    assert {"no_server_spans", "trial_span_missing"} <= flags(missing)


def test_rendered_error_messages_and_exception_priority():
    parent = {
        **trial_row("h7 trial {cell} {unit} t{trial}"),
        "otel_status_code": "ERROR",
        "message": "h7 trial c u t4",
    }
    stream = {
        **trial_row("Stream error: {error_message}"),
        "parent_span_id": "root",
        "message": "Stream error: connection lost",
        "attributes": json.dumps({"error_message": "connection lost"}),
    }
    exception = {
        **trial_row("execute_tool run_python_ui"),
        "parent_span_id": "root",
        "is_exception": True,
        "exception_type": "ValueError",
        "exception_message": "bad program",
    }
    packet = review._packet_from_rows(
        "trace-1",
        [parent, stream, exception],
        record=None,
        review_reason="failed",
        runaway_tokens=30_000,
    )
    assert [item["message"] for item in packet["errors"]] == [
        "bad program",
        "h7 trial c u t4",
        "connection lost",
    ]
    assert (
        next(
            item["detail"]
            for item in packet["anomalies"]
            if item["flag"] == "stream_error"
        )
        == "connection lost"
    )
    assert "span_name, message," in review._standard_sql("true")


def test_executor_attributes_are_lifted_without_duplication():
    tool = {
        **trial_row("executor call {tool}"),
        "parent_span_id": "root",
        "message": "executor call run_monty",
        "attributes": json.dumps(
            {
                "tool": "run_monty",
                "code": "print(6 * 7)",
                "output": "42",
                "value": 42,
                "kept": "detail",
            }
        ),
    }
    packet = review._packet_from_rows(
        "trace-1",
        [trial_row(), tool],
        record=None,
        review_reason="failed",
        runaway_tokens=30_000,
    )
    program = packet["programs"][0]
    assert (program["tool"], program["code"], program["output"]) == (
        "run_monty",
        "print(6 * 7)",
        "42",
    )
    assert program["attributes"] == {"kept": "detail"}


def test_model_call_token_counts_survive_safe_projection():
    chat = {
        **trial_row("chat model"),
        "parent_span_id": "root",
        "attr_gen_ai__usage__input_tokens": "12",
        "attr_gen_ai__usage__output_tokens": "34",
        "attr_gen_ai__usage__details__reasoning_tokens": "5",
        "attr_gen_ai__response__finish_reasons": ["stop"],
    }
    packet = review._safe(
        review._packet_from_rows(
            "trace-1",
            [trial_row(), chat],
            record=None,
            review_reason="failed",
            runaway_tokens=30_000,
        )
    )
    call = packet["model_calls"][0]
    assert call["input_tokens"] == 12
    assert call["output_tokens"] == 34
    assert call["reasoning_tokens"] == 5
    assert call["finish_reasons"] == ["stop"]


@pytest.mark.parametrize("program_kind", ["tool", "fence"])
def test_last_user_turn_with_program_is_not_flagged(program_kind):
    messages = [
        {"role": "user", "parts": [{"type": "text", "content": "old"}]},
        {"role": "assistant", "parts": [{"type": "text", "content": "no"}]},
        {"role": "user", "parts": [{"type": "text", "content": "new"}]},
    ]
    if program_kind == "tool":
        messages.append(
            {
                "role": "assistant",
                "parts": [{"type": "tool_call", "name": "run_monty"}],
            }
        )
        messages.append(
            {"role": "tool", "parts": [{"type": "text", "content": "42"}]}
        )
    else:
        messages.append(
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": "```python\n42\n```"}],
            }
        )
        messages.append(
            {
                "role": "user",
                "parts": [
                    {"type": "text", "content": "[program results]\n42"}
                ],
            }
        )
    messages.append(
        {"role": "assistant", "parts": [{"type": "text", "content": "42"}]}
    )
    packet = review._packet_from_rows(
        "trace-1",
        [trial_row(**{"pydantic_ai.all_messages": messages})],
        record=None,
        review_reason="failed",
        runaway_tokens=30_000,
    )
    assert "answered_without_program" not in flags(packet)


def test_unanswered_user_turn_is_not_flagged_as_answered_without_program():
    messages = [
        {
            "role": "user",
            "parts": [{"type": "text", "content": "Solve this"}],
        }
    ]

    packet = review._packet_from_rows(
        TRACE_1,
        [trial_row(**{"pydantic_ai.all_messages": messages})],
        record=None,
        review_reason="failed",
        runaway_tokens=30_000,
    )

    assert "answered_without_program" not in flags(packet)


@pytest.mark.parametrize(
    "result_text",
    [
        '[{"value": null, "output": "44\\n"}]',
        "PROGRAM RESULT:\n44",
        "program results:\n44",
        'monty result:\n[{"value": null, "output": "44\\n"}]',
    ],
)
def test_fence_program_result_continuation_is_not_a_new_turn(result_text):
    messages = [
        {"role": "user", "parts": [{"type": "text", "content": "new"}]},
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "```python\n44\n```"}],
        },
        {
            "role": "user",
            "parts": [{"type": "text", "content": result_text}],
        },
        {"role": "assistant", "parts": [{"type": "text", "content": "44"}]},
    ]
    packet = review._packet_from_rows(
        "trace-1",
        [trial_row(**{"pydantic_ai.all_messages": messages})],
        record=None,
        review_reason="failed",
        runaway_tokens=30_000,
    )

    assert "answered_without_program" not in flags(packet)


def test_missing_trial_identity_comes_from_record_with_sources():
    packet = review._packet_from_rows(
        "trace-1",
        [trial_row(model="span-model")],
        record={
            "question": "q",
            "expected": "e",
            "check": "c",
            "final_text": "a",
            "unit": "u",
            "task": "t",
            "cell": "cell-record",
            "arm": "arm",
            "model": "record-model",
        },
        review_reason="failed",
        runaway_tokens=30_000,
    )
    assert packet["trial"]["question"] == "q"
    assert packet["trial"]["model"] == "span-model"
    assert packet["trial_sources"] == {
        key: "record"
        for key in (
            "question",
            "expected",
            "check",
            "final_text",
            "task",
            "arm",
        )
    }


def test_extract_links_split_trace_and_strips_secrets():
    root = trial_row(
        correct=False,
        thread_id="thread-1",
        authorization="Bearer secret-token",
        **{"audit.dir": "/private/place"},
    )
    linked = {
        **trial_row(),
        "trace_id": TRACE_2,
        "parent_span_id": "parent",
        "span_name": "chat model",
        "service_name": "soliplex-server",
    }

    def fetch(params):
        sql = params["sql"]
        if "gen_ai.conversation.id" in sql:
            return columnar([linked])
        if f"trace_id = '{TRACE_2}'" in sql:
            return columnar([linked])
        if f"trace_id = '{TRACE_1}'" in sql:
            return columnar([root])
        return columnar([])

    packet = review.extract_packet(
        review.LogfireQuery("secret-token", fetch=fetch), TRACE_1
    )
    assert packet["linked_traces"][0]["trace_id"] == TRACE_2
    assert "server_spans_in_other_traces" in flags(packet)
    assert "secret-token" not in json.dumps(packet)
    assert "/private/place" not in json.dumps(packet)


def test_linked_trace_preserves_driver_anomaly_and_supplies_evidence():
    root = trial_row(correct=False)
    messages = [
        {"role": "user", "parts": [{"type": "text", "content": "q"}]},
        {"role": "assistant", "parts": [{"type": "text", "content": "a"}]},
    ]
    invoke = {
        **trial_row("invoke_agent agent"),
        "trace_id": TRACE_2,
        "parent_span_id": "server-root",
        "service_name": "soliplex-server",
        "start_timestamp": "2026-01-01T00:00:02Z",
        "attributes": json.dumps({"pydantic_ai.all_messages": messages}),
        "attr_gen_ai__usage__output_tokens": None,
    }
    chat = {
        **invoke,
        "span_id": "chat",
        "span_name": "chat model",
        "attr_gen_ai__usage__output_tokens": "7",
    }

    def fetch(params):
        sql = params["sql"]
        if f"trace_id = '{TRACE_1}'" in sql:
            return columnar([root])
        if f"trace_id = '{TRACE_2}'" in sql:
            return columnar([invoke, chat])
        return columnar([])

    packet = review.extract_packet(
        review.LogfireQuery("secret", fetch=fetch),
        TRACE_1,
        related_trace_ids=[TRACE_2],
    )
    assert packet["transcript"] == ["user: q", "assistant: a"]
    assert packet["model_calls"][0]["trace_id"] == TRACE_2
    assert packet["model_calls"][0]["output_tokens"] == 7
    assert "no_server_spans" in flags(packet)


def test_driver_only_linked_trace_is_dropped():
    root = trial_row(correct=False)
    false_match = {
        **trial_row("case: example"),
        "trace_id": TRACE_2,
        "parent_span_id": "eval-root",
    }

    def fetch(params):
        sql = params["sql"]
        if f"trace_id = '{TRACE_1}'" in sql:
            return columnar([root])
        if f"trace_id = '{TRACE_2}'" in sql:
            return columnar([false_match])
        return columnar([])

    packet = review.extract_packet(
        review.LogfireQuery("secret", fetch=fetch),
        TRACE_1,
        related_trace_ids=[TRACE_2],
    )
    assert packet["linked_traces"] == []
    assert "server_spans_in_other_traces" not in flags(packet)


@pytest.mark.parametrize("trace_id", ["x" * 50_000, "trace-1", "g" * 32])
def test_extract_rejects_invalid_trace_ids(trace_id):
    query = review.LogfireQuery(
        "secret", fetch=lambda _params: pytest.fail("query must not run")
    )

    with pytest.raises(review.InvalidTraceIdError):
        review.extract_packet(query, trace_id)


def test_extract_rejects_invalid_related_trace_id_before_querying_it():
    calls = []

    def fetch(params):
        calls.append(params["sql"])
        return columnar([])

    with pytest.raises(review.InvalidTraceIdError):
        review.extract_packet(
            review.LogfireQuery("secret", fetch=fetch),
            TRACE_1,
            related_trace_ids=["not-a-trace"],
        )

    assert calls == []


def test_extract_rejects_invalid_trace_id_discovered_by_thread_lookup():
    def fetch(params):
        if f"trace_id = '{TRACE_1}'" in params["sql"]:
            return columnar([trial_row(thread_id="thread-1")])
        return columnar([{"trace_id": "not-a-trace"}])

    with pytest.raises(review.InvalidTraceIdError):
        review.extract_packet(
            review.LogfireQuery("secret", fetch=fetch), TRACE_1
        )


def test_anomaly_details_are_clipped_when_created():
    stream = {
        **trial_row("Stream error: disconnected"),
        "message": "x" * 1_000,
    }

    packet = review._packet_from_rows(
        TRACE_1,
        [stream],
        record=None,
        review_reason="failed",
        runaway_tokens=30_000,
    )

    assert all(len(item["detail"]) <= 300 for item in packet["anomalies"])


def test_thread_lookup_rejects_invalid_id_and_escapes_like_wildcards():
    seen = []

    def fetch(params):
        seen.append(params["sql"])
        if f"trace_id = '{TRACE_1}'" in params["sql"]:
            return columnar([trial_row(thread_id="_")])
        return columnar([])

    review.extract_packet(review.LogfireQuery("secret", fetch=fetch), TRACE_1)
    lookup = next(sql for sql in seen if "gen_ai.conversation.id" in sql)
    assert "LIKE '%\\_%' ESCAPE '\\'" in lookup

    seen.clear()

    def invalid_fetch(params):
        seen.append(params["sql"])
        return columnar([trial_row(thread_id="%")])

    review.extract_packet(
        review.LogfireQuery("secret", fetch=invalid_fetch), TRACE_1
    )
    assert not any("gen_ai.conversation.id" in sql for sql in seen)


def test_clipping_packet_from_record_and_failed_iterator():
    record = {
        "trace_id": "offline",
        "correct": False,
        "question": "q",
        "final_text": "answer",
        "programs": [{"result": "x" * 10_000}],
    }
    packet = review.packet_from_record(record, review_reason="audit")
    review._fit_packet(packet, 2_000)
    assert (
        len(json.dumps(packet, sort_keys=True, separators=(",", ":"))) <= 2_000
    )
    assert packet["truncations"]
    assert list(review.iter_failed([record, {"correct": True}])) == [record]


def test_fit_packet_recursively_shrinks_and_never_raises():
    packet = {
        "trace_id": "trace-1",
        "review_reason": "failed",
        "trial": {"question": "q" * 8_000, "correct": False},
        "record_scoring": {"detail": "s" * 8_000},
        "transcript": [f"entry-{index}-" + "t" * 2_000 for index in range(12)],
        "programs": [{"attributes": {"nested": "p" * 8_000}}],
        "model_calls": [
            {
                "output_text": "m" * 8_000,
                "output_tokens": 30_000,
                "finish_reasons": ["length"],
            }
        ],
        "errors": [{"message": "e" * 8_000}],
        "linked_traces": [
            {
                "trace_id": "trace-2",
                "spans": [{"span_name": "l" * 8_000}],
            }
        ],
        "spans": [{"span_name": "z" * 8_000}],
        "anomalies": [{"flag": "runaway", "detail": "a" * 8_000}],
        "truncations": [],
    }

    review._fit_packet(packet, 2_000)

    assert packet["trace_id"] == "trace-1"
    assert packet["review_reason"] == "failed"
    assert packet["trial"]["correct"] is False
    assert packet["model_calls"][0]["output_tokens"] == 30_000
    assert packet["model_calls"][0]["finish_reasons"] == ["length"]
    assert packet["anomalies"][0]["detail"] == "a" * 8_000
    assert len(packet["programs"][0]["attributes"]["nested"]) <= 200
    assert len(packet["trial"]["question"]) <= 2_000
    assert packet["linked_traces"][0]["trace_id"] == "trace-2"
    assert "...[4 entries removed]..." in packet["transcript"]
    paths = {item["path"] for item in packet["truncations"]}
    assert "programs[0].attributes.nested" in paths
    assert packet["truncations"]
    assert packet.get("over_cap") is True


def test_fit_packet_caps_runaway_program_list_under_cap():
    packet = {
        "trace_id": "trace-1",
        "programs": [
            {"code": f"program-{index}-" + "x" * 1_300} for index in range(140)
        ],
        "model_calls": [{"output_text": "m" * 1_000} for _ in range(20)],
        "errors": [{"message": f"error-{index}"} for index in range(12)],
        "transcript": [],
        "linked_traces": [],
        "spans": [],
        "truncations": [],
    }

    review._fit_packet(packet, 40_000)

    encoded = json.dumps(packet, sort_keys=True, separators=(",", ":"))
    assert len(encoded) <= 40_000
    assert packet["programs"][6] == {"removed": 128}
    assert packet["model_calls"][8] == {"removed": 4}
    assert len(packet["errors"]) == 10
    truncations = {item["path"]: item for item in packet["truncations"]}
    assert truncations["programs"]["original_count"] == 140
    assert truncations["model_calls"]["original_count"] == 20
    assert truncations["errors"]["original_count"] == 12
    assert "over_cap" not in packet


def test_evidence_with_newlines():
    packet = {"trace_id": "t", "detail": "first line\nsecond line"}
    assert review.evidence_in_packet("first line second line", packet)


def test_evidence_searches_decoded_string_values_not_keys_or_short_text():
    packet = {
        "trace_id": "t",
        "detail": "failed",
        "encoded": json.dumps({"message": "quoted packet evidence"}),
    }

    assert not review.evidence_in_packet("trace_id", packet)
    assert not review.evidence_in_packet("a", packet)
    assert review.evidence_in_packet("quoted packet evidence", packet)


class FakeAgent:
    name = "fake"

    def __init__(self, output=None, error=None, usage_method=False):
        self.output = output
        self.error = error
        self.usage_method = usage_method

    async def run(self, prompt):
        if self.error:
            raise self.error
        usage = SimpleNamespace(input_tokens=3, output_tokens=4)
        if self.usage_method:
            return SimpleNamespace(output=self.output, usage=lambda: usage)
        return SimpleNamespace(output=self.output, usage=usage)


def test_classify_success_missing_evidence_and_error():
    verdict = review.Verdict(
        "infra", "quoted packet line", "Output vanished.", 0.8
    )
    found = asyncio.run(
        review.classify(
            FakeAgent(verdict), {"trace_id": "t", "x": "quoted packet line"}
        )
    )
    missing = asyncio.run(
        review.classify(FakeAgent(verdict), {"trace_id": "t", "x": "other"})
    )
    failed = asyncio.run(
        review.classify(FakeAgent(error=ValueError("bad")), {"trace_id": "t"})
    )
    assert found.evidence_found
    assert found.input_tokens == 3
    assert not missing.evidence_found
    assert failed.verdict is None
    assert failed.error == "bad"


def test_classify_supports_usage_method():
    verdict = review.Verdict("infra", "line", "Output vanished.", 0.8)
    result = asyncio.run(
        review.classify(
            FakeAgent(verdict, usage_method=True),
            {"trace_id": "t", "detail": "line"},
        )
    )

    assert result.verdict == verdict
    assert result.input_tokens == 3
    assert result.output_tokens == 4


def test_classify_preserves_verdict_when_usage_raises():
    verdict = review.Verdict("infra", "line", "Output vanished.", 0.8)

    class Result:
        output = verdict

        @property
        def usage(self):
            raise RuntimeError

    class Agent:
        name = "fake"

        async def run(self, prompt):
            return Result()

    result = asyncio.run(
        review.classify(Agent(), {"trace_id": "t", "detail": "line"})
    )

    assert result.verdict == verdict
    assert result.error is None
    assert result.input_tokens is None
    assert result.output_tokens is None


def make_review(trace_id, cause, confidence=0.9, found=True, reviewer="a"):
    verdict = review.Verdict(cause, "line", f"{cause} occurred.", confidence)
    return review.Review(trace_id, reviewer, verdict, None, 0.1, 1, 1, found)


def test_aggregate_propose_escalate_and_render():
    reviews = [
        make_review("a", "harness_bug"),
        make_review("b", "harness_bug"),
        make_review("c", "model_failure"),
    ]
    trials = {
        key: {"cell": "cell", "model": "model", "unit": "unit"}
        for key in "abc"
    }
    clusters = review.aggregate(reviews, trials)
    assert len(clusters) == 2
    proposals = review.propose(clusters, reviews)
    assert {item.gate for item in proposals} == {"failing test first"}
    assert all(item.guardrail == review.GUARDRAIL for item in proposals)

    second = make_review(
        "a", "infra", confidence=0.4, found=False, reviewer="b"
    )
    escalated = review.escalations({"a": [reviews[0]], "b": [second]})
    assert "reviewers disagree on cause" in escalated[0].reasons
    assert any("low confidence" in item for item in escalated[0].reasons)
    assert any("evidence not found" in item for item in escalated[0].reasons)
    assert review.render_markdown(proposals, escalated).startswith(
        review.GUARDRAIL
    )


def test_aggregate_excludes_escalated_reviews_and_honors_threshold():
    trials = {
        key: {"cell": "cell", "model": "model", "unit": "unit"}
        for key in "abcd"
    }
    reviews = [
        make_review("a", "harness_bug", confidence=0.0, found=False),
        make_review("b", "harness_bug", confidence=0.0, found=False),
        make_review("c", "harness_bug", confidence=0.7),
        make_review("d", "harness_bug", confidence=0.7),
    ]

    assert review.aggregate(reviews[:2], trials) == []
    assert review.aggregate(reviews, trials, threshold=0.8) == []
    assert len(review.aggregate(reviews, trials)) == 2


def test_aggregate_skips_empty_evaluation_scope_keys():
    reviews = [make_review("a", "doc_gap"), make_review("b", "doc_gap")]

    assert review.aggregate(reviews, {}) == []


def test_proposal_gate_for_each_non_model_cause():
    expected = {
        "question_flaw": "question panel review",
        "scorer_flaw": "failing test first",
        "harness_bug": "failing test first",
        "doc_gap": "doc change, then rerun the affected units",
        "infra": "serving-stack check, then rerun",
    }
    clusters = [
        review.Cluster(cause, "unit", ("u",), (cause,)) for cause in expected
    ]
    reviews = [make_review(cause, cause) for cause in expected]
    proposals = review.propose(clusters, reviews)
    assert {item.cause: item.gate for item in proposals} == expected
