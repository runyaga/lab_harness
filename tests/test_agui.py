import json
import pathlib

import httpx
import pytest

from soliplex_lab_harness import agui
from soliplex_lab_harness import preconditions
from soliplex_lab_harness import scoring
from soliplex_lab_harness.collect import ToolCall
from soliplex_lab_harness.collect import ToolResult

ROOM = "mc_ctx"
THREAD = "11111111-1111-4111-8111-111111111111"


def text(message_id, content):
    return [
        {"type": "TEXT_MESSAGE_START", "messageId": message_id},
        {
            "type": "TEXT_MESSAGE_CONTENT",
            "messageId": message_id,
            "delta": content,
        },
        {"type": "TEXT_MESSAGE_END", "messageId": message_id},
    ]


def call(call_id, name, args, parent="a1"):
    return [
        {
            "type": "TOOL_CALL_START",
            "toolCallId": call_id,
            "toolCallName": name,
            "parentMessageId": parent,
        },
        {"type": "TOOL_CALL_ARGS", "toolCallId": call_id, "delta": args},
        {"type": "TOOL_CALL_END", "toolCallId": call_id},
    ]


def result(call_id, content):
    return [
        {
            "type": "TOOL_CALL_RESULT",
            "toolCallId": call_id,
            "messageId": f"m-{call_id}",
            "content": content,
        }
    ]


def framed(events):
    return [
        {"type": "RUN_STARTED"},
        *events,
        {"type": "RUN_FINISHED"},
    ]


class FakeSoliplex:
    """A scripted AG-UI server: one event list (or status) per run."""

    def __init__(self, scripts, *, initial_state=None):
        self.scripts = list(scripts)
        self.initial_state = initial_state
        self.bodies = []
        self.run_ids = []
        self.threads_created = 0

    def handler(self, request):
        path = request.url.path
        base = f"/api/v1/rooms/{ROOM}/agui"
        if request.method == "POST" and path == base:
            self.threads_created += 1
            run = self._new_run_id()
            return httpx.Response(
                200,
                json={
                    "thread_id": THREAD,
                    "runs": {
                        run: {"run_input": {"state": self.initial_state}}
                    },
                },
            )
        if request.method == "POST" and path == f"{base}/{THREAD}":
            return httpx.Response(200, json={"run_id": self._new_run_id()})
        if request.method == "POST" and path.startswith(f"{base}/{THREAD}/"):
            self.bodies.append(json.loads(request.content))
            script = self.scripts.pop(0)
            if isinstance(script, int):
                return httpx.Response(script, text="refused")
            lines = "".join(f"data: {json.dumps(e)}\n\n" for e in script)
            return httpx.Response(
                200,
                text=lines,
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(404)

    def _new_run_id(self):
        run = f"run-{len(self.run_ids)}"
        self.run_ids.append(run)
        return run

    def server(self):
        return agui.Server(
            "http://soliplex.test",
            transport=httpx.MockTransport(self.handler),
        )


def echo_tool(**overrides):
    fields = {
        "name": "run_monty",
        "description": "run code",
        "parameters": {"type": "object", "properties": {}},
        "handler": lambda args: json.dumps({"value": args["code"]}),
    }
    fields.update(overrides)
    return agui.ClientTool(**fields)


class HostFunctionMissing(ValueError):
    def __init__(self):
        super().__init__("no such host function")


# -- run input shape ---------------------------------------------------


def test_run_input_has_every_ag_ui_field():
    body = agui.run_input(
        "t",
        "r",
        [{"id": "u", "role": "user", "content": "hi"}],
        tools=[{"name": "x"}],
        context=[{"description": "d", "value": "v"}],
        state={"k": 1},
        forwarded_props={"trigger": "button"},
    )

    assert body == {
        "threadId": "t",
        "runId": "r",
        "state": {"k": 1},
        "messages": [{"id": "u", "role": "user", "content": "hi"}],
        "tools": [{"name": "x"}],
        "context": [{"description": "d", "value": "v"}],
        "forwardedProps": {"trigger": "button"},
    }


def test_client_tool_definition_carries_metadata_only_when_set():
    plain = echo_tool()
    tagged = echo_tool(metadata={"skill": "ledger"})

    assert "metadata" not in plain.definition()
    assert tagged.definition()["metadata"] == {"skill": "ledger"}


def test_first_run_sends_profile_inputs():
    fake = FakeSoliplex([framed(text("a1", "hello"))])
    profile = agui.ClientProfile(
        tools=(echo_tool(metadata={"skill": "ledger"}),),
        context=({"description": "d", "value": "v"},),
        state={"client_prefs": {"units": "metric"}},
        forwarded_props={"trigger": "button"},
    )
    session = agui.Session(fake.server(), ROOM, profile)

    turn = session.turn("hi")

    (body,) = fake.bodies
    assert body["threadId"] == THREAD
    assert body["runId"] == "run-0"
    assert body["tools"][0]["name"] == "run_monty"
    assert body["tools"][0]["metadata"] == {"skill": "ledger"}
    assert body["context"] == [{"description": "d", "value": "v"}]
    assert body["state"] == {"client_prefs": {"units": "metric"}}
    assert body["forwardedProps"] == {"trigger": "button"}
    assert body["messages"][0]["content"] == "hi"
    assert turn.final_text == "hello"
    assert turn.error is None
    assert turn.rounds == 0


def test_turn_overrides_apply_to_that_turn_only():
    fake = FakeSoliplex([framed(text("a1", "one")), framed(text("a2", "two"))])
    profile = agui.ClientProfile(
        context=({"description": "base", "value": "v"},),
        forwarded_props={"trigger": "button"},
    )
    session = agui.Session(fake.server(), ROOM, profile)

    session.turn("first", forwarded_props={"trigger": "shortcut"}, context=[])
    session.turn("second")

    first, second = fake.bodies
    assert first["forwardedProps"] == {"trigger": "shortcut"}
    assert first["context"] == []
    assert second["forwardedProps"] == {"trigger": "button"}
    assert second["context"] == [{"description": "base", "value": "v"}]


def test_forwarded_props_override_may_be_none():
    fake = FakeSoliplex([framed(text("a1", "one"))])
    profile = agui.ClientProfile(forwarded_props={"trigger": "button"})
    session = agui.Session(fake.server(), ROOM, profile)

    session.turn("first", forwarded_props=None)

    assert fake.bodies[0]["forwardedProps"] is None


# -- client-tool round loop --------------------------------------------


def test_client_tool_round_sends_result_in_next_run_of_same_thread():
    fake = FakeSoliplex(
        [
            framed(call("c1", "run_monty", '{"code": "1 + 1"}')),
            framed(text("a2", "It is 2.")),
        ]
    )
    session = agui.Session(
        fake.server(), ROOM, agui.ClientProfile(tools=(echo_tool(),))
    )

    turn = session.turn("add")

    assert fake.threads_created == 1
    assert [b["runId"] for b in fake.bodies] == ["run-0", "run-1"]
    assert {b["threadId"] for b in fake.bodies} == {THREAD}
    sent = fake.bodies[1]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "tool"]
    assert sent[1]["toolCalls"][0]["function"] == {
        "name": "run_monty",
        "arguments": '{"code": "1 + 1"}',
    }
    assert sent[2]["toolCallId"] == "c1"
    assert sent[2]["content"] == json.dumps({"value": "1 + 1"})
    assert turn.rounds == 1
    assert turn.final_text == "It is 2."
    assert turn.client_calls == [
        {
            "id": "c1",
            "name": "run_monty",
            "args": '{"code": "1 + 1"}',
            "round": 1,
            "result": json.dumps({"value": "1 + 1"}),
        }
    ]
    assert [r["run_id"] for r in turn.runs] == ["run-0", "run-1"]


def test_second_turn_continues_the_thread_and_history():
    fake = FakeSoliplex([framed(text("a1", "one")), framed(text("a2", "two"))])
    session = agui.Session(fake.server(), ROOM)

    session.turn("first")
    session.turn("second")

    assert fake.threads_created == 1
    contents = [m.get("content") for m in fake.bodies[1]["messages"]]
    assert contents == ["first", "one", "second"]


def test_server_calls_are_kept_with_their_results():
    fake = FakeSoliplex(
        [
            framed(
                call("s1", "agui_context", "{}")
                + result("s1", "ctx text")
                + text("a2", "done")
            )
        ]
    )
    session = agui.Session(
        fake.server(), ROOM, agui.ClientProfile(tools=(echo_tool(),))
    )

    turn = session.turn("go")

    assert turn.server_calls == [
        {
            "id": "s1",
            "name": "agui_context",
            "args": "{}",
            "result": "ctx text",
        }
    ]
    assert turn.client_calls == []
    assert turn.rounds == 0


def test_round_cap_stops_and_says_so():
    loop = [
        framed(call(f"c{i}", "run_monty", '{"code": "x"}', parent=f"a{i}"))
        for i in range(3)
    ]
    fake = FakeSoliplex(loop)
    profile = agui.ClientProfile(tools=(echo_tool(),), max_rounds=2)
    session = agui.Session(fake.server(), ROOM, profile)

    turn = session.turn("loop")

    assert turn.error == agui.MAX_ROUNDS
    assert turn.rounds == 2
    assert len(fake.bodies) == 3


def test_handler_failure_goes_back_to_the_model():
    def boom(args):
        raise HostFunctionMissing

    fake = FakeSoliplex(
        [
            framed(call("c1", "run_monty", '{"code": "x"}')),
            framed(text("a2", "sorry")),
        ]
    )
    profile = agui.ClientProfile(tools=(echo_tool(handler=boom),))
    session = agui.Session(fake.server(), ROOM, profile)

    turn = session.turn("go")

    sent = json.loads(fake.bodies[1]["messages"][-1]["content"])
    assert sent == {
        "error": {
            "type": "HostFunctionMissing",
            "message": "no such host function",
        }
    }
    assert turn.error is None


def test_unparseable_arguments_go_back_to_the_model():
    fake = FakeSoliplex(
        [
            framed(call("c1", "run_monty", '{"code": ')),
            framed(text("a2", "sorry")),
        ]
    )
    session = agui.Session(
        fake.server(), ROOM, agui.ClientProfile(tools=(echo_tool(),))
    )

    session.turn("go")

    sent = json.loads(fake.bodies[1]["messages"][-1]["content"])
    assert sent["error"]["type"] == "JSONDecodeError"


def test_unknown_call_beside_a_client_call_gets_an_error_result():
    fake = FakeSoliplex(
        [
            framed(
                call("c1", "run_monty", '{"code": "x"}')
                + call("c2", "invented", "{}")
            ),
            framed(text("a2", "ok")),
        ]
    )
    session = agui.Session(
        fake.server(), ROOM, agui.ClientProfile(tools=(echo_tool(),))
    )

    turn = session.turn("go")

    assert [c["name"] for c in turn.client_calls] == ["run_monty", "invented"]
    assert json.loads(turn.client_calls[1]["result"]) == {
        "error": "unknown tool invented"
    }


# -- error paths -------------------------------------------------------


def test_http_refusal_is_a_turn_error():
    fake = FakeSoliplex([422])
    session = agui.Session(fake.server(), ROOM)

    turn = session.turn("go")

    assert turn.error == "HTTP 422: refused"
    assert turn.runs[0]["status"] == 422
    assert turn.runs[0]["error_body"] == "refused"


def test_run_error_event_is_a_turn_error():
    fake = FakeSoliplex(
        [[{"type": "RUN_STARTED"}, {"type": "RUN_ERROR", "message": "bad"}]]
    )
    session = agui.Session(fake.server(), ROOM)

    turn = session.turn("go")

    assert turn.error == "bad"


def test_thread_creation_failure_raises():
    def refuse(request):
        return httpx.Response(500)

    server = agui.Server(
        "http://soliplex.test", transport=httpx.MockTransport(refuse)
    )
    session = agui.Session(server, ROOM)

    with pytest.raises(httpx.HTTPStatusError):
        session.turn("go")


# -- state carry -------------------------------------------------------


def test_state_snapshot_is_adopted_and_sent_next_run():
    snapshot = {"client_prefs": {"units": "imperial"}, "server": 1}
    fake = FakeSoliplex(
        [
            framed(
                call("c1", "run_monty", '{"code": "x"}')
                + [{"type": "STATE_SNAPSHOT", "snapshot": snapshot}]
            ),
            framed(text("a2", "done")),
        ]
    )
    profile = agui.ClientProfile(
        tools=(echo_tool(),), state={"client_prefs": {"units": "metric"}}
    )
    session = agui.Session(fake.server(), ROOM, profile)

    turn = session.turn("go")

    assert fake.bodies[1]["state"] == snapshot
    assert turn.state == snapshot
    assert session.state == snapshot


def test_snapshots_ignored_when_not_adopting():
    fake = FakeSoliplex(
        [
            framed(
                call("c1", "run_monty", '{"code": "x"}')
                + [{"type": "STATE_SNAPSHOT", "snapshot": {"other": 1}}]
            ),
            framed(text("a2", "done")),
        ]
    )
    profile = agui.ClientProfile(
        tools=(echo_tool(),), state={"mine": 1}, adopt_snapshots=False
    )
    session = agui.Session(fake.server(), ROOM, profile)

    session.turn("go")

    assert fake.bodies[1]["state"] == {"mine": 1}


def test_no_profile_state_takes_the_new_thread_state():
    fake = FakeSoliplex(
        [framed(text("a1", "x"))], initial_state={"from": "server"}
    )
    session = agui.Session(fake.server(), ROOM)

    session.turn("go")

    assert fake.bodies[0]["state"] == {"from": "server"}


def test_profile_is_not_mutated_by_a_session():
    fake = FakeSoliplex(
        [framed([{"type": "STATE_SNAPSHOT", "snapshot": {"new": 1}}])]
    )
    profile = agui.ClientProfile(state={"old": 1})
    agui.Session(fake.server(), ROOM, profile).turn("go")

    assert profile.state == {"old": 1}


# -- fence mode --------------------------------------------------------


def test_fence_blocks_are_extracted_in_order():
    reply = (
        "First:\n```monty\na = 1\n```\nthen\n```python\nignored\n```\n"
        "```monty\nb = 2\n```"
    )

    assert agui.fenced_blocks(reply, "monty") == ["a = 1\n", "b = 2\n"]
    assert agui.fenced_blocks(reply, "python") == ["ignored\n"]
    assert agui.fenced_blocks("no fences", "monty") == []


def test_fence_tags():
    reply = "```monty\nx\n```\n```python\ny\n```\n```\nz\n```"

    assert agui.fence_tags(reply) == ["monty", "python", ""]


def test_fence_round_sends_result_as_next_user_message():
    seen = []

    def run_block(code):
        seen.append(code)
        return json.dumps({"value": 2})

    fake = FakeSoliplex(
        [
            framed(text("a1", "Running:\n```monty\n1 + 1\n```")),
            framed(text("a2", "The answer is 2.")),
        ]
    )
    profile = agui.ClientProfile(fence=agui.Fence("monty", run_block))
    session = agui.Session(fake.server(), ROOM, profile)

    turn = session.turn("add")

    assert seen == ["1 + 1\n"]
    sent = fake.bodies[1]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert sent[2]["content"] == 'monty result:\n{"value": 2}'
    assert fake.bodies[1]["tools"] == []
    assert turn.rounds == 1
    assert turn.final_text == "The answer is 2."
    assert turn.fence_blocks == [
        {
            "round": 1,
            "tag": "monty",
            "code": "1 + 1\n",
            "result": '{"value": 2}',
        }
    ]
    assert turn.tool_calls == []


def test_fence_reads_only_this_runs_text():
    """A run without text must not re-run the previous run's fence."""
    fake = FakeSoliplex(
        [
            framed(text("a1", "```monty\nx\n```")),
            framed([]),
        ]
    )
    profile = agui.ClientProfile(fence=agui.Fence("monty", lambda code: "1"))
    session = agui.Session(fake.server(), ROOM, profile)

    turn = session.turn("go")

    assert len(fake.bodies) == 2
    assert turn.rounds == 1
    assert turn.error is None


def test_fence_several_blocks_one_message():
    fake = FakeSoliplex(
        [
            framed(text("a1", "```monty\na\n```\n```monty\nb\n```")),
            framed(text("a2", "done")),
        ]
    )
    profile = agui.ClientProfile(
        fence=agui.Fence("monty", lambda code: code.strip().upper())
    )
    session = agui.Session(fake.server(), ROOM, profile)

    turn = session.turn("go")

    assert fake.bodies[1]["messages"][-1]["content"] == (
        "monty result:\nA\n\nmonty result:\nB"
    )
    assert [b["code"] for b in turn.fence_blocks] == ["a\n", "b\n"]


def test_fence_reply_format_is_pluggable():
    fake = FakeSoliplex(
        [
            framed(text("a1", "```sql\nselect 1\n```")),
            framed(text("a2", "done")),
        ]
    )
    fence = agui.Fence(
        "sql",
        lambda code: "1",
        reply=lambda results: "rows: " + ",".join(results),
    )
    session = agui.Session(
        fake.server(), ROOM, agui.ClientProfile(fence=fence)
    )

    session.turn("go")

    assert fake.bodies[1]["messages"][-1]["content"] == "rows: 1"


def test_fence_handler_failure_is_reported_as_result():
    def boom(code):
        raise RuntimeError("halt")

    fake = FakeSoliplex(
        [
            framed(text("a1", "```monty\nx\n```")),
            framed(text("a2", "done")),
        ]
    )
    session = agui.Session(
        fake.server(),
        ROOM,
        agui.ClientProfile(fence=agui.Fence("monty", boom)),
    )

    turn = session.turn("go")

    assert json.loads(turn.fence_blocks[0]["result"]) == {
        "error": {"type": "RuntimeError", "message": "halt"}
    }


def test_fence_round_cap():
    fake = FakeSoliplex(
        [framed(text(f"a{i}", "```monty\nx\n```")) for i in range(2)]
    )
    profile = agui.ClientProfile(
        fence=agui.Fence("monty", lambda code: "1"), max_rounds=1
    )
    session = agui.Session(fake.server(), ROOM, profile)

    turn = session.turn("go")

    assert turn.error == agui.MAX_ROUNDS
    assert turn.rounds == 1


# -- collector mapping -------------------------------------------------


def two_round_fake():
    return FakeSoliplex(
        [
            framed(
                call("s1", "invented_tool", "{}")
                + result(
                    "s1",
                    "Unknown tool name: 'invented_tool'. Available tools: "
                    "'run_monty'\n\nFix the errors and try again.",
                )
                + call("c1", "run_monty", '{"code": "x"}', parent="a2")
                + [{"type": "STATE_SNAPSHOT", "snapshot": {"s": 1}}]
            ),
            framed(text("a3", "Total 769.59")),
        ]
    )


def test_turn_maps_calls_and_results_in_order():
    fake = two_round_fake()
    session = agui.Session(
        fake.server(), ROOM, agui.ClientProfile(tools=(echo_tool(),))
    )

    turn = session.turn("go")

    assert turn.tool_calls == [
        ToolCall(name="invented_tool", arguments="{}", id="s1"),
        ToolCall(name="run_monty", arguments='{"code": "x"}', id="c1"),
    ]
    assert [r.name for r in turn.tool_results] == [
        "invented_tool",
        "run_monty",
    ]
    assert turn.tool_results[1] == ToolResult(
        name="run_monty", id="c1", result=json.dumps({"value": "x"})
    )


def test_run_trial_builds_a_trial_record():
    fake = two_round_fake()
    profile = agui.ClientProfile(tools=(echo_tool(),))

    record = agui.run_trial(
        fake.server(),
        ROOM,
        "go",
        cell="C-ctx",
        trial=3,
        profile=profile,
        metadata={"model": "glimmer"},
    )

    assert record.cell == "C-ctx"
    assert record.trial == 3
    assert record.ok
    assert record.error is None
    assert record.response == "Total 769.59"
    assert record.call_names == ["invented_tool", "run_monty"]
    assert record.tool_results is not None
    assert record.metadata["model"] == "glimmer"
    assert record.metadata["thread_id"] == THREAD
    assert record.metadata["run_ids"] == ["run-0", "run-1"]
    assert record.metadata["rounds"] == 1
    assert record.metadata["state"] == {"s": 1}
    assert "trace_id" in record.metadata
    assert record.metadata["outcomes"] == agui.OUTCOMES_SOURCE


def test_run_trial_record_scores_retries_and_invented_names():
    fake = two_round_fake()
    profile = agui.ClientProfile(tools=(echo_tool(),))
    record = agui.run_trial(
        fake.server(), ROOM, "go", cell="c", trial=0, profile=profile
    )

    assert scoring.retry_count(record) == 1
    assert scoring.invented_tool_name({"run_monty"}).test(record)
    assert scoring.retried().test(record)


def test_run_trial_runs_each_prompt_as_a_turn():
    fake = FakeSoliplex([framed(text("a1", "one")), framed(text("a2", "two"))])

    record = agui.run_trial(
        fake.server(), ROOM, ["first", "second"], cell="c", trial=0
    )

    assert record.response == "two"
    assert record.metadata["run_ids"] == ["run-0", "run-1"]
    assert fake.threads_created == 1


def test_run_trial_records_a_turn_error():
    fake = FakeSoliplex([422])

    record = agui.run_trial(fake.server(), ROOM, "go", cell="c", trial=0)

    assert not record.ok
    assert record.error == "HTTP 422: refused"


def test_run_trial_records_an_exception():
    def refuse(request):
        return httpx.Response(500)

    server = agui.Server(
        "http://soliplex.test", transport=httpx.MockTransport(refuse)
    )

    record = agui.run_trial(server, ROOM, "go", cell="c", trial=0)

    assert not record.ok
    assert record.error.startswith("HTTPStatusError")
    assert record.tool_results is None


def test_turn_result_as_dict_is_json():
    fake = two_round_fake()
    session = agui.Session(
        fake.server(), ROOM, agui.ClientProfile(tools=(echo_tool(),))
    )

    data = json.loads(json.dumps(session.turn("go").as_dict()))

    assert data["tool_calls"][0]["name"] == "invented_tool"
    assert data["runs"][0]["request"]["runId"] == "run-0"


# -- serve -------------------------------------------------------------


class FakeProcess:
    def __init__(self, exit_after=None):
        self.polls = 0
        self.exit_after = exit_after
        self.returncode = None
        self.terminated = False

    def poll(self):
        self.polls += 1
        if self.exit_after is not None and self.polls > self.exit_after:
            self.returncode = 3
        return self.returncode

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):  # pragma: no cover -- wait() never times out here
        raise AssertionError


def test_serve_launches_and_waits_for_rooms(tmp_path):
    launched = {}
    process = FakeProcess()

    def spawn(argv, **kwargs):
        launched["argv"] = argv
        launched["env"] = kwargs["env"]
        return process

    answers = iter([False, False, True])

    with agui.serve(
        tmp_path / "installation.yaml",
        log=tmp_path / "server.log",
        command=["uv", "run", "soliplex-cli"],
        port=8123,
        env={"EXTRA": "1"},
        audit_runner=FakeAudit(),
        spawn=spawn,
        ready=lambda url: next(answers),
        sleep=lambda s: None,
    ) as served:
        assert served.url == "http://127.0.0.1:8123"
        assert served.log == tmp_path / "server.log"

    assert launched["argv"] == [
        "uv",
        "run",
        "soliplex-cli",
        "serve",
        str(tmp_path / "installation.yaml"),
        "--no-auth-mode",
        "--port",
        "8123",
    ]
    assert launched["env"]["EXTRA"] == "1"
    assert process.terminated


def test_serve_uses_the_environments_cli(tmp_path):
    from soliplex_lab_harness import environs

    env = environs.Environment(
        root=tmp_path / "venv", pin=environs.Pin(name="v", version="1")
    )
    launched = {}

    def spawn(argv, **kwargs):
        launched["argv"] = argv
        return FakeProcess()

    with agui.serve(
        tmp_path / "i.yaml",
        log=tmp_path / "s.log",
        environment=env,
        port=1,
        audit_runner=FakeAudit(),
        spawn=spawn,
        ready=lambda url: True,
        sleep=lambda s: None,
    ):
        pass

    assert launched["argv"][0] == str(env.python.parent / "soliplex-cli")


def test_serve_refuses_command_and_environment(tmp_path):
    from soliplex_lab_harness import environs

    env = environs.Environment(
        root=tmp_path, pin=environs.Pin(name="v", version="1")
    )

    with (
        pytest.raises(agui.ServeError),
        agui.serve(
            tmp_path / "i.yaml",
            log=tmp_path / "s.log",
            command=["x"],
            environment=env,
        ),
    ):
        pass  # pragma: no cover


def test_serve_reports_an_exited_server(tmp_path):
    with (
        pytest.raises(agui.ServerExited, match="exited with 3"),
        agui.serve(
            tmp_path / "i.yaml",
            log=tmp_path / "s.log",
            port=1,
            audit_runner=FakeAudit(),
            spawn=lambda argv, **kw: FakeProcess(exit_after=0),
            ready=lambda url: False,
            sleep=lambda s: None,
        ),
    ):
        pass  # pragma: no cover


def test_serve_gives_up_after_the_timeout(tmp_path):
    clock = iter(range(0, 1000, 10))
    process = FakeProcess()

    with (
        pytest.raises(agui.ServerNotReady),
        agui.serve(
            tmp_path / "i.yaml",
            log=tmp_path / "s.log",
            port=1,
            timeout_s=30,
            audit_runner=FakeAudit(),
            spawn=lambda argv, **kw: process,
            ready=lambda url: False,
            sleep=lambda s: None,
            clock=lambda: next(clock),
        ),
    ):
        pass  # pragma: no cover
    assert process.terminated


def test_serve_errors_are_runtime_errors():
    assert issubclass(agui.ServeError, RuntimeError)


def test_free_port_is_an_int():
    assert isinstance(agui.free_port(), int)


def test_module_imports_httpx_lazily():
    source = pathlib.Path(agui.__file__).read_text()

    top = source.split("\ndef ", 1)[0].split("\nclass ", 1)[0]
    assert "import httpx" not in top
    assert "import logfire" not in top


# -- telemetry ---------------------------------------------------------


class StubContext:
    trace_id = 0xABC


class StubSpan:
    def __init__(self, name, attributes):
        self.name = name
        self.attributes = dict(attributes)
        self.level = "info"

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_level(self, level):
        self.level = level

    def get_span_context(self):
        return StubContext()


def test_trial_and_rounds_open_spans(monkeypatch):
    import contextlib

    spans = []

    @contextlib.contextmanager
    def stub_span(name, **attributes):
        span = StubSpan(name, attributes)
        spans.append(span)
        yield span

    monkeypatch.setattr(agui, "_span", stub_span)
    fake = two_round_fake()
    profile = agui.ClientProfile(tools=(echo_tool(),))

    record = agui.run_trial(
        fake.server(), ROOM, "go", cell="c", trial=1, profile=profile
    )

    assert record.metadata["trace_id"] == f"{0xABC:032x}"
    assert [s.name for s in spans] == [
        "trial {cell} t{trial}",
        "client round {round}",
        "execute_tool run_monty",
        "client round {round}",
    ]
    assert spans[0].attributes["thread_id"] == THREAD
    rounds = [s for s in spans if s.name == "client round {round}"]
    assert [s.attributes["run_id"] for s in rounds] == ["run-0", "run-1"]


def test_trace_id_of_no_span_is_none():
    assert agui.trace_id(None) is None


def test_to_record_maps_a_session_driven_by_hand():
    fake = two_round_fake()
    session = agui.Session(
        fake.server(), ROOM, agui.ClientProfile(tools=(echo_tool(),))
    )
    turn = session.turn("go")

    record = agui.to_record(
        session,
        [turn],
        cell="C-ctx",
        trial=2,
        elapsed_s=1.5,
        trace_id="abc",
        metadata={"arm": "C-ctx"},
    )

    assert record.ok
    assert record.elapsed_s == 1.5
    assert record.call_names == ["invented_tool", "run_monty"]
    assert scoring.retry_count(record) == 1
    assert record.metadata["arm"] == "C-ctx"
    assert record.metadata["trace_id"] == "abc"
    assert record.metadata["run_ids"] == ["run-0", "run-1"]


def test_to_record_with_a_failure_has_no_outcomes():
    fake = FakeSoliplex([])
    session = agui.Session(fake.server(), ROOM)

    record = agui.to_record(
        session, [], cell="c", trial=0, elapsed_s=0.0, failure="boom"
    )

    assert not record.ok
    assert record.error == "boom"
    assert record.tool_results is None
    assert record.response is None


# -- audit -------------------------------------------------------------


class FakeAudit:
    """An injected audit runner: one (returncode, stdout, stderr) per call."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, argv, env):
        self.calls.append((list(argv), dict(env)))
        if self.outcomes:
            return self.outcomes.pop(0)
        return (0, "", "")


def test_audit_runs_three_sections_with_the_servers_launcher(tmp_path):
    runner = FakeAudit()

    results = agui.audit(
        tmp_path / "installation.yaml",
        command=["uv", "run", "soliplex-cli"],
        env={"PYTHONPATH": "/bench"},
        cell="baseline",
        runner=runner,
    )

    assert [argv for argv, _ in runner.calls] == [
        [
            "uv",
            "run",
            "soliplex-cli",
            "audit",
            "-q",
            section,
            str(tmp_path / "installation.yaml"),
        ]
        for section in ("installation", "rooms", "skills")
    ]
    assert all(env["PYTHONPATH"] == "/bench" for _, env in runner.calls)
    assert [r.what for r in results] == [
        "soliplex-cli audit installation",
        "soliplex-cli audit rooms",
        "soliplex-cli audit skills",
    ]
    assert all(r.ok and r.cell == "baseline" for r in results)
    preconditions.assert_ok(results)


def test_audit_uses_the_environments_cli(tmp_path):
    from soliplex_lab_harness import environs

    env = environs.Environment(
        root=tmp_path / "venv", pin=environs.Pin(name="v", version="1")
    )
    runner = FakeAudit()

    agui.audit(tmp_path / "i.yaml", environment=env, runner=runner)

    assert runner.calls[0][0][0] == str(env.python.parent / "soliplex-cli")


def test_audit_stops_at_the_first_failing_section(tmp_path):
    runner = FakeAudit(
        (0, "", ""),
        (1, '{"room": {"mc_ctx": "unknown tool"}}', "warning text"),
    )

    results = agui.audit(tmp_path / "i.yaml", runner=runner)

    assert len(runner.calls) == 2
    assert [r.ok for r in results] == [True, False]
    assert "unknown tool" in results[1].detail
    assert "warning text" in results[1].detail
    with pytest.raises(preconditions.Failed, match="audit rooms"):
        preconditions.assert_ok(results)


def test_audit_cell_defaults_to_the_installation_path(tmp_path):
    results = agui.audit(tmp_path / "i.yaml", runner=FakeAudit())

    assert results[0].cell == str(tmp_path / "i.yaml")


def test_serve_audits_before_it_spawns(tmp_path):
    order = []
    runner = FakeAudit()

    def audit_runner(argv, env):
        order.append("audit")
        return runner(argv, env)

    def spawn(argv, **kwargs):
        order.append("spawn")
        return FakeProcess()

    with agui.serve(
        tmp_path / "i.yaml",
        log=tmp_path / "s.log",
        command=["soliplex-cli"],
        port=1,
        env={"PYTHONPATH": "/bench"},
        spawn=spawn,
        audit_runner=audit_runner,
        ready=lambda url: True,
        sleep=lambda s: None,
    ) as served:
        assert len(served.preconditions) == 3

    assert order == ["audit", "audit", "audit", "spawn"]
    assert runner.calls[0][1]["PYTHONPATH"] == "/bench"


def test_serve_refuses_to_start_on_a_failed_audit(tmp_path):
    def spawn(argv, **kwargs):  # pragma: no cover -- must not be called
        raise AssertionError

    with (
        pytest.raises(preconditions.Failed, match="audit installation"),
        agui.serve(
            tmp_path / "i.yaml",
            log=tmp_path / "s.log",
            port=1,
            spawn=spawn,
            audit_runner=FakeAudit((1, "{}", "bad yaml")),
        ),
    ):
        pass  # pragma: no cover


def test_serve_without_audit_sections_skips_the_audit(tmp_path):
    with agui.serve(
        tmp_path / "i.yaml",
        log=tmp_path / "s.log",
        port=1,
        audit_sections=(),
        spawn=lambda argv, **kw: FakeProcess(),
        audit_runner=FakeAudit((1, "", "")),
        ready=lambda url: True,
        sleep=lambda s: None,
    ) as served:
        assert served.preconditions == []


# -- review fixes ------------------------------------------------------


def test_unanswered_non_client_call_ends_the_turn_with_an_error():
    fake = FakeSoliplex([framed(call("x1", "needs_approval", "{}"))])
    session = agui.Session(fake.server(), ROOM)

    turn = session.turn("go")

    assert turn.error == "unanswered tool calls: needs_approval"
    assert len(fake.bodies) == 1


def test_parentless_calls_in_one_run_share_one_assistant_message():
    events = []
    for call_id in ("c1", "c2"):
        events += call(call_id, "run_monty", '{"code": "x"}', parent=None)
    fake = FakeSoliplex([framed(events), framed(text("a2", "done"))])
    session = agui.Session(
        fake.server(), ROOM, agui.ClientProfile(tools=(echo_tool(),))
    )

    session.turn("go")

    sent = fake.bodies[1]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "tool", "tool"]
    assert [c["id"] for c in sent[1]["toolCalls"]] == ["c1", "c2"]


def test_text_start_without_content_on_a_tool_parent():
    events = call("c1", "run_monty", '{"code": "x"}', parent="a1") + [
        {"type": "TEXT_MESSAGE_START", "messageId": "a1"},
        {"type": "TEXT_MESSAGE_END", "messageId": "a1"},
    ]
    fake = FakeSoliplex([framed(events), framed(text("a2", "done"))])
    profile = agui.ClientProfile(
        tools=(echo_tool(),), fence=agui.Fence("monty", lambda code: "1")
    )
    session = agui.Session(fake.server(), ROOM, profile)

    turn = session.turn("go")

    assert turn.final_text == "done"


def test_repeated_text_start_does_not_run_a_fence_twice():
    events = text("a1", "```monty\nx\n```") + [
        {"type": "TEXT_MESSAGE_START", "messageId": "a1"},
    ]
    fake = FakeSoliplex([framed(events), framed(text("a2", "done"))])
    profile = agui.ClientProfile(fence=agui.Fence("monty", lambda code: "1"))
    session = agui.Session(fake.server(), ROOM, profile)

    turn = session.turn("go")

    assert len(turn.fence_blocks) == 1


def test_results_keep_stream_order():
    events = (
        call("s1", "first", "{}")
        + call("s2", "second", "{}")
        + result("s2", "two")
        + result("s1", "one")
        + text("a2", "done")
    )
    fake = FakeSoliplex([framed(events)])
    session = agui.Session(fake.server(), ROOM)

    turn = session.turn("go")

    assert [r.id for r in turn.tool_results] == ["s2", "s1"]
    assert [r.name for r in turn.tool_results] == ["second", "first"]


def test_negative_max_rounds_is_refused():
    with pytest.raises(agui.BadMaxRounds):
        agui.ClientProfile(max_rounds=-1)


def test_serve_checks_the_process_after_ready(tmp_path):
    with (
        pytest.raises(agui.ServerExited),
        agui.serve(
            tmp_path / "i.yaml",
            log=tmp_path / "s.log",
            port=1,
            audit_runner=FakeAudit(),
            spawn=lambda argv, **kw: FakeProcess(exit_after=0),
            ready=lambda url: True,
            sleep=lambda s: None,
        ),
    ):
        pass  # pragma: no cover


# -- trace linking -----------------------------------------------------


def test_serve_links_traces_through_the_asgi_wrapper(tmp_path):
    launched = {}

    def spawn(argv, **kwargs):
        launched["argv"] = argv
        launched["env"] = kwargs["env"]
        return FakeProcess()

    with agui.serve(
        tmp_path / "i.yaml",
        log=tmp_path / "work" / "s.log",
        command=["soliplex-cli"],
        port=8,
        env={"PYTHONPATH": "/bench"},
        link_traces=True,
        audit_runner=FakeAudit(),
        spawn=spawn,
        ready=lambda url: True,
        sleep=lambda s: None,
    ):
        pass

    argv = launched["argv"]
    assert argv[-4:] == [
        "--workers",
        "1",
        "--app-factory-name",
        f"{agui.ASGI_TRACE_MODULE}:app",
    ]
    shim = launched["env"]["PYTHONPATH"].split(":")
    assert shim[1] == "/bench"
    copied = pathlib.Path(shim[0]) / f"{agui.ASGI_TRACE_MODULE}.py"
    assert "class TraceparentMiddleware" in copied.read_text()


def test_serve_does_not_link_traces_when_off(tmp_path):
    launched = {}

    def spawn(argv, **kwargs):
        launched["argv"] = argv
        return FakeProcess()

    with agui.serve(
        tmp_path / "i.yaml",
        log=tmp_path / "s.log",
        port=8,
        link_traces=False,
        audit_runner=FakeAudit(),
        spawn=spawn,
        ready=lambda url: True,
        sleep=lambda s: None,
    ):
        pass

    assert "--app-factory-name" not in launched["argv"]


def test_link_traces_defaults_to_whether_telemetry_is_configured(
    tmp_path, monkeypatch
):
    seen = []

    def spawn(argv, **kwargs):
        seen.append("--app-factory-name" in argv)
        return FakeProcess()

    for configured in (False, True):
        monkeypatch.setattr(agui, "_configured", configured)
        with agui.serve(
            tmp_path / "i.yaml",
            log=tmp_path / "s.log",
            port=8,
            audit_runner=FakeAudit(),
            spawn=spawn,
            ready=lambda url: True,
            sleep=lambda s: None,
        ):
            pass

    assert seen == [False, True]


def recording_spans(monkeypatch):
    import contextlib

    spans = []

    @contextlib.contextmanager
    def stub_span(name, **attributes):
        span = StubSpan(name, attributes)
        spans.append(span)
        yield span

    monkeypatch.setattr(agui, "_span", stub_span)
    return spans


def test_client_tool_runs_in_an_execute_tool_span(monkeypatch):
    spans = recording_spans(monkeypatch)
    fake = FakeSoliplex(
        [
            framed(call("c1", "run_monty", '{"code": "1 + 1"}')),
            framed(text("a2", "2")),
        ]
    )
    session = agui.Session(
        fake.server(), ROOM, agui.ClientProfile(tools=(echo_tool(),))
    )

    session.turn("go")

    (tool,) = [s for s in spans if s.name.startswith("execute_tool")]
    assert tool.name == "execute_tool run_monty"
    assert tool.attributes == {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": "run_monty",
        "gen_ai.tool.call.id": "c1",
        "gen_ai.tool.call.arguments": {"code": "1 + 1"},
        "gen_ai.conversation.id": THREAD,
        "gen_ai.tool.call.result": {"value": "1 + 1"},
    }
    assert tool.level == "info"


def test_execute_tool_span_is_an_error_when_the_handler_raises(monkeypatch):
    spans = recording_spans(monkeypatch)

    def boom(args):
        raise HostFunctionMissing

    fake = FakeSoliplex(
        [
            framed(call("c1", "run_monty", '{"code": "x"}')),
            framed(text("a2", "sorry")),
        ]
    )
    profile = agui.ClientProfile(tools=(echo_tool(handler=boom),))
    agui.Session(fake.server(), ROOM, profile).turn("go")

    (tool,) = [s for s in spans if s.name.startswith("execute_tool")]
    assert tool.level == "error"
    assert tool.attributes["gen_ai.tool.call.result"]["error"]["type"] == (
        "HostFunctionMissing"
    )


def test_execute_tool_span_is_an_error_when_the_result_is_one(monkeypatch):
    spans = recording_spans(monkeypatch)
    failing = echo_tool(handler=lambda args: json.dumps({"error": "bad"}))
    fake = FakeSoliplex(
        [
            framed(call("c1", "run_monty", '{"code": "x"}')),
            framed(text("a2", "sorry")),
        ]
    )
    agui.Session(
        fake.server(), ROOM, agui.ClientProfile(tools=(failing,))
    ).turn("go")

    (tool,) = [s for s in spans if s.name.startswith("execute_tool")]
    assert tool.level == "error"


def test_run_post_carries_the_injected_traceparent(monkeypatch):
    spans = recording_spans(monkeypatch)
    headers = []

    def handler(request):
        headers.append(request.headers.get("traceparent"))
        return fake.handler(request)

    def inject(carrier):
        carrier["traceparent"] = "00-trace-span-01"

    fake = FakeSoliplex([framed(text("a1", "hi"))])
    server = agui.Server(
        "http://soliplex.test",
        transport=httpx.MockTransport(handler),
        inject=inject,
    )

    server.post_run(ROOM, THREAD, "run-0", {"threadId": THREAD})

    assert headers == ["00-trace-span-01"]
    assert spans == []


def test_server_injects_with_opentelemetry_when_present(monkeypatch):
    import sys
    import types

    propagate = types.ModuleType("opentelemetry.propagate")
    propagate.inject = lambda carrier: carrier.update(traceparent="00-x")
    package = types.ModuleType("opentelemetry")
    package.propagate = propagate
    monkeypatch.setitem(sys.modules, "opentelemetry", package)
    monkeypatch.setitem(sys.modules, "opentelemetry.propagate", propagate)
    seen = []

    def handler(request):
        seen.append(request.headers.get("traceparent"))
        return httpx.Response(200, json={"run_id": "r"})

    server = agui.Server(
        "http://soliplex.test", transport=httpx.MockTransport(handler)
    )
    server.new_run(ROOM, THREAD)

    assert seen == ["00-x"]


def test_configure_telemetry_does_not_instrument_httpx(monkeypatch):
    import sys
    import types

    calls = []
    fake_logfire = types.ModuleType("logfire")
    fake_logfire.configure = lambda **kw: calls.append(("configure", kw))
    fake_logfire.instrument_httpx = lambda **kw: calls.append(("httpx", kw))
    monkeypatch.setitem(sys.modules, "logfire", fake_logfire)
    monkeypatch.setattr(agui, "_configured", False)

    agui.configure_telemetry("svc", environment="env")

    assert [c[0] for c in calls] == ["configure"]
    assert calls[0][1]["distributed_tracing"] is True


# -- conversation on the trial span -------------------------------------


def test_conversation_attributes_in_pydantic_ai_run_span_format():
    fake = two_round_fake()
    profile = agui.ClientProfile(
        tools=(echo_tool(),),
        context=({"description": "units", "value": "metric"},),
    )
    session = agui.Session(fake.server(), ROOM, profile)
    turn = session.turn("go")

    attrs = agui.conversation_attributes(session, [turn])

    assert attrs["gen_ai.operation.name"] == "invoke_agent"
    assert attrs["gen_ai.conversation.id"] == THREAD
    assert attrs["logfire.agent_run_candidate.v1"] is True
    assert attrs["final_result"] == "Total 769.59"
    retry = (
        "Unknown tool name: 'invented_tool'. Available tools: "
        "'run_monty'\n\nFix the errors and try again."
    )

    def calls(*parts):
        return {"role": "assistant", "parts": list(parts)}

    def responses(*parts):
        return {"role": "user", "parts": list(parts)}

    # The fixture's two calls have different parent messages (a1, a2).
    assert attrs["pydantic_ai.all_messages"] == [
        {"role": "user", "parts": [{"type": "text", "content": "go"}]},
        calls(
            {
                "type": "tool_call",
                "id": "s1",
                "name": "invented_tool",
                "arguments": "{}",
            }
        ),
        responses(
            {
                "type": "tool_call_response",
                "id": "s1",
                "name": "invented_tool",
                "result": retry,
            }
        ),
        calls(
            {
                "type": "tool_call",
                "id": "c1",
                "name": "run_monty",
                "arguments": '{"code": "x"}',
            }
        ),
        responses(
            {
                "type": "tool_call_response",
                "id": "c1",
                "name": "run_monty",
                "result": {"value": "x"},
            }
        ),
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "Total 769.59"}],
        },
    ]
    assert (
        attrs["gen_ai.input.messages"]
        == (attrs["pydantic_ai.all_messages"][:-1])
    )
    assert (
        attrs["gen_ai.output.messages"]
        == (attrs["pydantic_ai.all_messages"][-1:])
    )
    assert attrs["gen_ai.tool.definitions"] == [
        {
            "type": "function",
            "name": "run_monty",
            "description": "run code",
            "parameters": {"type": "object", "properties": {}},
        }
    ]
    assert attrs["gen_ai.system_instructions"] == [
        {"type": "text", "content": "units: metric"}
    ]


def test_conversation_attributes_include_fence_exchanges():
    fake = FakeSoliplex(
        [
            framed(text("a1", "```monty\n1 + 1\n```")),
            framed(text("a2", "2")),
        ]
    )
    profile = agui.ClientProfile(fence=agui.Fence("monty", lambda c: "2"))
    session = agui.Session(fake.server(), ROOM, profile)
    turn = session.turn("add")

    attrs = agui.conversation_attributes(session, [turn])

    texts = [
        (m["role"], m["parts"][0]["content"])
        for m in attrs["pydantic_ai.all_messages"]
    ]
    assert texts == [
        ("user", "add"),
        ("assistant", "```monty\n1 + 1\n```"),
        ("user", "monty result:\n2"),
        ("assistant", "2"),
    ]
    assert "gen_ai.tool.definitions" not in attrs
    assert "gen_ai.system_instructions" not in attrs


def test_conversation_without_a_final_answer_has_no_output():
    fake = FakeSoliplex([422])
    session = agui.Session(fake.server(), ROOM)
    turn = session.turn("go")

    attrs = agui.conversation_attributes(session, [turn])

    assert attrs["gen_ai.output.messages"] == []
    assert "final_result" not in attrs


def test_run_trial_puts_the_conversation_on_the_trial_span(monkeypatch):
    spans = recording_spans(monkeypatch)
    fake = two_round_fake()

    agui.run_trial(
        fake.server(),
        ROOM,
        "go",
        cell="c",
        trial=0,
        profile=agui.ClientProfile(tools=(echo_tool(),)),
    )

    trial = spans[0]
    assert trial.name == "trial {cell} t{trial}"
    assert trial.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert len(trial.attributes["pydantic_ai.all_messages"]) == 6


def test_consecutive_tool_results_fold_into_one_message():
    messages = [
        {"id": "u", "role": "user", "content": "go"},
        {
            "id": "a",
            "role": "assistant",
            "content": "Checking.",
            "toolCalls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "one", "arguments": "{}"},
                },
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "two", "arguments": "{}"},
                },
            ],
        },
        {"id": "t1", "role": "tool", "toolCallId": "c1", "content": "1"},
        {"id": "t2", "role": "tool", "toolCallId": "c2", "content": "x"},
    ]

    out = agui.otel_messages(messages)

    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert [p["type"] for p in out[1]["parts"]] == [
        "text",
        "tool_call",
        "tool_call",
    ]
    assert out[2]["parts"] == [
        {"type": "tool_call_response", "id": "c1", "name": "one", "result": 1},
        {
            "type": "tool_call_response",
            "id": "c2",
            "name": "two",
            "result": "x",
        },
    ]


# -- agy review fixes --------------------------------------------------


def test_duplicate_tool_call_id_in_a_run_is_an_error():
    events = (
        call("c1", "agui_context", "{}")
        + result("c1", "ctx")
        + call("c1", "run_monty", '{"code": "x"}', parent="a2")
    )
    fake = FakeSoliplex([framed(events)])
    session = agui.Session(
        fake.server(), ROOM, agui.ClientProfile(tools=(echo_tool(),))
    )

    turn = session.turn("go")

    assert turn.error == f"{agui.DUPLICATE_CALL_ID}: c1"
    assert turn.client_calls == []


def test_message_id_from_an_earlier_run_starts_a_new_message():
    fake = FakeSoliplex(
        [
            framed(text("m1", "first") + call("c1", "run_monty", "{}", "m1")),
            framed(text("m1", "second")),
        ]
    )
    session = agui.Session(
        fake.server(), ROOM, agui.ClientProfile(tools=(echo_tool(),))
    )

    turn = session.turn("go")

    contents = [
        m.get("content") for m in session.messages if m["role"] == "assistant"
    ]
    assert contents == ["first", "second"]
    assert turn.final_text == "second"


def test_final_text_comes_from_this_turn_only():
    fake = FakeSoliplex([framed(text("a1", "one")), framed([])])
    session = agui.Session(fake.server(), ROOM)

    session.turn("first")
    second = session.turn("second")

    assert second.final_text is None


def test_record_has_no_outcomes_when_a_call_has_none():
    loop = [
        framed(call(f"c{i}", "run_monty", '{"code": "x"}', parent=f"a{i}"))
        for i in range(2)
    ]
    fake = FakeSoliplex(loop)
    profile = agui.ClientProfile(tools=(echo_tool(),), max_rounds=1)

    record = agui.run_trial(
        fake.server(), ROOM, "go", cell="c", trial=0, profile=profile
    )

    assert record.error == agui.MAX_ROUNDS
    assert record.call_names == ["run_monty", "run_monty"]
    assert record.tool_results is None
    assert scoring.retry_count(record) is None


def test_record_keeps_outcomes_when_every_call_has_one():
    fake = FakeSoliplex([422])

    record = agui.run_trial(fake.server(), ROOM, "go", cell="c", trial=0)

    assert record.tool_calls == []
    assert record.tool_results == []


def test_server_reads_threads_and_runs():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, request.content))
        return httpx.Response(200, json={})

    server = agui.Server(
        "http://soliplex.test", transport=httpx.MockTransport(handler)
    )
    server.get_thread(ROOM, THREAD)
    server.get_run(ROOM, THREAD, "r1")
    server.new_run(ROOM, THREAD, parent_run_id="r1")
    server.close()

    assert seen == [
        ("GET", f"/api/v1/rooms/{ROOM}/agui/{THREAD}", b""),
        ("GET", f"/api/v1/rooms/{ROOM}/agui/{THREAD}/r1", b""),
        (
            "POST",
            f"/api/v1/rooms/{ROOM}/agui/{THREAD}",
            b'{"parent_run_id":"r1"}',
        ),
    ]


def test_no_top_level_import_of_server_side_packages():
    import ast

    for module in (agui,):
        tree = ast.parse(pathlib.Path(module.__file__).read_text())
        names = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        assert not names & {"httpx", "logfire", "opentelemetry", "soliplex"}
