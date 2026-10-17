import json
import pathlib

import httpx
import pytest

from soliplex_lab_harness import agui
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

    def set_attribute(self, key, value):
        self.attributes[key] = value

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
        "client round {round}",
    ]
    assert spans[0].attributes["thread_id"] == THREAD
    assert [s.attributes["run_id"] for s in spans[1:]] == ["run-0", "run-1"]


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
