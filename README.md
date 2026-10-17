# `soliplex_lab_harness`

Reusable machinery for running experiments against a soliplex installation.

See the sibling [lab_bench](https://github.com/soliplex/lab_bench) repository
for how it is intended to be used: this package is the part that does not
change per experiment, and a jig depends on a pinned version of it.

## Installing

There is no PyPI release. Pin a tag:

```toml
dependencies = [
    "soliplex-lab-harness @ git+https://github.com/soliplex/lab_harness@vX.Y",
]
```

`vX.Y` is a placeholder: substitute the version you mean to pin. This page
deliberately names no particular one, so that it cannot come to recommend a
release that has been superseded.

Release notes live in
[GitHub Releases](https://github.com/soliplex/lab_harness/releases), which is
where the current tag is; there is no `CHANGELOG.md` to keep in sync.

## What is here

| module | does |
| --- | --- |
| `collect` | captures the tool calls a run made, and what came back |
| `records` | one record per trial, appended to JSONL; tops a file up to N |
| `drive` | runs one turn against an installation, in process |
| `agui` | drives a running server over AG-UI HTTP, with client tools |
| `scoring` | turns records into rates, with pluggable checks |
| `environs` | builds one virtualenv per code-axis value |
| `preconditions` | refuses a run whose cells cannot answer the question |

Import the modules, not names from the package root.

## The one non-obvious thing

An **invented tool name never produces a tool span.**
`ToolManager._resolve_tool` rejects an unknown name before
`wrap_tool_execute` opens one, so a harness that harvests spans named
`running tool` reports a clean sheet no matter what the model did -- and a
null result then looks like a finding.

`collect` reads the agent run span's `pydantic_ai.all_messages` attribute
instead. That carries the whole message history, and every tool-call part in
it is recorded under the name the model actually used. It is also what
Logfire renders as a run's "events" list.

## Retries are the error signal that has room to move

`ok` and `correct` sit at the ceiling in a room that works: the interesting
question is what a run had to recover from on the way. Every `ModelRetry`
reaches the model through `RetryPromptPart.model_response`, which appends
`Fix the errors and try again.` -- and that suffix is the **only** marker
distinguishing a retry from a success, because `ToolReturnPart` and
`RetryPromptPart` both serialize as `type='tool_call_response'` with no
discriminator. `collect` therefore keeps response payloads, and
`scoring.classify_result` reads them:

| shape | means |
| --- | --- |
| `retry-unknown-tool` | the model invented a tool name |
| `retry-already-available` | a `load_capability` bounce |
| `retry-validation` | argument validation rejected the call |
| `retry-other` | some other `ModelRetry` |
| `tool-error` | the tool ran and handed back a failure |

Reading outcomes needs `include_content=True`, which
`drive.install_collector` sets.

## `None` is not zero

`TrialRecord.tool_results` is `None` when outcomes were not captured and
`[]` when they were captured and there were none. Anything counting them --
`scoring.retry_count`, the `retried()` check -- returns `None` rather than
`0` for a record that cannot answer, and `tally` drops those trials from
that check's denominator instead of scoring them as misses. A record written
before outcomes were captured therefore reports `-`, not a clean sheet.

Same reasoning as the tool-span problem above: a null that reads as a
result is the failure mode this package exists to avoid.

## Dispersion, not just means

`Tally` carries `turns`, `secs` and `retries` as `Dist` summaries -- mean,
sd, median, min, max -- and `secs_per_turn`, which separates "more
round-trips" from "slower round-trips". The first experiment to use this
package found its result in a standard deviation collapsing from 8.6 to 1.5
while the median barely moved; means alone would have shown nothing.
`render_distributions` draws one metric at a time, apart from the rates
table.

## No dependencies, on purpose

`soliplex` is the software under test. An experiment pins it per code-axis
value -- `soliplex==0.78.1`, or `soliplex @ git+<url>@<sha>` -- so a pin here
would fight the axis. Both `soliplex` and `logfire` are imported lazily by
`drive`, which raises a pointed error if they are missing.

## Sketch

```python
from pathlib import Path

from soliplex_lab_harness import drive
from soliplex_lab_harness import records
from soliplex_lab_harness import scoring

collector = drive.install_collector()   # once per process
target = drive.Target(
    installation=Path("cells/before-gemma4/example/minimal.yaml"),
    room_id="bwrap_sandbox",
    cwd=Path("cells/before-gemma4"),
    env={"OPENAI_API_KEY": "unused-by-vllm"},
)
installation = drive.load_installation(target)

out = Path("results/before-gemma4.jsonl")
records.top_up(
    out,
    20,
    lambda trial: drive.run_trial(
        target,
        "What is the total order value for the Southeast region?",
        cell="before-gemma4",
        trial=trial,
        collector=collector,
        installation=installation,
        metadata={"ref": "v0.78.1", "model": "gemma4-26b"},
    ),
    on_trial=lambda record: print(record.cell, record.trial, record.ok),
)

checks = [
    scoring.succeeded(),
    scoring.response_contains("40935.89"),
    scoring.invented_tool_name({"run", "run_python", "list_environments"}),
    scoring.retried(),
]
result = scoring.tally("before-gemma4", records.read(out), checks)
print(scoring.render([result], checks))
print(scoring.render_distributions([result], "turns"))
```

`top_up` takes a **target, not a count**: records already in `out` count
toward the 20, so a smoke trial can be run and verified before extending to
the full N without discarding it, and an interrupted run resumes rather than
restarting. Each record is appended as it arrives, so an interrupt loses at
most one trial.

`metadata` is open on purpose: stamp whatever identifies the arm -- resolved
commit sha, exact model id, endpoint. A result whose arm cannot be identified
afterwards is not a result.

## Driving a room over AG-UI

`drive` runs `soliplex ask` in process, and that path sends no client
tools, context, state or forwarded props. `agui` is the client side of the
protocol. It talks to a running `soliplex-cli serve`, runs the model's calls
to client tools itself, and sends the results in the next run of the same
thread.

```python
import asyncio
import json
from pathlib import Path

from soliplex_lab_harness import agui
from soliplex_lab_harness import records

agui.configure_telemetry("my-jig")  # optional, once per process


def run_code(args):
    return json.dumps({"value": len(args["code"])})


profile = agui.ClientProfile(
    tools=(
        agui.ClientTool(
            name="run_code",
            description="Run a program on the user's device.",
            parameters={
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
            handler=run_code,
            metadata={"skill": "ledger"},
        ),
    ),
    context=({"description": "units", "value": "metric"},),
    state={"client_prefs": {"units": "metric"}},
    forwarded_props={"trigger": "button"},
    max_rounds=6,
)

with agui.serve(
    Path("cells/tip/installation.yaml"),
    log=Path("work/tip/server.log"),
    environment=env,  # an environs.Environment, or command=[...]
) as served:
    server = agui.Server(served.url)

    async def trial(index):
        return await asyncio.to_thread(
            agui.run_trial,
            server,
            "ledger",
            ["What is my pending total?", "And last month?"],
            cell="tip",
            trial=index,
            profile=profile,
            metadata=env.metadata(),
        )

    asyncio.run(
        records.top_up_async(
            Path("results/tip.jsonl"), 20, trial, max_concurrency=4
        )
    )
```

- `Session(server, room, profile).turn(text)` is the lower level: one user
  message, then client-tool rounds until the model stops calling client
  tools or `max_rounds` is reached (`error == agui.MAX_ROUNDS`). A second
  `turn` continues the thread with its history and state. Its
  `TurnResult` keeps every run's request body and raw SSE lines.
- The session sends its `state` with every run and, with
  `adopt_snapshots=True`, replaces it with each `STATE_SNAPSHOT`.
- `Fence(tag, handler)` is a protocol without client tools. Each
  `` ```tag `` block in a run's reply goes to `handler`, and the results go
  back as the next user message (`"<tag> result:\n..."` by default). Fence
  rounds count toward `max_rounds`, and the blocks land in
  `metadata["fence_blocks"]`, not in `tool_calls`.
- `run_trial` stamps `thread_id`, `run_ids`, `rounds`, `trace_id` and the
  final `state` into `metadata`. Nothing process-global changes, so trials
  may run concurrently. `records.top_up_async` runs them under a
  concurrency bound and resumes by trial index.
- `configure_telemetry` instruments httpx, so every request carries a W3C
  `traceparent`. It sends only when `LOGFIRE_TOKEN` is set.

### What an AG-UI record can see

The records come from the event stream, because
`pydantic_ai.all_messages` stays in the server process. pydantic-ai's AG-UI
adapter streams each tool-call part under the name the model used, before
`_resolve_tool` runs, and streams each `RetryPromptPart` that answers a
tool call as a `TOOL_CALL_RESULT` carrying `model_response()`. Invented
tool names and tool retries therefore reach `tool_calls` and
`tool_results` with the text that `collect` reads, and
`scoring.invented_tool_name`, `scoring.retried` and
`scoring.classify_result` work unchanged. This is the behaviour of
pydantic-ai 2.56. A code axis that moves pydantic-ai needs it checked again.

A `ModelRetry` that answers no tool call, such as one from an output
validator, appears in neither source. The event stream does not carry it,
and `collect` reads it as a plain text part.

## Preconditions: refuse before spending trials

A cell that structurally cannot exhibit the behaviour under test still
produces numbers, and those numbers read as a finding. Two nulls in the work
that motivated this package were burned exactly that way, and every defect
caught since has been silent in the same manner: the run completed and the
table looked plausible.

`preconditions` owns the part of that which is not experiment-specific -- a
result record, a renderer, and a way to refuse:

```python
from soliplex_lab_harness import preconditions

def check(cell, trials):
    return [
        preconditions.Result(
            cell, "a turn completes", trials[0].ok, trials[0].error or ""
        ),
        preconditions.Result(
            cell,
            "deferral engages",
            any("load_capability" in t.call_names for t in trials),
        ),
    ]

results = [r for cell in cells for r in check(cell, read(cell))]
print(preconditions.render(results))
preconditions.assert_ok(results)   # raises Failed, naming each one
```

**What a check *is* stays with the experiment.** A check is any function
returning `Result` values, because what is worth asserting differs entirely
between sets. This module deliberately does not define a check type, a
registry, or a runner: there is one working example to generalise from, and
that is enough to justify the record but not an abstraction over it.

`render([])` reports `no preconditions were checked` rather than saying
everything holds. Nothing checked is not the same as nothing wrong, and
losing that distinction is the failure this module exists to prevent.

Nothing here drives a turn -- a check reads what a run already recorded. The
checks that need no turn belong with whatever they check, on the principle of
make-the-thing-then-verify-the-thing: `environs.verify_install` compares an
install against its own RECORD at build time and raises there, where the
error cannot be skipped or mistaken for a measurement failure.

## Code axes install; they do not check out

Each code-axis value is a `Pin` -- an exact released version, or a git ref --
installed into its own virtualenv:

```python
from pathlib import Path

from soliplex_lab_harness import environs

released = environs.Pin(name="v078", version="0.78.1")
from_ref = environs.Pin(
    name="tip",
    url="https://github.com/soliplex/soliplex",
    ref="v0.78.1",
)

env = environs.build(
    released,
    Path("envs/v078"),
    extra_requirements=(
        "soliplex-lab-harness @ git+https://github.com/soliplex/lab_harness@vX.Y",
    ),
)
print(env.python)        # interpreter to drive trials with
print(env.metadata())    # stamp this onto every record from this arm
```

`extra_requirements` normally includes this package, because a trial is
driven in process: the harness has to be importable in the environment that
holds the software under test. Passing it explicitly also forces the
experiment to record *which harness version measured the run*. The example
here uses the same `X.Y` placeholders used elsewhere: they must be replaced
with the specific version of this package which the trial requires.

`build()` resolves a tag or branch to a **commit sha** via `git ls-remote`
and puts it in `metadata()`. Recording a tag records nothing -- tags can be
moved, and branches certainly are.

### Overlays: isolating co-landed changes

A release bundles every change in it. To attribute an effect to one of them
you need an arm that is "ref A, but with this one file from ref B":

```python
env = environs.build(
    environs.Pin(name="v077skill", version="0.77.2"),
    Path("envs/v077skill"),
    overlays=[
        environs.Overlay(
            source=Path("overlays/SKILL.md"),          # committed with the experiment
            destination="soliplex/skills/bwrap_sandbox/SKILL.md",
            note="SKILL.md as of v0.78.1",
        )
    ],
    extra_requirements=(
        "soliplex-lab-harness @ git+https://github.com/soliplex/lab_harness@vX.Y",
    ),
)
```

The overlay refuses to *create* a file: a typo in `destination` would
otherwise silently add something the software never shipped, and the arm
would measure software nobody runs. It also lands in `metadata()`, so the
arm describes itself.

### Verify an install before spending trials

A built environment is only trustworthy while it still matches what was
installed. Two ways it stops matching, both silent:

```python
environs.verify_install(env)
```

- **Something wrote into it.** Raises `InstallDiverged`, naming the files.
  The case this was built for: an overlay applied through the hardlink uv
  shares with its cache, which rewrites the same file in every sibling
  environment holding that distribution -- and poisons the cache for later
  installs.
- **A declared overlay did not land.** Raises `OverlayNotApplied`, so an arm
  cannot quietly measure the unmodified software while reporting that it
  does not.

By default the accepted changes are the environment's own overlay
destinations, so the ordinary call takes no arguments and an overlay arm
passes while a corrupted one does not. Pass `expect_modified=[...]` to check
against a different set.

Neither failure produces an error on its own -- both produce *plausible
numbers* -- which is why this is worth asserting rather than assuming.

### Everything shells out through an injected runner

`build()` and `Pin.resolve_ref()` take a `runner`, so the whole module is
testable without a network or a real `uv`. The default runner raises
`CommandFailed` with the captured stderr.

## Caveats

- `drive.run_trial` adjusts the process working directory and environment
  around the call, so it is **not thread-safe**. Trials run sequentially.
- `drive.install_collector` configures logfire, which is process-global.
  Call it once.
- Instrumentation is local only (`send_to_logfire=False`): nothing leaves
  the machine.
- `environs.build` shells out to `uv`, which must be on `PATH`.
- Confirm that behavior-affecting **non-Python assets are packaged**. An
  editable checkout exposes every file in the tree; an installed
  distribution exposes only what its packaging includes, so an unpackaged
  asset means an installed arm measures different software than a checkout
  would. soliplex ships `SKILL.md` via `MANIFEST.in`'s `global-include`.

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check
```
