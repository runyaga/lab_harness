import asyncio

import pytest

from soliplex_lab_harness import records


class EndpointDown(Exception):
    pass


def make_record(trial):
    return records.TrialRecord(cell="c", trial=trial, elapsed_s=0.1, ok=True)


def run_async(coro):
    return asyncio.run(coro)


def test_runs_every_missing_trial(tmp_path):
    path = tmp_path / "c.jsonl"

    async def run(trial):
        return make_record(trial)

    ran = run_async(records.top_up_async(path, 4, run, max_concurrency=2))

    assert ran == 4
    assert sorted(r.trial for r in records.read(path)) == [0, 1, 2, 3]


def test_resumes_by_recorded_trial_index(tmp_path):
    """A gap left by an interrupted parallel run is filled, not skipped."""
    path = tmp_path / "c.jsonl"
    records.append(path, make_record(0))
    records.append(path, make_record(2))
    asked = []

    async def run(trial):
        asked.append(trial)
        return make_record(trial)

    ran = run_async(records.top_up_async(path, 4, run, max_concurrency=4))

    assert sorted(asked) == [1, 3]
    assert ran == 2
    assert sorted(r.trial for r in records.read(path)) == [0, 1, 2, 3]


def test_a_full_file_runs_nothing(tmp_path):
    path = tmp_path / "c.jsonl"
    for trial in range(3):
        records.append(path, make_record(trial))

    async def run(trial):  # pragma: no cover -- must not be called
        raise AssertionError

    assert run_async(records.top_up_async(path, 3, run)) == 0


def test_concurrency_is_bounded(tmp_path):
    path = tmp_path / "c.jsonl"
    active = 0
    peak = 0

    async def run(trial):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        active -= 1
        return make_record(trial)

    run_async(records.top_up_async(path, 9, run, max_concurrency=3))

    assert peak == 3


def test_records_arrive_in_completion_order_one_per_line(tmp_path):
    path = tmp_path / "c.jsonl"
    order = []

    async def run(trial):
        # Later trials finish first.
        for _ in range(5 - trial):
            await asyncio.sleep(0)
        return make_record(trial)

    run_async(
        records.top_up_async(
            path,
            5,
            run,
            max_concurrency=5,
            on_trial=lambda record: order.append(record.trial),
        )
    )

    assert order == [4, 3, 2, 1, 0]
    assert [r.trial for r in records.read(path)] == order


def test_on_trial_sees_the_record_on_disk(tmp_path):
    path = tmp_path / "c.jsonl"
    on_disk = []

    async def run(trial):
        return make_record(trial)

    run_async(
        records.top_up_async(
            path,
            2,
            run,
            max_concurrency=2,
            on_trial=lambda record: on_disk.append(
                record.trial in {r.trial for r in records.read(path)}
            ),
        )
    )

    assert on_disk == [True, True]


def test_a_failing_trial_keeps_the_others(tmp_path):
    path = tmp_path / "c.jsonl"

    async def run(trial):
        if trial == 1:
            raise EndpointDown
        await asyncio.sleep(0)
        return make_record(trial)

    with pytest.raises(records.TrialsFailed) as info:
        run_async(records.top_up_async(path, 3, run, max_concurrency=3))

    assert [type(e) for e in info.value.exceptions] == [EndpointDown]
    assert sorted(r.trial for r in records.read(path)) == [0, 2]


def test_a_record_for_the_wrong_trial_is_refused(tmp_path):
    path = tmp_path / "c.jsonl"

    async def run(trial):
        return make_record(trial + 10)

    with pytest.raises(records.TrialsFailed) as info:
        run_async(records.top_up_async(path, 1, run))

    (error,) = info.value.exceptions
    assert isinstance(error, records.WrongTrial)
    assert not path.exists()


def test_max_concurrency_must_be_positive(tmp_path):
    async def run(trial):  # pragma: no cover -- must not be called
        raise AssertionError

    with pytest.raises(records.BadConcurrency):
        run_async(
            records.top_up_async(
                tmp_path / "c.jsonl", 1, run, max_concurrency=0
            )
        )


def test_completed_trials(tmp_path):
    path = tmp_path / "c.jsonl"
    assert records.completed_trials(path) == set()
    records.append(path, make_record(2))
    records.append(path, make_record(2))
    records.append(path, make_record(0))

    assert records.completed_trials(path) == {0, 2}
