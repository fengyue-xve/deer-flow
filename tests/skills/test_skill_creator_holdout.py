"""Small evaluation sets must retain examples for description improvement."""

import runpy
from pathlib import Path

import pytest


@pytest.fixture
def loop_script(monkeypatch):
    root = Path(__file__).resolve().parents[2] / "skills/public/skill-creator"
    monkeypatch.syspath_prepend(str(root))
    return runpy.run_path(str(root / "scripts/run_loop.py"))


@pytest.mark.parametrize("positive,negative", [(1, 1), (1, 4), (4, 1), (0, 1), (1, 0), (4, 4)])
def test_holdout_retains_training_example_per_class(loop_script, positive, negative):
    examples = [
        {"query": f"{label}-{i}", "should_trigger": label}
        for label, count in [(True, positive), (False, negative)]
        for i in range(count)
    ]
    train, test = loop_script["split_eval_set"](examples, 0.4)

    assert len(train) + len(test) == len(examples)
    assert {e["query"] for e in train}.isdisjoint(e["query"] for e in test)
    for label, count in [(True, positive), (False, negative)]:
        if count:
            assert any(e["should_trigger"] == label for e in train)
        if count > 1:
            assert any(e["should_trigger"] == label for e in test)


def test_singletons_do_not_report_vacuous_success(loop_script, tmp_path, monkeypatch):
    run_loop = loop_script["run_loop"]
    namespace = run_loop.__globals__
    monkeypatch.setitem(namespace, "find_project_root", lambda: tmp_path)
    monkeypatch.setitem(namespace, "parse_skill_md", lambda _: ("example", "original", "body"))
    examples = [{"query": "yes", "should_trigger": True}, {"query": "no", "should_trigger": False}]
    monkeypatch.setitem(namespace, "run_eval", lambda **kwargs: {
        "results": [{**e, "pass": False, "triggers": 0, "runs": 1} for e in kwargs["eval_set"]]
    })
    improvements = []
    def improve(**kwargs):
        improvements.append(kwargs["eval_results"])
        return "improved"
    monkeypatch.setitem(namespace, "improve_description", improve)

    result = run_loop(examples, tmp_path, None, 1, 1, 2, 1, 0.5, 0.4, "test", False)

    assert len(improvements) == 1
    assert improvements[0]["summary"]["failed"] == 2
    assert result["iterations_run"] == 2
