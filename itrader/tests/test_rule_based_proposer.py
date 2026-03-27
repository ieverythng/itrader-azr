from pathlib import Path

from itrader.proposers.factory import build_proposer
from itrader.proposers.openai_compat import OpenAICompatProposer
from itrader.proposers.ollama import OllamaProposer
from itrader.proposers.rule_based import RuleBasedProposer
from itrader.rl_lib.tasks.task_pool import TaskPool
from itrader.rl_lib.tasks.spec import TaskSpec
from itrader.rl_lib.tasks.validation import validate_and_clamp_task


def test_rule_based_proposer_is_deterministic_for_seed():
    p1 = RuleBasedProposer(seed=2026, window_size=30, n_features=5, t_embed_dim=32, action_dim=3)
    p2 = RuleBasedProposer(seed=2026, window_size=30, n_features=5, t_embed_dim=32, action_dim=3)
    t1 = p1.propose_tasks("summary-a", 4)
    t2 = p2.propose_tasks("summary-b", 4)
    assert t1 == t2


def test_rule_based_tasks_have_expected_shape_fields():
    p = RuleBasedProposer(seed=7, window_size=30, n_features=5, t_embed_dim=32, action_dim=3)
    tasks = p.propose_tasks("summary", 3)
    assert len(tasks) == 3
    for task in tasks:
        assert task["obs_dim"] == 30 * 5 + 32
        assert task["action_dim"] == 3
        assert task["reward_id"] == "pnl_linear"
        assert task["start_idx"] is None
        assert task["bar_resolution"] == "1m"


def test_rule_based_proposer_uses_reward_specific_param_shape():
    proposer = RuleBasedProposer(
        seed=7,
        window_size=30,
        n_features=5,
        t_embed_dim=32,
        action_dim=3,
        reward_id="pnl_sharpe",
    )
    task = proposer.propose_tasks("summary", 1)[0]
    assert task["reward_id"] == "pnl_sharpe"
    assert "dd_w" in task["reward_params"]
    assert "turnover_w" in task["reward_params"]
    assert "dd_mode" not in task["reward_params"]


def test_factory_builds_rule_based_proposer():
    proposer = build_proposer("rule_based", seed=1)
    assert isinstance(proposer, RuleBasedProposer)


def test_factory_builds_ollama_proposer():
    proposer = build_proposer("ollama", model="qwen2.5:7b-instruct")
    assert isinstance(proposer, OllamaProposer)


def test_factory_builds_vllm_proposer():
    proposer = build_proposer("vllm", model="Qwen/Qwen3.5-9B", base_url="http://localhost:8000/v1")
    assert isinstance(proposer, OpenAICompatProposer)


def test_openai_compat_parse_json_with_repair_handles_prefixed_text():
    p = OpenAICompatProposer(model="dummy", base_url="http://localhost:8000/v1")
    text = 'agent reply [{"family_id":"00","price_id":"SIM"},{"family_id":"00","price_id":"BTC"}]'
    out = p._parse_json_with_repair(text, expected_n=2)
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0]["family_id"] == "00"


def test_ollama_parse_json_with_repair_handles_prefixed_text():
    p = OllamaProposer(model="dummy", base_url="http://localhost:11434")
    text = 'some text [{"family_id":"00","price_id":"SIM"},{"family_id":"00","price_id":"BTC"}]'
    out = p._parse_json_with_repair(text, expected_n=2)
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0]["family_id"] == "00"


def test_ollama_generated_tasks_can_be_added_to_task_pool(tmp_path: Path, monkeypatch):
    p = OllamaProposer(model="dummy", base_url="http://localhost:11434")
    fake_json = (
        '[{"family_id":"00","price_id":"SIM","horizon":128,"reward_id":"pnl_linear",'
        '"reward_params":{"pnl_w":1.0},"seed":123,"obs_dim":182,"action_dim":3,"start_idx":null}]'
    )
    monkeypatch.setattr(p, "_generate_text", lambda prompt: fake_json)
    tasks = p.propose_tasks("market", 1)
    assert len(tasks) == 1

    env_cfg = {"window_size": 30, "n_features": 5, "t_embed_dim": 32, "action_dim": 3}
    validated = validate_and_clamp_task(tasks[0], env_cfg, fallback_seed=1)
    TaskSpec.from_dict(validated)  # schema check

    pool = TaskPool(tmp_path / "pool.jsonl", read_only=False)
    uid = pool.add(validated, meta={"backend": "ollama"})
    assert isinstance(uid, str)
    assert pool.size() == 1
