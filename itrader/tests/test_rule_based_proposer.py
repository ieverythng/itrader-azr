import json
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
        assert task["family_id"] == "00"
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


def test_rule_based_proposer_preserves_registered_reward():
    proposer = RuleBasedProposer(seed=1, reward_id="pnl_sharpe")
    tasks = proposer.propose_tasks("summary", 2)
    assert len(tasks) == 2
    for task in tasks:
        assert task["reward_id"] == "pnl_sharpe"


def test_factory_builds_ollama_proposer():
    proposer = build_proposer("ollama", model="qwen2.5:7b-instruct")
    assert isinstance(proposer, OllamaProposer)


def test_factory_builds_vllm_proposer():
    proposer = build_proposer("vllm", model="Qwen/Qwen3.5-9B", base_url="http://localhost:8000/v1")
    assert isinstance(proposer, OpenAICompatProposer)


def test_factory_builds_llama_cpp_proposer():
    proposer = build_proposer("llama_cpp", model="qwen36-27b-q3km", base_url="http://localhost:8080/v1")
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


def test_ollama_generate_text_sends_think_false(monkeypatch):
    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"response": "[]"}).encode("utf-8")

    def _fake_urlopen(req, timeout):
        captured["timeout"] = timeout
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _Resp()

    monkeypatch.setattr("itrader.proposers.ollama.request.urlopen", _fake_urlopen)

    p = OllamaProposer(model="dummy", base_url="http://localhost:11434", timeout_s=12.0)
    out = p._generate_text("prompt")
    assert out == "[]"
    assert captured["timeout"] == 12.0
    assert captured["payload"]["think"] is False


def test_ollama_generate_text_raises_on_empty_response_with_thinking(monkeypatch):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "response": "",
                    "thinking": "reasoning without final answer",
                    "done_reason": "length",
                }
            ).encode("utf-8")

    monkeypatch.setattr("itrader.proposers.ollama.request.urlopen", lambda req, timeout: _Resp())

    p = OllamaProposer(model="dummy", base_url="http://localhost:11434")
    try:
        p._generate_text("prompt")
        assert False, "Expected RuntimeError for empty response with thinking payload"
    except RuntimeError as exc:
        msg = str(exc)
        assert "empty response text" in msg
        assert "done_reason=length" in msg
