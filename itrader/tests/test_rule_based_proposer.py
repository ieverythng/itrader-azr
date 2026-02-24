from itrader.proposers.factory import build_proposer
from itrader.proposers.rule_based import RuleBasedProposer


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


def test_factory_builds_rule_based_proposer():
    proposer = build_proposer("rule_based", seed=1)
    assert isinstance(proposer, RuleBasedProposer)
