from itrader.rl_lib.tasks.families import (
    ACTIVE_FAMILY_IDS,
    PRIMARY_FAMILY_ID,
    PRIMARY_REWARD_ID,
    TASK_FAMILY_REGISTRY,
    enforce_active_family_id,
    enforce_active_reward_id,
)


def test_task_family_registry_contains_expected_roadmap_families():
    assert set(TASK_FAMILY_REGISTRY.keys()) == {"00", "01", "02", "03", "04"}
    assert PRIMARY_FAMILY_ID == "00"
    assert PRIMARY_REWARD_ID == "pnl_linear"


def test_task_family_active_set_is_only_00_for_now():
    assert ACTIVE_FAMILY_IDS == ("00",)
    assert enforce_active_family_id("03") == "00"


def test_reward_enforcement_is_pnl_linear_for_now():
    assert enforce_active_reward_id("pnl_sharpe", family_id="00") == "pnl_sharpe"
    assert enforce_active_reward_id("pnl_linear", family_id="00") == "pnl_linear"
    assert enforce_active_reward_id("not_registered", family_id="00") == "pnl_linear"
