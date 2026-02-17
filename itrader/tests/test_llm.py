import pytest


def _proposer_cls():
    pytest.importorskip("transformers")
    from itrader.llm.proposer_llm import ProposerLLM
    return ProposerLLM


def test_parse_json_with_repair_direct_array():
    ProposerLLM = _proposer_cls()
    p = ProposerLLM.__new__(ProposerLLM)
    txt = '[{"family_id":"00","price_id":"SIM"},{"family_id":"00","price_id":"BTCUSDT"}]'
    out = p._parse_json_with_repair(txt, expected_n=2)
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0]["family_id"] == "00"


def test_parse_json_with_repair_embedded_object():
    ProposerLLM = _proposer_cls()
    p = ProposerLLM.__new__(ProposerLLM)
    txt = "some preface {\"family_id\":\"00\",\"price_id\":\"SIM\"} trailing text"
    out = p._parse_json_with_repair(txt, expected_n=1)
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0]["price_id"] == "SIM"
