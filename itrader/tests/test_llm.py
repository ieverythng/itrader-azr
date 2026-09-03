from itrader.proposers._shared import parse_json_with_repair


def test_parse_json_with_repair_direct_array():
    txt = '[{"family_id":"00","price_id":"SIM"},{"family_id":"00","price_id":"BTCUSDT"}]'
    out = parse_json_with_repair(txt, expected_n=2)
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0]["family_id"] == "00"


def test_parse_json_with_repair_embedded_object():
    txt = "some preface {\"family_id\":\"00\",\"price_id\":\"SIM\"} trailing text"
    out = parse_json_with_repair(txt, expected_n=1)
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0]["price_id"] == "SIM"
