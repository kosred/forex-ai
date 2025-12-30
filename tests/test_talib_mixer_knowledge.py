import json

from forex_bot.features import talib_mixer as tm


def test_talib_mixer_save_load_knowledge(tmp_path, monkeypatch):
    monkeypatch.setattr(tm, "TALIB_AVAILABLE", False)
    mixer = tm.TALibStrategyMixer()

    payload = {
        "synergy_matrix": {"RSI_SMA": 1.23},
        "regime_performance": {"trend": {"RSI": 0.5}},
    }
    path = tmp_path / "knowledge.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    mixer.load_knowledge(path)
    assert mixer.indicator_synergy_matrix[("RSI", "SMA")] == 1.23
    assert mixer.regime_performance["trend"]["RSI"] == 0.5

    out_path = tmp_path / "out.json"
    mixer.save_knowledge(out_path)
    assert out_path.exists()
