from forex_bot.core.system import HardwareProbe, HardwareProfile


def test_hardware_probe_detect(monkeypatch):
    fake_profile = HardwareProfile(
        cpu_cores=8,
        total_ram_gb=32.0,
        available_ram_gb=16.0,
        gpu_names=["FakeGPU"],
        num_gpus=1,
    )

    monkeypatch.setattr(HardwareProbe, "detect", lambda self: fake_profile)

    probe = HardwareProbe()
    profile = probe.detect()

    assert isinstance(profile, HardwareProfile)
    assert profile.cpu_cores == 8
    assert profile.total_ram_gb == 32.0
    assert profile.available_ram_gb == 16.0
    assert profile.gpu_names == ["FakeGPU"]
    assert profile.num_gpus == 1
