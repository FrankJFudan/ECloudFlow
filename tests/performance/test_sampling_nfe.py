from ecloudflow.training.benchmark import measured_stub_nfe


def test_sampling_profiles_record_expected_nfe() -> None:
    assert measured_stub_nfe("fast") == 20
    assert measured_stub_nfe("balanced") == 82
    assert measured_stub_nfe("quality") == 208
