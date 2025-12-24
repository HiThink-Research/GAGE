from gage_eval.assets.datasets.validation import SampleValidator, DefaultEnvelopeModel, ValidationMode


def test_validator_rejects_wrong_inputs_type():
    validator = SampleValidator(envelope_model=DefaultEnvelopeModel, mode=ValidationMode.WARN)
    sample = {"id": "s1", "messages": [], "choices": [], "inputs": []}
    result = validator.validate_envelope(sample, dataset_id="ds", sample_id="s1", trace=None)
    assert result is None  # in WARN mode, None means the sample is skipped


def test_validator_passes_valid_envelope():
    validator = SampleValidator(envelope_model=DefaultEnvelopeModel, mode=ValidationMode.STRICT)
    sample = {"id": "s1", "messages": [], "choices": [], "inputs": {}, "metadata": {}, "data_tag": {}}
    result = validator.validate_envelope(sample, dataset_id="ds", sample_id="s1", trace=None)
    assert result["id"] == "s1"
