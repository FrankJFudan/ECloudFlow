from ecloudflow.pipeline import ECloudFlowPipeline, GenerationMode, GenerationRequest


class _ModeGenerator:
    def __init__(self):
        self.modes = []

    def __call__(self, **kwargs):
        self.modes.append(kwargs["mode"])
        return "CCO"


def test_one_checkpoint_supports_all_generation_modes():
    generator = _ModeGenerator()
    pipeline = ECloudFlowPipeline(
        candidate_generator=generator,
        checkpoint_hash="same-checkpoint",
    )
    for mode in GenerationMode:
        request = GenerationRequest(
            pocket="toy-pocket.pdb",
            num_molecules=1,
            mode=mode,
            fragment="fragment.sdf" if mode is not GenerationMode.DE_NOVO else None,
        )
        result = pipeline.generate_request(request)
        assert result.model_checkpoint_hash == pipeline.checkpoint_hash
    assert generator.modes == list(GenerationMode)
