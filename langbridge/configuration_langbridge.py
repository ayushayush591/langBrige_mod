from transformers.configuration_utils import PretrainedConfig


class LangBridgeConfig(PretrainedConfig):

    def __init__(
        self,
        enc: str = 'DKYoon/mt5-base-lm-adapt',
        lm: str = 'facebook/opt-125m',
        multi_lm: str = 'bigscience/bloomz-1b1',
        dim_enc: int = 768,
        dim_lm: int = 768,
        dim_multi_lm: int = 1536,
        freeze_language_model: bool = True,
        freeze_encoder: bool = True,
        freeze_multi_lm: bool = True,
        alignments: str = 'linear',
        alignments1: str = 'linear',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.lm = lm
        self.enc = enc
        self.multi_lm = multi_lm
        self.dim_enc = dim_enc
        self.dim_lm = dim_lm
        self.dim_multi_lm = dim_multi_lm
        self.freeze_language_model = freeze_language_model
        self.freeze_encoder = freeze_encoder
        self.alignments = alignments
        self.alignments1 = alignments1
