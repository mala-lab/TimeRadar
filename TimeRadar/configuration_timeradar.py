from transformers import PretrainedConfig


class TimeRadarConfig(PretrainedConfig):
    """Configuration for a pretrained TimeRadar anomaly detector."""

    model_type = "timeradar"

    # Store architecture and default inference parameters in Hugging Face format.
    def __init__(
        self,
        seq_len: int = 100,
        patch_len: int = 5,
        hidden_dim: int = 64,
        d_model: int = 256,
        depth: int = 10,
        mask_mode: str = "c",
        copies: int = 10,
        norm: bool = False,
        anomaly_score_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if seq_len <= 0 or patch_len <= 0:
            raise ValueError("seq_len and patch_len must be positive")
        if copies <= 0:
            raise ValueError("copies must be positive")
        if mask_mode == "c" and copies % 2:
            raise ValueError("copies must be even when mask_mode='c'")
        if not 0.0 <= anomaly_score_weight <= 1.0:
            raise ValueError("anomaly_score_weight must be between 0 and 1")

        self.seq_len = seq_len
        self.patch_len = patch_len
        self.hidden_dim = hidden_dim
        self.d_model = d_model
        self.depth = depth
        self.mask_mode = mask_mode
        self.copies = copies
        self.norm = norm
        self.anomaly_score_weight = anomaly_score_weight
