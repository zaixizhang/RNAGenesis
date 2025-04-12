# modified from https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/models/esm/configuration_esm.py


from omegaconf import OmegaConf

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)



class EncDecConfig(PretrainedConfig):
    r"""
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:

    ```"""
    model_type = "EncDec"

    def __init__(
        self,
        # protein
        encoder_config = "facebook/esm2_t6_8M_UR50D", # string or OmegaConf config
        # rna
        rna_encoder_type = "rna-fm", #"utr-lm", "rna-fm"
        decoder_config = "config-progen2-small-rna",
        num_query_tokens = 16,
        hidden_size = 320,
        num_attn_heads = 20,
        data_type = "protein",
        rna_tokenizer_config="tokenizer_rna",
        load_weights = True, 
        rm_encoder = False, 
        onehot_dim = None,
        onehot_layernorm = False,
        rm_qt = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_config = encoder_config if isinstance(encoder_config, str) or isinstance(encoder_config, dict) else OmegaConf.to_container(encoder_config) # string or dict
        self.decoder_config = decoder_config
        self.num_query_tokens = num_query_tokens
        self.hidden_size = hidden_size
        self.num_attn_heads = num_attn_heads
        self.data_type = data_type
        self.rna_encoder_type = rna_encoder_type
        self.rna_tokenizer_config = rna_tokenizer_config
        self.load_weights = load_weights 
        self.rm_encoder = rm_encoder 
        self.onehot_dim = onehot_dim
        self.rm_qt = rm_qt
        self.onehot_layernorm = onehot_layernorm

