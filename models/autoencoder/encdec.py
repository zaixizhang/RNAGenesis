import fm.modules
from transformers import PreTrainedModel, AutoTokenizer, EsmForMaskedLM, AutoTokenizer, AutoModel
import fm
import torch 
import torch.nn as nn
import time
from .decoder.progen.progen2.models.progen.modeling_progen import ProGenForCausalLM
from .decoder.progen.progen2.models.progen.configuration_progen import ProGenConfig
from tokenizers import Tokenizer
from .configuration_qt import QtConfig
from .configuration_encdec import EncDecConfig
from .modeling_qt import QtModel
from util import ProGenPath, instantiate_from_config
from .encoder.esm import Alphabet
import os
from util import truncate_op

class DataCollatorEncDec:
    def __init__(self, num_query_tokens, data_type="protein", rna_encoder_type=None, rna_tokenizer_config="tokenizer_rna"):
        self.data_type = data_type

        if self.data_type == "protein":

            self.enc_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
            self.enc_pad_token = self.enc_tokenizer.pad_token_id
            self.dec_tokenizer = Tokenizer.from_str(open(f'{ProGenPath}/tokenizer.json').read())

        elif self.data_type == "rna":
            assert rna_encoder_type is not None
            assert os.path.exists(f'{ProGenPath}/{rna_tokenizer_config}.json')
            if rna_encoder_type == "rna-fm":
                _, alphabet = fm.pretrained.rna_fm_t12()
                self.enc_pad_token = alphabet.padding_idx
            elif rna_encoder_type == "utr-lm":
                # change from "AGCT" to "AGCU"
                alphabet = Alphabet(standard_toks = 'AGCU', mask_prob = 0)
                assert alphabet.tok_to_idx == {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'U': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}
                self.enc_pad_token = alphabet.padding_idx
            elif rna_encoder_type == "biomap":
                self.enc_pad_token = 0

            self.dec_tokenizer = Tokenizer.from_str(open(f'{ProGenPath}/{rna_tokenizer_config}.json').read())

        else:
            print(f"{self.data_type} is not supported!")
            raise ValueError        
        self.dec_tokenizer.enable_padding()
        self.dec_prefix = '1' # TODO: Progen2 compatible, check protein seq direction
        self.dec_suffix = '2' # TODO
        self.num_query_tokens = num_query_tokens

    def __call__(self, instances):
        # merge and pad seq.

        bs = len(instances)

        #print(instances)
        input_ids = [torch.tensor(ins['input_ids'], dtype=torch.long) for ins in instances]
        decoder_input_ids = [torch.tensor(ins['decoder_input_ids'], dtype=torch.long) for ins in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value = self.enc_pad_token)
        attention_mask = input_ids.ne(self.enc_pad_token)

        decoder_input_ids = torch.nn.utils.rnn.pad_sequence(decoder_input_ids, batch_first=True, padding_value = self.dec_tokenizer.padding['pad_id'])
        decoder_attention_mask = decoder_input_ids.ne(self.dec_tokenizer.padding['pad_id'])

        labels = torch.where(decoder_attention_mask.bool(), decoder_input_ids, -100)
        query_mask = torch.ones(bs, self.num_query_tokens, device = labels.device, dtype=torch.int)
        labels = torch.cat([query_mask * -100, labels], dim = 1)
        return dict(
                input_ids = input_ids,
                attention_mask = attention_mask,
                decoder_input_ids = decoder_input_ids,
                decoder_attention_mask = decoder_attention_mask,
                labels = labels
                )

class EncDec(PreTrainedModel):
    config_class = EncDecConfig

    def __init__(self, config, decoder_config=None):
        super(EncDec, self).__init__(config)
        self.data_type = config.data_type
        self.rna_encoder_type = None
        self.rna_tokenizer_config = None
        self.onehot_layernorm = config.onehot_layernorm
        if decoder_config is not None: # Support another Decoder Config which are not used during training
            config.decoder_config = decoder_config
        if self.data_type == "protein":
            self.encoder = EsmForMaskedLM.from_pretrained(config.encoder_config)
            self.decoder = ProGenForCausalLM.from_pretrained(config.decoder_config)
            self.dec_tokenizer = Tokenizer.from_str(open(f'{ProGenPath}/tokenizer.json').read())
            qt_from_hidden_size = self.encoder.config.hidden_size
        elif self.data_type == "rna":
            self.rna_encoder_type = config.rna_encoder_type
            self.rna_tokenizer_config = config.rna_tokenizer_config
            if self.rna_encoder_type == "rna-fm":
                # [YYK] 0801 update the codes
                self.encoder, Alphabet = fm.pretrained.rna_fm_t12()
                if config.rm_encoder: # 0802 Update the codes [WARNING] Currently, this is only for RNA-FM
                    print("Remove the RNA-FM model and instead use one-hot encoding")
                    qt_from_hidden_size = len(Alphabet.all_toks) if config.onehot_dim is None else config.onehot_dim
                    if self.onehot_layernorm:
                        self.emb_layer_norm_before = nn.LayerNorm(qt_from_hidden_size)
                        # self.emb_layer_norm_before = fm.modules.ESM1bLayerNorm(qt_from_hidden_size)
                    self.encoder = None
                else:
                    qt_from_hidden_size = self.encoder.args.embed_dim
                    self.num_layers = self.encoder.args.layers
            elif self.rna_encoder_type == "utr-lm":
                # self.encoder = instantiate_from_config(OmegaConf.create(config.encoder_config))
                self.encoder = instantiate_from_config(config.encoder_config)
                qt_from_hidden_size = self.encoder.embed_dim
                self.num_layers = self.encoder.num_layers
            elif self.rna_encoder_type == "biomap":
                qt_from_hidden_size = 1280
                self.num_layers = 32
            else:
                print(f"{self.rna_encoder_type} is not a supported rna encoder type!")
                raise ValueError
            ###### fix the bug ########
            if "rna" not in os.path.basename(config.decoder_config):
                self.decoder = ProGenForCausalLM(config=ProGenConfig.from_json_file(f"{ProGenPath}/config-progen2-small-rna.json"))
            else:
                self.decoder = ProGenForCausalLM(config=ProGenConfig.from_json_file(f"{ProGenPath}/{config.decoder_config}.json"))
            print(f"{ProGenPath}/{config.decoder_config}.json")  # models/autoencoder/decoder/progen_configs/config-progen2-small-rna_tokenizer_v2.json
            self.dec_tokenizer = Tokenizer.from_str(open(f'{ProGenPath}/{self.rna_tokenizer_config}.json').read())
            ### decoder setting should match the setting of the decoder tokenizer
        else:
            print(f"{self.data_type} is not supported!")
            raise ValueError
        self.dec_tokenizer.enable_padding() #TODO: check if this is necessary
        if config.rm_qt:
            self.qt = None
            enc_qt_out_dim = qt_from_hidden_size
            print("Remove the Query Transformer Model and instead average all the token embeddings")
        else:
            default_qt_config = QtConfig(
                qt_from_hidden_size = qt_from_hidden_size, 
                num_query_tokens = config.num_query_tokens, 
                hidden_size = config.hidden_size, 
                num_attention_heads = config.num_attn_heads
                )
            self.qt = QtModel(default_qt_config)
            enc_qt_out_dim = config.hidden_size
        self.transform = nn.Linear(enc_qt_out_dim, self.decoder.config.hidden_size)
        ### [0602] Remove the linear layer if the hidden size of the encoder and the decoder are the same
        if config.hidden_size == self.decoder.config.hidden_size:
            nn.init.eye_(self.transform.weight)
            nn.init.zeros_(self.transform.bias)
            self.transform.weight.requires_grad = False
            self.transform.bias.requires_grad = False
        self.config = config
        self.qt_from_hidden_size = qt_from_hidden_size

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels):

        emb = self.get_latent(input_ids, attention_mask)
        emb = self.transform(emb)

        batch_size, num_queries, _ = emb.size()
        input_embs = torch.cat([emb, self.decoder.transformer.wte(decoder_input_ids)], dim = 1)
        query_attention_mask = torch.ones((batch_size, num_queries), dtype=torch.long, device=decoder_attention_mask.device)
        decoder_attention_mask = torch.cat([query_attention_mask, decoder_attention_mask ], dim=1)

        decoder_output = self.decoder(inputs_embeds = input_embs, labels = labels, attention_mask=decoder_attention_mask)

        return decoder_output

    
    def onehot(self, input_ids):
        # emb = torch.zeros(list(input_ids.shape) +[self.qt_from_hidden_size], device=input_ids.device)
        # emb.scatter_(dim=-1, index=input_ids.unsqueeze(-1), value=1)
        ## or
        emb = torch.nn.functional.one_hot(input_ids, num_classes=self.qt_from_hidden_size).float()
        return emb 

    def generate(
            self, latent_gen, 
            max_seq_len=150, do_sample=False, top_k=50, top_p=1, 
            eos_token="<|eos|>", bos_token="<|bos|>", truncate=False, extend_input=False, multiple_eos_tokens=False
            ):
        # decode latent to sequence
        pred_original_sample_1 = self.transform(latent_gen)
        batch_size, num_queries, _ = pred_original_sample_1.size()

        # pad_token_id, bos_token_id, eos_token_id = self.dec_tokenizer.encode("".join(["<|pad|>", "<|bos|>",  eos_token])).ids
        
        pad_token_id = self.dec_tokenizer.encode("<|pad|><pad>").ids
        if pad_token_id == []:
            pad_token_id = None
        else:
            pad_token_id = pad_token_id[0]
        
        try:
            bos_token_id = self.dec_tokenizer.encode(bos_token).ids
        except:
            print(bos_token, "not in the tokenizer")
            bos_token_id = self.dec_tokenizer.encode("<|bos|><bos><cls>").ids
        assert len(bos_token_id) >0
        bos_token_id = bos_token_id[0]
        
        if not multiple_eos_tokens:
            if "eos" in eos_token:
                eos_token_id = self.dec_tokenizer.encode("<|eos|><eos>").ids
            else:
                eos_token_id = self.dec_tokenizer.encode(eos_token).ids
            if eos_token_id == []:
                eos_token_id = None
            else:
                eos_token_id = eos_token_id[0]
        else:
            eos_token_id = self.dec_tokenizer.encode("<|eos|><eos><|bos|><bos><cls><|pad|><pad>12").ids
        # pad_token_id, bos_token_id, eos_token_id = None, None, None
        # pad_token_id, eos_token_id = None, None
        # print("eos_token_id", eos_token_id, "bos_token_id", bos_token_id, "pad_token_id", pad_token_id)
        if extend_input:
            assert bos_token == "1"
            with torch.no_grad():
                bos_embeds = self.decoder.transformer.wte(torch.LongTensor([bos_token_id] * batch_size).to(device=pred_original_sample_1.device)).unsqueeze(1)
            pred_original_sample_1 = torch.cat([pred_original_sample_1, bos_embeds], dim=1)

        start_time = time.time()
        dec_output = self.decoder.generate(
            inputs_embeds = pred_original_sample_1,
            do_sample=do_sample, # False: greedy decoding
            top_k=top_k, # top_k sampling
            top_p=top_p, # top_p sampling
            pad_token_id = pad_token_id, 
            eos_token_id = eos_token_id,
            bos_token_id= bos_token_id,  
            max_new_tokens = max_seq_len, #TODO: check if this is correct
            output_scores = True, # get the scores of the tokens
            return_dict_in_generate=True
            )
        # print("total decoder time: ", time.time() - start_time)
        tokens_batch = dec_output.sequences
        as_lists = lambda batch: [batch[i, ...].detach().cpu().numpy().tolist() for i in range(batch.shape[0])]
        seqs = self.dec_tokenizer.decode_batch(as_lists(tokens_batch))
        if truncate: # TODO: check if this is necessary
            seqs = [truncate_op(seq) for seq in seqs]
        logits = dec_output.scores # args.max_seq_len * (batch_size, vocab_size)
        return logits, seqs
    

class DataCollatorEncDec_Mix(DataCollatorEncDec):
    def __call__(self, instances):
        dict_ = super().__call__(instances)
        dict_["seqs"] = [ins["seqs"] for ins in instances]
        return dict_
