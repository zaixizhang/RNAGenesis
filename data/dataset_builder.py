from datasets import load_dataset, concatenate_datasets, features
from transformers import AutoTokenizer #, EsmForMaskedLM,  AutoModelForCausalLM, Trainer, TrainingArguments
from tokenizers import Tokenizer
import torch
# import time 
import fm
from util import ProGenPath, DATA_CACHE_DIR
from models.autoencoder.encoder.esm import Alphabet
### DEPRECATED ###
def build_dataset(file='./uniref50_max1000.txt'):
    uniref50 = load_dataset("text", data_files = file, cache_dir=DATA_CACHE_DIR) #, cache_mode="no_cache")
    
    enc_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D", use_fast=True)
    dec_tokenizer = Tokenizer.from_str(open(f'{ProGenPath}/tokenizer.json').read())
    dec_tokenizer.enable_padding()
    dec_prefix = '1' # TODO: Progen2 compatible, check protein seq direction
    dec_suffix = '2' # TODO

    def preprocess(x):
        """
            x['text']: protein string.
        """

        x = x['text']
        # encoder
        # current = time.time()

        enc_inputs = enc_tokenizer(x, return_tensors='pt', padding=False)
        input_ids = enc_inputs['input_ids'][0]

        # enc_t = time.time()-current
        # print("encod token: {}".format(enc_t))
        
        # decoder
        x = dec_prefix + x + dec_suffix 
        decoder_input_ids = torch.LongTensor(dec_tokenizer.encode(x).ids)
        # print("decod token: {}".format(time.time()-current-enc_t))
        return dict(
                input_ids = input_ids,
                decoder_input_ids = decoder_input_ids,
                )


    uniref50_preprocessed = uniref50.map(preprocess, remove_columns=['text'], num_proc=40)

    def preprocess_2(x_all):
        """
            x['text']: protein string.
        """

        x_all = x_all['text']
        input_ids_list = []
        decoder_input_ids_list = []

        enc_inputs_all = enc_tokenizer(x_all, return_tensors='pt', padding=True)
        input_ids_list = enc_inputs_all["input_ids"]
        input_ids_list = list(torch.split(input_ids_list, split_size_or_sections=1, dim=0))

        x_all = [dec_prefix + x + dec_suffix for x in x_all]
        decoder_input_ids_list = [torch.LongTensor(emb.ids) for emb in dec_tokenizer.encode_batch(x_all)]

        # for x in x_all:
        #     # encoder
        #     enc_inputs = enc_tokenizer(x, return_tensors='pt', padding=False)
        #     input_ids = enc_inputs['input_ids']
        #     input_ids_list.append(input_ids)

        #     # decoder
        #     x = dec_prefix + x + dec_suffix 
        #     decoder_input_ids = torch.LongTensor(dec_tokenizer.encode(x).ids)
        #     decoder_input_ids_list.append(decoder_input_ids)
        
        return dict(
                input_ids = input_ids_list,
                decoder_input_ids = decoder_input_ids_list,
                )


    # uniref50_preprocessed = uniref50.map(preprocess_2, batched=True, batch_size=1000, remove_columns=['text'], num_proc=8)
    return uniref50_preprocessed['train'].train_test_split(test_size=0.1, shuffle=True, seed=42)


############################################
def build_dataset_rna(
        file='../../data/greengenes_filterNone.txt', 
        encoder_type="rna-fm", 
        tokenizer_config="tokenizer_rna",
        split=True
        ):
    rnacentral = load_dataset("text", data_files = file, cache_dir=DATA_CACHE_DIR) #, cache_mode="no_cache")
    
    # enc_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D", use_fast=True)
    if encoder_type == "rna-fm":
        _, alphabet = fm.pretrained.rna_fm_t12()
        # batch_converter = alphabet.get_batch_converter()
    elif encoder_type == "utr-lm":
        # change from "AGCT" to "AGCU"
        alphabet = Alphabet(standard_toks = 'AGCU', mask_prob = 0)
        assert alphabet.tok_to_idx == {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'U': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}
    batch_converter = alphabet.get_batch_converter()

    dec_tokenizer = Tokenizer.from_str(open(f'{ProGenPath}/{tokenizer_config}.json').read())
    dec_tokenizer.enable_padding()
    dec_prefix = '1' # TODO: Progen2 compatible, check protein seq direction
    dec_suffix = '2' # TODO
    # if tokenizer_config == "tokenizer_rna":
    #     dec_prefix = '1' # TODO: Progen2 compatible, check protein seq direction
    #     dec_suffix = '2' # TODO
    # else:
    #     dec_prefix = dec_suffix = '' # Have to include "1" and "2" as prefix and suffix

    def preprocess(x):
        """
            x['text']: rna string.
        """

        x = x['text']
        # encoder
        # current = time.time()
        #### [UPDATE 0604] ####
        # assert "T" not in set(x)  # only U
        # or 
        x = x.replace('T','U')
        assert "T" not in set(x)  # only U
        _, _, input_ids = batch_converter([("rna1", x)])
        input_ids = input_ids[0]
        # input_ids = enc_inputs['input_ids'][0]

        # enc_t = time.time()-current
        # print("encod token: {}".format(enc_t))
        
        # decoder
        x = dec_prefix + x + dec_suffix 
        # if sum([1 for item in x if item not in "AGCU12"]) > 0:
        #     print(x)
        decoder_input_ids = torch.LongTensor(dec_tokenizer.encode(x).ids)
        # print("decod token: {}".format(time.time()-current-enc_t))
        return dict(
                input_ids = input_ids,
                decoder_input_ids = decoder_input_ids,
                )


    rnacentral_preprocessed = rnacentral.map(preprocess, remove_columns=['text'], num_proc=1) #40)
    # # [DEPERECATED]
    # def preprocess_2(x_all):
    #     """
    #         x['text']: protein string.
    #     """

    #     x_all = x_all['text']
    #     input_ids_list = []
    #     decoder_input_ids_list = []

    #     # enc_inputs_all = enc_tokenizer(x_all, return_tensors='pt', padding=True)
    #     # input_ids_list = enc_inputs_all["input_ids"]
    #     _, _, input_ids_list = batch_converter([(f"rna{i}", x) for i, x in enumerate(x_all)])
    #     input_ids_list = list(torch.split(input_ids_list, split_size_or_sections=1, dim=0))

    #     x_all = [dec_prefix + x + dec_suffix for x in x_all]
    #     decoder_input_ids_list = [torch.LongTensor(emb.ids) for emb in dec_tokenizer.encode_batch(x_all)]


    #     return dict(
    #             input_ids = input_ids_list,
    #             decoder_input_ids = decoder_input_ids_list,
    #             )


    # rnacentral_preprocessed = rnacentral.map(preprocess_2, batched=True, batch_size=1000, remove_columns=['text'], num_proc=8)
    if split:
        return rnacentral_preprocessed['train'].train_test_split(test_size=0.1, shuffle=True, seed=42)
    else:
        return rnacentral_preprocessed['train']

### Adapted from Kaidi ###
### Build dataset for reward training ###
def build_reward_dataset(
        filelist=['RP_293T_MTL_transfer.csv'], encoder_type="rna-fm",
        tokenizer_config="tokenizer_rna", split=True, 
        remove_pad = False, reward_name="te_log" # or "log_te" or "rl_log2"
        ):
    if encoder_type == "rna-fm":
        _, alphabet = fm.pretrained.rna_fm_t12()
        # batch_converter = alphabet.get_batch_converter()
    elif encoder_type == "utr-lm":
        # change from "AGCT" to "AGCU"
        alphabet = Alphabet(standard_toks = 'AGCU', mask_prob = 0)
        assert alphabet.tok_to_idx == {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'U': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}
    batch_converter = alphabet.get_batch_converter()

    dec_tokenizer = Tokenizer.from_str(open(f'{ProGenPath}/{tokenizer_config}.json').read())
    dec_tokenizer.enable_padding()
    dec_prefix = '1' # TODO: Progen2 compatible, check protein seq direction
    dec_suffix = '2' # TODO
    # dec_tokenizer = Tokenizer.from_str(open(f'{ProGenPath}/tokenizer_rna.json').read())
    # dec_tokenizer.enable_padding()
    # dec_prefix = '1' # TODO: Progen2 compatible, check protein seq direction
    # dec_suffix = '2' # TODO
    def preprocess(x):
        """
            1. Replace 'T' with 'U' in the RNA sequence.
        """
        y=x[reward_name]
        x=x['utr']
        if remove_pad: # IMPORTANT: remove padding token
            x = x.rstrip('<pad>')
        x=x.replace('T','U')
        # encoder       
        _, _, input_ids = batch_converter([("rna1", x)]) # bos_token + seq + eos_token
        input_ids = input_ids[0]
        #print(len(input_ids))

        # decoder
        x = dec_prefix + x + dec_suffix 
        decoder_input_ids = torch.LongTensor(dec_tokenizer.encode(x).ids) # dec_prefix + seq + dec_suffix
        assert len(input_ids) == len(decoder_input_ids)
        return dict(
                input_ids = input_ids,
                log_te=y,
                decoder_input_ids = decoder_input_ids,
                )   
    
     
    for i, file in enumerate(filelist):
        RP_TE = load_dataset("csv", data_files = file, cache_dir=DATA_CACHE_DIR)
        #RP_TE = RP_TE.filter(lambda x: x['te'] <= 10)

        # List of all columns that you need to remove
        columns_to_remove =RP_TE.column_names['train'].remove(reward_name)
        

        # Code to preprocess and retain only 'log_te', 'input_ids', and 'decoder_input_ids'
        if i==0:
            RP_TE_preprocessed = RP_TE.map(preprocess, remove_columns=columns_to_remove, num_proc=1)['train']
        else:
            RP_TE_preprocessed = concatenate_datasets([RP_TE_preprocessed, RP_TE.map(preprocess, remove_columns=columns_to_remove, num_proc=1)['train']])
    if split:
        return RP_TE_preprocessed.train_test_split(test_size=0.1, shuffle=True, seed=42)
    return RP_TE_preprocessed  


def build_reward_dataset_naive(
        filelist=['RP_293T_MTL_transfer.csv'], 
        split=True, 
        remove_pad = False, reward_name="te_log" # or "log_te" or "rl_log2"
        ):
    def preprocess(x):
        """
            1. Replace 'T' with 'U' in the RNA sequence.
        """
        y=x[reward_name]
        x=x['utr']
        if remove_pad: # IMPORTANT: remove padding token
            x = x.rstrip('<pad>')
        x=x.replace('T','U')
        return dict(
            seqs = x,
            log_te=y,
        )
     
    for i, file in enumerate(filelist):
        RP_TE = load_dataset("csv", data_files = file, cache_dir=DATA_CACHE_DIR)
        #RP_TE = RP_TE.filter(lambda x: x['te'] <= 10)

        # List of all columns that you need to remove
        columns_to_remove =RP_TE.column_names['train'].remove(reward_name)
        

        # Code to preprocess and retain only 'log_te', 'input_ids', and 'decoder_input_ids'
        if i==0:
            RP_TE_preprocessed = RP_TE.map(preprocess, remove_columns=columns_to_remove, num_proc=1)['train']
        else:
            RP_TE_preprocessed = concatenate_datasets([RP_TE_preprocessed, RP_TE.map(preprocess, remove_columns=columns_to_remove, num_proc=1)['train']])
    if split:
        return RP_TE_preprocessed.train_test_split(test_size=0.1, shuffle=True, seed=42)
    return RP_TE_preprocessed  


def build_dataset_rna_naive(
        file='../../data/greengenes_filterNone.txt', 
        split=True
        ):
    rnacentral = load_dataset("text", data_files = file, cache_dir=DATA_CACHE_DIR) #, cache_mode="no_cache")

    def preprocess(x):
        """
            x['text']: rna string.
        """

        x = x['text']
        # assert "T" not in set(x)  # only U
        # or 
        x = x.replace('T','U') # replace T with U [UPDATE 0611]
        assert "T" not in set(x)  # only U
        return dict(
                seqs = x,
                )
    rnacentral_preprocessed = rnacentral.map(preprocess, remove_columns=['text'], num_proc=1) #40)
    if split:
        return rnacentral_preprocessed['train'].train_test_split(test_size=0.1, shuffle=True, seed=42)
    else:
        return rnacentral_preprocessed['train']


def build_dataset_rna_mix(
        file='../../data/greengenes_filterNone.txt', 
        encoder_type="rna-fm", 
        tokenizer_config="tokenizer_rna",
        split=True
        ):
    rnacentral = load_dataset("text", data_files = file, cache_dir=DATA_CACHE_DIR) #, cache_mode="no_cache")
    
    if encoder_type == "rna-fm":
        _, alphabet = fm.pretrained.rna_fm_t12()
    elif encoder_type == "utr-lm":
        # change from "AGCT" to "AGCU"
        alphabet = Alphabet(standard_toks = 'AGCU', mask_prob = 0)
        assert alphabet.tok_to_idx == {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'U': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}
    batch_converter = alphabet.get_batch_converter()

    dec_tokenizer = Tokenizer.from_str(open(f'{ProGenPath}/{tokenizer_config}.json').read())
    dec_tokenizer.enable_padding()
    dec_prefix = '1' # TODO: Progen2 compatible, check protein seq direction
    dec_suffix = '2' # TODO
    def preprocess(x):
        """
            x['text']: rna string.
        """

        x = x['text']
        ## store original sequence
        # assert "T" not in set(x)  # only U
        # or 
        x = x.replace('T','U') # replace T with U [UPDATE 0611]
        assert "T" not in set(x)  # only U
        x_original = x
        _, _, input_ids = batch_converter([("rna1", x)])
        input_ids = input_ids[0]
        
        # decoder
        x = dec_prefix + x + dec_suffix 
        decoder_input_ids = torch.LongTensor(dec_tokenizer.encode(x).ids)
        return dict(
                input_ids = input_ids,
                decoder_input_ids = decoder_input_ids,
                seqs = x_original ## store original sequence
                )


    rnacentral_preprocessed = rnacentral.map(preprocess, remove_columns=['text'], num_proc=1) #40)
    if split:
        return rnacentral_preprocessed['train'].train_test_split(test_size=0.1, shuffle=True, seed=42)
    else:
        return rnacentral_preprocessed['train']
    
##### classifier #####
def build_dataset_classifier(
        file='../../data/greengenes_filterNone.txt',
        encoder_type="rna-fm", 
        split=True
        ):
    ori_data = load_dataset("text", data_files = file, cache_dir=DATA_CACHE_DIR) #, cache_mode="no_cache")
    
    # enc_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D", use_fast=True)
    if encoder_type == "rna-fm":
        _, alphabet = fm.pretrained.rna_fm_t12()
        # batch_converter = alphabet.get_batch_converter()
    elif encoder_type == "utr-lm":
        # change from "AGCT" to "AGCU"
        alphabet = Alphabet(standard_toks = 'AGCU', mask_prob = 0)
        assert alphabet.tok_to_idx == {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'U': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}
    batch_converter = alphabet.get_batch_converter()
    # class_labels = []
    def preprocess(x):
        """
            x['text']: rna string.
        """

        x = x['text']
        x, class_label_x = x.strip().split('\t')
        # class_labels.append(class_label_x)
        x = x.replace('T','U')
        assert "T" not in set(x)  # only U
        _, _, input_ids = batch_converter([("rna1", x)])
        input_ids = input_ids[0]
    
        return dict(
                input_ids = input_ids,
                class_label_x = class_label_x,
                )

    ori_data_prep = ori_data.map(preprocess, remove_columns=['text'], num_proc=1)["train"]
    class_labels = ori_data_prep.unique("class_label_x")
    class_labels = sorted_cls(class_labels)
    feat_cls = features.ClassLabel(names=class_labels)
    ori_data_prep = ori_data_prep.cast_column("class_label_x", feat_cls)
    # ori_data_prep = ori_data_prep.class_encode_column("class_label_x")
    if split:
        return ori_data_prep.train_test_split(test_size=0.1, shuffle=True, seed=42, stratify_by_column="class_label_x"), class_labels ## 
    else:
        return ori_data_prep, class_labels

def sorted_cls(class_labels):
    class_labels_sorted = []
    if "others" in class_labels:
        class_labels_sorted.append("others")
        class_labels.remove("others")
    class_labels_sorted += sorted(class_labels)
    return class_labels_sorted

if __name__ == "__main__":
    import sys
    my_dataset = build_dataset(sys.argv[1])
    further_subsampled_test_dataset = my_dataset['test'].train_test_split(test_size=0.1, shuffle=True, seed=42)
    print(len(further_subsampled_test_dataset['test']))
    print(further_subsampled_test_dataset['test'][0])
