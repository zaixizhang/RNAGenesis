import torch
import numpy, random
import importlib
ProGenPath = "models/autoencoder/decoder/progen_configs"
DATA_CACHE_DIR = ".cache/data" ## change this to your cache directory
XDG_CACHE_HOME = ".cache" ## change this to your cache directory

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def load_model_ckpt(model, ckpt, verbose=True, map_location="cpu"):
    # only load the ckpt into the built model
    # map_location = "cpu" if not torch.cuda.is_available() else "cuda"
    print("checkpoint map location:", map_location)
    sd = torch.load(ckpt, map_location=map_location)
    keys_ = list(sd.keys())[:]
    for k in keys_:
        if k.startswith("module."): # remove prefix module. if it is present
            nk = k[7:]
            sd[nk] = sd[k]
            del sd[k]

    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys: {}".format(len(m)))
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys: {}".format(len(u)))
        print(u)
    return model

def truncate_op(sample, terminals = ["1", "2"]):
    pos = []
    for terminal in terminals:
        find_pos = sample.find(terminal, 1)
        if find_pos != -1:
            pos.append(find_pos)
    if len(pos) > 0:
        return sample[:(min(pos)+1)]
    else:
        return sample

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True #TODO: check if this is necessary
    torch.backends.cudnn.benchmark = False
    numpy.random.seed(seed)
    random.seed(seed)   

def compute_metrics(eval_pred):
    logits = eval_pred.predictions[0]
    predictions = numpy.argmax(logits, axis=-1)
    labels = eval_pred.label_ids[:, 1:]
    predictions = predictions[:, : -1]
    masks = (labels != -100)
    correctness = (predictions == labels) & masks
    return {"accuracy": numpy.sum(correctness) / numpy.sum(masks)}

def filter_seqs(data, 
            max_len=-1, min_len=-1, 
            remove_bos_eos=False, filter_vocab=False, 
            max_len_limit=None, filter_unstopped_generation=False, 
            eos_token = "2", do_bos_filter=False, bos_token="1",
            ):
    ### filter the generated sequences ###
    print(len(data), "vanilla samples")
    if max_len_limit is None: max_len_limit = 10**10
    if filter_unstopped_generation:
        data = [d for d in data if len(d) < max_len_limit and d.endswith(eos_token) and (not do_bos_filter or d.startswith(bos_token))] #TODO: check if this is correct
        print(f"Removing the sequences with length reaching the max len limit for generation or the sequences which don't end with eos token {eos_token} or start with bos token {bos_token}")
        print(len(data))
    if remove_bos_eos:
        n_data = []
        for d in data:
            seq_len = len(d)
            d_ = d.strip("12")
            try:
                assert len(d_) >= seq_len - 2
            except:
                print(len(d_), seq_len, d)
            d = d_
            n_data.append(d)
        data = n_data
    if filter_vocab: # filter the tokens which are not one of {A, C, G, T}
        for i in range(len(data)-1, -1, -1):
            d = data[i]
            if numpy.sum([1 for item in set(d) if item not in "ACGU"]) > 0: # "1" may in the mid of sequences
                data.remove(d)
        print(f"filter the tokens which are not one of [A, C, G, U]: {len(data)}")
    if max_len == -1: max_len = 10**10
    if min_len == -1: min_len = 1 # at least 1
    data = [d for d in data if len(d) < max_len and len(d) >= min_len] # <= max_len --> < max_len
    print(f"the sequences with length range {min_len} to {max_len}: {len(data)}")
    print("Filtering Process ends and we have", len(data), "samples")

    return data

def load_data_from_files(real_data_path, gen_data_path, random_data_path):
    datasets = []
    for path in [real_data_path, gen_data_path, random_data_path]:
        data = load_data_from_file(path)
        datasets.append(data)
    return datasets 

def load_data_from_file(data_path, replace=True):
    with open(data_path, "r") as f:
        data = f.readlines()
        if replace:
            data = [(d.strip()).replace("U", "T") for d in data]
            assert set("".join(data)).issubset(set("ACGT"))
        else:
            # data = [d.strip() for d in data]
            data = [(d.strip()).replace("T", "U") for d in data] # replace T with U
            assert set("".join(data)).issubset(set("ACGU"))
    return data

def count_params(model, verbose=True, print_modules=False):
    print("------------------------\nCounting the number of parameters in the model")
    if print_modules:
        total_params = 0
        for name, module in model.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            total_params += module_params
            if verbose:
                print(f"{name} ({module.__class__.__name__}) has {module_params*1.e-6:.2f} M params.")
    else:
        total_params = sum(p.numel() for p in model.parameters())
        
    if verbose:
        print("------\nSummary:")
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    
    return total_params
