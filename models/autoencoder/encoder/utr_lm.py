import os
encoder_dir = os.path.dirname(os.path.relpath(__file__))
import sys
sys.path.append(encoder_dir)
from util import get_obj_from_str, load_model_ckpt
from esm import Alphabet


def build_utr_lm(esm2_modelfile, class_name="esm.model.esm2", load_weights=True):

    esm2_modelfile = os.path.join(encoder_dir, esm2_modelfile)
    esm2_model_info = os.path.splitext(esm2_modelfile)[0].split('/')[-1].split('_')
    for info in esm2_model_info:
        if 'layers' in info: 
            layers = int(info[:-6])
        elif 'heads' in info:
            heads = int(info[:-5])
        elif 'embedsize' in info:
            embed_dim = int(info[:-9])
        # elif 'batchToks' in info:
        #     batch_toks = int(info[:-9])
    alphabet = Alphabet(standard_toks = 'AGCU', mask_prob = 0)

    class_name = encoder_dir.replace('/', '.') + "." + class_name
    ESM2_model = get_obj_from_str(class_name)(num_layers = layers,
                    embed_dim = embed_dim,
                    attention_heads = heads,
                    alphabet = alphabet) # .to(device)
    # ESM2_model.load_state_dict(torch.load(esm2_modelfile, map_location="cpu"), strict = True)
    if load_weights:
        load_model_ckpt(ESM2_model, esm2_modelfile, map_location="cpu")
    else:
        print("Not loading pretrained model weights")
    return ESM2_model



