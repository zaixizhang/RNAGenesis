import os
import sys
import pickle
main_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(main_dir)
print('add %s to path'%main_dir)

colors=["#8C736F","#DDD0C8","#D4B8B4","#A08887","#ADAAA5","#E2C6C4","#B7B7BD","#B1ACB3","#AAB8AB"
"#53565C"]

def read_karollus_data(key=None):
    with open("/data/users/wergillius/UTR_VAE/Alex_framepool/data_dict.pkl", 'rb') as handle:
        data_dict = pickle.load(handle)
    
    if key is not None:
        data = data_dict[key]
    else:
        data = data_dict
    return data