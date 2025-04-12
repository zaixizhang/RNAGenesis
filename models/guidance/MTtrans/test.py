import copy
from models.popen import Auto_popen
from models import reader, train_val
import argparse
import torch
from glob import glob
import os
os.chdir(os.path.dirname(__file__))
parser = argparse.ArgumentParser('the main to train model')
parser.add_argument('--config_file',type=str,required=True)
parser.add_argument('--ckpt_path',type=str,default=None)
parser.add_argument('--ckpt_folder',type=str,default=None)
# parser.add_argument('--cuda',type=str,default=0,required=False)
parser.add_argument("--kfold_index",type=int,default=1)
parser.add_argument("--pad_to",type=int,default=0)

args = parser.parse_args()

def test(args):
    POPEN = Auto_popen(args.config_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    POPEN.cuda_id = device
    POPEN.pad_to = args.pad_to

    # read data                                                                                                                                                                                                                                                                                                              
    base_path = ['cycle_train_val.csv', 'cycle_test.csv']
    base_csv = copy.deepcopy(POPEN.csv_path) #'cycle_MTL_transfer.csv'

    pretrain_model = torch.load(args.ckpt_path, map_location=torch.device('cpu'))['state_dict']
    model = pretrain_model.to(device)
    print(model.all_tasks)
    print(POPEN.cycle_set)
    # POPEN.cycle_set = model.all_tasks
    POPEN.cycle_set = [t for t in POPEN.cycle_set if t in model.all_tasks]
    print("Updated:", POPEN.cycle_set)
    loader_set = {}
    for task in POPEN.cycle_set:
        if (task in ['MPA_U', 'MPA_H', 'MPA_V', 'SubMPA_H']):
            datapopen = Auto_popen('log/Backbone/RL_hard_share/3M/schedule_lr.ini')
            datapopen.split_like = [path.replace('cycle', task) for path in base_path]
            datapopen.kfold_index = args.kfold_index
            datapopen.other_input_columns = POPEN.other_input_columns
            datapopen.n_covar = POPEN.n_covar

        elif (task in ['RP_293T', 'RP_muscle', 'RP_PC3']):
            # base_csv = 'cycle_protein_coding.csv'
            datapopen = Auto_popen('log/Backbone/RL_hard_share/3R/schedule_MTL.ini')
            datapopen.csv_path = base_csv.replace("cycle",task)
            datapopen.kfold_index = args.kfold_index
            datapopen.kfold_cv = POPEN.kfold_cv
            datapopen.pad_to = POPEN.pad_to
            datapopen.aux_task_columns = POPEN.aux_task_columns
            datapopen.other_input_columns = POPEN.other_input_columns
            datapopen.n_covar = POPEN.n_covar

        elif (task in ['pcr3', '293']):
            datapopen = Auto_popen('log/Backbone/RL_hard_share/karollus_RPs/rp_cycle.ini')
            datapopen.csv_path = base_csv.replace("cycle",task)
            datapopen.kfold_index = args.kfold_index
            datapopen.kfold_cv = POPEN.kfold_cv
            datapopen.aux_task_columns = POPEN.aux_task_columns
            datapopen.other_input_columns = POPEN.other_input_columns
            datapopen.pad_to = POPEN.pad_to
            datapopen.n_covar = POPEN.n_covar
        

        loader_set[task] = reader.get_dataloader(datapopen)
    if len(loader_set) == 0:
        print("No task to test!")
        return None
    test_dict = train_val.cycle_validate(loader_set,model,None,popen=POPEN,epoch=None,which_set=0, ckpt_path=os.path.splitext(os.path.basename(args.ckpt_path))[0])
    print("Train: ")
    print(test_dict)
    test_dict = train_val.cycle_validate(loader_set,model,None,popen=POPEN,epoch=None,which_set=1, ckpt_path=os.path.splitext(os.path.basename(args.ckpt_path))[0])
    print("Validation: ")
    print(test_dict)
    test_dict = train_val.cycle_validate(loader_set,model,None,popen=POPEN,epoch=None,which_set=2, ckpt_path=os.path.splitext(os.path.basename(args.ckpt_path))[0])
    print("Test: ")
    print(test_dict)

    return test_dict
assert args.ckpt_path is not None or args.ckpt_folder is not None
if args.ckpt_folder is not None:
    ckpt_paths = glob(args.ckpt_folder + '/*.pth')
    for ckpt_path in ckpt_paths:
        # if ckpt_path != "/scratch/gpfs/yy1325/codes/biodiff/models/guidance/MTtrans/checkpoint/V_muscle_RMSprop-model_best_cv1.pth":
        #     continue
        print("----------------------------")
        args.ckpt_path = ckpt_path
        print(ckpt_path)
        test_dict = test(args)
else:
    test_dict = test(args)
print("trial ends!")