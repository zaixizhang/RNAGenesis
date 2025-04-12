import os
import numpy as np
import pandas as pd
import PATH
import utils
from Bio import SeqIO
from scipy import sparse


pj = lambda x: os.path.join(utils.data_dir,x)


# read in the sequence 
fa = SeqIO.parse(pj('gencode_v17_5utr_15bpcds.fa'),'fasta')
tx_seq=dict()
for seq_record in fa:
    tx=seq_record.id
    seq=seq_record.seq
    if len(seq)<30: #skip when it too short
        continue
    if "ATG" not in seq:
        continue
    if tx not in tx_seq or len(seq)>len(tx_seq[tx]):
        tx_seq[tx]=seq

# txt records TE and rkpm
RP_data_path = {}
RP_data_path['muscle'] = pj('df_counts_and_len.TE_sorted.Muscle.with_annot.txt')
RP_data_path['PC3'] = pj('df_counts_and_len.TE_sorted.pc3.with_annot.txt')
RP_data_path['293T'] = pj('df_counts_and_len.TE_sorted.HEK_Andrev2015.with_annot.txt')

# read the hand crafted features
# (57415, 33)
feat_mat = pd.read_csv("util/Zhang_et_al_features.csv",index_col=0)
feat_mat.insert(0, 'seq', [tx_seq[x].upper().__str__() for x in feat_mat.index.values])


processed_dict = {}
for cell_line, csv_path in RP_data_path.items():

    assert os.path.exists(csv_path), f"The raw data file for {cell_line} does not exist!"
    
    print(f"processing {cell_line}...\n")
    RP_raw_data = pd.read_table(csv_path, sep=' ')
    RP_raw_data.loc[:,'T_id'] = RP_raw_data.index.values
    # adding sequence in 
    RP_raw_data = RP_raw_data.query('rpkm_rnaseq >5 & rpkm_riboseq > 0.1')
    RP_raw_data['log_te'] = np.log(RP_raw_data.te.values)

    RP_feat_merge = RP_raw_data.merge(feat_mat,left_on=['T_id'],right_index=True,suffixes=["",""])
    RP_feat_merge.sort_values('rpkm_rnaseq', ascending=False, inplace=True)
    
    # drop duplicated UTRs
    RP_raw_dedup = RP_feat_merge.drop_duplicates(RP_feat_merge.columns[17:], keep='first')
    RP_raw_dedup['utr'] = RP_raw_dedup['seq'].apply(lambda x: x[-216:-16]) # max len 200
    RP_raw_dedup['utr_len'] = RP_raw_dedup.utr.apply(len)
    RP_raw_dedup.query('`utr_len`>30')
    RP_raw_dedup = RP_raw_dedup.drop_duplicates(['utr'], keep='first') 
    
    # save them
    processed_dict[cell_line] = RP_raw_dedup
    processed_dict[cell_line].to_csv(pj(f"RP_{cell_line}_MTL_transfer.csv"))

print("The preprocssing for RP tasks is Finished !!")
print(f"The files are saved to {utils.data_dir}")