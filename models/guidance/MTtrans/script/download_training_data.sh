DATA_DIR=$(cat machine_configure.json |grep data_dir|awk  -F ': "' '{print $2}'|awk -F '",' '{print $1}')

if [ ! -d ${DATA_DIR} ]; then
    echo "=============================================================="
    echo "                   "
    echo ${DATA_DIR} does not exist! Redirect to ./data
    echo "                   "
    echo "=============================================================="
    DATA_DIR=$(pwd)/data
    fi

cd ${DATA_DIR}

## download the ribosome profiling datasest 
# RP-muscle
wget https://raw.githubusercontent.com/zzz2010/5UTR_Optimizer/master/data/df_counts_and_len.TE_sorted.Muscle.with_annot.txt 
# RP-293T
wget https://raw.githubusercontent.com/zzz2010/5UTR_Optimizer/master/data/df_counts_and_len.TE_sorted.HEK_Andrev2015.with_annot.txt 
# RP-PC3
wget https://raw.githubusercontent.com/zzz2010/5UTR_Optimizer/master/data/df_counts_and_len.TE_sorted.pc3.with_annot.txt 
# ref
wget https://raw.githubusercontent.com/zzz2010/5UTR_Optimizer/master/data/gencode_v17_5utr_15bpcds.fa

## download the Massively parallel report assay datasets
# MPA_U
wget ftp.ncbi.nlm.nih.gov/geo/samples/GSM3130nnn/GSM3130435/suppl/GSM3130435_egfp_unmod_1.csv.gz 
# MPA_H
wget ftp.ncbi.nlm.nih.gov/geo/samples/GSM3130nnn/GSM3130443/suppl/GSM3130443_designed_library.csv.gz 
# MPA_V
wget ftp.ncbi.nlm.nih.gov/geo/samples/GSM4084nnn/GSM4084997/suppl/GSM4084997_varying_length_25to100.csv.gz


## dowlaod the yeast dataset
wget ftp.ncbi.nlm.nih.gov/geo/samples/GSM2793nnn/GSM2793752/suppl/GSM2793752_Random_UTRs.csv.gz


gzip -d *.gz

echo "Finished ! All data ready~"
echo "Please step to the pre-proccessing"