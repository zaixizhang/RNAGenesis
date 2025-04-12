echo "please enter config path"
read filename
for i in {0..2} ;do 
    out_name=$(echo $filename | cut -d "." -f 1)
    
    nohup python script/iter_train.py --cuda 0 --config_file $filename --kfold_index $i > ${out_name}_cv${i}.out &
    pids=$!
    echo "PID: "$pids
    done