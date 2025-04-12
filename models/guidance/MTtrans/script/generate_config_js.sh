script_dir=$(pwd)
echo "{" > machine_configure.json
echo '"script_dir": "'${script_dir}'",' >> machine_configure.json
echo '"data_dir": "'${script_dir}'/data",' >> machine_configure.json
echo '"log_dir": "'${script_dir}'/log",' >> machine_configure.json
echo '"pth_dir": "'${script_dir}'/checkpoint"' >> machine_configure.json
echo "}" >> machine_configure.json

if [ ! -d ${script_dir}/data ]; then
    mkdir data
    mkdir checkpoint    
    fi