nohup python3 -u main.py /export/data/yliu/dataset_tick/ >log.out 2>&1 &
job_pid=$!

echo "PID of the job: $job_pid"
