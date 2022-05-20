#!/bin/zsh

export CUDA_VISIBLE_DEVICES=4
export CPU_NUM=10
export NVIDIA_TF32_OVERRIDE=0
export FLAGS_use_cinn=1
export FLAGS_allow_cinn_ops="fill_constant_p;broadcast_p;add_p;sub_p;div_p;mul_p;sqrt_p;tanh_p;matmul_p;reduce_p;concat_p;reshape_p;transpose_p;slice_select_p;slice_assign_p;split_p;index_select_p;index_assign_p"
#FLAGS_new_executor_sequential_run=true FLAGS_host_trace_level=10 FLAGS_static_executor_perfstat_filepath=./perfstat_old 
export PYTHONPATH=/work/somecode/Science/manual/PaddleScience:$PYTHONPATH
python3 laplace2d_static.py 2>&1 | tee -a single_workerlog.0