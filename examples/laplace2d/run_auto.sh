
export CUDA_VISIBLE_DEVICES=4,5
export CPU_NUM=10
export NVIDIA_TF32_OVERRIDE=0
export FLAGS_use_cinn=1
export FLAGS_allow_cinn_ops="fill_constant_p;broadcast_p;add_p;sub_p;div_p;mul_p;sqrt_p;tanh_p;matmul_p;reduce_p;concat_p;reshape_p;transpose_p;slice_select_p;slice_assign_p;split_p;index_select_p;index_assign_p;fill_any_like"


output_dir="./output/laplace2d_manual_base_auto_with_norm_allreduce"
export PYTHONPATH=/work/somecode/Science/manual/PaddleScience:$PYTHONPATH

mkdir -p $output_dir
rm -rf $output_dir/*
rm -rf ./data/*.npy
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export FLAGS_call_stack_level=2
# export GLOG_vmodule=operator=4
export GLOG_v=4
export FLAGS_START_PORT=6687
export FLAGS_cinn_use_new_fusion_pass=1
python3 -m paddle.distributed.fleet.launch \
    --log_dir ${output_dir} \
    --gpus=${CUDA_VISIBLE_DEVICES} \
    laplace2d_static_auto.py \
    --num_epoch 10 \
    --output_dir $output_dir \
    --norm_allreduce true
