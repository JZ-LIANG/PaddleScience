
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=4,5
export CPU_NUM=10
export NVIDIA_TF32_OVERRIDE=0


output_dir="./output/final_dp2"
export PYTHONPATH=/work/somecode/Science/final/PaddleScience:$PYTHONPATH

mkdir -p $output_dir
rm -rf $output_dir/*
rm -rf ./data/*.npy
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export FLAGS_call_stack_level=2
# export GLOG_vmodule=operator=4
# export GLOG_v=5
export FLAGS_START_PORT=6687
# export FLAGS_cinn_use_new_fusion_pass=1
python3 -m paddle.distributed.fleet.launch \
    --log_dir ${output_dir} \
    --gpus=${CUDA_VISIBLE_DEVICES} \
    cylinder3d_unsteady_ad.py
