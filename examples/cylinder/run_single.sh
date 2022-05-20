export PYTHONPATH=/work/somecode/Science/final/PaddleScience:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=5 


rm -rf single_final_workerlog.0
python3 cylinder3d_unsteady_ad.py | tee -a single_final_workerlog.0