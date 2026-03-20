#python train.py --gpu_no 4 \
CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 \
torchrun --nproc_per_node=4 --master_port=23458 \
train.py --is_multigpu 1 \
--cfg ./Config/sam2_globalscale.py \
--save_path work_dir/sam2_globalscale_seg \
--iter_display 1000 \
--train_vector 0

CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 \
torchrun --nproc_per_node=4 --master_port=23458 \
train.py --is_multigpu 1 \
--cfg ./Config/sam2_globalscale.py \
--load_path work_dir/sam2_globalscale_seg \
--load_no 4 5 \
--save_path work_dir/sam2_globalscale_vector \
--iter_display 1000 \
--train_vector 1

