#python train.py --gpu_no 4 \
CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 \
torchrun --nproc_per_node=4 --master_port=23458 \
train.py --is_multigpu 1 \
--cfg ./Config/sam_cityscale.py \
--save_path work_dir/sam_cityscale_seg \
--train_vector 0

CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 \
torchrun --nproc_per_node=4 --master_port=23458 \
train.py --is_multigpu 1 \
--cfg ./Config/sam_cityscale.py \
--load_path work_dir/sam_cityscale_seg \
--load_no 11 12 13 \
--save_path work_dir/sam_cityscale_vector \
--train_vector 1

