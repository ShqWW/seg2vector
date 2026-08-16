#python test.py --gpu_no 4 \
CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 \
torchrun --nproc_per_node=4 --master_port=1234 \
test.py --is_multigpu 1 \
--cfg ./Config/sam_spacenet.py \
--load_path work_dir/sam_spacenet_vector \
--load_no 4 \
--view_path ./view_dir/view_sam_spacenet \
--result_path ./result_dir/result_sam_spacenet

python road_graph_metric/apls.py \
--dataset spacenet \
--data_path /home/wsq/dataset/spacenet \
--result_path ./result_dir/result_sam_spacenet \
--num_processes 32

python road_graph_metric/topo.py \
--dataset spacenet \
--data_path /home/wsq/dataset/spacenet \
--result_path ./result_dir/result_sam_spacenet \
--num_processes 32







