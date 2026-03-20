# python test.py --gpu_no 4 \
CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 \
torchrun --nproc_per_node=4 --master_port=1234 \
test.py --is_multigpu 1 \
--cfg ./Config/sam2_spacenet.py \
--load_path work_dir/sam2_spacenet_vector \
--load_no 4 \
--view_path ./view_dir/view_sam2_spacenet2 \
--is_view 1 \
--result_path ./result_dir/result_sam2_spacenet

python road_graph_metric/apls.py \
--dataset spacenet \
--data_path /yourdataset/spacenet \
--result_path ./result_dir/result_sam2_spacenet \
--num_processes 32

python road_graph_metric/topo.py \
--dataset spacenet \
--data_path /yourdataset/spacenet \
--result_path ./result_dir/result_sam2_spacenet \
--num_processes 32





