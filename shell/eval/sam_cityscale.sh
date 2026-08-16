# python test.py --gpu_no 4 \
CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 \
torchrun --nproc_per_node=4 --master_port=1235 \
test.py --is_multigpu 1 \
--cfg ./Config/sam_cityscale.py \
--load_path work_dir/sam_cityscale_vector \
--load_no 6 \
--view_path ./view_dir/view_sam_cityscale \
--result_path ./result_dir/result_sam_cityscale

python road_graph_metric/apls.py \
--dataset cityscale \
--data_path /home/wsq/dataset/cityscale \
--result_path ./result_dir/result_sam_cityscale \
--num_processes 32

python road_graph_metric/topo.py \
--dataset cityscale \
--data_path /home/wsq/dataset/cityscale \
--result_path ./result_dir/result_sam_cityscale \
--num_processes 32


