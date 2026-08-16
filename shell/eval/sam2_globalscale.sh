CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 \
torchrun --nproc_per_node=4 --master_port=1234 \
test.py --is_multigpu 1 \
--cfg ./Config/sam2_globalscale.py \
--load_path work_dir/sam2_globalscale_vector \
--load_no 1 \
--view_path ./view_dir/view_sam2_globalscale \
--result_path ./result_dir/result_sam2_globalscale \
--in_domain_test 1

python road_graph_metric/apls.py \
--dataset globalscale \
--data_path /home/wsq/dataset/Global-Scale \
--result_path ./result_dir/result_sam2_globalscale \
--num_processes 128 \
--in_domain_test 1

python road_graph_metric/topo.py \
--dataset globalscale \
--data_path /home/wsq/dataset/Global-Scale \
--result_path ./result_dir/result_sam2_globalscale \
--num_processes 128 \
--in_domain_test 1


