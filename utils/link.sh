 rm -rf dataset
 mkdir dataset
 mkdir ./dataset/cityscale
 ln -s ~/dataset/cityscale/20cities ./dataset/cityscale
 mkdir ~/dataset/cityscale_preprocess
 ln -s ~/dataset/cityscale_preprocess ./dataset


ln -s ~/dataset/spacenet/RGB_1.0_meter ./dataset/spacenet
mkdir ~/dataset/spacenet_preprocess
ln -s ~/dataset/spacenet_preprocess ./dataset



ln -s ~/dataset/Global-scale ./dataset/globalscale
ln -s ~/dataset/globalscale_preprocess ./dataset