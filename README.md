PyTorch IHGNN
=============

About
-----

PyTorch implementation of IHGNN.


Installation
------------

This implementation is based on Hanjun Dai's structure2vec graph backend. Under the "lib/" directory, type

    make -j4

to compile the necessary c++ files.

After that, under the root directory of this repository, type

    ./run_DGCNN.sh DATANAME

to run DGCNN on dataset DATANAME with the default setting.


Datasets
--------

Default graph datasets are stored in "data/DSName/DSName.txt". Check the "data/README.md" for the format. 
