# NoScope

This is the official project page for the NoScope project.

Please read the [blog post](http://dawn.cs.stanford.edu/2017/06/22/noscope/) and [paper](https://arxiv.org/abs/1703.02529) for more details!


# Requirements

This repository contains the code for the optimization step in the paper. The inference code is
[here](https://github.com/stanford-futuredata/tensorflow-noscope/tree/speedhax).

You will need the following installed:
- python-setuptools python-tk
- CUDA, CUDNN
- tensorflow-gpu, OpenCV 3.2 with FFmpeg bindings

Your machine will need at least:
- AVX2 capabilities
- 300+GB of memory
- 500+GB of space
- A GPU (this has only been tested with NVIDIA K80 and P100)


# Setting up the inference engine

To set up the inference engine, do the following:
```
git clone https://github.com/stanford-futuredata/tensorflow-noscope.git
cd tensorflow-noscope
git checkout speedhax
git submodule init
git submodule update
./configure
cd tensorflow
bazel build -c opt --copt=-mavx2 --config=cuda noscope
```
The build will fail. To fix this, update the BUILD file to point towards your OpenCV install and add
this directory to your PATH environmental variable. Please encourage the Tensorflow developers to
support non-bazel building and linking. Due to a quirk in bazel, it may occasionally "forget" that
tensorflow-noscope was built. If this happens, rebuild.


# Running the example

Once you have inference engine set up, the `example/` subfolder contains the script to reproduce
Figure 5d in the paper.

First, download the coral-reef video and labels:
```
wget https://storage.googleapis.com/noscope-data/csvs-yolo/coral-reef-long.csv
wget https://storage.googleapis.com/noscope-data/videos/coral-reef-long.mp4
```

You will need to update the `code` and `data` paths in run.sh. Additionally, you will need to update
`noscope_motherdog.py` to point to the YOLO configuration files and weights.
