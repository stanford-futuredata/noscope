# NoScope

This is the official project page for the NoScope project.

Please read the [blog post](http://dawn.cs.stanford.edu/2017/06/22/noscope/) and [paper](https://arxiv.org/abs/1703.02529) for more details!


# Requirements

This repository contains the code for the optimization step in the paper. The inference code is
[here](https://github.com/stanford-futuredata/tensorflow-noscope/tree/speedhax).

You will need the following installed:
- python 2.7
- pip python-setuptools python-tk
- keras
- CUDA, CUDNN, tensorflow-gpu
- OpenCV 3.2 with FFmpeg bindings
- g++ 5.4 or later

Your machine will need at least:
- AVX2 capabilities
- 300+GB of memory
- 500+GB of space
- A GPU (this has only been tested with NVIDIA K80 and P100)

## Guides on Installing the Requirements

- python 2.7 - For Linux, use your package manager. For Mac: http://docs.python-guide.org/en/latest/starting/install/osx/.
- pip python-setuptools python-tk - https://packaging.python.org/tutorials/installing-packages/
- Install version 1.2 of keras, as it must interface with TensorFlow 0.12 - pip install keras==1.2
- CUDA, CUDNN, tensorflow-gpu
  - https://www.tensorflow.org/versions/r0.12/get_started/os_setup - YOU MUST INSTALL THESE VERSIONS, not more recent ones. Go to the section **Download and Install cuDNN** for cuDNN installation instructions.
  - Note: Having both TensorFlow-gpu 0.12 and more recent versions installed is complicated. It may not be easy to have the multiple versions of cuDNN installed. This project requires cuDNN 5.1 and more recent versions of TensorFlow will break with that installed. Therefore, it is recommended that users uninstall more recent versions of tensorflow and delete the cuDNN so files if they are installed.
- OpenCV 3.2 with FFmpeg bindings - https://github.com/BVLC/caffe/wiki/OpenCV-3.2-Installation-Guide-on-Ubuntu-16.04
- g++ 5.4 or later - For Linux, use your package manager. For Mac, http://braumeister.org/formula/gcc@5 should work, though the developers haven't tested this.


# Setting up the inference engine

To set up the inference engine, do the following:
Note: It is recommended that you create a folder that contains this repository, the tensorflow-noscope
repository, and the data folder referred to below.
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
this directory to your PATH environmental variable. The BUILD file is in the tensorflow-noscope
git repository at tensorflow/noscope/BUILD. You will need to change all references to "/lfs/0/ddkang/".
You will probably need to change these to "/usr/" if you installed OpenCV using the directions above.

Please encourage the Tensorflow developers to support non-bazel building and linking. Due to a quirk
in bazel, it may occasionally "forget" that tensorflow-noscope was built. If this happens, rebuild.

# Setting up the optimization engine

To set up the optimization engine, install the NoScope python package by going to the root directory
of where you checked out https://github.com/stanford-futuredata/noscope and running
"pip install -e ./"

# Running the example
Once you have inference engine set up, the `example/` subfolder within this repository contains the
script to reproduce Figure 5d in the paper.

In order to run this:
1. Create a folder named data that sits in the same directory as your noscope and tensorflow-noscope
folders
2. Create the following folders within the data folder: videos, csv, cnn-avg, cnn-models, and experiments
3. Download the coral-reef video and labels, putting the csv file in the csv folder and the mp4 file in
the videos folder:
```
wget https://storage.googleapis.com/noscope-data/csvs-yolo/coral-reef-long.csv
wget https://storage.googleapis.com/noscope-data/videos/coral-reef-long.mp4
```
4. Update the `code` and `data` paths in example/run.sh. `code` should point to the folder that contains
both the noscope and tensorflow-noscope folders. This value is how the optimization and inference engines
find eachother. `data` should point to the data folder created in this section.
5. Download the YOLO neural network weights file from https://pjreddie.com/media/files/yolo.weights.
It is suggested that you place the file at the location tensorflow-noscope/tensorflow/noscope/darknet/weights/.
Note that you will need to make the weights folder.
6. Update example/noscope_motherdog.py to point to the YOLO configuration and weights files. The config
file is tensorflow-noscope/tensorflow/noscope/darknet/cfg/yolo.cfg and the weights file is the one you
downloaded. If you put the weights file in the suggested location, this step should be unnecessary.
7. Run example/run.sh. The outputted summary.csv will be in the location data/experiments/$VIDEO_NAME.

# Datasets
The datasets that are currently available are `coral-reef-long` and `jackson-town-square`.

The mp4 video files are available at `https://storage.googleapis.com/noscope-data/videos/VIDEO_NAME.mp4`

The CSVs with ground truth labels are available at
`https://storage.googleapis.com/noscope-data/csvs-yolo/VIDEO_NAME.csv`
