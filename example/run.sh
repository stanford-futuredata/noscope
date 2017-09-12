# DO NOT USE RELATIVE PATHS
CODE_DIR="/home/daniel_d_kang/code/"
DATA_DIR="/home/daniel_d_kang/data/"

# Move this down to run jackson-town-square
VIDEO_NAME="jackson-town-square"
OBJECT="car"
NUM_FRAMES="918000"
START_FRAME="0"
GPU_NUM="0"


VIDEO_NAME="coral-reef-long"
OBJECT="person"
NUM_FRAMES="1188000"
START_FRAME="648000"
GPU_NUM="0"




# Generate the models
export CUDA_VISIBLE_DEVICES="$GPU_NUM"
python $CODE_DIR/noscope/exp/shuffled_small_cnn.py \
  --avg_fname ${VIDEO_NAME}.npy \
  --csv_in $DATA_DIR/csv/${VIDEO_NAME}.csv \
  --video_in $DATA_DIR/videos/${VIDEO_NAME}.mp4 \
  --output_dir $DATA_DIR/cnn-models/ \
  --base_name ${VIDEO_NAME} \
  --objects $OBJECT \
  --num_frames $NUM_FRAMES \
  --start_frame $START_FRAME

# Convert keras models to TF models
python $CODE_DIR/noscope/scripts/export-tf-graphs.py \
  --model_dir $DATA_DIR/cnn-models/ \
  --output_dir $DATA_DIR/cnn-models/

# Generate the avg
python $CODE_DIR/noscope/exp/to_avg.py \
  --npy_fname ${VIDEO_NAME}.npy \
  --txt_fname $DATA_DIR/cnn-avg/${VIDEO_NAME}.txt

# Run the inference
rm -r $DATA_DIR/experiments/$VIDEO_NAME
python $CODE_DIR/noscope/examples/noscope_motherdog.py \
  --code_dir $CODE_DIR \
  --data_dir $DATA_DIR \
  --video_name $VIDEO_NAME \
  --gpu_num $GPU_NUM
