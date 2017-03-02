#! /usr/bin/env bash

if [ "$#" -ne 2 ]; then
    echo "Two arguments required: [m3u8 video url] [output name]"
    exit 1
fi

set -x

count=0
while true; 
do
  /opt/ffmpeg-3.2.2-64bit-static/ffmpeg -i $1 -framerate 30 -crf 15 -crf_max 20 -c copy ${2}.${count}.mp4
  count=$((count+1))
  sleep  5
done


