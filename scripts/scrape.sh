#! /usr/bin/env bash

if [ "$#" -ne 2 ]; then
    echo "Two arguments required: [video url] [output name]"
    exit 1
fi

set -x

count=0
while true; 
  #do youtube-dl --no-overwrites --proxy socks5://127.0.0.1:8123/ \
  #  --hls-prefer-native -c --no-part --keep-video --retries infinite $1 -o ${2}.${count}.mp4
  do youtube-dl --no-overwrites --ffmpeg-location /opt/ffmpeg-3.2.2-64bit-static/ \
    -c --no-part --keep-video --retries infinite $1 -o ${2}.${count}.mp4
  count=$((count+1))
  sleep  5
done


