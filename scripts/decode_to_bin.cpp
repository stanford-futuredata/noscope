#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgproc/imgproc.hpp"

constexpr size_t kResol_ = 50;
constexpr size_t kFrameSize_ = kResol_ * kResol_ * 3;

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " FNAME NUM_FRAMES START_FRAME\n";
    return 0;
  }

  const std::string fname(argv[1]);
  const size_t kNbFrames_ = std::stoi(argv[2]);
  const size_t kStartFrame_ = std::stoi(argv[3]);

  std::cout << "Running on:\n";
  std::cout << fname << "\n";
  std::cout << kNbFrames_ << "\n";
  std::cout << kStartFrame_ << "\n";

  std::vector<char> frame_data(kNbFrames_ * kFrameSize_);

  cv::VideoCapture cap(fname);
  cap.set(CV_CAP_PROP_POS_FRAMES, kStartFrame_);

  for (size_t i = 0; i < kNbFrames_; i++) {
    cv::Mat frame, resized;

    const bool success = cap.read(frame);
    if (!success) {
      std::cerr << "Failed to read frame " << i << "\n";
      throw std::runtime_error("Failed to read frame");
    }
    if (!frame.isContinuous()) {
      throw std::runtime_error("Frame is not continuous");
    }
    if (i % 500 == 0) {
      std::cout << i << "\n";
    }

    cv::resize(frame, resized, cv::Size(kResol_, kResol_), 0, 0, cv::INTER_NEAREST);

    memcpy(&frame_data[i * kFrameSize_], resized.data, kFrameSize_);
  }

  std::ofstream outfile("data.bin", std::ios::out | std::ios::binary);
  outfile.write(&frame_data[0], frame_data.size());

  return 0;
}
