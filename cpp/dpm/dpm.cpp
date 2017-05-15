// Adopted from https://github.com/opencv/opencv_contrib/blob/master/modules/dpm/samples/cascade_detect_camera.cpp

#include <opencv2/dpm.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/videoio/videoio_c.h>

#include <stdio.h>
#include <iostream>

using namespace cv::dpm;

int main(int argc, char** argv) {
	cv::Ptr<DPMDetector> detector = DPMDetector::create(
      std::vector<std::string>(1, "bus.xml"));

  cv::VideoCapture capture("/lfs/1/ddkang/noscope/data/videos/taipei-long.mp4");
	// capture.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	// capture.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

	if (!capture.isOpened()) {
    throw std::runtime_error("Fail to open video\n");
    return -1;
	}

#ifdef HAVE_TBB
  std::cout << "Running with TBB.\n";
#else
#ifdef _OPENMP
  std::cout << "Running with OpenMP.\n";
#else
  std::cout << "Running without OpenMP and without TBB.\n";
#endif
#endif

  constexpr int kNbFrames = 100;
  double total_time = 0;
  cv::Mat frame;
  for (int i = 0; i < kNbFrames; i++) {
    const bool read_frame = capture.read(frame);
    if (!read_frame) {
      throw std::runtime_error("Failed to read frame.");
      return -1;
    }

    std::vector<DPMDetector::ObjectDetection> ds;

    cv::Mat image;
    frame.copyTo(image);

    double t = (double) cv::getTickCount();
    detector->detect(image, ds);
    t = ((double) cv::getTickCount() - t) / cv::getTickFrequency(); //elapsed time

    total_time += t;
  }

  std::cout << "FPS: " << total_time / kNbFrames << "\n";

  return 0;
}

