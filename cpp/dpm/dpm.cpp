// Adopted from https://github.com/opencv/opencv_contrib/blob/master/modules/dpm/samples/cascade_detect_camera.cpp

#include <opencv2/dpm.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/videoio/videoio_c.h>

#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace cv::dpm;

constexpr size_t kNbFrames = 10000;
constexpr size_t kSkip = 30;


int main(int argc, char** argv) {
  std::string xml_fname = "person.xml";
  std::string video_fname = "/lfs/1/ddkang/noscope/data/videos/elevator.mp4";

  cv::Ptr<DPMDetector> detector = DPMDetector::create(
      std::vector<std::string>(1, xml_fname));

  cv::VideoCapture capture(video_fname);
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

  std::ofstream csv_file("elevator.csv", std::ofstream::out);

  double total_time = 0;
  cv::Mat frame;
  size_t cur_frame = 0;
  for (size_t i = 0; i < kNbFrames; i++) {
    for (size_t j = 0; j < kSkip; j++) {
      const bool read_frame = capture.read(frame);
      if (!read_frame) {
        throw std::runtime_error("Failed to read frame.");
        return -1;
      }
      cur_frame++;
    }

    std::vector<DPMDetector::ObjectDetection> ds;

    cv::Mat image;
    frame.copyTo(image);

    double t = (double) cv::getTickCount();
    detector->detect(image, ds);
    t = ((double) cv::getTickCount() - t) / cv::getTickFrequency(); //elapsed time

    // FIXME
    if (ds.size() > 0) {
      csv_file << cur_frame << ",1\n";
    } else {
      csv_file << cur_frame << ",0\n";
    }

    total_time += t;
  }

  std::cout << "FPS: " << total_time / kNbFrames << "\n";

  return 0;
}

