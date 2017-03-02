#include <sys/mman.h>
#include <xmmintrin.h>
#include <immintrin.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>

#define RESOL (50)
#define NB_CHAN (3)

using namespace cv;

float MSE_CV(const Mat& I1, const Mat& I2) {
  Mat s1 = I1 - I2;
  // absdiff(I1, I2, s1);
  // s1.convertTo(s1, CV_32F);
  s1 = s1.mul(s1);

  Scalar s = sum(s1);

  float sse = s.val[0] + s.val[1] + s.val[2];
  return sse / (I1.channels() * I1.total());
}

float MSE_DDKANG(const float *f1, const float *f2) {
  float tmp = 0;
  const int N = RESOL * RESOL * NB_CHAN;
  for (int i = 0; i < N; i++) {
    const float diff = f1[i] - f2[i];
    tmp += diff * diff;
  }

  return tmp / N;
}

float MSE_DDKANG_MULTI(const float *f1, const float *f2) {
  const int N = RESOL * RESOL * NB_CHAN;
  float acc[8] = {0};
  for (int i = 0; i < N; i += 8) {
    for (int j = 0; j < 8; j++) {
      const float diff = f1[i + j] - f2[i + j];
      acc[j] += diff * diff;
    }
  }

  float mse = 0;
  for (int i = 0; i < 8; i++) mse += acc[i];
  return mse / N;
}

// http://stackoverflow.com/questions/23189488/horizontal-sum-of-32-bit-floats-in-256-bit-avx-vector
static inline float _mm256_reduce_add_ps(__m256 x) {
    /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    /* Conversion to float is a no-op on x86-64 */
    return _mm_cvtss_f32(x32);
}

float MSE_AVX(const float *f1, const float *f2) {
  const int N = RESOL * RESOL * NB_CHAN;

  __m256 acc =  _mm256_set1_ps(0);
  for (int i = 0; i < N; i += 8) {
    __m256 tmp = _mm256_sub_ps(_mm256_loadu_ps(f1 + i), _mm256_loadu_ps(f2 + i));
    tmp = _mm256_mul_ps(tmp, tmp);
    acc = _mm256_add_ps(acc, tmp);
  }

  float mse = _mm256_reduce_add_ps(acc);
  return mse / N;
}

float MSE_AVX_MULTI(const float *f1, const float *f2) {
  const int N = RESOL * RESOL * NB_CHAN;
  const int NB_ACC = 2;

  __m256 acc[NB_ACC];
  for (int i = 0; i < NB_ACC; i++) acc[i] = _mm256_set1_ps(0);
  for (int i = 0; i < N; i += 8 * NB_ACC) {
    for (int j = 0; j < NB_ACC; j++) {
      __m256 tmp = _mm256_sub_ps(_mm256_loadu_ps(f1 + i + j * 8), _mm256_loadu_ps(f2 + i + j * 8));
      tmp = _mm256_mul_ps(tmp, tmp);
      acc[j] = _mm256_add_ps(acc[j], tmp);
    }
  }

  __m256 tmp = acc[0];
  for (int i = 1; i < NB_ACC; i++)
    tmp = _mm256_add_ps(acc[i], tmp);

  float mse = _mm256_reduce_add_ps(tmp);
  return mse / N;
}

/*float MSE_AVX2(const float *f1, const float *f2) {
  const int N = RESOL * RESOL * NB_CHAN;

  float buf[16];
  __m512 acc;
  _mm512_set1_ps(0);
  for (int i = 0; i < N; i += 16) {
    __m512 tmp = _mm512_sub_ps(_mm512_loadu_ps(f1 + i), _mm512_loadu_ps(f2 + i));
    tmp = _mm512_mul_ps(tmp, tmp);
    acc = _mm512_add_ps(acc, tmp);
  }
  _mm512_storeu_ps(buf, acc);

  float mse = 0;
  for (int i = 0; i < 16; i++) mse += buf[i];
  return mse / N;
}*/

int main() {
  const int NB_FRAMES = 100000;
  const int DELAY = 10;

  std::vector<float> data(NB_FRAMES * RESOL * RESOL * NB_CHAN);
  std::vector<float> mses(NB_FRAMES);

  int dims[] = {RESOL, RESOL};
  std::vector<Mat> frames(100000);
  for (int i = 0; i < NB_FRAMES; i++) {
    frames[i] = Mat(RESOL, RESOL, CV_32FC3, &data[i * RESOL * RESOL * NB_CHAN]);
  }

  if (mlockall(MCL_CURRENT) != -1) {
    int errsv = errno;
    std::cerr << "mlockall failed: " << errsv << "\n";
  }

  auto start = std::chrono::high_resolution_clock::now();
  // #pragma omp parallel for num_threads(32) schedule(static)
  for (int i = DELAY; i < NB_FRAMES; i++) {
    // mses[i] = MSE_CV(frames[i], frames[i - DELAY]);
    // mses[i] = MSE_DDKANG(&data[i * RESOL * RESOL * NB_CHAN], &data[(i - DELAY) * RESOL * RESOL * NB_CHAN]);
    mses[i] = MSE_AVX(&data[i * RESOL * RESOL * NB_CHAN], &data[(i - DELAY) * RESOL * RESOL * NB_CHAN]);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end-start;
  std::cout << "Time: " << diff.count() << " s" << std::endl;

  return 0;
}
