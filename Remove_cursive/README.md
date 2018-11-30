#Algo to remove cursive style

**Algorithm removes cursive writing style**

 removes the cursive writing style.
 preprocessing step for handwritten text recognition.
 input and output of the algorithm for a given image (```data/test1.png```) is done as follows:

![deslanting](./doc/deslanting.png)
* CPU: all computations are done on the CPU using OpenCV.





Build **CPU** implementation on Linux (OpenCV must be installed):
```g++ --std=c++11 src/cpp/main.cpp src/cpp/DeslantImgCPU.cpp `pkg-config --cflags --libs opencv` -o DeslantImg ```


Deslant the two images provided in data/  directory, writes output to the root directory of repo
```./DeslantImg


Call function ```deslantImg(img, bgcolor)``` with the input image (grayscale) and the background color (to fill empty image space).
It returns the deslanted image

```
#include "DeslantImgCPU.hpp"
...

// read grayscale image
const cv::Mat img = cv::imread("data/test1.png", cv::IMREAD_GRAYSCALE);

// deslant it
const cv::Mat res = htr::deslantImg(img, 255);

// save the result
cv::imwrite("out1.png", res);
```
