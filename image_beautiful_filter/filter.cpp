#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void sketch_filter(Mat &img);

int main(int argc, char **argv)
{
    Mat image = imread("../reba.jpg");
    namedWindow("input", WINDOW_AUTOSIZE);
    imshow("input", image);
    sketch_filter(image);
    waitKey(0);

    VideoCapture cap(0);
    Mat frame;
    while (true)
    {
        bool ret = cap.read(frame);
        imshow("input", frame);
        if (!ret)
            break;
        sketch_filter(frame);
        char c = waitKey(1);
        if (c == 27)
        {
            break;
        }
    }
    waitKey(0);
    return 0;
}

void sketch_filter(Mat &img)
{
    double t1 = getTickCount();
    //灰度去色
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    //取反与高斯模糊
    Mat invert;
    bitwise_not(gray, invert);
    GaussianBlur(invert, invert, Size(15, 15), 0);

    //减淡公式 C = Min(A + (A x B) / (255 - B), 255)
    Mat result(gray.size(), CV_8UC1);
    for (size_t row = 0; row < gray.rows; row++)
    {
        uchar *g_pixel = gray.data + row * gray.step;
        uchar *in_pixel = invert.data + row * invert.step;
        uchar *result_row = result.data + row * result.step;

        for (size_t col = 0; col < gray.cols; col++)
        {
            int a = *g_pixel++;
            int b = *in_pixel++;
            int c = std::min(a + (a * b) / (256 - b), 255);
            *result_row++ = c;
        }
    }

    gray.release();
    invert.release();

    double t2 = getTickCount();
    double fps = getTickFrequency() / (t2 - t1);
    putText(result, format("current fps: %.2f", fps), Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 2, 8);
    imshow("sketch effect", result);
}