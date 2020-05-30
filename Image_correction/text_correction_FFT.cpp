/*********************************************************************************
*FileName:  // 文档偏斜矫正基于FFT变换的频率域梯度 
*Author:  //richard_chen
*Version:  //1.0
*Date:  //2020-05-30
*Description: // 1. 图片灰度化
              // 2. 离散傅里叶变换得到频率域空间的振幅
              // 3. 二值化
              // 4. 霍夫直线检测得到角度
              // 5. 根据角度完成倾斜矫正
*Others: // 
*Function List:  //主要函数列表，每条记录应包含函数名及功能简要说明
               1. 
               2. 
               3. 
*History:  //修改历史记录列表，每条修改记录应包含修改日期、修改者及修改内容简介

**********************************************************************************/
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void GetRotatedImg(const char *pSrcFileName)
{
    Mat src = imread(pSrcFileName);
    imshow("original pic", src);

    Mat gray, binary;

    cvtColor(src, gray, COLOR_BGR2GRAY);

    //expand input image to optimal size
    Mat padded;

    //离散弗利特变换
    int m = getOptimalDFTSize(gray.rows);
    int n = getOptimalDFTSize(gray.cols);

    // on the border add zero values
    copyMakeBorder(gray, padded, 0, m - gray.rows, 0, n - gray.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};

    Mat complexI;

    //Add to the expanded another plane with zero
    merge(planes, 2, complexI);

    //离散傅里叶变换
    dft(complexI, complexI);

    // 实部与虚部得到梯度图像
    split(complexI, planes);
    magnitude(planes[0], planes[1], planes[0]);
    Mat magI = planes[0];

    magI += Scalar::all(1);

    log(magI, magI);

    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols / 2;
    int cy = magI.rows / 2;
    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;
    // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);

    q3.copyTo(q0);

    tmp.copyTo(q3);

    q1.copyTo(tmp);

    q2.copyTo(q1);

    tmp.copyTo(q2);

    // 归一化与阈值化显示
    normalize(magI, magI, 0, 1.0, NORM_MINMAX);

    Mat dst;
    magI.convertTo(dst, CV_8UC1, 255, 0);
    threshold(dst, binary, 160, 255, THRESH_BINARY);

    //霍夫直线判断
    vector<Vec2f> lines;
    Mat linImg = Mat::zeros(binary.size(), CV_8UC3);

    HoughLines(binary, lines, 1, (float)CV_PI / 180, 30, 0, 0);
    int numLines = lines.size();
    float degree = 0.0;

    for (int i = 0; i < numLines; i++)
    {
        float rho = lines[i][0];
        float theta = lines[i][1];
        float offset = CV_PI / 12.0;

        if (abs(theta) > offset && abs(theta) < (CV_PI / 2.0 - offset))
        {
            cout << "theta: " << theta << endl;
            degree = (theta)*180 - 90;
        }
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;

        pt1.x = cvRound(x0 + 1000 * (-b));

        pt1.y = cvRound(y0 + 1000 * (a));

        pt2.x = cvRound(x0 - 1000 * (-b));

        pt2.y = cvRound(y0 - 1000 * (a));

        line(linImg, pt1, pt2, Scalar(0, 255, 0), 3, 8, 0);
    }
    imshow("lines", linImg);

    // 旋转调整
    Mat rot_mat = getRotationMatrix2D(Point(binary.cols / 2, binary.rows / 2), degree, 1);
    Mat rotated;
    warpAffine(src, rotated, rot_mat, src.size(), cv::INTER_CUBIC, 0, Scalar(255, 255, 255));
    imshow("rotated image", rotated);
    imwrite("../roated_image_1.jpg", rotated);
}

int main()
{
    GetRotatedImg("../3.jpg");
    waitKey(0);
    return 0;
}