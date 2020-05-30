/*********************************************************************************
*FileName:  // image correction 
*Author:  //richard_chen
*Version:  //1.0
*Date:  //2020-05-30
*Description: // 1. 图片灰度化
              // 2. 阈值二值化
              // 3. 检测轮廓
              // 4. 寻找轮廓的包围矩阵，并且获取角度
              // 5. 根据角度进行旋转矫正
              // 6. 对旋转后的图像进行轮廓提取
              // 7. 对轮廓内的图像区域抠图，成为一张独立的图像，完成矫正
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

#define ERROR 1234

//度数转换
double DegreeTrans(double theta)
{
    double res = theta / CV_PI * 180;
    return res;
}

//逆时针旋转图像
void rotateImage(Mat src, Mat &img_rotate, double degree)
{
    //旋转中心为图像中心
    Point2f center;
    center.x = float(src.cols / 2.0);
    center.y = float(src.rows / 2.0);

    int length = 0;
    length = sqrt(src.cols * src.cols + src.rows * src.rows);

    //计算二维旋转的仿射变换矩阵
    Mat M = getRotationMatrix2D(center, degree, 1);
    warpAffine(src, img_rotate, M, Size(length, length), 1, 0, Scalar(255, 255, 255)); //背景填充为白色
}

//通过霍夫变换计算变换角度
double CalcDegree(const Mat &srcImage, Mat &dst)
{
    Mat midImage, dstImage;

    Canny(srcImage, midImage, 50, 200, 3);
    cvtColor(midImage, dstImage, COLOR_GRAY2BGR);

    //通过霍夫变换检测直线
    vector<Vec2f> lines;
    HoughLines(midImage, lines, 1, CV_PI / 180, 300, 0, 0); //第5个参数为阈值，阈值越大，检测精度越高

    //由于图像不同，阈值不好设定，因为阈值设定过高导致无法检测直线，阈值过低直线太多，速度很慢
    //所以根据阈值由大到小设置了三个阈值，如果经过大量试验后，可以固定一个适合的阈值。

    if (!lines.size())
    {
        HoughLines(midImage, lines, 1, CV_PI / 180, 200, 0, 0);
    }

    if (!lines.size())
    {
        HoughLines(midImage, lines, 1, CV_PI / 180, 150, 0, 0);
    }

    if (!lines.size())
    {
        cout << "没有检测到直线！" << endl;
        return ERROR;
    }

    float sum = 0;
    //依次画出每条线段
    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0];
        float theta = lines[i][1];
        Point pt1, pt2;

        double a = cos(theta);
        double b = sin(theta);
        double x0 = a * rho;
        double y0 = b * rho;

        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));

        //只选择角度最小的作为旋转角度
        sum += theta;
        line(dstImage, pt1, pt2, Scalar(55, 100, 195), 1, LINE_AA);

        imshow("straight line effect", dstImage);
    }

    float average = sum / lines.size(); //对所有角度求平均，这样做旋转效果会更好

    cout << "average theta: " << average << endl;

    double angle = DegreeTrans(average) - 90;

    rotateImage(dstImage, dst, angle);
    return angle;
}

void ImageRecify(const char *pInFileName)
{
    double degree;
    Mat src = imread(pInFileName);
    imshow("origin image", src);
    Mat dst;
    degree = CalcDegree(src, dst);

    if (degree == ERROR)
    {
        cout << "correction error!" << endl;
        return;
    }

    rotateImage(src, dst, degree);
    cout << "angle: " << degree << endl;
    imshow("rotated image", dst);

    Mat resultImage = dst(Rect(0, 0, dst.cols, 500)); //根据先验知识，估计好文本的长宽，再裁剪下来
    imshow("clipped image", resultImage);
    imwrite("../rectified.jpg", resultImage);
}

int main()
{
    ImageRecify("../3.jpg");
    waitKey(0);
    return 0;
}
