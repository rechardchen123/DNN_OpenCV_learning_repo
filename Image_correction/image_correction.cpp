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
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types_c.h>

using namespace std;
using namespace cv;

void GetContoursPic(const char *pSrcFileName)
{
    //第一个参数:输入图片的名称, 第二个参数:输出图片的名称
    Mat srcImg = imread(pSrcFileName);
    imshow("origin img", srcImg);

    Mat gray, binImg;
    //灰度化
    cvtColor(srcImg, gray, COLOR_RGB2GRAY);
    imshow("gray img", gray);

    //二值化
    threshold(gray, binImg, 100, 200, THRESH_BINARY);
    imshow("binary img", binImg);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    vector<Rect> boundRect(contours.size());
    //注意第5个参数只CV_RETR_EXTERNAL，只检索外框
    findContours(binImg, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE); //find the co
    cout << contours.size() << endl;

    Point2f rectpoint[4];
    RotatedRect rect;
    for (size_t i = 0; i < contours.size(); i++)
    {
        //Rotated rectangle
        rect = minAreaRect(contours[i]);
        rect.points(rectpoint); //获取4个顶点坐标

        float angle = rect.angle; //获取包围盒与水平方向的角度
        cout << angle << endl;

        int line1 = sqrt((rectpoint[1].y - rectpoint[0].y) * (rectpoint[1].y - rectpoint[0].y) + (rectpoint[1].x -
                                                                                                  rectpoint[0].x) *
                                                                                                     (rectpoint[1].x - rectpoint[0].x));
        int line2 = sqrt((rectpoint[3].y - rectpoint[0].y) * (rectpoint[3].y - rectpoint[0].y) + (rectpoint[3].x -
                                                                                                  rectpoint[0].x) *
                                                                                                     (rectpoint[3].x - rectpoint[0].x));
        //计算面积
        if (line1 * line2 < 600)
        {
            continue;
        }

        //为了让正方形横着放，所以旋转角度是不一样的。竖放的，给他加90度，翻过来
        if (line1 > line2)
        {
            angle = 90 + angle;
        }

        //新建一个感兴趣的区域图，大小跟原图一样大
        Mat ROISrcImg(srcImg.rows, srcImg.cols, CV_8UC3); //这里必须选择CV_8UC3
        ROISrcImg.setTo(0);                               //颜色都设置为黑色

        //对得到的轮廓进行填充
        drawContours(binImg, contours, -1, Scalar(255), FILLED);

        //抠图得到ROISrcImg
        srcImg.copyTo(ROISrcImg, binImg);

        //显示看看效果，除了感兴趣的区域，其他部分都为黑色
        namedWindow("ROISrcImg", 1);
        imshow("ROISrcImg", ROISrcImg);

        //创建一个旋转后的图像
        Mat RotationedImg(ROISrcImg.rows, ROISrcImg.cols, CV_8UC1);
        RotationedImg.setTo(0);

        //对ROISrcImg进行旋转
        Point2f center = rect.center;                                                //中心点
        Mat M2 = getRotationMatrix2D(center, angle, 1);                              // 计算旋转加缩放的变换矩阵
        warpAffine(ROISrcImg, RotationedImg, M2, ROISrcImg.size(), 1, 0, Scalar(0)); //仿射变换

        imshow("roated image", RotationedImg);
        imwrite("../rotated_image.jpg", RotationedImg);
    }

    //对ROI区域进行抠图

    //对旋转后的图片进行轮廓提取
    vector<vector<Point>> contours2;
    Mat raw = imread("../rotated_image.jpg");
    Mat secondFindImg;

    cvtColor(raw, secondFindImg, COLOR_BGR2GRAY); //灰度化
    threshold(secondFindImg, secondFindImg, 80, 200, THRESH_BINARY);
    findContours(secondFindImg, contours2, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    for (int j = 0; j < contours2.size(); j++)
    {
        Rect rect = boundingRect(Mat(contours2[j]));

        if (rect.area() < 600)
        {
            continue;
        }

        Mat dstImg = raw(rect);
        imshow("dst", dstImg);
        imwrite("../pDstFileName.jpg", dstImg);
    }
}

int main()
{
    GetContoursPic("../1.png");
    waitKey(0);
    return 0;
}