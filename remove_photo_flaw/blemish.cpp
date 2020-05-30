#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int patch_width = 60;
int patch_height = 60;
static void onMouse(int event, int x, int y, int flags, void *data)
{
    Mat image = *((Mat *)data);
    if (event == EVENT_LBUTTONDOWN)
    {
        printf("current location Point(%d, %d)\n", x, y);
        Rect patch_rect(x - patch_width / 2, y - patch_height / 2, patch_width, patch_height);
        Mat patch = image(patch_rect);

        Mat sub_patch[9];
        sub_patch[0] = patch(Range(0, patch_height / 3), Range(0, patch_width / 3));
        sub_patch[1] = patch(Range(0, patch_height / 3), Range(patch_width / 3, 2 * patch_width / 3));
        sub_patch[2] = patch(Range(0, patch_height / 3), Range(2 * patch_width / 3, patch_width));
        sub_patch[3] = patch(Range(patch_height / 3, 2 * patch_height / 3), Range(0, patch_width / 3));
        sub_patch[4] = patch(Range(patch_height / 3, 2 * patch_height / 3), Range(patch_width / 3, 2 * patch_width / 3));
        sub_patch[5] = patch(Range(patch_height / 3, 2 * patch_height / 3), Range(2 * patch_width / 3, patch_width));
        sub_patch[6] = patch(Range(2 * patch_height / 3, patch_height), Range(0, patch_width / 3));
        sub_patch[7] = patch(Range(2 * patch_height / 3, patch_height), Range(patch_width / 3, 2 * patch_width / 3));
        sub_patch[8] = patch(Range(2 * patch_height / 3, patch_height), Range(2 * patch_width / 3, patch_width));

        Mat sub_patch_laplacian[9];
        for (int i = 0; i < 9; i++)
        {
            Laplacian(sub_patch[i], sub_patch_laplacian[i], CV_32F, 3, 1, 0);
        }

        float sub_patch_mean[9];
        for (int i = 0; i < 9; i++)
        {
            sub_patch_mean[i] = (mean(sub_patch_laplacian[i])).val[0];
        }

        float min_mean = sub_patch_mean[0];
        int min_index = 0;

        for (int i = 0; i < 9; i++)
        {
            if (sub_patch_mean[i] < min_mean)
            {
                min_mean = sub_patch_mean[i];
                min_index = i;
            }
        }

        Mat output;
        Mat src_mask = Mat::ones(sub_patch[min_index].rows, sub_patch[min_index].cols, sub_patch[min_index].depth()); //生成掩膜
        src_mask = 255 * src_mask;
        seamlessClone(sub_patch[min_index], image, src_mask, Point(x, y), image, NORMAL_CLONE); //泊松融合算法
        imshow("after effect", image);
    }

    if (event == EVENT_RBUTTONDOWN)
    {
        Mat dst;
        bilateralFilter(image, dst, 0, 50, 5);
        imshow("Dermabrasion effect", dst);
    }
}

int main()
{
    Mat image = imread("../blemish.png");
    namedWindow("input", WINDOW_AUTOSIZE);
    imshow("input", image);
    setMouseCallback("input", onMouse, &image);
    waitKey(0);
    return 0;
}