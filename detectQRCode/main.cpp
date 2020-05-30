#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

int main() {
    Mat image = imread("C:\\Users\\ucesxc0\\CLionProjects\\detectQRCode\\1.jpg");
    if (image.empty()) {
        cout << "Please confirm the filename" << endl;
        return -1;
    }
    Mat gray, qrcode_1;
    cvtColor(image, gray, COLOR_BGR2GRAY); // binary the image
    QRCodeDetector qrCodeDetector;
    vector<Point> points;
    string information;
    bool isQRCode;
    isQRCode = qrCodeDetector.detect(gray, points); // identify the QR code
    if (isQRCode) {
        information = qrCodeDetector.decode(gray, points, qrcode_1);
        cout << points << endl; //output the four coordination
    } else {
        cout << "Cannot identify the QR code, please input the correct format." << endl;
        return -1;
    }
    // draw the QRcode frame
    for (int i = 0; i < points.size(); i++) {
        if (i == points.size() - 1) {
            line(image, points[i], points[0], Scalar(0, 0, 255), 2, 8);
            break;
        }
        line(image, points[i], points[i + 1], Scalar(0, 0, 255), 2, 8);
    }
    //output the information
    putText(image, information.c_str(), Point(20, 30), 0, 1.0, Scalar(0, 0, 255), 2, 8);
    //output the images
    imshow("result", image);
    namedWindow("QRCode_identify", WINDOW_NORMAL);
    imshow("QRCode_identify", qrcode_1);
    waitKey(0);
    return 0;
}
