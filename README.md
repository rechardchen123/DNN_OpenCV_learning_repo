- [<span id="head1"> DNN_OpenCV_learning_repo</span>](#head1)
	- [<span id="head2">face mask detection</span>](#head2)
	- [<span id="head3"> detectQRCode</span>](#head3)
	- [<span id="head4">face detector</span>](#head4)
	- [<span id="head5"> image_beautiful_filter</span>](#head5)
	- [<span id="head6">Image_correction </span>](#head6)
	- [<span id="head7"> remove_photo_flaw</span>](#head7)
	- [ social_distance_detector](#head8)
	
	
# <span id="head1"><span id="head1"> DNN_OpenCV_learning_repo</span></span>

This is an repository to learn the OpenCV and Deep Neural network.  This repo is used to implement some useful applications like `face mask detection`, `image beatiful filter` and `image correction`. It will be continuously updated. 

## <span id="head2"><span id="head2">face mask detection</span></span>

This is a face mask detection repository and it is derived and improved from the website [COVD-19:FACE Mask DETECTOR](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/).


The structure of the project is below:

```
.
├── dataset
│   ├── with_mask
│   └── without_mask
├── detect_mask_image.py
├── detect_mask_video.py
├── examples
│   ├── example_01.png
│   ├── example_02.png
│   ├── example_03.png
│   └── example_04.png
├── face_detector
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── LICENSE
├── mask_detector.model
├── plot.png
├── README.md
└── train_mask_detector.py

5 directories, 13 files
```

## <span id="head3"><span id="head3"> detectQRCode</span></span>
This is a QR code detection based on the OpenCV 4.2. The OpenCV 4.2 provides the function to detect. 

```
bool cv::QRCodeDetector::detect(InputArray  img,
                                OutputArray  points 
                                )

string cv::QRCodeDetector::decode(InputArray  img,
                                    InputArray  points,
                                    OutputArray  straight_qrcode = noArray() 
                                    )

string cv::QRCodeDetector::detectAndDecode(InputArray  img,
                                            OutputArray  points = noArray(),
                                            OutputArray  straight_qrcode = noArray() 
                                            )

```


The steps for the QR code detection are:
1. Image binary. 
2. Identify the QR code using QRCodeDetector.detect().
3. Using the QRCodeDetector.decode() to detect the QR code information.
4. Output the QR code information.

## <span id="head4"><span id="head4">face detector</span></span>
This is a face detector to use OpenCV to perform face recognition. To build the face recognition system, firstly it performs face detection, extract face embeddings from each fae using deep learning, train a face recognition model on the embeddings, and then finally recognize faces in both images and video streams with OpenCV. 

```
.
│  deploy.prototxt.txt
│  detect_faces.py
│  detect_faces_video.py
│  iron_chic.jpg
│  list.txt
│  res10_300x300_ssd_iter_140000.caffemodel
│  rooster.jpg
│  
└─
```

## <span id="head5"><span id="head5"> image_beautiful_filter</span></span>
This is an application using the OpenCV to revise the image called beautiful filter. 



## <span id="head6"><span id="head6">Image_correction </span></span>

The project contains image correction, text correction and the text correction has two methods. The method is derived from the [blog](https://www.cnblogs.com/skyfsm/p/6902524.html). 

The file contains:

```
│  1.png
│  2.png
│  3.jpg
│  CMakeLists.txt
│  image_correction.cpp
│  list.txt
│  pDstFileName.jpg
│  rectified.jpg
│  roated_image_1.jpg
│  rotated_image.jpg
│  text_correction.cpp
│  text_correction_FFT.cpp
```



**Image correction:**

1. image grayscale
2. the image binary
3. detect the contour
4. find the outline matrix and get the angle
5. rotation correction based on the angle
6. extract the contour of the rotated image
7. cut out the image area within the outline and output the image. 

The effect after the `image_correction` is shown below:

`raw image`：

![image-20200601051717054](README.assets/image-20200601051717054.png)

`binary_image`:

![image-20200601051753334](README.assets/image-20200601051753334.png)

`rotated_image`

![image-20200601051851314](README.assets/image-20200601051851314.png)

`output_image`

![image-20200601051909266](README.assets/image-20200601051909266.png)



**text_correction:**

1. image grayscale and binary
2. contour detection 
3. using the Hough transformation to detect the center of the image.
4. clip the image and rotate

`original image`

![image-20200601052610113](README.assets/image-20200601052610113.png)



`Hough transformation`

![image-20200601052646673](README.assets/image-20200601052646673.png)



`output`

![image-20200601052731871](README.assets/image-20200601052731871.png)



## <span id="head7"><span id="head7"> remove_photo_flaw</span></span>

The `remove_photo_flaw` is used to remove the image flaw. 

`raw image`

![image-20200601054047358](README.assets/image-20200601054047358.png)

`output`

![image-20200601054126502](README.assets/image-20200601054126502.png)

![image-20200601054137396](README.assets/image-20200601054137396.png)

## <span id="head8"> social_distance_detector</span>

The COVID-19 social distancing detector uses the OpenCV, Deep learning, and Computer vision. It is derived from the [Pyimageresearch](https://www.pyimagesearch.com/2020/06/01/opencv-social-distancing-detector/).

The step is as follow:

![image-20200603022212363](README.assets/image-20200603022212363.png)

1. apply object detection to detect all people in a video stream. 
2. compute the pairwise distances between all detected people.
3. based on these distances, check to see if any two people are less than `N` pixels apart. 

```
.
├── pyimagesearch
│   ├── __init__.py
│   ├── detection.py
│   └── social_distancing_config.py
├── yolo-coco
│   ├── coco.names
│   ├── yolov3.cfg
│   └── yolov3.weights
├── output.avi
├── pedestrians.mp4
└── social_distance_detector.py
2 directories, 9 files
```

The `yolo-coco` is saved in the [social_distance_detector](https://drive.google.com/drive/folders/1Qxs2m8hFMI8EIcrcxQDCzpYaM5g7Xb2V?usp=sharing).

`social_distancing_config.py`: a python file holding a number of constants in one convenient place. 

`detection.py`: YOLO object detection with OpenCV involves more lines of code that some easier model. 

`social_distance_detector.py`: This file is for looping over frames of a video stream and ensuring that people are maintaining a healthy distance from one another during a pandemic. It is compatible with both video files and webcam streams. 

