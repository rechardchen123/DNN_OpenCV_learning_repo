# base path to yolo directory 
MODEL_PATH = "yolo-coco"

#initialise minimum probability to filter weak detection along with
# the threshold when applying non-maxima suppression 
MIN_CONF = 0.3
NMS_THRESH = 0.3

# boolean indicating if NVIDIA CUDA GPU should be used
USE_GPU = False

# define the minimum safe distance that two people can be 
# from each other 
MIN_DISTANCE = 50

