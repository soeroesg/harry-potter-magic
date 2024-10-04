# code adapted from https://github.com/kaustubh-sadekar/Invisibility_Cloak

import cv2
import datetime
from pathlib import Path
import time
import numpy as np


def setBackground(img):
    print("Set background")
    global background
    background = img.copy()

def setWand(img):
    print("Set wand")
    global wand
    wand = img.copy()

def getAverageColor(patch):
    averageColor = patch.mean(axis=0).mean(axis=0) # float
    return np.round(averageColor).astype(np.uint8)


def getDominantColor(patch):
    # code adapted from https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv
    pixels = np.float32(patch.reshape(-1, 3))
    # We apply k-means clustering to create a palette with the most 
    # n_colors representative colors of the image.
    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    # And finally the dominant color is the palette color which occurs most frequently 
    # on the quantized image:
    dominantColor = palette[np.argmax(counts)] # float
    return np.round(dominantColor).astype(np.uint8)


def sampleCenterRect(img):
    height, width, channels = img.shape
    
    # 20 percent rectangle around the middle of the image
    top = np.uint32(height/2 - height*0.1)
    bottom = np.uint32(height/2 + height*0.1)
    left = np.uint32(width/2 - width*0.1)
    right = np.uint32(width/2 + width*0.1)
    global roi
    #roi = img.copy() # full image
    roi = img[top:bottom, left:right, :].copy()
    # average color of the patch
    targetColorBgr = getAverageColor(roi)
    # dominant color of the image
    #targetColorBgr = getDominantColor(roi)
    return targetColorBgr


def bgr2hsv(colorBgr):
    tmpImgBgr = np.uint8([[colorBgr]])
    tmpImgHsv = cv2.cvtColor(tmpImgBgr, cv2.COLOR_BGR2HSV)
    return tmpImgHsv[0,0]


recording = False
#fourcc = cv2.VideoWriter_fourcc('X','V','I','D'); videoExt='avi'
#fourcc = cv2.VideoWriter_fourcc('m','p','4','v'); videoExt='mp4' # mov
fourcc = cv2.VideoWriter_fourcc('M','J','P','G'); videoExt='avi'
#fourcc = cv2.VideoWriter_fourcc('X','2','6','4'); videoExt='avi'
outVideo = cv2.VideoWriter()
recordMode = 'cloak'
def toggleRecording():
    global recording
    recording = not recording
    if not recording:
        if outVideo.isOpened():
            outVideo.release()
        return
    videoFileName = f"varazslat_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.{videoExt}"
    videoFilePath = str(Path.home() / videoFileName)
    outVideo.open(videoFilePath, fourcc, 25.0, (img.shape[1], img.shape[0]))
    if outVideo.isOpened():
        print("Writing video file " + videoFilePath)
    else:
        print("Cannot open video file " + videoFilePath)


cap = cv2.VideoCapture(0)
time.sleep(4) # need to wait a bit until the camera can be opened



# capture background, but first wait a few frames for AE/AF/AWB camera control to settle
kNumFramesToSkipAtStartup = 60
background = None
roi = None
flip = False
wand = None

targetColorBgr = [0,255,0]
targetColorHsv = bgr2hsv(targetColorBgr)

count = 0
while (cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        print("Cannot read image from camera")
        break
    if flip:
        img = np.flip(img, axis=1)
    
    count += 1
    if count < kNumFramesToSkipAtStartup:
        continue
    if count == kNumFramesToSkipAtStartup:
        setBackground(img)
        setWand(img)


    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # TODO: does blur help at all?
    #blur_size = (11, 11)
    #hsv = cv2.GaussianBlur(hsv, blur_size, 3) # TODO: does it make sense to blur HSV?

    targetHue = targetColorHsv[0]
    targetSaturation = targetColorHsv[1]
    targetValue = targetColorHsv[2]
    kHueRadius = 10
    kMinValue = 25
    # note: HSV color space wraps around and both ends are red,
    # therefore we create two masks and merge them if the targetHue is close to red
    if targetHue < kHueRadius: # red in the lower end of the hue axis
        low = np.array([0, 120, kMinValue])
        high = np.array([targetHue, 255, 255])
        mask1 = cv2.inRange(hsv, low, high)
        #cv2.imshow("mask1", mask1)
        low = np.array([180 - kHueRadius + targetHue, 120, kMinValue])
        high = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, low, high)
        #cv2.imshow("mask2", mask2)
        # Addition of the two masks to generate the final mask.
        mask = mask1 + mask2
    elif targetHue > 170:  # red in the upper end of the hue axis
        low = np.array([0, 120, kMinValue])
        high = np.array([targetHue - 180 + kHueRadius, 255, 255])
        mask1 = cv2.inRange(hsv, low, high)
        #cv2.imshow("mask1", mask1)
        low = np.array([targetHue, 120, kMinValue])
        high = np.array([180, 255, 255]) # max HSV
        mask2 = cv2.inRange(hsv, low, high)
        #cv2.imshow("mask2", mask2)
        # Addition of the two masks to generate the final mask.
        mask = mask1 + mask2
    else: # not red
        low = np.array([targetHue - kHueRadius, 120, kMinValue])
        high = np.array([targetHue + kHueRadius, 255, 255])
        mask = cv2.inRange(hsv, low, high)

    # TODO: experiment with others
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7,7),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((7,7), np.uint8))
    cv2.imshow("mask", mask)

    maskBright = cv2.inRange(hsv, np.array([0, 0, 252]), np.array([180, 255, 255]))
    cv2.imshow("lightMask", maskBright)
    wand[np.where(maskBright==255)] = img[np.where(maskBright==255)]
    
    if background is not None:
        cv2.imshow("background", background)
    if roi is not None:
        cv2.imshow("ROI", roi)
    
    #mask_inv = cv2.bitwise_not(mask)
    #res1 = cv2.bitwise_and(img, img, mask=mask_inv)
    #res2 = cv2.bitwise_and(background, background, mask=mask)
    
    # Replacing pixels corresponding to cloak with the background pixels.
    magic = img.copy()
    magic[np.where(mask==255)] = background[np.where(mask==255)]

    cloak = np.zeros(img.shape, dtype=np.uint8)
    cloak[np.where(mask==255)] = img[np.where(mask==255)]
    msg =  'RGB: ' + str(targetColorBgr[2]) + ', ' + str(targetColorBgr[1]) +  ', ' + str(targetColorBgr[0])
    cv2.putText(cloak, msg, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
    msg =  'HSV: ' + str(targetColorHsv[0]) + ', ' + str(targetColorHsv[1]) +  ', ' + str(targetColorHsv[2])
    cv2.putText(cloak, msg, (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
    if recording:
        cv2.circle(cloak, (30,30), 20, (0,0,255), -1)

    if recording:
        if recordMode == 'wand':
            outVideo.write(wand)
        else:
            outVideo.write(magic)

    #merged=cv2.addWeighted(res1, 1, res2, 1,0)
    #out.write(merged)
    #cv2.imshow("merged", merged)
    cv2.imshow("varazslat", magic)
    cv2.imshow("kopeny", cloak)
    cv2.imshow("palca", wand)

    key = cv2.waitKey(1)
    if key == 27: # Esc: end
        break
    elif key == ord('q'): # end
        break
    elif key == ord('b'): # set background
        setBackground(img)
        setWand(img)
    elif key == ord('f'): # flip image for front camera
        flip = not flip
    elif key == ord('k'): # set cloak color from middle region of the current image
        targetColorBgr = sampleCenterRect(img)
        targetColorHsv = bgr2hsv(targetColorBgr)
    elif key == ord('r'): # toggle recording
        toggleRecording()
    elif key == ord('w'):
        if recordMode == 'cloak':
            recordMode = 'wand'
        else:
            recordMode = 'cloak'
    elif key == ord(' '): # save image
        imgFileName = f"varazslat_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.jpg"
        imgFilePath = str(Path.home() / imgFileName)
        cv2.imwrite(imgFilePath, magic )
    
cap.release()
cv2.destroyAllWindows()
