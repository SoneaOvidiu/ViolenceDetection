from time import sleep

import numpy as np
import cv2 as cv
import openpose as op

# parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
#                                               The example file can be downloaded from: \
#                                               https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
# parser.add_argument('image', type=str, help='path to image file')
# args = parser.parse_args()

# cap = cv.VideoCapture(args.image)
cap = cv.VideoCapture("V_119-unmodifiable.mp4")

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_frame = old_frame[9:710, 443:836]
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

net = op.initialize_network()
FRAME_HEIGHT = 701
FRAME_WIDTH = 836 - 443
inHeight = 368
inWidth = int((inHeight / FRAME_HEIGHT) * FRAME_WIDTH)
inpBlob = cv.dnn.blobFromImage(old_frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)
output = net.forward()

detected_key_points = []
key_points_list = np.zeros((0, 3))
keypoint_id = 0
threshold = 0.1
for part in range(op.nPoints):
    probMap = output[0, part, :, :]
    probMap = cv.resize(probMap, (old_frame.shape[1], old_frame.shape[0]))
    key_points = op.get_key_points(probMap, threshold)
    key_points_with_id = []
    for o in range(len(key_points)):
        key_points_with_id.append(key_points[o] + (keypoint_id,))
        key_points_list = np.vstack([key_points_list, key_points[o]])
        keypoint_id += 1
    detected_key_points.append(key_points_with_id)
frameClone = old_frame.copy()
for o in range(op.nPoints):
    for m in range(len(detected_key_points[o])):
        cv.circle(frameClone, detected_key_points[o][m][0:2], 5, op.colors[o], -1, cv.LINE_AA)

p0 = []
for key_point in detected_key_points:
    for person_kp in key_point:
        p0.append(np.matrix(person_kp[0:2]).astype(np.float32))
p0 = np.array(p0)
print(type(cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)[0]))
print(type(p0[0]))

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while 1:
    ret, frame = cap.read()
    frame = frame[9:710, 443:836]
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv.circle(frame, (a, b), 5, color[i].tolist(), -1)
    img = cv.add(frame, mask)

    sleep(0.5)

    cv.imshow('frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
