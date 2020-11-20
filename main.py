import cv2
import time
import numpy as np

protoFile = "pose/coco/pose_deploy_linevec.prototxt"
weightsFile = "pose/coco/pose_iter_440000.caffemodel"
nPoints = 18
# COCO Output Format
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank',
                    'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
              [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
              [2, 17], [5, 16]]

# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44],
          [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],
          [47, 48], [49, 50], [53, 54], [51, 52], [55, 56],
          [37, 38], [45, 46]]

colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255],
          [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255],
          [0, 0, 255], [255, 0, 0], [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]


def getKeypoints(probMap, threshold=0.1):
    mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)

    mapMask = np.uint8(mapSmooth > threshold)
    keypoints = []

    # find the blobs
    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints


# Find valid connections between the different joints of a all persons present
def getValidPairs(output):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid

        if (nA != 0 and nB != 0):
            valid_pair = np.zeros((0, 3))
            for i in range(nA):
                max_j = -1
                maxScore = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))]])
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores) / len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                    if (len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples) > conf_th:
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else:  # If no keypoints are detected
            # print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs


def getPersonwiseKeypoints(valid_pairs, invalid_pairs):
    # the last number in each row is the overall score
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:, 0]
            partBs = valid_pairs[k][:, 1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][
                        2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score
                    row[-1] = sum(keypoints_list[valid_pairs[k][i, :2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints


def validate_actual_person(key_points):
    cont = 0
    for elem in key_points:
        if elem[0] != -1:
            cont += 1
    return cont > 5


def center_of_mass_4_person(keypoins):
    cont_top = 0
    center_top = [0, 0]
    top = [1, 2, 5]
    for index in top:
        if keypoins[index][0] != -1:
            cont_top += 1
            center_top[0] += keypoins[index][0]
            center_top[1] += keypoins[index][1]
    if cont_top < 1:
        return [-1, -1]
    center_top[0] /= cont_top
    center_top[1] /= cont_top
    cont_bottom = 0
    center_bottom = [0, 0]
    bottom = [8, 11]
    for index in bottom:
        if keypoins[index][0] != -1:
            cont_bottom += 1
            center_bottom[0] += keypoins[index][0]
            center_bottom[1] += keypoins[index][1]
    if cont_bottom < 1:
        return [-1, -1]
    center_bottom[0] /= cont_bottom
    center_bottom[1] /= cont_bottom
    return [(center_bottom[0] * 3 + center_top[0] * 2) / 5, (center_bottom[1] * 3 + center_top[1] * 2) / 5]


def estimate_height(keypoints):
    cont = 0
    r_sh_el_dist = 0
    l_sh_el_dist = 0
    l_ank_eye_dist = 0
    r_ank_eye_dist = 0
    r_ank_ear_dist = 0
    l_ank_ear_dist = 0
    if keypoints[2] != (-1, -1) and keypoints[3] != (-1, -1):
        r_sh_el_dist = np.sqrt((keypoints[2][0] - keypoints[3][0]) ** 2 + (keypoints[2][1] - keypoints[3][1]) ** 2) * 8
        cont += 1
    if keypoints[5] != (-1, -1) and keypoints[6] != (-1, -1):
        l_sh_el_dist = np.sqrt((keypoints[5][0] - keypoints[6][0]) ** 2 + (keypoints[5][1] - keypoints[6][1]) ** 2) * 8
        cont += 1
    if keypoints[13] != (-1, -1) and keypoints[15] != (-1, -1):
        l_ank_eye_dist = np.sqrt(
            (keypoints[13][0] - keypoints[15][0]) ** 2 + (keypoints[13][1] - keypoints[15][1]) ** 2)
        cont += 1
    if keypoints[13] != (-1, -1) and keypoints[17] != (-1, -1):
        l_ank_ear_dist = np.sqrt(
            (keypoints[13][0] - keypoints[17][0]) ** 2 + (keypoints[13][1] - keypoints[17][1]) ** 2)
        cont += 1
    if keypoints[10] != (-1, -1) and keypoints[14] != (-1, -1):
        r_ank_eye_dist = np.sqrt(
            (keypoints[10][0] - keypoints[14][0]) ** 2 + (keypoints[10][1] - keypoints[14][1]) ** 2)
        cont += 1
    if keypoints[10] != (-1, -1) and keypoints[16] != (-1, -1):
        l_ank_ear_dist = np.sqrt(
            (keypoints[10][0] - keypoints[16][0]) ** 2 + (keypoints[10][1] - keypoints[16][1]) ** 2)
        cont += 1
    if cont != 0:
        # print(r_sh_el_dist, l_sh_el_dist, l_ank_eye_dist, r_ank_eye_dist, r_ank_ear_dist, l_ank_ear_dist,(r_sh_el_dist+l_sh_el_dist+l_ank_eye_dist+r_ank_eye_dist+r_ank_ear_dist+l_ank_ear_dist)/cont)
        return (r_sh_el_dist + l_sh_el_dist + l_ank_eye_dist + r_ank_eye_dist + r_ank_ear_dist + l_ank_ear_dist) / cont
    else:
        return -1


def distance_beetween_2_points(x0, y0, x1, y1):
    return np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)


def map_persons(persons, found_persons):
    # persons= nr total de persoane distincte din video
    # found_persons= nr de persone distince din frame-ul curent
    avg_human_height = 1.6  # average human height is 168.4cm but our highest points are the ears so we "shaved off" 8.4 cm => 1.6m
    speed = 0  # measured in pixels per frame
    possibilities = []
    assignment = []
    for i in range(len(found_persons)):
        possibilities.append([])
        for j in range(len(persons)):
            possibilities[i].append([0, 0, 0])
            possibilities[i][j][0] = 1 - persons[j][0][len(persons[j][0]) - 1][20] / found_persons[i][20]
            possibilities[i][j][1] = 1 - persons[j][0][len(persons[j][0]) - 1][18][0] / found_persons[i][18][0]
            possibilities[i][j][2] = distance_beetween_2_points(persons[j][1][0], persons[j][1][1],
                                                                possibilities[i][j][0], possibilities[i][j][1])
    for i in range(len(possibilities)):
        assignment.append([100000, -1, -1])
    to_assign = len(found_persons)
    cont = 0
    while to_assign > 0:
        for i in range(len(possibilities)):
            for j in range(len(possibilities[i])):
                if possibilities[i][j][2] < assignment[i][0] and (
                        len([x[0] for x in assignment if x[2] == j]) == 0 or possibilities[i][j][2] < min(
                    [x[0] for x in assignment if x[2] == j])):
                    if assignment[i][0] == 100000:
                        to_assign -= 1
                    assignment[i][0] = possibilities[i][j][2]
                    assignment[i][1] = i
                    assignment[i][2] = j
        for i in range(len(assignment) - 1):
            for j in range(i + 1, len(assignment)):
                if assignment[i][2] == assignment[j][2] and assignment[i][2] != -1:
                    if assignment[i][0] > assignment[j][0]:
                        assignment[i][0] = 100000
                        assignment[i][1] = -1
                        assignment[i][2] = -1
                    else:
                        assignment[j][0] = 100000
                        assignment[j][1] = -1
                        assignment[j][2] = -1
                    to_assign += 1
        cont += 1
        if cont >= 10:
            break
    for i in range(len(assignment)):
        if assignment[i][2] == -1:
            persons.append([[found_persons[i]], [0, 0]])
        else:
            persons[assignment[i][2]][0].append(found_persons[assignment[i][1]])
            persons[assignment[i][2]][1][0] = possibilities[assignment[i][1]][assignment[i][2]][0]
            persons[assignment[i][2]][1][1] = possibilities[assignment[i][1]][assignment[i][2]][1]
    return persons


def statistical_approach(person_tracker, found_persons, frame_count):
    actual_persons = []
    center_of_mass = []
    for elem in found_persons:
        com = center_of_mass_4_person(elem)
        if validate_actual_person(elem) and com != [-1, -1] and estimate_height(elem) != -1:
            actual_persons.append(elem)
            center_of_mass.append(com)
    for i in range(len(actual_persons)):
        actual_persons[i].append(tuple(center_of_mass[i]))
        actual_persons[i].append(frame_count)  # needed for speed
        actual_persons[i].append(estimate_height(actual_persons[i]))  # needed for person velocity using vectors
    if len(person_tracker) == 0:
        person_tracker = [[[person], [0, 0]] for person in actual_persons]
    else:
        person_tracker = map_persons(person_tracker, actual_persons)
    return person_tracker


def person_tracking(person_tracker, person_keypoints, all_keypoints, frame_count):
    found_persons = []
    for person in person_keypoints:
        found_persons.append([(-1, -1)] * 18)
        for index in range(len(person) - 1):
            if person[index] != -1:
                for elem in all_keypoints[index]:
                    if elem[3] == person[index]:
                        found_persons[len(found_persons) - 1][index] = (elem[0], elem[1])
    person_tracker = statistical_approach(person_tracker, found_persons, frame_count)
    return person_tracker


def min_dist_between_persons_in_frame(frame_number, persons):
    distances = []
    for i in range(len(persons) - 1):
        distances.append([0, 0, 0, 0, 10000])
        if persons[i][0][len(persons[i][0]) - 1][19] == frame_number:
            for j in range(i + 1, len(persons)):
                if persons[j][0][len(persons[j][0]) - 1][19] == frame_number:
                    for x in range(18):
                        if persons[i][0][len(persons[i][0]) - 1][x] != (-1, -1):
                            for y in range(18):
                                if persons[j][0][len(persons[j][0]) - 1][y] != (-1, -1):
                                    if distance_beetween_2_points(persons[i][0][len(persons[i][0]) - 1][x][0],
                                                                  persons[i][0][len(persons[i][0]) - 1][x][1],
                                                                  persons[j][0][len(persons[j][0]) - 1][y][0],
                                                                  persons[j][0][len(persons[j][0]) - 1][y][1]) < \
                                            distances[i][4]:
                                        distances[i][4] = distance_beetween_2_points(
                                            persons[i][0][len(persons[i][0]) - 1][x][0],
                                            persons[i][0][len(persons[i][0]) - 1][x][1],
                                            persons[j][0][len(persons[j][0]) - 1][y][0],
                                            persons[j][0][len(persons[j][0]) - 1][y][1])
                                        distances[i][0] = i
                                        distances[i][1] = j
                                        distances[i][2] = x
                                        distances[i][3] = y
    return distances


def dist_between_persons_gravity_center(frame_number, persons):
    distances = []
    for i in range(len(persons) - 1):
        distances.append([0, 0, 10000])
        if persons[i][0][len(persons[i][0]) - 1][19] == frame_number:
            for j in range(i + 1, len(persons)):
                if persons[j][0][len(persons[j][0]) - 1][19] == frame_number:
                    distances[i][2] = distance_beetween_2_points(persons[i][0][len(persons[i][0]) - 1][18][0],
                                                                 persons[i][0][len(persons[i][0]) - 1][18][1],
                                                                 persons[j][0][len(persons[j][0]) - 1][18][0],
                                                                 persons[j][0][len(persons[j][0]) - 1][18][1])
                    distances[i][0] = i
                    distances[i][0] = j
    return distances


def close_persons(min_dist, persons, frameWidth, frameHeight):
    valid = []
    for elem in min_dist:
        if persons[elem[0]][0][len(persons[elem[0]][0]) - 1][20] / persons[elem[1]][0][len(persons[elem[1]][0]) - 1][
            20] > 0.5 and persons[elem[0]][0][len(persons[elem[0]][0]) - 1][20] / \
                persons[elem[1]][0][len(persons[elem[1]][0]) - 1][20] < 2:
            if elem[4] < 10000 and elem[4] < frameWidth * frameHeight / 26331:
                valid.append(elem)
    return valid


def get_max_wrist_speed(person):
    dist = 0
    last_appearance_of_right_wrist = (-1, -1)
    prev_last_appearance_of_right_wrist = (-1, -1)
    frame_of_last_r_w_a = -1
    frame_of_prev_last_r_w_a = -1
    last_appearance_of_left_wrist = (-1, -1)
    prev_last_appearance_of_left_wrist = (-1, -1)
    frame_of_last_l_w_a = -1
    frame_of_prev_last_l_w_a = -1
    frame = len(person) - 1
    while (frame_of_prev_last_l_w_a == -1 and frame_of_prev_last_r_w_a == -1) and frame >= 0:
        if last_appearance_of_right_wrist != (-1, -1) and prev_last_appearance_of_right_wrist == (-1, -1) and \
                person[frame][4] != (-1, -1):
            prev_last_appearance_of_right_wrist = person[frame][4]
            frame_of_prev_last_r_w_a = frame
        if person[frame][4] != (-1, -1) and last_appearance_of_right_wrist == (-1, -1):
            last_appearance_of_right_wrist = person[frame][4]
            frame_of_last_r_w_a = frame
        if last_appearance_of_left_wrist != (-1, -1) and prev_last_appearance_of_left_wrist == (-1, -1) and \
                person[frame][7] != (-1, -1):
            prev_last_appearance_of_left_wrist = person[frame][7]
            frame_of_prev_last_l_w_a = frame
        if person[frame][7] != (-1, -1) and last_appearance_of_left_wrist == (-1, -1):
            last_appearance_of_left_wrist = person[frame][7]
            frame_of_last_l_w_a = frame
        frame -= 1
    dist1 = -1
    dist2 = -1
    if frame_of_prev_last_r_w_a != -1:
        dist1 = distance_beetween_2_points(last_appearance_of_right_wrist[0], last_appearance_of_right_wrist[1],
                                           prev_last_appearance_of_right_wrist[0],
                                           prev_last_appearance_of_right_wrist[1]) / (
                            frame_of_last_r_w_a - frame_of_prev_last_r_w_a)
    if frame_of_prev_last_l_w_a != -1:
        dist2 = distance_beetween_2_points(last_appearance_of_left_wrist[0], last_appearance_of_left_wrist[1],
                                           prev_last_appearance_of_left_wrist[0],
                                           prev_last_appearance_of_left_wrist[1]) / (
                            frame_of_last_l_w_a - frame_of_prev_last_l_w_a)
    # print(last_appearance_of_right_wrist, prev_last_appearance_of_right_wrist, frame_of_last_r_w_a,
    #       frame_of_prev_last_r_w_a, last_appearance_of_left_wrist, prev_last_appearance_of_left_wrist,
    #       frame_of_last_l_w_a, frame_of_prev_last_l_w_a)
    return max([dist1, dist2])


def agitated_persons(persons, persons_filtered_by_proximity,frameWidth,frameHeight):
    to_return = []
    for elem in persons_filtered_by_proximity:
        potential_attacker = -1
        val = -1
        val1 = get_max_wrist_speed(persons[elem[0]][0])
        val2 = get_max_wrist_speed(persons[elem[1]][0])
        if val1 > val2:
            potential_attacker = elem[0]
            val = val1
        else:
            potential_attacker = elem[1]
            val = val2
        if val != -1 and val > frameWidth * frameHeight / 36864:
            print(val, potential_attacker)
    return to_return


video_file = cv2.VideoCapture("V_119.mp4")
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
# net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
person_tracker = []
frame_count = 0
while True:
    frame_count += 1
    # _, _ = video_file.read()
    # _, _ = video_file.read()
    # _, _ = video_file.read()
    ret, image1 = video_file.read()
    # cv2.imshow("Unaltered", image1)
    if not ret:
        break
    frameWidth = image1.shape[1]
    frameHeight = image1.shape[0]

    # Fix the input Height and get the width according to the Aspect Ratio
    inHeight = 368
    inWidth = int((inHeight / frameHeight) * frameWidth)
    inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)
    output = net.forward()

    detected_keypoints = []
    keypoints_list = np.zeros((0, 3))
    keypoint_id = 0
    threshold = 0.1
    for part in range(nPoints):
        probMap = output[0, part, :, :]
        probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
        keypoints = getKeypoints(probMap, threshold)
        # print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1
        detected_keypoints.append(keypoints_with_id)
    frameClone = image1.copy()
    for i in range(nPoints):
        for j in range(len(detected_keypoints[i])):
            cv2.circle(frameClone, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)

    valid_pairs, invalid_pairs = getValidPairs(output)
    personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs)
    person_tracker = person_tracking(person_tracker, personwiseKeypoints, detected_keypoints, frame_count)
    min_distances = min_dist_between_persons_in_frame(frame_count, person_tracker)
    distances_between_gravity_points = dist_between_persons_gravity_center(frame_count, person_tracker)
    persons_filtered_by_proximity = close_persons(min_distances, person_tracker, frameWidth, frameHeight)
    violent_persons = agitated_persons(person_tracker, persons_filtered_by_proximity,frameWidth,frameHeight)
    for i in range(17):
        for n in range(len(personwiseKeypoints)):
            index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
            if -1 in index:
                continue
            B = np.int32(keypoints_list[index.astype(int), 0])
            A = np.int32(keypoints_list[index.astype(int), 1])
            cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
    cv2.imshow("Detected Pose", frameClone)
    if cv2.waitKey(1) == ord('q'):
        break
video_file.release()
cv2.destroyAllWindows()
