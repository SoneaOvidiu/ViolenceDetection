import json
import os
from math import floor, ceil

import cv2
import numpy as np
# STEP 1
import openpose as op


# STEP 2
def validate_actual_person(key_points_to_validate):
    cont = 0
    for elem in key_points_to_validate:
        if elem[0] != -1:
            cont += 1
    return cont > 5


# STEP 2
def center_of_mass_for_person(key_points_for_person):
    cont_top = 0
    center_top = [0, 0]
    top = [1, 2, 5]
    for i in top:
        if key_points_for_person[i][0] != -1:
            cont_top += 1
            center_top[0] += key_points_for_person[i][0]
            center_top[1] += key_points_for_person[i][1]
    if cont_top < 1:
        return [-1, -1]
    center_top[0] /= cont_top
    center_top[1] /= cont_top
    cont_bottom = 0
    center_bottom = [0, 0]
    bottom = [8, 11]
    for i in bottom:
        if key_points_for_person[i][0] != -1:
            cont_bottom += 1
            center_bottom[0] += key_points_for_person[i][0]
            center_bottom[1] += key_points_for_person[i][1]
    if cont_bottom < 1:
        return [-1, -1]
    center_bottom[0] /= cont_bottom
    center_bottom[1] /= cont_bottom
    return [(center_bottom[0] * 3 + center_top[0] * 2) / 5, (center_bottom[1] * 3 + center_top[1] * 2) / 5]


# STEP 2
def estimate_height(key_points_for_person):
    cont = 0
    r_sh_el_dist = 0
    l_sh_el_dist = 0
    l_ank_eye_dist = 0
    r_ank_eye_dist = 0
    r_ank_ear_dist = 0
    l_ank_ear_dist = 0
    if key_points_for_person[2] != (-1, -1) and key_points_for_person[3] != (-1, -1):
        r_sh_el_dist = np.sqrt((key_points_for_person[2][0] - key_points_for_person[3][0]) ** 2 + (
                key_points_for_person[2][1] - key_points_for_person[3][1]) ** 2) * 8
        cont += 1
    if key_points_for_person[5] != (-1, -1) and key_points_for_person[6] != (-1, -1):
        l_sh_el_dist = np.sqrt((key_points_for_person[5][0] - key_points_for_person[6][0]) ** 2 + (
                key_points_for_person[5][1] - key_points_for_person[6][1]) ** 2) * 8
        cont += 1
    if key_points_for_person[13] != (-1, -1) and key_points_for_person[15] != (-1, -1):
        l_ank_eye_dist = np.sqrt((key_points_for_person[13][0] - key_points_for_person[15][0]) ** 2 + (
                key_points_for_person[13][1] - key_points_for_person[15][1]) ** 2)
        cont += 1
    if key_points_for_person[13] != (-1, -1) and key_points_for_person[17] != (-1, -1):
        l_ank_ear_dist = np.sqrt((key_points_for_person[13][0] - key_points_for_person[17][0]) ** 2 + (
                key_points_for_person[13][1] - key_points_for_person[17][1]) ** 2)
        cont += 1
    if key_points_for_person[10] != (-1, -1) and key_points_for_person[14] != (-1, -1):
        r_ank_eye_dist = np.sqrt((key_points_for_person[10][0] - key_points_for_person[14][0]) ** 2 + (
                key_points_for_person[10][1] - key_points_for_person[14][1]) ** 2)
        cont += 1
    if key_points_for_person[10] != (-1, -1) and key_points_for_person[16] != (-1, -1):
        l_ank_ear_dist = np.sqrt((key_points_for_person[10][0] - key_points_for_person[16][0]) ** 2 + (
                key_points_for_person[10][1] - key_points_for_person[16][1]) ** 2)
        cont += 1
    if cont != 0:
        # print(r_sh_el_dist, l_sh_el_dist, l_ank_eye_dist, r_ank_eye_dist, r_ank_ear_dist, l_ank_ear_dist,
        # (r_sh_el_dist+l_sh_el_dist+l_ank_eye_dist+r_ank_eye_dist+r_ank_ear_dist+l_ank_ear_dist)/cont)
        return (r_sh_el_dist + l_sh_el_dist + l_ank_eye_dist + r_ank_eye_dist + r_ank_ear_dist + l_ank_ear_dist) / cont
    else:
        return -1


# STEP 2
def distance_between_2_points(x0, y0, x1, y1):
    return np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)


# STEP 2
def map_persons(persons, found_persons):
    # persons = total no of distinct persons in the video (up to the current frame)
    # found_persons = number of distinct persons in the current frame

    # average human height is 168.4cm but our highest points are the ears so we "shaved off" 8.4 cm => 1.6m
    # avg_human_height = 1.6
    # speed measured in pixels per frame
    # speed = 0
    possibilities = []
    assignment = []
    for i in range(len(found_persons)):
        possibilities.append([])
        for j in range(len(persons)):
            possibilities[i].append([0, 0, 0])
            possibilities[i][j][0] = 1 - persons[j][0][len(persons[j][0]) - 1][20] / found_persons[i][20]
            possibilities[i][j][1] = 1 - persons[j][0][len(persons[j][0]) - 1][18][0] / found_persons[i][18][0]
            possibilities[i][j][2] = distance_between_2_points(persons[j][1][0], persons[j][1][1],
                                                               possibilities[i][j][0], possibilities[i][j][1])
    for i in range(len(possibilities)):
        assignment.append([100000, -1, -1])
    to_assign = len(found_persons)
    cont = 0
    while to_assign > 0:
        for i in range(len(possibilities)):
            for j in range(len(possibilities[i])):
                if possibilities[i][j][2] < assignment[i][0] and (
                        len([x[0] for x in assignment if x[2] == j]) == 0 or possibilities[i][j][2] <
                        min([x[0] for x in assignment if x[2] == j])):
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


# STEP 2
def statistical_approach(person_tracker, found_persons, frame_count):
    actual_persons = []
    center_of_mass = []
    for elem in found_persons:
        com = center_of_mass_for_person(elem)
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


# STEP 2
def person_tracking(person_tracker, person_key_points, all_key_points, frame_count):
    found_persons = []
    for person in person_key_points:
        found_persons.append([(-1, -1)] * 18)
        for index in range(len(person) - 1):
            if person[index] != -1:
                for elem in all_key_points[index]:
                    if elem[3] == person[index]:
                        found_persons[len(found_persons) - 1][index] = (elem[0], elem[1])
    person_tracker = statistical_approach(person_tracker, found_persons, frame_count)
    return person_tracker


# STEP 3
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
                                    x1 = persons[i][0][len(persons[i][0]) - 1][x][0]
                                    x2 = persons[i][0][len(persons[i][0]) - 1][x][1]
                                    y1 = persons[j][0][len(persons[j][0]) - 1][y][0]
                                    y2 = persons[j][0][len(persons[j][0]) - 1][y][1]

                                    if distance_between_2_points(x1, x2, y1, y2) < distances[i][4]:
                                        distances[i][4] = distance_between_2_points(x1, x2, y1, y2)
                                        distances[i][0] = i
                                        distances[i][1] = j
                                        distances[i][2] = x
                                        distances[i][3] = y
    return distances


# STEP 3
def dist_between_centers_of_mass_for_persons(frame_number, persons):
    distances = []
    for i in range(len(persons) - 1):
        distances.append([0, 0, 10000])
        if persons[i][0][len(persons[i][0]) - 1][19] == frame_number:
            for j in range(i + 1, len(persons)):
                if persons[j][0][len(persons[j][0]) - 1][19] == frame_number:
                    distances[i][2] = distance_between_2_points(persons[i][0][len(persons[i][0]) - 1][18][0],
                                                                persons[i][0][len(persons[i][0]) - 1][18][1],
                                                                persons[j][0][len(persons[j][0]) - 1][18][0],
                                                                persons[j][0][len(persons[j][0]) - 1][18][1])
                    distances[i][0] = i
                    distances[i][0] = j
    return distances


# STEP 3
def get_close_persons(min_dist, persons, frame_width, frame_height):
    valid = []
    for elem in min_dist:
        if 0.5 < persons[elem[0]][0][len(persons[elem[0]][0]) - 1][20] / \
                persons[elem[1]][0][len(persons[elem[1]][0]) - 1][20] < 2:
            if elem[4] < 10000 and elem[4] < frame_width * frame_height / 26331:
                valid.append(elem)
    return valid


# STEP 3
def get_max_wrist_speed(person):
    last_seen_right_wrist = (-1, -1)
    last_seen_left_wrist = (-1, -1)
    prev_last_seen_right_wrist = (-1, -1)
    prev_last_seen_left_wrist = (-1, -1)
    frame_last_r_w = -1
    frame_last_l_w = -1
    frame_prev_last_r_w = -1
    frame_prev_last_l_w = -1
    frame = len(person) - 1
    while (frame_prev_last_l_w == -1 and frame_prev_last_r_w == -1) and frame >= 0:
        right_wrist_current_frame = person[frame][4]
        left_wrist_current_frame = person[frame][7]
        if last_seen_right_wrist != (-1, -1) and prev_last_seen_right_wrist == (-1, -1) and \
                right_wrist_current_frame != (-1, -1):
            prev_last_seen_right_wrist = right_wrist_current_frame
            frame_prev_last_r_w = frame
        if right_wrist_current_frame != (-1, -1) and last_seen_right_wrist == (-1, -1):
            last_seen_right_wrist = right_wrist_current_frame
            frame_last_r_w = frame
        if last_seen_left_wrist != (-1, -1) and prev_last_seen_left_wrist == (-1, -1) and \
                left_wrist_current_frame != (-1, -1):
            prev_last_seen_left_wrist = left_wrist_current_frame
            frame_prev_last_l_w = frame
        if left_wrist_current_frame != (-1, -1) and last_seen_left_wrist == (-1, -1):
            last_seen_left_wrist = left_wrist_current_frame
            frame_last_l_w = frame
        frame -= 1

    dist1 = -1
    dist2 = -1
    if frame_prev_last_r_w != -1:
        dist1 = distance_between_2_points(last_seen_right_wrist[0], last_seen_right_wrist[1],
                                          prev_last_seen_right_wrist[0], prev_last_seen_right_wrist[1])
        dist1 /= (frame_last_r_w - frame_prev_last_r_w)
    if frame_prev_last_l_w != -1:
        dist2 = distance_between_2_points(last_seen_left_wrist[0], last_seen_left_wrist[1],
                                          prev_last_seen_left_wrist[0], prev_last_seen_left_wrist[1])
        dist2 /= (frame_last_l_w - frame_prev_last_l_w)
    # print(last_seen_right_wrist, prev_last_seen_right_wrist, frame_last_r_w,
    #       frame_prev_last_r_w, last_seen_left_wrist, prev_last_seen_left_wrist,
    #       frame_last_l_w, frame_prev_last_l_w)
    return max([dist1, dist2])


# STEP 3
def get_agitated_persons(persons, close_persons, frame_width, frame_height):
    to_return = []
    for elem in close_persons:
        val1 = get_max_wrist_speed(persons[elem[0]][0])
        val2 = get_max_wrist_speed(persons[elem[1]][0])

        # the faster of the two close persons is marked as a possible threat
        potential_attacker = elem[0] if val1 > val2 else elem[1]
        val = max(val1, val2)
        if val != -1 and val > frame_width * frame_height / 36864:
            to_return.append([potential_attacker, val])
    return to_return


# def test_on_119():
#     net = op.initialize_network()
#
#     INPUT_FILE_NAME = "V_119"
#     INPUT_VIDEO = cv2.VideoCapture("videos/" + INPUT_FILE_NAME + ".mp4")
#     FRAME_WIDTH = int(INPUT_VIDEO.get(3))
#     FRAME_HEIGHT = int(INPUT_VIDEO.get(4))
#     OUTPUT_VIDEO = cv2.VideoWriter("videos/" + INPUT_FILE_NAME + "_out.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 10,
#                                    (FRAME_WIDTH, FRAME_HEIGHT))
#
#     PERSON_TRACKER = []
#     FRAME_COUNT = 0
#
#     while INPUT_VIDEO.isOpened():
#         FRAME_COUNT += 1
#         # _, _ = video_file.read()
#         ret, image1 = INPUT_VIDEO.read()
#         # cv2.imshow("Unaltered", image1)
#         if not ret:
#             break
#
#         # Fix the input Height and get the width according to the Aspect Ratio
#         inHeight = 368
#         inWidth = int((inHeight / FRAME_HEIGHT) * FRAME_WIDTH)
#         inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
#
#         net.setInput(inpBlob)
#         output = net.forward()
#
#         detected_key_points = []
#         key_points_list = np.zeros((0, 3))
#         keypoint_id = 0
#         threshold = 0.1
#         for part in range(op.nPoints):
#             probMap = output[0, part, :, :]
#             probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
#             key_points = op.get_key_points(probMap, threshold)
#             key_points_with_id = []
#             for o in range(len(key_points)):
#                 key_points_with_id.append(key_points[o] + (keypoint_id,))
#                 key_points_list = np.vstack([key_points_list, key_points[o]])
#                 keypoint_id += 1
#             detected_key_points.append(key_points_with_id)
#         frameClone = image1.copy()
#         for o in range(op.nPoints):
#             for m in range(len(detected_key_points[o])):
#                 cv2.circle(frameClone, detected_key_points[o][m][0:2], 5, op.colors[o], -1, cv2.LINE_AA)
#
#         valid_pairs, invalid_pairs = op.get_valid_pairs(detected_key_points, output, FRAME_WIDTH, FRAME_HEIGHT)
#         person_wise_key_points = op.get_person_wise_key_points(key_points_list, valid_pairs, invalid_pairs)
#
#         PERSON_TRACKER = person_tracking(PERSON_TRACKER, person_wise_key_points, detected_key_points, FRAME_COUNT)
#
#         min_distances_between_persons = min_dist_between_persons_in_frame(FRAME_COUNT, PERSON_TRACKER)
#         distances_between_centers_of_mass = dist_between_centers_of_mass_for_persons(FRAME_COUNT, PERSON_TRACKER)
#
#         persons_filtered_by_proximity = get_close_persons(min_distances_between_persons, PERSON_TRACKER,
#                                                           FRAME_WIDTH, FRAME_HEIGHT)
#         possible_violent_persons = get_agitated_persons(PERSON_TRACKER, persons_filtered_by_proximity,
#                                                         FRAME_WIDTH, FRAME_HEIGHT)
#
#         possible_violent_persons_ids = [p[0] for p in possible_violent_persons]
#         print("Violent persons in frame {}: {}".format(FRAME_COUNT, possible_violent_persons_ids))
#         for o in range(17):
#             for person_id in range(len(person_wise_key_points)):
#                 INDEX = person_wise_key_points[person_id][np.array(op.POSE_PAIRS[o])]
#                 if -1 in INDEX:
#                     continue
#                 B = np.int32(key_points_list[INDEX.astype(int), 0])
#                 A = np.int32(key_points_list[INDEX.astype(int), 1])
#                 if o == 0 and person_id in possible_violent_persons_ids:
#                     bg_width = 100
#                     bg_height = 35
#                     cv2.rectangle(frameClone, (B[0], A[0]), (B[0] + bg_width, A[0] + bg_height), (0, 0, 255), -1)
#                     cv2.putText(frameClone, "Threat", (B[0] + int(bg_width / 10), A[0] + int(bg_height / 1.5)),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
#                     # Save the frame containing a threat
#                     if INPUT_FILE_NAME not in os.listdir("videos"):
#                         os.mkdir("videos/" + INPUT_FILE_NAME)
#                     cv2.imwrite("videos/" + INPUT_FILE_NAME + "/" + INPUT_FILE_NAME + "_" + str(FRAME_COUNT) + ".jpg",
#                                 frameClone)
#                 # Uncomment to draw lines corresponding to the body of the persons:
#                 # cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), pn.colors[o], 3, cv2.LINE_AA)
#
#         # Uncomment to save processed video to file:
#         # OUTPUT_VIDEO.write(frameClone)
#
#         # Uncomment to display annotated frames:
#         cv2.imshow("Detected Pose", frameClone)
#         if cv2.waitKey(1) == ord('q'):
#             break
#
#     INPUT_VIDEO.release()
#     OUTPUT_VIDEO.release()
#     cv2.destroyAllWindows()


def compare_predictions(ground_truth, predictions):
    tp = 0
    fp = 0
    fn = 0

    for prediction in predictions:
        found = False
        for truth in ground_truth:
            if truth[0] <= prediction <= truth[1]:
                tp += 1
                found = True
                break
        if not found:
            fp += 1

    for truth in ground_truth:
        found = False
        for prediction in predictions:
            if truth[0] <= prediction <= truth[1]:
                found = True
                break
        if not found:
            fn += 1

    return tp, fp, fn


if __name__ == "__main__":
    net = op.initialize_network()

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    file_names = [name[:-5] for name in os.listdir("CCTV") if name.endswith(".mpeg")]
    with open('ground-truth.json') as json_file:
        data = json.load(json_file)['database']
        annotations = []
        for n in file_names:
            ann = data[n]['annotations']
            annotations.append((n, [(floor(s['segment'][0]), ceil(s['segment'][1])) for s in ann]))
            print(annotations[-1])

    for video_index in range(len(file_names)):
        INPUT_FILE_NAME = file_names[video_index]
        INPUT_VIDEO = cv2.VideoCapture("CCTV/" + INPUT_FILE_NAME + ".mpeg")
        FRAME_WIDTH = int(INPUT_VIDEO.get(3))
        FRAME_HEIGHT = int(INPUT_VIDEO.get(4))
        FPS = INPUT_VIDEO.get(cv2.CAP_PROP_FPS)

        PERSON_TRACKER = []
        FRAME_COUNT = 0

        times_with_violent_persons = []

        while INPUT_VIDEO.isOpened():
            FRAME_COUNT += 1
            ret, image1 = INPUT_VIDEO.read()
            if not ret:
                break
            if (FRAME_COUNT - 1) % 5 == 0:
                # Fix the input Height and get the width according to the Aspect Ratio
                inHeight = 368
                inWidth = int((inHeight / FRAME_HEIGHT) * FRAME_WIDTH)
                inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

                net.setInput(inpBlob)
                output = net.forward()

                detected_key_points = []
                key_points_list = np.zeros((0, 3))
                keypoint_id = 0
                threshold = 0.1
                for part in range(op.nPoints):
                    probMap = output[0, part, :, :]
                    probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
                    key_points = op.get_key_points(probMap, threshold)
                    key_points_with_id = []
                    for o in range(len(key_points)):
                        key_points_with_id.append(key_points[o] + (keypoint_id,))
                        key_points_list = np.vstack([key_points_list, key_points[o]])
                        keypoint_id += 1
                    detected_key_points.append(key_points_with_id)
                frameClone = image1.copy()
                for o in range(op.nPoints):
                    for m in range(len(detected_key_points[o])):
                        cv2.circle(frameClone, detected_key_points[o][m][0:2], 5, op.colors[o], -1, cv2.LINE_AA)

                valid_pairs, invalid_pairs = op.get_valid_pairs(detected_key_points, output, FRAME_WIDTH, FRAME_HEIGHT)
                person_wise_key_points = op.get_person_wise_key_points(key_points_list, valid_pairs, invalid_pairs)

                PERSON_TRACKER = person_tracking(PERSON_TRACKER, person_wise_key_points, detected_key_points, FRAME_COUNT)

                min_distances_between_persons = min_dist_between_persons_in_frame(FRAME_COUNT, PERSON_TRACKER)
                distances_between_centers_of_mass = dist_between_centers_of_mass_for_persons(FRAME_COUNT, PERSON_TRACKER)

                persons_filtered_by_proximity = get_close_persons(min_distances_between_persons, PERSON_TRACKER,
                                                                  FRAME_WIDTH, FRAME_HEIGHT)
                possible_violent_persons = get_agitated_persons(PERSON_TRACKER, persons_filtered_by_proximity,
                                                                FRAME_WIDTH, FRAME_HEIGHT)

                possible_violent_persons_ids = [p[0] for p in possible_violent_persons]
                print("Violent persons in frame {}: {}".format(FRAME_COUNT, possible_violent_persons_ids))

                if len(possible_violent_persons_ids):
                    times_with_violent_persons.append(FRAME_COUNT / FPS)

                for o in range(17):
                    for person_id in range(len(person_wise_key_points)):
                        INDEX = person_wise_key_points[person_id][np.array(op.POSE_PAIRS[o])]
                        if -1 in INDEX:
                            continue
                        B = np.int32(key_points_list[INDEX.astype(int), 0])
                        A = np.int32(key_points_list[INDEX.astype(int), 1])
                        if o == 0 and person_id in possible_violent_persons_ids:
                            bg_width = 100
                            bg_height = 35
                            cv2.rectangle(frameClone, (B[0], A[0]), (B[0] + bg_width, A[0] + bg_height), (0, 0, 255), -1)
                            cv2.putText(frameClone, "Threat", (B[0] + int(bg_width / 10), A[0] + int(bg_height / 1.5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                            # Save the frame containing a threat
                            if INPUT_FILE_NAME not in os.listdir("CCTV/threats"):
                                os.mkdir("CCTV/threats/" + INPUT_FILE_NAME)
                            cv2.imwrite("CCTV/threats/" + INPUT_FILE_NAME + "/" + INPUT_FILE_NAME + "_" + str(FRAME_COUNT)
                                        + ".jpg", frameClone)

                        # Uncomment to draw lines corresponding to the body of the persons:
                        # cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), pn.colors[o], 3, cv2.LINE_AA)

                # Uncomment to display annotated frames:
                # cv2.imshow("Detected Pose", frameClone)
                # if cv2.waitKey(1) == ord('q'):
                #     break

        INPUT_VIDEO.release()
        cv2.destroyAllWindows()

        print(times_with_violent_persons)
        TP, FP, FN = compare_predictions(annotations[video_index][1], times_with_violent_persons)
        print(TP, FP, FN)
        true_positives += TP
        false_positives += FP
        false_negatives += FN

    print("TP =", true_positives)
    print("FP =", false_positives)
    print("FN =", false_negatives)
    print("---")
    print("Precision", true_positives / (true_positives + false_positives))
    print("Recall", true_positives / (true_positives + false_negatives))
    print("F1-score", true_positives / (true_positives + (false_positives + false_negatives) / 2))
