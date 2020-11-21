import cv2
import numpy as np

protoFile = "pose/coco/pose_deploy_linevec.prototxt"
weightsFile = "pose/coco/pose_iter_440000.caffemodel"
nPoints = 18

# COCO Output Format
keyPointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank',
                    'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 17], [5, 16]]

# index of PAFs corresponding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], [23, 24], [25, 26], [27, 28],
          [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], [55, 56], [37, 38], [45, 46]]

colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255],
          [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255],
          [0, 0, 255], [255, 0, 0], [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]


def initialize_network(preferable_backend=cv2.dnn.DNN_BACKEND_CUDA, preferable_target=cv2.dnn.DNN_TARGET_CUDA):
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    net.setPreferableBackend(preferable_backend)
    net.setPreferableTarget(preferable_target)

    return net


def get_key_points(prob_map, threshold=0.1):
    map_smooth = cv2.GaussianBlur(prob_map, (3, 3), 0, 0)

    map_mask = np.uint8(map_smooth > threshold)
    key_points = []

    # find the blobs
    contours, _ = cv2.findContours(map_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for each blob find the maxima
    for cnt in contours:
        blob_mask = np.zeros(map_mask.shape)
        blob_mask = cv2.fillConvexPoly(blob_mask, cnt, 1)
        masked_prob_map = map_smooth * blob_mask
        _, max_val, _, max_loc = cv2.minMaxLoc(masked_prob_map)
        key_points.append(max_loc + (prob_map[max_loc[1], max_loc[0]],))

    return key_points


# Find valid connections between the different joints of a all persons present
def get_valid_pairs(detected_key_points, output, frame_width, frame_height):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        paf_a = output[0, mapIdx[k][0], :, :]
        paf_b = output[0, mapIdx[k][1], :, :]
        paf_a = cv2.resize(paf_a, (frame_width, frame_height))
        paf_b = cv2.resize(paf_b, (frame_width, frame_height))

        # Find the key-points for the first and second limb
        candidate_a = detected_key_points[POSE_PAIRS[k][0]]
        candidate_b = detected_key_points[POSE_PAIRS[k][1]]
        n_a = len(candidate_a)
        n_b = len(candidate_b)

        # If key-points for the joint-pair is detected
        # check every joint in candidate_a with every joint in candidate_b
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid

        if n_a != 0 and n_b != 0:
            valid_pair = np.zeros((0, 3))
            for i in range(n_a):
                max_j = -1
                max_score = -1
                found = 0
                for j in range(n_b):
                    # Find d_ij
                    d_ij = np.subtract(candidate_b[j][:2], candidate_a[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candidate_a[i][0], candidate_b[j][0], num=n_interp_samples),
                                            np.linspace(candidate_a[i][1], candidate_b[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for m in range(len(interp_coord)):
                        paf_interp.append([paf_a[int(np.round(interp_coord[m][1])), int(np.round(interp_coord[m][0]))],
                                           paf_b[int(np.round(interp_coord[m][1])), int(np.round(interp_coord[m][0]))]])
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores) / len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                    if (len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples) > conf_th:
                        if avg_paf_score > max_score:
                            max_j = j
                            max_score = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:
                    valid_pair = np.append(valid_pair, [[candidate_a[i][3], candidate_b[max_j][3], max_score]], axis=0)

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else:  # If no key-points are detected
            # print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs


def get_person_wise_key_points(key_points_list, valid_pairs, invalid_pairs):
    # the last number in each row is the overall score
    person_wise_key_points = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            part_a = valid_pairs[k][:, 0]
            part_b = valid_pairs[k][:, 1]
            index_a, index_b = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(person_wise_key_points)):
                    if person_wise_key_points[j][index_a] == part_a[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    person_wise_key_points[person_idx][index_b] = part_b[i]
                    person_wise_key_points[person_idx][-1] += key_points_list[part_b[i].astype(int), 2] + \
                                                              valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[index_a] = part_a[i]
                    row[index_b] = part_b[i]
                    # add the keypoint_scores for the two key-points and the paf_score
                    row[-1] = sum(key_points_list[valid_pairs[k][i, :2].astype(int), 2]) + valid_pairs[k][i][2]
                    person_wise_key_points = np.vstack([person_wise_key_points, row])
    return person_wise_key_points
