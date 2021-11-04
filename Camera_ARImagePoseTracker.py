import cv2
import numpy as np
from numpy.core.fromnumeric import transpose
import CalibrationHelpers as calib
import glob
import time
import open3d as o3d


def Transform3D(intrinsics, points):
    n = len(points)

    fx = intrinsics[0][0]
    fy = intrinsics[1][1]
    cx = intrinsics[0][2]
    cy = intrinsics[1][2]

    M = []

    for point_idx in range(n):
        u, v = points[point_idx]
        x = [(u-cx)/fx, (v-cy)/fy, 1]
        M.append(x)


    M = np.array(M)
    return M 







def ReturnMRow(intrinsics, points1, points2, Rx1, Tx1):


    # your code here
    assert(len(points1) == len(points2))
    n = len(points1)

    fx = intrinsics[0][0]
    fy = intrinsics[1][1]
    cx = intrinsics[0][2]
    cy = intrinsics[1][2]

    M = np.array([])

    for point_idx in range(n):
        u1, v1 = points1[point_idx]
        u2, v2 = points2[point_idx]

        x_1 = [(u1-cx)/fx, (v1-cy)/fy, 1]
        x_2 = [(u2-cx)/fx, (v2-cy)/fy, 1]

        result_1 = np.matmul(Rx1, x_1)
        result_1= np.cross(x_2, result_1)
        result_1=result_1[:,np.newaxis]
        result_2 = np.cross(x_2, Tx1)
        result_2=result_2[:,np.newaxis]


        zero_left = np.zeros((3, point_idx))
        zero_right = np.zeros((3, (n - 1) - point_idx))

        m_row = np.append(zero_left, result_1, axis=1)

        m_row = np.append(m_row, zero_right, axis=1)
        m_row = np.append(m_row, result_2, axis=1)

        if(M.size == 0):
            M = m_row
        else:
            M = np.vstack((M, m_row))
    print(M.shape)
    return M

def FilterByEpipolarConstraint(intrinsics, matches, points1, points2, Rx1, Tx1,
                               threshold = 0.01):


    # your code here
    assert(len(points1) == len(points2))
    length = len(points1)

    mask = []

    fx = intrinsics[0][0]
    fy = intrinsics[1][1]
    cx = intrinsics[0][2]
    cy = intrinsics[1][2]

    # E = T x R
    essential_matrix = np.cross(Tx1,Rx1,axisa=0,axisb=0)

    for point_idx in range(length):
        u1, v1 = points1[point_idx]
        u2, v2 = points2[point_idx]

        x_1 = [(u1-cx)/fx, (v1-cy)/fy, 1]
        x_2 = [(u2-cx)/fx, (v2-cy)/fy, 1]

        # x2^T * E 
        result = np.matmul(np.transpose(x_2), essential_matrix)
        result = abs(np.matmul(result, x_1))
        # print(result)
        if(result <= threshold):
            # we want to keep the value
            mask.append(1)
        else:
            # we don't want this value 
            mask.append(0)
                
    # print(mask)
    return mask

# This function is yours to complete
# it should take in a set of 3d points and the intrinsic matrix
# rotation matrix(R) and translation vector(T) of a camera
# it should return the 2d projection of the 3d points onto the camera defined
# by the input parameters    
def ProjectPoints(points3d, new_intrinsics, R, T):
    
    # your code here!

    # multiply points3d with R

    points2d = []
    for points in points3d:
        # print(points)
        # print(T)
        result = np.matmul(R, points)
        result = np.add(result, T)
        result = np.matmul(new_intrinsics, result)
        z = result[-1]
        result = [i/z for i in result]
        points2d.append(result[:-1])


    points2d = np.array(points2d)
    return (points2d)
    
# This function will render a cube on an image whose camera is defined
# by the input intrinsics matrix, rotation matrix(R), and translation vector(T)
def renderCube(img_in, new_intrinsics, R, T):
    # Setup output image
    img = np.copy(img_in)

    # We can define a 10cm cube by 4 sets of 3d points
    # these points are in the reference coordinate frame
    scale = 0.1
    face1 = np.array([[0,0,0],[0,0,scale],[0,scale,scale],[0,scale,0]],
                     np.float32)
    face2 = np.array([[0,0,0],[0,scale,0],[scale,scale,0],[scale,0,0]],
                     np.float32)
    face3 = np.array([[0,0,scale],[0,scale,scale],[scale,scale,scale],
                      [scale,0,scale]],np.float32)
    face4 = np.array([[scale,0,0],[scale,0,scale],[scale,scale,scale],
                      [scale,scale,0]],np.float32)
    # using the function you write above we will get the 2d projected 
    # position of these points
    face1_proj = ProjectPoints(face1, new_intrinsics, R, T)
    # this function simply draws a line connecting the 4 points
    img = cv2.polylines(img, [np.int32(face1_proj)], True, 
                              tuple([255,0,0]), 3, cv2.LINE_AA) 
    # repeat for the remaining faces
    face2_proj = ProjectPoints(face2, new_intrinsics, R, T)
    img = cv2.polylines(img, [np.int32(face2_proj)], True, 
                              tuple([0,255,0]), 3, cv2.LINE_AA) 
    
    face3_proj = ProjectPoints(face3, new_intrinsics, R, T)
    img = cv2.polylines(img, [np.int32(face3_proj)], True, 
                              tuple([0,0,255]), 3, cv2.LINE_AA) 
    
    face4_proj = ProjectPoints(face4, new_intrinsics, R, T)
    img = cv2.polylines(img, [np.int32(face4_proj)], True, 
                              tuple([125,125,0]), 3, cv2.LINE_AA) 
    return img

# This function takes in an intrinsics matrix, and two sets of 2d points
# if a pose can be computed it returns true along with a rotation and 
# translation between the sets of points. 
# returns false if a good pose estimate cannot be found
def ComputePoseFromHomography(new_intrinsics, referencePoints, imagePoints):
    # compute homography using RANSAC, this allows us to compute
    # the homography even when some matches are incorrect
    homography, mask = cv2.findHomography(referencePoints, imagePoints, 
                                          cv2.RANSAC, 5.0)
    # check that enough matches are correct for a reasonable estimate
    # correct matches are typically called inliers
    MIN_INLIERS = 30
    if(sum(mask)>MIN_INLIERS):
        # given that we have a good estimate
        # decompose the homography into Rotation and translation
        # you are not required to know how to do this for this class
        # but if you are interested please refer to:
        # https://docs.opencv.org/master/d9/dab/tutorial_homography.html
        RT = np.matmul(np.linalg.inv(new_intrinsics), homography)
        norm = np.sqrt(np.linalg.norm(RT[:,0])*np.linalg.norm(RT[:,1]))
        RT = -1*RT/norm
        c1 = RT[:,0]
        c2 = RT[:,1]
        c3 = np.cross(c1,c2)
        T = RT[:,2]
        R = np.vstack((c1,c2,c3)).T
        W,U,Vt = cv2.SVDecomp(R)
        R = np.matmul(U,Vt)
        return True, R, T
    # return false if we could not compute a good estimate
    return False, None, None

# Load the reference image that we will try to detect in the webcam
# reference = cv2.imread('ARTrackerImage.jpg',0)
reference = cv2.imread('camera_ref_data/reference.png',0)
RES = 480
reference = cv2.resize(reference,(RES,RES))

# create the feature detector. This will be used to find and describe locations
# in the image that we can reliably detect in multiple images
feature_detector = cv2.BRISK_create(octaves=5)
# compute the features in the reference image
reference_keypoints, reference_descriptors = \
        feature_detector.detectAndCompute(reference, None)
# make image to visualize keypoints
keypoint_visualization = cv2.drawKeypoints(
        reference,reference_keypoints,outImage=np.array([]), 
        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# keypoint_visualization = cv2.drawKeypoints(
#         reference,reference_keypoints,outImage=np.array([]))

# display the image
cv2.imshow("Keypoints",keypoint_visualization)
# wait for user to press a key before proceeding
# cv2.waitKey(0)

# create the matcher that is used to compare feature similarity
# Brisk descriptors are binary descriptors (a vector of zeros and 1s)
# Thus hamming distance is a good measure of similarity        
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Load the camera calibration matrix
intrinsics, distortion, new_intrinsics, roi = \
        calib.LoadCalibrationData('camera_calibration_data')

# initialize video capture
# the 0 value should default to the webcam, but you may need to change this
# for your camera, especially if you are using a camera besides the default
# cap = cv2.VideoCapture(0)

ref_poses = []


feature_sets = []

images = glob.glob("camera_ref_data"+'/*.png')

for image in images: 
    current_frame = cv2.imread(image)
    # current_frame = cv2.resize(current_frame,(RES,RES))
    # undistort the current frame using the loaded calibration
    current_frame = cv2.undistort(current_frame, intrinsics, distortion, None,\
                                  new_intrinsics)
    # apply region of interest cropping
    x, y, w, h = roi
    current_frame = current_frame[y:y+h, x:x+w]
    
    # detect features in the current image
    current_keypoints, current_descriptors = \
        feature_detector.detectAndCompute(current_frame, None)

    # match the features from the reference image to the current image

    if current_descriptors is None:
        continue 

    matches = matcher.match(reference_descriptors, current_descriptors)
    feature_sets.append(set([i.queryIdx for i in matches]))


assert(len(feature_sets) != 0)

feature_dict = {}

for feature in feature_sets:
    for f in feature:
        if f in feature_dict:
            feature_dict[f] += 1
        else:
            feature_dict[f] = 1


feature_set = []
for i in feature_dict:
    if feature_dict[i] >= 4:
        feature_set.append(i)

# print(list(feature_set))

for image in images: 
    # # read the current frame from the webcam
    # ret, current_frame = cap.read()
    
    print("processing  " + image)

    # # ensure the image is valid
    # if not ret:
    #     print("Unable to capture video")
    #     break
    current_frame = cv2.imread(image)
    # current_frame = cv2.resize(current_frame,(RES,RES))
    # undistort the current frame using the loaded calibration
    current_frame = cv2.undistort(current_frame, intrinsics, distortion, None,\
                                  new_intrinsics)
    # apply region of interest cropping
    x, y, w, h = roi
    current_frame = current_frame[y:y+h, x:x+w]
    
    # detect features in the current image
    current_keypoints, current_descriptors = \
        feature_detector.detectAndCompute(current_frame, None)
        
    # match the features from the reference image to the current image

    if current_descriptors is None:
        print("skipping " + image)
        continue 

    # detector = cv2.ORB_create()
    # kp1, des1 = detector.detectAndCompute(current_frame, None)
    # kp2, des2 = detector.detectAndCompute(reference, None)

    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches = bf.match(des1, des2)
    # matches = matches[:40]
    # matches = sorted(matches, key = lambda x: x.distance)



    # print(reference_descriptors)
    matches = matcher.match(reference_descriptors, current_descriptors)
    # matches returns a vector where for each element there is a 
    # query index matched with a train index. I know these terms don't really
    # make sense in this context, all you need to know is that for us the 
    # query will refer to a feature in the reference image and train will
    # refer to a feature in the current image

    # inlier_mask = FilterByEpipolarConstraint(intrinsics, matches, , threshold=0.01)

    # # create a visualization of the matches between the reference and the
    # # current image
    # match_visualization = cv2.drawMatches(reference, reference_keypoints, current_frame,
    #                         current_keypoints, matches, 0, 
    #                         flags=
    #                         cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
    #                         matchesMask=inlier_mask)
    # cv2.imshow('matches',match_visualization)
    # time.sleep(1.5)

    matches_tmp = []

    for i in range(len(matches)):
        if matches[i].queryIdx in feature_set:
            matches_tmp.append(matches[i])

    matches = matches_tmp
 

    # set up reference points and image points
    # here we get the 2d position of all features in the reference image
    referencePoints = np.float32([reference_keypoints[m.queryIdx].pt \
                                  for m in matches])
    # # convert positions from pixels to meters
    SCALE = 0.1 # this is the scale of our reference image: 0.1m x 0.1m
    referencePoints = SCALE*referencePoints/RES
    
    imagePoints = np.float32([current_keypoints[m.trainIdx].pt \
                                  for m in matches])
    
    # compute homography
    if(len(referencePoints) < 4 or len(imagePoints) < 4 ):
        print("skipping " + image)
        continue

    ret, R, T = ComputePoseFromHomography(new_intrinsics,referencePoints,
                                          imagePoints)

    render_frame = current_frame
    if(ret):
        # compute the projection and render the cube
        if(len(ref_poses) == 0):
            ref_poses.append((R,T))
        else:
            R_1R = ref_poses[0][0]
            T_1R = ref_poses[0][1]

            R_new = np.matmul(R, np.transpose(R_1R))
            T_new = np.matmul(R, np.matmul(np.transpose(R_1R), T_1R))
            T_new = np.subtract(T, T_new)


            M = ReturnMRow(intrinsics, referencePoints, imagePoints, R_new, T_new)
            M = np.asmatrix(M)

            W,U,Vt = cv2.SVDecomp(M)  
            depths = Vt[-1,:]/Vt[-1,-1]
            # Remove last element of depth vector (not needed)
            depths = depths[:len(depths) - 1]
            transform = Transform3D(intrinsics, referencePoints)
            depths_result = transform*depths[:, np.newaxis]

            print(depths_result.shape)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(depths_result)
            o3d.visualization.draw_geometries([pcd])

            inlier_mask = FilterByEpipolarConstraint(intrinsics, matches, referencePoints, imagePoints, R_new, T_new, threshold=0.015)

            # create a visualization of the matches between the reference and the
            # current image
            match_visualization = cv2.drawMatches(reference, reference_keypoints, current_frame,
                                    current_keypoints, matches, 0, 
                                    flags=
                                    cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
                                    matchesMask=inlier_mask)

            cv2.imshow('matches', match_visualization)
            time.sleep(2)


            ref_poses.append((R_new, T_new))
            render_frame = renderCube(current_frame,new_intrinsics,R,T) 
    else:
        print("skipping " + image)
    
        
    # display the current image frame
    # cv2.imshow('frame', render_frame)
    time.sleep(1.5)

    k = cv2.waitKey(1)
    if k == 27 or k==113:  #27, 113 are ascii for escape and q respectively
        #exit
        break
# while(True):
#     pass
cv2.destroyAllWindows() 
# cap.release()

