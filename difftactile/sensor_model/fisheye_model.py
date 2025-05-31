"""
a class to model fisheye camera
"""

import numpy as np
import cv2
import taichi as ti


@ti.func
def project_3d_2d(a, f=8.627e-04, m=173913.04, cx=3.320e02, cy=2.400e02):
    # ref. Universal Semantic Segmentation for Fisheye Urban Driving Images Ye et al.
    # a is 3d vec
    a[2] += 2.0 * 0.01  # distance to the image plane
    a_norm = a.norm(1e-10)
    cos = a[2] / a_norm

    cos = ti.min(1.0, cos)
    cos = ti.max(-1.0, cos)
    theta = ti.acos(cos)
    omega = ti.atan2(a[1], a[0] + 1e-8) + ti.math.pi
    r = m * f * theta

    p = ti.Vector([0.0, 0.0])
    p[0] = r * ti.cos(omega) + cx
    p[1] = r * ti.sin(omega) + cy

    return p


def project_points_to_pix(a, f=8.627e-04, m=173913.04, cx=3.320e02, cy=2.400e02):
    # ref. Universal Semantic Segmentation for Fisheye Urban Driving Images Ye et al.
    # a is a point cloud if (n, 3)
    a[:, 2] += 2.0 * 0.01  # (14-0.7-9)* 0.01 # distance to the image plane
    b = np.array([[0.0, 0.0, 1.0]]).repeat(len(a), axis=0)
    inner_product = (a * b).sum(axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    cos = inner_product / (a_norm * b_norm)

    theta = np.arccos(cos)
    omega = np.arctan2(a[:, 1], a[:, 0]) + np.pi

    r = m * f * theta

    p = np.zeros((len(a), 2))
    p[:, 0] = r * np.cos(omega) + cx
    p[:, 1] = r * np.sin(omega) + cy

    return p


def get_marker_image(img):
    curve1 = 50
    curve2 = 100
    mask = img < curve1
    img1 = (curve2 / curve1) * img
    img2 = 255 - (255 - curve2) / (255 - curve1) * (255 - img)
    img = img1 * mask + img2 * (1 - mask)
    img = img.astype("uint8")

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.minThreshold = 0

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)


    MarkerCenter = []
    for pt in keypoints:
        MarkerCenter.append([pt.pt[0], pt.pt[1]])
    MarkerCenter = np.array(MarkerCenter)

    # filter out invalid markers
    if False:
        center_coordinates = np.array([320, 240])
        w_length = 150
        h_length = 100
        start_point = (320 - w_length, 240 - h_length)
        end_point = (320 + w_length, 240 + h_length)

        offset = np.abs(MarkerCenter[:, 0:2] - center_coordinates)
        valid_marker_mask = np.logical_and(offset[:, 0] < w_length, offset[:, 1] < h_length)
        MarkerCenter = MarkerCenter[valid_marker_mask]

    return MarkerCenter

if __name__ == '__main__':
    # Load the init.png image from the same directory
    img = cv2.imread("system-id-screws-3-reps-0001.png")
    if img is None:
        print("Error: Could not load system-id-screws-3-reps-0001.png")
    else:
        # Get marker positions from the image
        marker_positions = get_marker_image(img)
        print("Detected marker positions:")
        print(marker_positions)
        
        # Create a copy of the image for visualization
        vis_img = img.copy()
        
        # Draw detected markers
        for pos in marker_positions:
            # Convert positions to integers for drawing
            center = (int(pos[0]), int(pos[1]))
            # Draw a circle at each marker position (red color)
            cv2.circle(vis_img, center, radius=5, color=(0, 0, 255), thickness=2)
            
        # Display the image with detected markers
        cv2.imshow("Detected Markers", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()