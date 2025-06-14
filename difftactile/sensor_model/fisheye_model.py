"""
a class to model fisheye camera
"""

import numpy as np
import cv2
from os import path as osp
import os
import math
import taichi as ti

@ti.func
def project_3d_2d(a, f=8.627e-04, m=173913.04, cx=3.320e+02, cy=2.400e+02):
    #ref. Universal Semantic Segmentation for Fisheye Urban Driving Images Ye et al.
    #a is 3d vec
    a[2] += 2.0*0.01 # distance to the image plane
    a_norm = a.norm(1e-10)
    cos = a[2] / a_norm

    cos = ti.min(1.0, cos)
    cos = ti.max(-1.0, cos)
    theta = ti.acos(cos)
    omega = ti.atan2(a[1],a[0]+1e-8) + ti.math.pi
    r = m * f * theta

    p = ti.Vector([0.0, 0.0])
    p[0] = r * ti.cos(omega) + cx
    p[1] = r * ti.sin(omega) + cy

    return p

def project_points_to_pix(a, f=8.627e-04, m=173913.04, cx=3.320e+02, cy=2.400e+02):
    #ref. Universal Semantic Segmentation for Fisheye Urban Driving Images Ye et al.
    #a is a point cloud if (n, 3)
    a[:,2] += 2.0*0.01 #(14-0.7-9)* 0.01 # distance to the image plane
    b = np.array([[0., 0., 1.]]).repeat(len(a), axis=0)
    inner_product = (a * b).sum(axis=1)
    a_norm = np.linalg.norm(a,axis=1)
    b_norm = np.linalg.norm(b,axis=1)
    cos = inner_product / (a_norm * b_norm)

    theta = np.arccos(cos)
    omega = np.arctan2(a[:,1],a[:,0]) + np.pi

    r = m * f * theta

    p = np.zeros((len(a),2))
    p[:,0] = r * np.cos(omega) + cx
    p[:,1] = r * np.sin(omega) + cy

    return p

def get_marker_image(img):
    params = cv2.SimpleBlobDetector_Params()

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)

    # Circle parameters
    circle_center = np.array([359, 266])
    circle_radius = 180

    MarkerCenter = []
    for pt in keypoints:
        point = np.array([pt.pt[0], pt.pt[1]])
        # Calculate distance from point to circle center
        distance = np.linalg.norm(point - circle_center)
        # Only add points that are inside the circle
        if distance < circle_radius:
            MarkerCenter.append([pt.pt[0], pt.pt[1]])
    MarkerCenter = np.array(MarkerCenter)

    return MarkerCenter

if __name__ == '__main__':
    img = cv2.imread("./system-id-screws-3-reps-0001.png")
    marker_positions = get_marker_image(img)

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