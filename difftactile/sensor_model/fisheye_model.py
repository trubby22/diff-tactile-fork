"""
a class to model fisheye camera
"""

import numpy as np
import cv2
import taichi as ti
import pickle
from tqdm import tqdm  # For progress bar


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
    # Open the video file
    video_path = "system-id-screws-3-reps.mkv"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        exit()
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration_secs = total_frames / fps
    print(f"Video properties:")
    print(f"- Total frames: {total_frames}")
    print(f"- FPS: {fps}")
    print(f"- Duration: {duration_secs:.2f} seconds")
    
    # Initialize variables
    frames_per_batch = 100  # Number of frames to process before saving
    current_batch = []
    batch_number = 0
    pickle_file = "marker_positions.pkl"
    frames_to_skip = fps - 1  # We'll process 1 frame and skip (fps-1) frames
    
    # Process frames
    frame_idx = 0
    with tqdm(total=int(duration_secs)) as pbar:
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                print(f"Error reading frame {frame_idx}")
                break
                
            # Process one frame per second
            if frame_idx % fps == 0:  # This frame is on a second boundary
                # Get marker positions for this frame
                marker_positions = get_marker_image(frame)
                current_batch.append(marker_positions)
                
                # If we've collected enough frames, save the batch
                if len(current_batch) >= frames_per_batch or frame_idx >= total_frames - fps:
                    # Save batch to pickle file in append mode
                    with open(pickle_file, 'ab' if batch_number > 0 else 'wb') as f:
                        pickle.dump(current_batch, f)
                    
                    # Clear the current batch and increment batch number
                    current_batch = []
                    batch_number += 1
                    print(f"\nSaved batch {batch_number} to {pickle_file}")
                
                # Optional: Display progress
                vis_frame = frame.copy()
                # Draw detected markers
                for pos in marker_positions:
                    center = (int(pos[0]), int(pos[1]))
                    cv2.circle(vis_frame, center, radius=5, color=(0, 0, 255), thickness=2)
                
                # Display frame with markers
                cv2.imshow("Processing Video (1 fps)", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                    break
                    
                pbar.update(1)  # Update progress bar for each second processed
            
            # Skip frames to get to the next second
            for _ in range(min(frames_to_skip, total_frames - frame_idx - 1)):
                cap.read()  # Read and discard frames
            frame_idx += fps  # Jump to next second
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\nProcessing complete!")
    print(f"Results saved to {pickle_file} in {batch_number} batches")