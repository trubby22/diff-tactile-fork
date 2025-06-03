import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from fisheye_model import get_marker_image
import tkinter as tk
from PIL import Image, ImageTk
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

class MarkerTracker:
    def __init__(self, video_path, output_path=None):
        """
        Initialize the marker tracker with input video path and optional output path.
        
        Args:
            video_path (str): Path to the input video file
            output_path (str, optional): Path for the output video file
        """
        self.video_path = Path(video_path)
        self.output_path = Path(output_path) if output_path else self.video_path.parent / f"{self.video_path.stem}_tracked.mkv"
        self.frame_markers = []  # List to store markers for each frame
        self.base_frame_mappings = []  # List to store mappings to frame 0
        self.frames = []  # List to store actual frames
        
        # LK optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
    def extract_frames(self):
        """Extract frames from video at 1.0 second intervals."""
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 1.0)  # Number of frames to skip for 1.0s interval
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.frames.append(frame)
                markers = get_marker_image(gray)
                if len(markers) > 0:  # Only append if markers were detected
                    self.frame_markers.append(markers)
                else:
                    self.frame_markers.append(np.array([]))
                
            frame_count += 1
            
        cap.release()
        
    def compute_base_frame_mappings(self):
        """
        Compute mappings between each frame and frame 0 using Lucas-Kanade optical flow
        with drift correction using blob detection and Hungarian algorithm matching.
        """
        if len(self.frames) < 2:
            return
            
        base_markers = self.frame_markers[0]
        if len(base_markers) == 0:
            return
            
        # Initialize tracking state
        prev_frame = cv2.cvtColor(self.frames[0], cv2.COLOR_BGR2GRAY)
        prev_pts = base_markers.reshape(-1, 1, 2).astype(np.float32)
        
        # Initialize identity mapping for frame 0
        identity_chain = [np.arange(len(base_markers))]  # Maps current indices to frame 0 indices
        
        for frame_idx in range(1, len(self.frames)):
            try:
                curr_frame = cv2.cvtColor(self.frames[frame_idx], cv2.COLOR_BGR2GRAY)
                curr_detected = self.frame_markers[frame_idx]
                
                # Handle empty detections
                if len(curr_detected) == 0:
                    print(f"No markers detected in frame {frame_idx}")
                    self.base_frame_mappings.append(np.array([]))
                    identity_chain.append(np.array([]))
                    prev_pts = np.array([]).reshape(-1, 1, 2).astype(np.float32)
                    prev_frame = curr_frame
                    continue
                
                # Ensure prev_pts is not empty and has correct shape
                if prev_pts.size == 0 or prev_pts.shape[0] == 0:
                    print(f"No previous points to track in frame {frame_idx}")
                    self.base_frame_mappings.append(np.full((len(curr_detected), 2), np.nan))
                    identity_chain.append(np.full(len(curr_detected), -1, dtype=int))
                    prev_pts = curr_detected.reshape(-1, 1, 2).astype(np.float32)
                    prev_frame = curr_frame
                    continue
                
                # Track points using LK optical flow
                curr_tracked, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_frame, curr_frame, prev_pts, None, **self.lk_params
                )
                
                # Filter out points where tracking failed
                good_tracked = curr_tracked[status.ravel() == 1]
                good_prev_idx = np.where(status.ravel() == 1)[0]
                
                if len(good_tracked) == 0:
                    print(f"No points successfully tracked in frame {frame_idx}")
                    self.base_frame_mappings.append(np.full((len(curr_detected), 2), np.nan))
                    identity_chain.append(np.full(len(curr_detected), -1, dtype=int))
                    prev_pts = curr_detected.reshape(-1, 1, 2).astype(np.float32)
                    prev_frame = curr_frame
                    continue
                
                # Reshape tracked points to 2D array for distance calculation
                good_tracked = good_tracked.reshape(-1, 2)
                
                # Match tracked points to detected points using Hungarian algorithm
                cost_matrix = cdist(curr_detected, good_tracked)
                
                if cost_matrix.size == 0:
                    print(f"Empty cost matrix in frame {frame_idx}")
                    self.base_frame_mappings.append(np.full((len(curr_detected), 2), np.nan))
                    identity_chain.append(np.full(len(curr_detected), -1, dtype=int))
                    prev_pts = curr_detected.reshape(-1, 1, 2).astype(np.float32)
                    prev_frame = curr_frame
                    continue
                
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                # Create mapping matrix
                mapping = np.full((len(curr_detected), 2), np.nan)
                new_identity = np.full(len(curr_detected), -1, dtype=int)
                
                for i, j in zip(row_ind, col_ind):
                    if cost_matrix[i, j] < 10:  # Only accept matches within reasonable distance
                        # Map through the chain to get frame 0 index
                        frame0_idx = identity_chain[-1][good_prev_idx[j]]
                        if frame0_idx >= 0:  # Valid chain to frame 0
                            mapping[i] = [frame0_idx, cost_matrix[i, j]]
                            new_identity[i] = frame0_idx
                
                # Update tracking state
                prev_frame = curr_frame.copy()
                prev_pts = curr_detected[row_ind].reshape(-1, 1, 2).astype(np.float32)
                
                # Store results
                self.base_frame_mappings.append(mapping)
                identity_chain.append(new_identity)
                
            except Exception as e:
                print(f"Warning: Optical flow tracking failed for frame {frame_idx}")
                print(f"Error details: {str(e)}")
                print(f"Shape of prev_pts: {prev_pts.shape}")
                print(f"Number of current detections: {len(curr_detected)}")
                self.base_frame_mappings.append(np.full((len(self.frame_markers[frame_idx]), 2), np.nan))
                identity_chain.append(np.array([]))
                
                # Reset tracking state
                if len(curr_detected) > 0:
                    prev_pts = curr_detected.reshape(-1, 1, 2).astype(np.float32)
                else:
                    prev_pts = np.array([]).reshape(-1, 1, 2).astype(np.float32)
                prev_frame = curr_frame
        
    def create_visualization(self):
        """Create visualization video with marker tracking."""
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        first_frame = self.frames[0]
        out = cv2.VideoWriter(str(self.output_path), fourcc, 2.0, 
                            (first_frame.shape[1], first_frame.shape[0]))
        
        # Process each frame
        for frame_idx in range(len(self.frames)):
            frame = self.frames[frame_idx].copy()
            base_frame = self.frames[0].copy()
            
            # Draw current frame markers
            for marker in self.frame_markers[frame_idx]:
                cv2.circle(frame, tuple(map(int, marker)), 5, (0, 255, 0), -1)
            
            # Draw base frame markers
            for marker in self.frame_markers[0]:
                cv2.circle(base_frame, tuple(map(int, marker)), 5, (255, 0, 0), -1)
            
            # Create blended image
            blended = cv2.addWeighted(frame, 0.7, base_frame, 0.3, 0)
            
            # Draw displacement arrows
            if frame_idx > 0:
                mapping = self.base_frame_mappings[frame_idx - 1]
                for i, map_entry in enumerate(mapping):
                    if not np.isnan(map_entry[0]):
                        start_point = tuple(map(int, self.frame_markers[0][int(map_entry[0])]))
                        end_point = tuple(map(int, self.frame_markers[frame_idx][i]))
                        cv2.arrowedLine(blended, start_point, end_point, (0, 0, 255), 2)
            
            out.write(blended)
            
        out.release()
        
    def process_video(self):
        """Process the video file through all steps."""
        print("Extracting frames...")
        self.extract_frames()
        print("Computing base frame mappings using Lucas-Kanade optical flow...")
        self.compute_base_frame_mappings()
        print("Creating visualization...")
        self.create_visualization()
        print("Processing complete!")
        
class VideoPlayer:
    def __init__(self, video_path):
        """Initialize video player with the given video path."""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(str(video_path))
        self.current_frame = 0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create GUI
        self.root = tk.Tk()
        self.root.title("Marker Tracking Viewer")
        
        # Create canvas for video display
        self.canvas = tk.Canvas(self.root)
        self.canvas.pack()
        
        # Add frame counter label
        self.frame_label = tk.Label(self.root, text=f"Frame: 0/{self.total_frames}")
        self.frame_label.pack()
        
        # Bind keyboard events
        self.root.bind('<Left>', self.prev_frame)
        self.root.bind('<Right>', self.next_frame)
        self.root.bind('<Escape>', self.quit)
        
        # Show first frame
        self.show_frame()
        
    def show_frame(self):
        """Display the current frame."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=img)
            
            # Update canvas size if needed
            self.canvas.config(width=img.width, height=img.height)
            
            # Update image
            self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.canvas.image = photo  # Keep reference
            
            # Update frame counter
            self.frame_label.config(text=f"Frame: {self.current_frame}/{self.total_frames}")
    
    def next_frame(self, event):
        """Show next frame."""
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.show_frame()
    
    def prev_frame(self, event):
        """Show previous frame."""
        if self.current_frame > 0:
            self.current_frame -= 1
            self.show_frame()
    
    def quit(self, event):
        """Close the video player."""
        self.root.quit()
        
    def run(self):
        """Start the video player."""
        self.root.mainloop()
        self.cap.release()

def process_and_view_video(input_path, output_path=None):
    """
    Process a video file and launch the interactive viewer.
    
    Args:
        input_path (str): Path to input video file
        output_path (str, optional): Path for output video file
    """
    # Process video
    tracker = MarkerTracker(input_path, output_path)
    tracker.process_video()
    
    # Launch viewer
    player = VideoPlayer(tracker.output_path)
    player.run()


if __name__ == '__main__':
    path = '/Users/piotrblaszyk/Documents/university/MRIGI/individual-project-70007/diff-tactile-fork/difftactile/sensor_model'
    process_and_view_video(f'{path}/system-id-screws-3-reps.mkv', f'{path}/marker-tracker.mkv')

    # player = VideoPlayer(f'{path}/marker-tracker.mkv')
    # player.run()
