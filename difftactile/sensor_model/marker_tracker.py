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
        self.k_neighbors = 5  # Number of nearest neighbors for geometric descriptors
        
    def compute_geometric_descriptors(self, points):
        """
        Compute geometric descriptors based on k-nearest neighbors distances.
        
        Args:
            points (np.ndarray): Array of shape (N, 2) containing point coordinates
            
        Returns:
            np.ndarray: Array of shape (N, k) containing sorted distances to k nearest neighbors
        """
        n_points = len(points)
        if n_points < 2:
            return np.zeros((n_points, self.k_neighbors))
            
        # Build KD-tree for efficient nearest neighbor search
        tree = KDTree(points)
        
        # Compute distances to k+1 nearest neighbors (including self)
        distances, _ = tree.query(points, k=min(self.k_neighbors + 1, n_points))
        
        # Remove self-distance (first column) and pad if necessary
        if n_points <= self.k_neighbors + 1:
            # Pad with max distance if we have fewer points than k_neighbors
            descriptors = np.zeros((n_points, self.k_neighbors))
            descriptors[:, :distances.shape[1]-1] = distances[:, 1:]
            descriptors[:, distances.shape[1]-1:] = np.max(distances) * 1.5
        else:
            descriptors = distances[:, 1:self.k_neighbors+1]
            
        # Sort distances for each point to ensure invariance to point ordering
        descriptors.sort(axis=1)
        
        # Normalize by max distance for scale invariance
        max_dist = np.max(descriptors)
        if max_dist > 0:
            descriptors = descriptors / max_dist
            
        return descriptors
        
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
                self.frame_markers.append(markers)
                
            frame_count += 1
            
        cap.release()
        
    def compute_base_frame_mappings(self):
        """
        Compute mappings between each frame and frame 0 using K-Nearest Neighbors
        geometric descriptors and the Hungarian algorithm for optimal assignment.
        """
        base_markers = self.frame_markers[0]
        
        for frame_idx in range(1, len(self.frame_markers)):
            current_markers = self.frame_markers[frame_idx]
            
            # Skip if either frame has no markers
            if len(base_markers) == 0 or len(current_markers) == 0:
                self.base_frame_mappings.append(np.full((len(current_markers), 2), np.nan))
                continue
                
            try:
                # Compute geometric descriptors for both frames
                base_desc = self.compute_geometric_descriptors(base_markers)
                current_desc = self.compute_geometric_descriptors(current_markers)
                
                # Compute descriptor distance matrix
                desc_dist = np.linalg.norm(current_desc[:, None] - base_desc[None, :], axis=2)
                
                # Compute spatial distances and normalize
                spatial_dist = cdist(current_markers, base_markers)
                spatial_dist = spatial_dist / spatial_dist.max() if spatial_dist.max() > 0 else spatial_dist
                
                # Combine geometric and spatial distances
                cost_matrix = 0.7 * desc_dist + 0.3 * spatial_dist
                
                # Solve optimal assignment using Hungarian algorithm
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                # Create mapping matrix with distances
                mapping = np.full((len(current_markers), 2), np.nan)
                for i, j in zip(row_ind, col_ind):
                    # Only keep matches if the cost is reasonable
                    if cost_matrix[i, j] < 0.5:  # Threshold for accepting matches
                        mapping[i] = [j, np.linalg.norm(current_markers[i] - base_markers[j])]
                        
                self.base_frame_mappings.append(mapping)
                
            except Exception as e:
                print(f"Warning: Geometric matching failed for frame {frame_idx}: {str(e)}")
                self.base_frame_mappings.append(np.full((len(current_markers), 2), np.nan))
                
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
        print("Computing base frame mappings using K-Nearest Neighbors geometric descriptors...")
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
