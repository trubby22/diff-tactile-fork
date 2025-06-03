import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from fisheye_model import get_marker_image
import tkinter as tk
from PIL import Image, ImageTk
from pycpd import DeformableRegistration
from scipy.spatial import cKDTree

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
        
    def extract_frames(self):
        """Extract frames from video at 0.5 second intervals."""
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 0.5)  # Number of frames to skip for 0.5s interval
        
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
        Compute mappings between each frame and frame 0 using Coherent Point Drift (CPD).
        Uses non-rigid registration to align markers and find correspondences.
        """
        base_markers = self.frame_markers[0]
        
        for frame_idx in range(1, len(self.frame_markers)):
            current_markers = self.frame_markers[frame_idx]
            
            # Skip if either frame has no markers
            if len(base_markers) == 0 or len(current_markers) == 0:
                self.base_frame_mappings.append(np.full((len(current_markers), 2), np.nan))
                continue
                
            try:
                # Apply CPD registration
                reg = DeformableRegistration(X=base_markers, Y=current_markers)
                reg.register()
                registered_points = reg.transform_point_cloud(current_markers)
                
                # Find nearest neighbors for correspondence
                tree = cKDTree(base_markers)
                distances, indices = tree.query(registered_points, k=1)
                
                # Create mapping matrix with distances
                mapping = np.full((len(current_markers), 2), np.nan)
                for i, (dist, idx) in enumerate(zip(distances, indices)):
                    mapping[i] = [idx, dist]
                        
                self.base_frame_mappings.append(mapping)
                
            except Exception as e:
                print(f"Warning: CPD registration failed for frame {frame_idx}: {str(e)}")
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
        print("Computing base frame mappings using CPD...")
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
