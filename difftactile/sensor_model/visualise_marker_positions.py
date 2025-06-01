import pickle
import cv2
import numpy as np

def load_marker_positions(pickle_file):
    """Load all marker positions from the pickle file."""
    all_frames = []
    with open(pickle_file, 'rb') as f:
        while True:
            try:
                batch = pickle.load(f)
                all_frames.extend(batch)
            except EOFError:
                break
    return all_frames

def draw_markers(canvas, markers, radius=5, color=(0, 0, 255)):
    """Draw markers on the canvas."""
    vis_frame = canvas.copy()
    for pos in markers:
        center = (int(pos[0]), int(pos[1]))
        cv2.circle(vis_frame, center, radius=radius, color=color, thickness=2)
    return vis_frame

def main():
    # Load the video to get resolution
    video_path = "system-id-screws-3-reps.mkv"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Load marker positions
    pickle_file = "marker_positions.pkl"
    try:
        marker_positions = load_marker_positions(pickle_file)
    except FileNotFoundError:
        print(f"Error: Could not find {pickle_file}")
        return
    
    num_frames = len(marker_positions)
    print(f"Loaded {num_frames} frames of marker positions")
    
    # Create blank canvas with same resolution as video
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Initialize visualization
    current_frame = 0
    window_name = "Marker Positions Visualization"
    cv2.namedWindow(window_name)
    
    print("\nControls:")
    print("- Left Arrow: Previous frame")
    print("- Right Arrow: Next frame")
    print("- ESC or 'q': Quit")
    
    while True:
        # Draw current frame
        vis_frame = draw_markers(canvas, marker_positions[current_frame])
        
        # Add frame counter to image
        text = f"Frame: {current_frame + 1}/{num_frames}"
        cv2.putText(vis_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow(window_name, vis_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == 83 or key == ord('d'):  # Right arrow or 'd'
            current_frame = min(current_frame + 1, num_frames - 1)
        elif key == 81 or key == ord('a'):  # Left arrow or 'a'
            current_frame = max(current_frame - 1, 0)
        
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
