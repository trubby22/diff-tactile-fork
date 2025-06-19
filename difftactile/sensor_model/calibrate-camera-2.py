import numpy as np
import cv2

# Load the saved parameters
params = np.load("fisheye_params.npz")

K = params["K"]
D = params["D"]

# Print the contents
print("K (Intrinsic Matrix):\n", params["K"])
print("D (Distortion Coefficients):\n", params["D"])

fx = K[0,0]
fy = K[1,1]
cx = K[0,2]
cy = K[1,2]

print(f"Focal Length (fx, fy): {fx:.2f}, {fy:.2f}")
print(f"Principal Point (cx, cy): {cx:.2f}, {cy:.2f}")

# Function to scale intrinsic parameters to a new resolution
def scale_resolution(fx, fy, cx, cy, orig_width, orig_height, new_width, new_height):
    scale_x = new_width / orig_width
    scale_y = new_height / orig_height
    fx_scaled = fx * scale_x
    fy_scaled = fy * scale_y
    cx_scaled = cx * scale_x
    cy_scaled = cy * scale_y
    print(f"Scaled Focal Length (fx, fy): {fx_scaled:.2f}, {fy_scaled:.2f}")
    print(f"Scaled Principal Point (cx, cy): {cx_scaled:.2f}, {cy_scaled:.2f}")

    # Open the image
    img = cv2.imread('../tasks/init.png')
    if img is not None:
        # Draw a small red dot at the scaled principal point
        center = (int(round(cx_scaled)), int(round(cy_scaled)))
        cv2.circle(img, center, radius=5, color=(0, 0, 255), thickness=-1)
        # Save the image
        cv2.imwrite('../tasks/output/init-principal-point.png', img)
        print("Saved image with principal point to './init-principal-point.png'")
    else:
        print("Could not open './init.png' to plot principal point.")

# Example usage: scale to 1280x720
scale_resolution(fx, fy, cx, cy, 1920, 1080, 640, 480)

