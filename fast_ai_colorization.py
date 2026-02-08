import cv2
import numpy as np
import os
from PIL import Image

# ============================================================
# COLOR PALETTES (LAB Space: A and B channels)
# ============================================================
# In OpenCV 8-bit LAB: L:0-255, A:0-255 (128=neutral), B:0-255 (128=neutral)
PALETTES = {
    'natural': {
        'sky': (120, 110),        # Slight blue
        'vegetation': (115, 135), # Green-yellow
        'ground': (135, 145),      # Brownish
        'skin': (140, 135),        # Peach
        'building': (128, 128)     # Neutral gray
    },
    'warm': {
        'sky': (128, 145),         # Golden/Yellow
        'vegetation': (120, 150),  # Warm green
        'ground': (140, 155),      # Warm brown
        'skin': (145, 145),        # Warm peach
        'building': (135, 135)     # Beige
    },
    'cinematic': {
        'sky': (110, 110),         # Cool blue-green
        'vegetation': (120, 120),  # Muted green
        'ground': (125, 125),      # Muted
        'skin': (135, 130),        # Muted peach
        'building': (120, 125)     # Cool gray
    }
}

def create_color_map(height, width, palette_name='natural', L_channel=None):
    """
    Creates a semantic color map based on vertical position and brightness.
    """
    palette = PALETTES.get(palette_name, PALETTES['natural'])

    # 1. Height-based segmentation
    y_coords = np.linspace(0, 1, height).reshape(-1, 1)
    height_map = np.tile(y_coords, (1, width))

    # 2. Brightness-based segmentation
    if L_channel is not None:
        brightness = L_channel / 255.0
    else:
        brightness = np.ones((height, width)) * 0.5

    # Initialize A and B channels with neutral gray
    A = np.full((height, width), 128, dtype=np.float32)
    B = np.full((height, width), 128, dtype=np.float32)

    # Define semantic regions (broad overlaps)
    # Sky: Top 40%, bright
    sky_mask = ((1.0 - height_map) * brightness).astype(np.float32)
    # Ground: Bottom 60%
    ground_mask = height_map.astype(np.float32)
    # Skin/Subject: Bright areas in the middle
    subject_mask = ((brightness > 0.5) * (height_map > 0.2) * (height_map < 0.8)).astype(np.float32)
    # Vegetation: Certain height and brightness
    veg_mask = ((height_map > 0.3) * (height_map < 0.7) * (brightness < 0.6)).astype(np.float32)

    # Normalize masks
    total_mask = sky_mask + ground_mask + subject_mask + veg_mask + 1e-6
    sky_mask /= total_mask
    ground_mask /= total_mask
    subject_mask /= total_mask
    veg_mask /= total_mask

    # Blend colors
    A = (sky_mask * palette['sky'][0] +
         ground_mask * palette['ground'][0] +
         subject_mask * palette['skin'][0] +
         veg_mask * palette['vegetation'][0])

    B = (sky_mask * palette['sky'][1] +
         ground_mask * palette['ground'][1] +
         subject_mask * palette['skin'][1] +
         veg_mask * palette['vegetation'][1])

    return A.astype(np.uint8), B.astype(np.uint8)

def colorize_frame(frame, palette_name='natural', prev_color=None, alpha=0.5):
    """
    Colorizes a single grayscale frame.
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # 1. Enhancement: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L = clahe.apply(gray)

    # 2. Semantic Color Assignment
    height, width = L.shape
    A, B = create_color_map(height, width, palette_name, L_channel=L)

    # 3. Merge to LAB and convert to BGR
    lab = cv2.merge([L, A, B])
    colorized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 4. Post-processing: Bilateral Filter (denoise)
    colorized = cv2.bilateralFilter(colorized, 7, 50, 50)

    # 5. Post-processing: Sharpening
    blurred = cv2.GaussianBlur(colorized, (0, 0), 3)
    colorized = cv2.addWeighted(colorized, 1.5, blurred, -0.5, 0)

    # 6. Temporal Stabilization
    if prev_color is not None:
        colorized = cv2.addWeighted(colorized, alpha, prev_color, 1.0 - alpha, 0)

    return colorized

def process_video(input_path, output_path, palette_name='natural', limit_frames=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 24

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    prev_frame = None
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        colorized = colorize_frame(frame, palette_name, prev_color=prev_frame)
        out.write(colorized)

        prev_frame = colorized
        count += 1
        if limit_frames and count >= limit_frames:
            break

    cap.release()
    out.release()
    print(f"Processed {count} frames. Saved to {output_path}")

def create_comparison(input_path, output_path, result_path):
    cap_in = cv2.VideoCapture(input_path)
    cap_out = cv2.VideoCapture(output_path)

    ret1, frame_in = cap_in.read()
    ret2, frame_out = cap_out.read()

    if ret1 and ret2:
        # Resize to same height if needed
        h1, w1 = frame_in.shape[:2]
        h2, w2 = frame_out.shape[:2]

        if h1 != h2:
            frame_in = cv2.resize(frame_in, (int(w1 * h2 / h1), h2))

        comparison = np.hstack((frame_in, frame_out))
        cv2.imwrite(result_path, comparison)
        print(f"Comparison saved to {result_path}")

    cap_in.release()
    cap_out.release()

if __name__ == "__main__":
    # Video 1: Natural Palette
    print("Processing Video 1 (Natural)...")
    process_video("enhanced_realism.mp4", "enhanced_realism_AI_COLORIZED.mp4", palette_name='natural')

    # Video 2: Warm Palette
    print("Processing Video 2 (Warm)...")
    process_video("enhanced_segmented_5s.mp4", "enhanced_segmented_5s_AI_COLORIZED.mp4", palette_name='warm')

    # Comparison Images
    print("Creating Comparison Images...")
    create_comparison("enhanced_realism.mp4", "enhanced_realism_AI_COLORIZED.mp4", "enhanced_realism_comparison.jpg")
    create_comparison("enhanced_segmented_5s.mp4", "enhanced_segmented_5s_AI_COLORIZED.mp4", "enhanced_segmented_5s_comparison.jpg")
