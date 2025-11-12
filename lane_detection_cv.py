"""
Traditional Computer Vision Lane Detection
Uses Canny edge detection + Hough transform for reliable lane detection
"""
import cv2
import numpy as np
import argparse
from pathlib import Path


def region_of_interest(img, vertices):
    """Apply ROI mask to keep only the region defined by vertices"""
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def detect_lane_edges(image, low_threshold=50, high_threshold=150):
    """Detect edges using Canny edge detector"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blur, low_threshold, high_threshold)
    
    return edges


def detect_lines(edges, rho=1, theta=np.pi/180, threshold=50, min_line_length=50, max_line_gap=150):
    """Detect lines using Hough transform"""
    lines = cv2.HoughLinesP(
        edges,
        rho=rho,
        theta=theta,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    return lines


def separate_left_right_lines(lines, img_width):
    """Separate lines into left and right based on slope"""
    left_lines = []
    right_lines = []
    
    if lines is None:
        return left_lines, right_lines
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Skip vertical lines
        if x2 - x1 == 0:
            continue
            
        slope = (y2 - y1) / (x2 - x1)
        
        # Filter by slope to remove horizontal and near-horizontal lines
        if abs(slope) < 0.5:
            continue
        
        # Separate by position and slope
        if slope < 0 and x2 < img_width * 0.6:  # Left lane (negative slope)
            left_lines.append((x1, y1, x2, y2, slope))
        elif slope > 0 and x1 > img_width * 0.4:  # Right lane (positive slope)
            right_lines.append((x1, y1, x2, y2, slope))
    
    return left_lines, right_lines


def fit_lane_polynomial(lines, img_height, img_width):
    """Fit a polynomial curve through line segments and detect line style"""
    if not lines:
        return None, None
    
    # Extract all points from line segments
    points = []
    for x1, y1, x2, y2, slope in lines:
        points.append((x1, y1))
        points.append((x2, y2))
    
    if len(points) < 4:
        return None, None
    
    # Sort points by y coordinate
    points = sorted(points, key=lambda p: p[1])
    x_coords = np.array([p[0] for p in points])
    y_coords = np.array([p[1] for p in points])
    
    # Fit 2nd degree polynomial for curves (x = f(y))
    try:
        poly = np.polyfit(y_coords, x_coords, deg=2)
    except:
        return None, None
    
    # Detect line style (solid vs dashed)
    # Calculate gaps between consecutive line segments
    gaps = []
    sorted_lines = sorted(lines, key=lambda l: (l[1] + l[3]) / 2)  # Sort by midpoint y
    for i in range(len(sorted_lines) - 1):
        # Distance between end of one segment and start of next
        _, _, _, y2_end, _ = sorted_lines[i]
        _, y1_start, _, _, _ = sorted_lines[i + 1]
        gap = abs(y1_start - y2_end)
        gaps.append(gap)
    
    # If we have significant gaps, it's likely dashed
    avg_gap = np.mean(gaps) if gaps else 0
    is_dashed = avg_gap > 20  # Threshold for dashed detection
    
    # Generate smooth curve points
    y_range = np.linspace(int(img_height * 0.6), img_height, num=50)
    curve_points = []
    
    for y in y_range:
        x = poly[0] * y**2 + poly[1] * y + poly[2]
        x = int(np.clip(x, 0, img_width - 1))
        curve_points.append((x, int(y)))
    
    return curve_points, is_dashed


def draw_lane_curve(image, curve_points, is_dashed=False, color=(0, 255, 0), thickness=8):
    """Draw a smooth lane curve (solid or dashed)"""
    if curve_points is None or len(curve_points) < 2:
        return image
    
    line_image = np.zeros_like(image)
    
    if is_dashed:
        # Draw dashed line
        dash_length = 20
        gap_length = 15
        
        for i in range(0, len(curve_points) - 1, dash_length + gap_length):
            end_idx = min(i + dash_length, len(curve_points) - 1)
            segment = curve_points[i:end_idx + 1]
            
            if len(segment) > 1:
                pts = np.array(segment, dtype=np.int32)
                cv2.polylines(line_image, [pts], False, color, thickness, cv2.LINE_AA)
    else:
        # Draw solid line
        pts = np.array(curve_points, dtype=np.int32)
        cv2.polylines(line_image, [pts], False, color, thickness, cv2.LINE_AA)
    
    return line_image


def draw_lanes(image, left_curve, left_dashed, right_curve, right_dashed):
    """Draw both lane curves on image"""
    line_image = np.zeros_like(image)
    
    # Color coding: solid = green, dashed = yellow
    if left_curve is not None:
        color = (0, 255, 255) if left_dashed else (0, 255, 0)
        line_img = draw_lane_curve(image, left_curve, left_dashed, color=color, thickness=8)
        line_image = cv2.add(line_image, line_img)
    
    if right_curve is not None:
        color = (0, 255, 255) if right_dashed else (0, 255, 0)
        line_img = draw_lane_curve(image, right_curve, right_dashed, color=color, thickness=8)
        line_image = cv2.add(line_image, line_img)
    
    # Combine with original image
    result = cv2.addWeighted(image, 0.8, line_image, 1.0, 0.0)
    return result


def process_image(image, debug=False):
    """
    Main lane detection pipeline
    """
    h, w = image.shape[:2]
    
    # Define region of interest (trapezoid focusing on bottom half)
    roi_vertices = np.array([[
        (int(w * 0.1), h),              # Bottom left
        (int(w * 0.45), int(h * 0.6)),  # Top left
        (int(w * 0.55), int(h * 0.6)),  # Top right
        (int(w * 0.9), h)                # Bottom right
    ]], dtype=np.int32)
    
    # Detect edges
    edges = detect_lane_edges(image)
    
    # Apply ROI mask
    masked_edges = region_of_interest(edges, roi_vertices)
    
    if debug:
        cv2.imwrite('/tmp/lane_debug_edges.jpg', masked_edges)
    
    # Detect lines
    lines = detect_lines(masked_edges)
    
    # Separate lines into left and right
    left_lines, right_lines = separate_left_right_lines(lines, w)
    
    # Fit polynomial curves and detect line styles
    left_curve, left_dashed = fit_lane_polynomial(left_lines, h, w)
    right_curve, right_dashed = fit_lane_polynomial(right_lines, h, w)
    
    # Draw lanes with curves and style detection
    result = draw_lanes(image, left_curve, left_dashed, right_curve, right_dashed)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Lane Detection using Computer Vision')
    parser.add_argument('--input', required=True, help='Input image or directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--debug', action='store_true', help='Save debug images')
    args = parser.parse_args()
    
    # Collect input images
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    else:
        extensions = ['*.jpg', '*.jpeg', '*.png']
        image_paths = []
        for ext in extensions:
            image_paths.extend(input_path.glob(ext))
        image_paths = sorted(image_paths)
    
    if not image_paths:
        print(f"No images found in {args.input}")
        return
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(image_paths)} images...")
    
    # Process each image
    for i, img_path in enumerate(image_paths):
        try:
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Failed to read: {img_path}")
                continue
            
            # Process
            result = process_image(image, debug=args.debug)
            
            # Save
            output_path = output_dir / f"{img_path.stem}_lanes{img_path.suffix}"
            cv2.imwrite(str(output_path), result)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(image_paths)} images")
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"Done! Results saved to {output_dir}")


if __name__ == '__main__':
    main()

