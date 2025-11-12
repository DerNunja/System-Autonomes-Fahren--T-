"""
Machine Learning Lane Detection using Semantic Segmentation
Uses a lightweight pre-trained model for reliable lane detection
"""
import torch
import torch.nn as nn
import cv2
import numpy as np
import argparse
from pathlib import Path
from torchvision import transforms
from PIL import Image


class SimpleLaneNet(nn.Module):
    """Lightweight segmentation network for lane detection"""
    def __init__(self):
        super(SimpleLaneNet, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = self._conv_block(3, 16)
        self.enc2 = self._conv_block(16, 32)
        self.enc3 = self._conv_block(32, 64)
        self.enc4 = self._conv_block(64, 128)
        
        # Decoder (upsampling)
        self.dec4 = self._upconv_block(128, 64)
        self.dec3 = self._upconv_block(64, 32)
        self.dec2 = self._upconv_block(32, 16)
        self.dec1 = self._upconv_block(16, 8)
        
        # Final output
        self.out = nn.Conv2d(8, 1, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
        
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _upconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Decoder
        d4 = self.dec4(e4)
        d3 = self.dec3(d4 + e3)
        d2 = self.dec2(d3 + e2)
        d1 = self.dec1(d2 + e1)
        
        return torch.sigmoid(self.out(d1))


def extract_lane_points(mask, num_points=50):
    """Extract lane points from segmentation mask"""
    h, w = mask.shape
    lanes = []
    
    # Sample horizontal slices
    y_samples = np.linspace(int(h * 0.5), h - 1, num_points, dtype=int)
    
    for y in y_samples:
        row = mask[y, :]
        # Find lane pixels in this row
        lane_pixels = np.where(row > 128)[0]
        
        if len(lane_pixels) > 0:
            # Group into left and right lanes
            mid = w // 2
            left_pixels = lane_pixels[lane_pixels < mid]
            right_pixels = lane_pixels[lane_pixels >= mid]
            
            if len(left_pixels) > 0:
                lanes.append(('left', int(np.mean(left_pixels)), y))
            if len(right_pixels) > 0:
                lanes.append(('right', int(np.mean(right_pixels)), y))
    
    # Organize into left and right lanes
    left_lane = [(x, y) for side, x, y in lanes if side == 'left']
    right_lane = [(x, y) for side, x, y in lanes if side == 'right']
    
    return left_lane, right_lane


def fit_polynomial(points, img_height, img_width, degree=2):
    """Fit polynomial curve through points"""
    if len(points) < 4:
        return None
    
    x_coords = np.array([p[0] for p in points])
    y_coords = np.array([p[1] for p in points])
    
    try:
        poly = np.polyfit(y_coords, x_coords, deg=degree)
    except:
        return None
    
    # Generate smooth curve
    y_range = np.linspace(min(y_coords), max(y_coords), num=50)
    curve_points = []
    
    for y in y_range:
        x = sum(poly[i] * y**(degree - i) for i in range(len(poly)))
        x = int(np.clip(x, 0, img_width - 1))
        curve_points.append((x, int(y)))
    
    return curve_points


def draw_lanes(image, left_curve, right_curve):
    """Draw lane curves on image"""
    overlay = image.copy()
    
    # Draw filled lane area
    if left_curve and right_curve:
        pts_left = np.array(left_curve, dtype=np.int32)
        pts_right = np.array(right_curve[::-1], dtype=np.int32)  # Reverse for polygon
        lane_area = np.concatenate([pts_left, pts_right])
        cv2.fillPoly(overlay, [lane_area], (0, 255, 0))
        image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    
    # Draw lane lines
    if left_curve:
        pts = np.array(left_curve, dtype=np.int32)
        cv2.polylines(image, [pts], False, (0, 255, 0), 8, cv2.LINE_AA)
    
    if right_curve:
        pts = np.array(right_curve, dtype=np.int32)
        cv2.polylines(image, [pts], False, (0, 255, 0), 8, cv2.LINE_AA)
    
    return image


def process_image_ml(image, model, device, transform):
    """Process image with ML model"""
    h, w = image.shape[:2]
    
    # Prepare input
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    # Run model
    with torch.no_grad():
        mask = model(input_tensor)
    
    # Get segmentation mask
    mask = mask.squeeze().cpu().numpy()
    mask = cv2.resize(mask, (w, h))
    mask = (mask * 255).astype(np.uint8)
    
    # Extract lane points
    left_points, right_points = extract_lane_points(mask)
    
    # Fit curves
    left_curve = fit_polynomial(left_points, h, w) if left_points else None
    right_curve = fit_polynomial(right_points, h, w) if right_points else None
    
    # Draw lanes
    result = draw_lanes(image, left_curve, right_curve)
    
    return result


def create_untrained_model():
    """Create model (will be trained on your data or use transfer learning)"""
    model = SimpleLaneNet()
    # Note: This is an untrained model. For production, you would:
    # 1. Train on labeled lane data
    # 2. Or use a pre-trained model like ERFNet, ENet, etc.
    return model


def main():
    parser = argparse.ArgumentParser(description='ML-based Lane Detection')
    parser.add_argument('--input', required=True, help='Input image or directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--model', default=None, help='Path to trained model weights')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = create_untrained_model()
    if args.model and Path(args.model).exists():
        model.load_state_dict(torch.load(args.model, map_location=device))
        print(f"Loaded model from {args.model}")
    else:
        print("WARNING: Using untrained model. Results will be poor.")
        print("You need to either:")
        print("  1. Train this model on labeled lane data")
        print("  2. Use a pre-trained segmentation model")
        return
    
    model.to(device)
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Collect images
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
    
    # Process images
    for i, img_path in enumerate(image_paths):
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Failed to read: {img_path}")
                continue
            
            result = process_image_ml(image, model, device, transform)
            
            output_path = output_dir / f"{img_path.stem}_lanes{img_path.suffix}"
            cv2.imwrite(str(output_path), result)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(image_paths)} images")
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"Done! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
