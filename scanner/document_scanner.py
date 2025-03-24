import cv2
import numpy as np
import imutils
from skimage.filters import threshold_local
from skimage import exposure
from PIL import Image
import os
import logging
from scipy import ndimage
import math

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB limit

def order_points(pts):
    # Initialize a list of coordinates that will be ordered
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum
    # The bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # The top-right point will have the smallest difference
    # The bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    # Obtain a consistent order of the points and unpack them
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Construct destination points for the transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Calculate the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def enhance_image_quality(image, strength=1.0):
    """
    Enhance image quality focusing specifically on text clarity
    """
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Analyze image quality
        mean_brightness = np.mean(gray)
        std_dev = np.std(gray)
        
        # Text-specific enhancement
        if mean_brightness < 127:
            # Brighten dark text carefully
            alpha = min(1.25, 1 + (127 - mean_brightness) / 200)
            beta = min(25, (127 - mean_brightness) / 4)
            gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        
        # Apply adaptive thresholding for better text separation
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,  # Larger block size for better text detection
            10   # Higher constant for clearer text
        )
        
        # Clean up noise while preserving text
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        if len(image.shape) == 3:
            # For color documents, preserve color information
            result = image.copy()
            # Apply the enhanced text mask
            result[cleaned == 0] = [0, 0, 0]
            result[cleaned == 255] = image[cleaned == 255]
            return result
        return cleaned
    except Exception as e:
        logging.warning(f"Image enhancement failed: {str(e)}")
        return image

def remove_shadows(image):
    """Remove shadows from image using advanced technique"""
    try:
        # Convert to LAB color space
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
        else:
            l = image.copy()
            
        # Apply bilateral filter to reduce noise while preserving edges
        smoothed = cv2.bilateralFilter(l, 9, 75, 75)
        
        # Create a normalized version to find shadows
        normalized = cv2.normalize(smoothed, None, 0, 255, cv2.NORM_MINMAX)
        
        # Calculate shadow mask
        shadow_mask = cv2.subtract(l, normalized)
        
        # Remove shadows from original L channel
        result_l = cv2.add(l, shadow_mask)
        
        if len(image.shape) == 3:
            # Merge back with color channels
            result = cv2.merge([result_l, a, b])
            result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        else:
            result = result_l
            
        return result
    except Exception as e:
        logging.warning(f"Shadow removal failed: {str(e)}")
        return image

def apply_smart_sharpen(image, amount=1.1):
    """Intelligent sharpening that preserves text quality"""
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply unsharp mask to L channel
        gaussian = cv2.GaussianBlur(l, (0, 0), 2.0)
        unsharp = cv2.addWeighted(l, amount + 0.5, gaussian, -0.5, 0)
        
        # Merge channels back
        lab = cv2.merge([unsharp, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except Exception:
        return image

def reduce_noise(image, strength='normal'):
    """
    Noise reduction optimized for text preservation
    """
    try:
        # More conservative parameters for text
        strength_params = {
            'light': {'h': 2, 'search': 7, 'window': 21},    # Very light for clear text
            'normal': {'h': 4, 'search': 7, 'window': 21},   # Balanced for most text
            'strong': {'h': 6, 'search': 7, 'window': 21}    # Still conservative for text
        }
        
        params = strength_params.get(strength, strength_params['normal'])
        
        if len(image.shape) == 3:
            # For color images, process luminance channel
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Denoise luminance while preserving edges
            l = cv2.fastNlMeansDenoising(
                l,
                None,
                h=params['h'],
                searchWindowSize=params['search'],
                templateWindowSize=params['window']
            )
            
            denoised = cv2.merge([l, a, b])
            return cv2.cvtColor(denoised, cv2.COLOR_LAB2BGR)
        else:
            # For B&W images, use edge-preserving denoising
            return cv2.fastNlMeansDenoising(
                image,
                None,
                h=params['h'],
                searchWindowSize=params['search'],
                templateWindowSize=params['window']
            )
            
    except Exception as e:
        logging.warning(f"Noise reduction failed: {str(e)}")
        return image

def apply_color_enhancement(image, mode='original'):
    """
    Basic color mode selection without enhancements
    Args:
        image: Input image
        mode: 'original', 'bw', 'grayscale', or 'color'
    """
    try:
        if mode == 'original':
            return image
        elif mode == 'bw':
            # Convert to black and white
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Simple thresholding for B&W
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return bw
            
        elif mode == 'grayscale':
            # Convert to grayscale
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image
            
        elif mode == 'color':
            # Enhance color
            if len(image.shape) == 3:
                return image
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        return image
        
    except Exception as e:
        logging.warning(f"Color enhancement failed: {str(e)}")
        return image

def deskew_image(image):
    """Deskew image using Hough lines"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    
    if lines is not None:
        angle = np.median([line[0][1] for line in lines]) * 180 / np.pi
        if angle > 45:
            angle = angle - 90
        elif angle < -45:
            angle = 90 + angle
            
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return image

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(image_path):
    try:
        with Image.open(image_path) as img:
            # Verify the image can be read
            img.verify()
            
            # Check if image is not corrupted
            img = Image.open(image_path)
            img.load()
            
            # Check file size
            if os.path.getsize(image_path) > MAX_IMAGE_SIZE:
                raise ValueError("Image file is too large")
            
            return True
    except Exception as e:
        logging.error(f"Image validation failed: {str(e)}")
        return False

def detect_document_corners(image):
    """Detect document corners in image"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Try multiple edge detection methods
        methods = [
            lambda: cv2.Canny(blurred, 75, 200),
            lambda: cv2.Canny(blurred, 50, 150),
            lambda: cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        ]
        
        for method in methods:
            try:
                edges = method()
                # Dilate edges to connect gaps
                edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

                # Find contours
                contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = imutils.grab_contours(contours)
                
                if not contours:
                    continue
                    
                # Sort by area and keep largest
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
                
                for contour in contours:
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                    
                    if len(approx) == 4:
                        # Check if contour is sufficiently large
                        area = cv2.contourArea(approx)
                        if area > (image.shape[0] * image.shape[1] * 0.25):  # At least 25% of image
                            return approx
                            
            except Exception as e:
                logging.warning(f"Corner detection method failed: {str(e)}")
                continue
                
        return None

    except Exception as e:
        logging.error(f"Corner detection failed: {str(e)}")
        return None

def line_intersection(rho1, theta1, rho2, theta2):
    """Helper function to find intersection of two lines in Hough space"""
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    try:
        x0, y0 = np.linalg.solve(A, b)
        return [int(x0), int(y0)]
    except:
        return None

def scan_document(image_path, enhance=True, options=None):
    """
    Document scanning with real-time enhancements
    Args:
        image_path: Path to image file or bytes
        enhance: Boolean to control enhancement
        options: Dictionary of processing options
    Returns:
        Dictionary containing processed image and success status
    """
    try:
        # Initialize default options
        default_options = {
            'mode': 'original',
            'contrast': 1.2,
            'clean_background': True,
            'straighten': True,
            'sharpen_text': True,
            'auto_crop': True  # Add auto-crop option
        }
        
        # Update with provided options
        if options:
            for key in default_options:
                if key in options:
                    default_options[key] = options[key]
        
        # Load image
        if isinstance(image_path, bytes):
            nparr = np.frombuffer(image_path, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(image_path)
            
        if image is None:
            return {'success': False, 'error': 'Could not read image'}

        # Keep original for comparison
        original = image.copy()
        
        try:
            # Resize for faster processing while maintaining quality
            max_dim = 1500
            height, width = image.shape[:2]
            if max(height, width) > max_dim:
                scale = max_dim / max(height, width)
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            # 1. Auto-crop and align document
            if default_options['auto_crop']:
                # Detect document corners
                corners = detect_document_corners(image)
                if corners is not None:
                    # Calculate scale ratio for original size
                    ratio = original.shape[0] / image.shape[0]
                    # Transform original image using detected corners
                    image = four_point_transform(original, corners.reshape(4, 2) * ratio)
                    original = image.copy()  # Update original to cropped version

            # 2. Straighten if enabled
            if default_options['straighten']:
                image = deskew_image(image)
            
            # 3. Apply color mode
            if default_options['mode'] != 'original':
                image = apply_color_enhancement(image, mode=default_options['mode'])
            
            # 4. Clean background if enabled
            if default_options['clean_background']:
                text_density = check_text_density(image)
                strength = 'light' if text_density > 0.2 else 'normal'
                image = reduce_noise(image, strength=strength)
            
            # 5. Enhance text if enabled
            if default_options['sharpen_text']:
                if check_text_density(image) > 0.1:
                    image = apply_smart_sharpen(image)
            
            # 6. Adjust contrast
            contrast = float(default_options['contrast'])
            if abs(contrast - 1.0) > 0.05:
                image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
            
            return {
                'success': True,
                'processed': image,
                'original': original
            }
            
        except Exception as e:
            logging.error(f"Enhancement failed: {str(e)}")
            # Return original if processing fails
            return {
                'success': True,
                'processed': original,
                'original': original
            }
            
    except Exception as e:
        logging.error(f"Document scanning failed: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def check_text_density(image):
    """Analyze image for text content"""
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Calculate text density
        text_pixels = np.sum(binary == 255)
        total_pixels = binary.size
        
        return text_pixels / total_pixels
    except Exception:
        return 0.0

def get_document_info(image):
    """
    Get information about the detected document
    Returns dict with:
        - success: bool
        - angle: rotation angle
        - confidence: detection confidence
        - corners: corner points
    """
    try:
        if isinstance(image, bytes):
            nparr = np.frombuffer(image, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Resize and process
        image = imutils.resize(image, height=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 75, 200)
        
        # Find contours
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(approx) == 4:
                # Calculate confidence based on contour area ratio
                area_ratio = cv2.contourArea(c) / (image.shape[0] * image.shape[1])
                confidence = min(area_ratio * 100 * 3, 100)  # Scale up but cap at 100%
                
                # Calculate rotation angle
                rect = cv2.minAreaRect(c)
                angle = rect[-1]
                if angle < -45:
                    angle = 90 + angle
                
                return {
                    'success': True,
                    'angle': angle,
                    'confidence': confidence,
                    'corners': approx.reshape(4, 2).tolist()
                }
                
        return {
            'success': False,
            'error': 'No document detected'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        } 