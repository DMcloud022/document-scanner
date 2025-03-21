import cv2
import numpy as np
import imutils
from skimage.filters import threshold_local
from skimage import exposure

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

def enhance_image(image):
    """Apply various image enhancement techniques"""
    # Adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    if len(image.shape) == 3:
        # Color image
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        # Grayscale image
        image = clahe.apply(image)
    
    # Denoise
    image = cv2.fastNlMeansDenoising(image) if len(image.shape) == 2 \
        else cv2.fastNlMeansDenoisingColored(image)
    
    return image

def adjust_gamma(image, gamma=1.0):
    """Adjust image gamma"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_color_enhancement(image, mode='auto'):
    """Enhance image colors"""
    if mode == 'auto':
        # Auto color correction
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    elif mode == 'bw':
        # High contrast B&W
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    elif mode == 'grayscale':
        # Enhanced grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray)
    return image

def remove_shadows(image):
    """Remove shadows from image"""
    rgb_planes = cv2.split(image)
    result_planes = []
    
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        result_planes.append(diff_img)
        
    return cv2.merge(result_planes)

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

def scan_document(image, enhance=True, options=None):
    """
    Enhanced document scanning with multiple options
    Args:
        image: Input image
        enhance: Whether to apply enhancement
        options: Dict with additional options:
            - color_mode: 'auto', 'bw', 'grayscale', 'original'
            - remove_shadows: bool
            - deskew: bool
            - sharpen: bool
            - denoise: bool
    """
    if options is None:
        options = {}
    
    # Default options
    default_options = {
        'color_mode': 'auto',
        'remove_shadows': False,
        'deskew': True,
        'sharpen': True,
        'denoise': True
    }
    options = {**default_options, **options}
    
    # Process image
    if isinstance(image, bytes):
        nparr = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Keep track of original
    orig = image.copy()
    
    # Resize while maintaining aspect ratio
    ratio = image.shape[0] / 500.0
    image = imutils.resize(image, height=500)
    
    # Convert to grayscale and blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply automatic threshold using Otsu's method
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Apply Canny edge detection with automatic threshold
    sigma = 0.33
    median = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    edged = cv2.Canny(thresh, lower, upper)
    
    # Dilate edges to connect broken contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated = cv2.dilate(edged, kernel, iterations=1)

    # Find contours
    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # Sort contours by area and keep only the largest ones
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    
    screenCnt = None

    # Loop over contours
    for c in cnts:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # If we have found a contour with 4 points, we can break
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        return None

    # Apply perspective transform
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    
    if options['deskew']:
        warped = deskew_image(warped)
    
    if options['remove_shadows']:
        warped = remove_shadows(warped)
    
    if enhance:
        warped = apply_color_enhancement(warped, options['color_mode'])
        
        if options['denoise']:
            warped = cv2.fastNlMeansDenoisingColored(warped)
            
        if options['sharpen']:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            warped = cv2.filter2D(warped, -1, kernel)
    
    # Convert to grayscale
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    T = threshold_local(warped_gray, 11, offset=10, method="gaussian")
    warped_thresh = (warped_gray > T).astype("uint8") * 255
    
    return warped_thresh

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