import cv2
import numpy as np


def extract_iris_features(image, original_color_image=None):
    denoised = cv2.bilateralFilter(image, 9, 75, 75)
    normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
    equalized = cv2.equalizeHist(normalized)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
    enhanced_image = clahe.apply(equalized)
    filtered_image = cv2.GaussianBlur(enhanced_image, (3, 3), 0)

    segmentation_result = detect_iris_hough(filtered_image)
    
    if segmentation_result is None:
        print("Hough detection failed. Attempting fallback to contour detection...")
        try:
            cX, cY, p_r_contour, binary_image = detectIris(filtered_image)
            i_r = int(p_r_contour * 2.5) 
            center = (cX, cY)
            segmentation_result = (center, p_r_contour, i_r)
            print("Fallback to contour detection successful.")
        except Exception as e:
            print(f"Contour detection fallback also failed: {e}")
            segmentation_result = None

    if segmentation_result is None:
        print("ERROR: All segmentation methods failed. Using crude fallback values.")
        h, w = image.shape
        center, p_r, i_r = (w//2, h//2), 30, 80
        binary_image = np.zeros_like(image)
    else:
        center, p_r, i_r = segmentation_result
        p_r = int(p_r)
        i_r = int(i_r)
        binary_image = np.zeros_like(image)
        cv2.circle(binary_image, center, p_r, 255, -1)
    
    display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.circle(display_image, center, p_r, (0, 255, 0), 2)
    cv2.circle(display_image, center, i_r, (0, 0, 255), 2)

    normalized_iris = polar_transform(enhanced_image, center, i_r, p_r)
    
    h_norm, w_norm = normalized_iris.shape
    occlusion_mask = np.ones_like(normalized_iris)
    top_mask_height = int(h_norm * 0.15)
    bottom_mask_height = int(h_norm * 0.85)
    occlusion_mask[0:top_mask_height, :] = 0
    occlusion_mask[bottom_mask_height:h_norm, :] = 0
    normalized_iris = cv2.bitwise_and(normalized_iris, normalized_iris, mask=occlusion_mask)

    gabor_features = apply_gabor_filters(normalized_iris)
    sobel_x, sobel_y = sobel_operator(normalized_iris)
    scharr_x, scharr_y = scharr_operator(normalized_iris)
    median_filtered_image = median_filter(normalized_iris)
    
    try:
        lbp_features = compute_lbp(normalized_iris)
    except Exception as e:
        print(f"LBP computation failed: {e}")
        lbp_features = np.zeros_like(normalized_iris)
    
    try:
        variance_features = compute_variance_features(normalized_iris)
    except Exception as e:
        print(f"Variance computation failed: {e}")
        variance_features = np.array([[0.0]])
    
    feature_vectors = []
    for f in gabor_features:
        feature_vectors.append(f.flatten() * 0.4)
    
    # LBP features (weight: 0.3) - most robust to illumination
    feature_vectors.append(lbp_features.flatten() * 0.3)
    
    # Edge features (weight: 0.2)
    feature_vectors.extend([scharr_x.flatten() * 0.1, scharr_y.flatten() * 0.1])
    
    # Texture variance (weight: 0.1)
    feature_vectors.append(variance_features.flatten() * 0.1)
    
    # Add color features if color image is provided
    color_features = None  # Default: R,G,B,H,S,V
    if original_color_image is not None:
        try:
            # Pass the detected iris radius to the color feature extractor
            color_features = extract_color_features(original_color_image, center, i_r)
            print(f"Successfully extracted color histogram.")
        except Exception as e:
            print(f"Color feature extraction failed: {e}")
            color_features = None
    
    feature_vector = np.concatenate(feature_vectors)
    feature_vector = robust_normalize(feature_vector)

    return (feature_vector, display_image, enhanced_image, filtered_image, binary_image, normalized_iris, gabor_features,
            sobel_x, sobel_y, scharr_x, scharr_y, median_filtered_image, color_features)


def detect_iris_hough(image):
    blurred = cv2.medianBlur(image, 7)
    
    pupil_params = {
        "dp": 1, "minDist": 200, "param1": 250, "param2": 20,
        "minRadius": 10, "maxRadius": 50
    }
    
    pupil_circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, **pupil_params)
    
    if pupil_circles is None:
        print("WARN: Hough Transform could not detect pupil.")
        return None
        
    pupil = np.uint16(np.around(pupil_circles[0, 0]))
    p_cx, p_cy, p_r = pupil[0], pupil[1], pupil[2]

    iris_params = {
        "dp": 1, "minDist": 200, "param1": 250, "param2": 50,
        "minRadius": p_r + 20, "maxRadius": p_r + 80
    }
    iris_circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, **iris_params)

    if iris_circles is None:
        print("WARN: Hough Transform could not detect iris.")
        return None

    # Find the iris circle most concentric with the pupil
    best_iris_r = None
    min_dist = float('inf')
    for iris in np.uint16(np.around(iris_circles[0, :])):
        i_cx, i_cy, i_r = iris[0], iris[1], iris[2]
        dist = np.sqrt((p_cx - i_cx)**2 + (p_cy - i_cy)**2)
        if dist < min_dist and dist < 25: # Concentricity tolerance
            min_dist = dist
            best_iris_r = i_r
            
    if best_iris_r is None:
        print("WARN: No concentric iris found.")
        return None

    return (p_cx, p_cy), p_r, best_iris_r


def detectIris(image):
    height, width = image.shape[:2]

    roi_margins = [0.2, 0.3, 0.4]
    threshold_values = [40, 50, 60, 80]
    
    best_contour = None
    best_binary = None
    best_area = 0
    
    for margin in roi_margins:
        roi_x_start = int(width * margin)
        roi_x_end = int(width * (1 - margin))
        roi_y_start = int(height * margin)
        roi_y_end = int(height * (1 - margin))

        cropped_image = image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        
        for threshold_val in threshold_values:
            _, binary_image = cv2.threshold(cropped_image, threshold_val, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > best_area and area > 100:
                    best_contour = largest_contour + np.array([roi_x_start, roi_y_start])
                    best_binary = binary_image
                    best_area = area

    if best_contour is None:
        roi_margin = 0.3
        roi_x_start = int(width * roi_margin)
        roi_x_end = int(width * (1 - roi_margin))
        roi_y_start = int(height * roi_margin)
        roi_y_end = int(height * (1 - roi_margin))

        cropped_image = image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        _, best_binary = cv2.threshold(cropped_image, 50, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(best_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return width // 2, height // 2, min(width, height) // 4, best_binary
        
        best_contour = max(contours, key=cv2.contourArea) + np.array([roi_x_start, roi_y_start])

    new_pupil_contour = best_contour

    M = cv2.moments(new_pupil_contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    dist = calculateDist(new_pupil_contour, (cX, cY))

    return cX, cY, int(dist), best_binary


def calculateDist(contour, center):
    cX, cY = center

    distances = []
    for point in contour:
        x, y = point[0]
        distance = np.sqrt((x - cX) ** 2 + (y - cY) ** 2)
        distances.append(distance)

    return np.max(distances)


def apply_gabor_filters(image):
    theta_values = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    gabor_features = []

    for theta in theta_values:
        filtered_iris = gabor_filter(image, th=theta)
        gabor_features.append(filtered_iris)

    return gabor_features


def gabor_filter(img, ksize=31, sigma=4.0, th=0, lambd=10.0, gamma=0.5, psi=0):
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, th, lambd, gamma, psi, ktype=cv2.CV_32F)
    filtered_image = cv2.filter2D(img, cv2.CV_8UC3, kernel)
    return filtered_image


def median_filter(image, ksize=5):
    median_filtered_image = cv2.medianBlur(image, ksize)
    return median_filtered_image


def scharr_operator(image):
    scharr_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
    scharr_x = cv2.convertScaleAbs(scharr_x)
    scharr_y = cv2.convertScaleAbs(scharr_y)
    return scharr_x, scharr_y


def sobel_operator(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    return sobel_x, sobel_y


def polar_transform(image, center, iris_radius, pupil_radius, output_size=(64, 512)):
    polar_image = np.zeros(output_size, dtype=np.uint8)
    theta_range = np.linspace(0, 2 * np.pi, output_size[1])

    for i in range(output_size[1]):
        for j in range(output_size[0]):
            theta = theta_range[i]

            r = pupil_radius + (j / float(output_size[0])) * (iris_radius - pupil_radius)

            x = int(center[0] + r * np.cos(theta))
            y = int(center[1] + r * np.sin(theta))

            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                polar_image[j, i] = image[y, x]

    return polar_image


def compute_lbp(image, radius=1, n_points=8):
    """Compute Local Binary Pattern for texture analysis (8-point LBP)"""
    h, w = image.shape
    lbp = np.zeros((h, w), dtype=np.uint8)
    
    # Pre-calculate sampling points for efficiency
    sampling_points = []
    for k in range(n_points):
        angle = 2 * np.pi * k / n_points
        x_offset = radius * np.cos(angle)
        y_offset = radius * np.sin(angle)
        sampling_points.append((x_offset, y_offset))
    
    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            center = float(image[i, j])
            code = 0
            
            for k, (x_offset, y_offset) in enumerate(sampling_points):
                # Calculate sampling point
                x = i + x_offset
                y = j + y_offset
                
                # Bilinear interpolation
                x1, y1 = int(x), int(y)
                x2, y2 = x1 + 1, y1 + 1
                
                # Bounds checking
                if 0 <= x1 < h-1 and 0 <= y1 < w-1:
                    # Interpolation weights
                    wx = x - x1
                    wy = y - y1
                    
                    # Bilinear interpolation
                    interpolated = ((1-wx)*(1-wy)*image[x1, y1] + 
                                  wx*(1-wy)*image[x2, y1] + 
                                  (1-wx)*wy*image[x1, y2] + 
                                  wx*wy*image[x2, y2])
                    
                    # Compare with center and set bit
                    if interpolated >= center:
                        code |= (1 << k)  # Set k-th bit
            
            # Ensure code fits in uint8 (0-255)
            lbp[i, j] = min(code, 255)
    
    return lbp

def compute_variance_features(image, block_size=8):
    """Compute variance-based texture features"""
    h, w = image.shape
    
    # Ensure we have at least one block
    if h < block_size or w < block_size:
        return np.array([[np.var(image)]])
    
    rows = max(1, h // block_size)
    cols = max(1, w // block_size)
    variance_map = np.zeros((rows, cols))
    
    for i in range(0, min(h - block_size + 1, rows * block_size), block_size):
        for j in range(0, min(w - block_size + 1, cols * block_size), block_size):
            block = image[i:i+block_size, j:j+block_size]
            if block.size > 0:  # Safety check
                variance_map[i//block_size, j//block_size] = np.var(block.astype(np.float32))
    
    return variance_map

def robust_normalize(feature_vector):
    """Robust normalization that handles outliers"""
    # Remove extreme outliers (beyond 3 standard deviations)
    mean = np.mean(feature_vector)
    std = np.std(feature_vector)
    
    # Clip extreme values
    clipped = np.clip(feature_vector, mean - 3*std, mean + 3*std)
    
    # Normalize
    norm = np.linalg.norm(clipped)
    if norm > 0:
        return clipped / norm
    else:
        # Fallback to uniform distribution
        return np.ones_like(feature_vector) / np.sqrt(len(feature_vector))

def extract_color_features(color_image, iris_center, iris_radius):
    """Extract a color histogram from the iris region"""
    h, w = color_image.shape[:2]
    cX, cY = int(iris_center[0]), int(iris_center[1])
    
    mask = np.zeros((h, w), dtype=np.uint8)
    if cX > 0 and cY > 0:
        cv2.circle(mask, (cX, cY), int(iris_radius), 255, -1)
    
    # Calculate a 3D BGR histogram. Using 8 bins per channel for 512 features.
    hist = cv2.calcHist([color_image], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    # Normalize the histogram to be comparable
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    return hist

def color_similarity(hist1, hist2):
    """Calculate color similarity between two color histograms using correlation"""
    if hist1 is None or hist2 is None or hist1.size == 0 or hist2.size == 0:
        return 0.0
        
    # Correlation returns 1 for perfect match, -1 for perfect mismatch.
    score = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL)
    
    # Normalize from [-1, 1] range to [0, 1] range
    return (score + 1) / 2

def enhanced_cosine_similarity(vector1, vector2, color_hist1=None, color_hist2=None):
    """Enhanced similarity with a dynamic color penalty and score fusion"""
    
    # --- DYNAMIC COLOR PENALTY ---
    color_multiplier = 1.0
    if color_hist1 is not None and color_hist2 is not None:
        color_sim = color_similarity(color_hist1, color_hist2)
        print(f"Color Histogram Similarity: {color_sim:.3f}")
        
        # This new penalty curve replaces the old "hard gate"
        if color_sim >= 0.8:
            color_multiplier = 1.0  # Very similar color, no penalty
        elif color_sim >= 0.6:
            # Moderately similar, gentle linear penalty
            color_multiplier = 0.8 + (color_sim - 0.6) * 1.0 
        elif color_sim >= 0.4:
            # Low similarity, aggressive non-linear penalty
            color_multiplier = 0.1 + (color_sim - 0.4) * 3.5
        else:
            # Very different colors, massive penalty to ensure no match
            color_multiplier = 0.05
            
        print(f"Applied Color Penalty Multiplier: {color_multiplier:.3f}")

    # --- TEXTURE SIMILARITY ---
    try:
        cos_sim = cosine_similarity(vector1, vector2)
        
        # Pearson correlation coefficient
        corr_matrix = np.corrcoef(vector1, vector2)
        corr_coef = 0.0
        if corr_matrix.shape == (2, 2) and not np.isnan(corr_matrix[0, 1]):
            corr_coef = corr_matrix[0, 1]
        
        # Euclidean distance
        euclidean_dist = np.linalg.norm(vector1 - vector2)
        euclidean_sim = max(0, 1 - (euclidean_dist / np.sqrt(2)))
        
        # Weighted combination of texture metrics
        texture_sim = (0.5 * cos_sim + 0.3 * abs(corr_coef) + 0.2 * euclidean_sim)
        
        # --- SCORE FUSION ---
        # The final score is the texture match attenuated by the color penalty.
        final_score = texture_sim * color_multiplier
        
        return max(0.0, min(1.0, final_score))
        
    except Exception as e:
        print(f"Texture similarity calculation failed: {e}")
        return 0.0


def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
