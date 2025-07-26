#!/usr/bin/env python3
"""
Studio Photography Detector

Analyzes an image to determine if it was taken in a studio based on lighting characteristics.
Examines shadows, highlights, color temperature consistency, and background separation.
"""

import sys
import json
import argparse
import os
import time
import numpy as np
import cv2
from scipy import ndimage, signal, fft
from scipy.ndimage import gaussian_filter
from scipy.stats import skew, kurtosis
from skimage import filters, measure, segmentation, exposure
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored output
init()

# Pre-compute constants for better performance (calculated once at module load)
import time
start_time = time.time()

# Pre-computed kernels for consistent operations
GAUSSIAN_KERNEL = cv2.getGaussianKernel(5, 0)
SOBEL_KERNEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SOBEL_KERNEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
MORPH_KERNEL = np.ones((5, 5), np.uint8)

# Analysis parameters
MAX_RESOLUTION = 512  # Optimal balance of speed vs accuracy
GRID_SIZE = 8

setup_time = time.time() - start_time

# ML-based feature detection algorithms
class MLFeatureAnalyzer:
    """Uses machine learning algorithms for better feature identification"""
    
    def __init__(self):
        self.kmeans_lighting = None
        self.gmm_background = None
    
    def analyze_lighting_clusters(self, image):
        """Use K-means clustering to identify lighting patterns"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Reshape for clustering
        pixels = l_channel.reshape(-1, 1)
        
        # Use K-means to find lighting clusters
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(pixels)
        
        # Analyze cluster characteristics
        cluster_centers = kmeans.cluster_centers_.flatten()
        cluster_std = [np.std(pixels[clusters == i]) for i in range(3)]
        
        # Studio lighting typically has more distinct lighting zones
        lighting_separation = np.std(cluster_centers)
        lighting_uniformity = 1.0 - np.mean(cluster_std) / 255.0
        
        return {
            'lighting_separation': min(lighting_separation / 50.0, 1.0),
            'lighting_uniformity': max(lighting_uniformity, 0.0),
            'cluster_count': len(np.unique(clusters))
        }
    
    def analyze_background_segmentation(self, image):
        """Use Gaussian Mixture Model for background/foreground segmentation"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use texture features for segmentation
        h, w = gray.shape
        
        # Create feature vector combining intensity and texture
        features = []
        
        # Add intensity
        features.append(gray.flatten())
        
        # Add local standard deviation (texture)
        texture = ndimage.generic_filter(gray, np.std, size=5)
        features.append(texture.flatten())
        
        # Add gradient magnitude
        grad_x = ndimage.sobel(gray, axis=1)
        grad_y = ndimage.sobel(gray, axis=0)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        features.append(gradient.flatten())
        
        # Combine features
        feature_matrix = np.column_stack(features)
        
        # Use Gaussian Mixture Model for segmentation
        gmm = GaussianMixture(n_components=2, random_state=42)
        segments = gmm.fit_predict(feature_matrix)
        
        # Reshape back to image
        segment_map = segments.reshape(h, w)
        
        # Analyze segmentation quality
        # Studio images typically have cleaner background/foreground separation
        segment_0_ratio = np.sum(segment_map == 0) / (h * w)
        segment_1_ratio = np.sum(segment_map == 1) / (h * w)
        
        # Better separation means one segment is clearly dominant (background)
        separation_quality = abs(segment_0_ratio - segment_1_ratio)
        
        # Check edge coherence between segments
        edges = cv2.Canny(gray, 50, 150)
        segment_edges = cv2.Canny((segment_map * 255).astype(np.uint8), 50, 150)
        
        # Studio images have segment boundaries that align with actual edges
        edge_alignment = np.sum((edges > 0) & (segment_edges > 0)) / max(np.sum(edges > 0), 1)
        
        return {
            'separation_quality': separation_quality,
            'edge_alignment': edge_alignment,
            'background_uniformity': self._analyze_segment_uniformity(gray, segment_map)
        }
    
    def _analyze_segment_uniformity(self, gray, segment_map):
        """Analyze uniformity within the larger segment (assumed background)"""
        # Find which segment is larger (likely background)
        segment_0_size = np.sum(segment_map == 0)
        segment_1_size = np.sum(segment_map == 1)
        
        bg_segment = 0 if segment_0_size > segment_1_size else 1
        bg_mask = segment_map == bg_segment
        
        if np.sum(bg_mask) > 0:
            bg_pixels = gray[bg_mask]
            uniformity = 1.0 - np.std(bg_pixels) / 128.0
            return max(uniformity, 0.0)
        return 0.5
    
    def detect_professional_patterns(self, image):
        """Use statistical analysis to detect professional photography patterns"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Analyze histogram characteristics
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist_norm = hist / hist.sum()
        
        # Calculate statistical moments
        histogram_skewness = skew(hist_norm)
        histogram_kurtosis = kurtosis(hist_norm)
        
        # Professional photos often have specific histogram characteristics
        # Studio photos tend to have more controlled dynamic range
        professional_histogram = (
            abs(histogram_skewness) < 0.5 and  # Not too skewed
            histogram_kurtosis > 1.5            # Some peaking (controlled lighting)
        )
        
        # Analyze spatial frequency content
        f_transform = fft.fft2(gray)
        f_shift = fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Professional photos have characteristic frequency distributions
        # Studio backgrounds create specific frequency signatures
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2
        
        # Create rings at different frequencies
        y, x = np.ogrid[:h, :w]
        
        # Low frequency (0-30 pixels from center)
        low_freq_mask = ((x - center_x)**2 + (y - center_y)**2) <= 30**2
        # Mid frequency (30-100 pixels)
        mid_freq_mask = (((x - center_x)**2 + (y - center_y)**2) > 30**2) & \
                       (((x - center_x)**2 + (y - center_y)**2) <= 100**2)
        # High frequency (>100 pixels)
        high_freq_mask = ((x - center_x)**2 + (y - center_y)**2) > 100**2
        
        low_power = np.sum(magnitude_spectrum[low_freq_mask])
        mid_power = np.sum(magnitude_spectrum[mid_freq_mask])
        high_power = np.sum(magnitude_spectrum[high_freq_mask])
        total_power = low_power + mid_power + high_power
        
        # Studio images typically have higher low-frequency content
        frequency_signature = low_power / (total_power + 1e-6)
        
        return {
            'professional_histogram': float(professional_histogram),
            'histogram_skewness': abs(histogram_skewness),
            'histogram_kurtosis': histogram_kurtosis,
            'frequency_signature': frequency_signature
        }

# Global ML analyzer instance
ml_analyzer = MLFeatureAnalyzer()

# Utility functions to reduce duplication
def analyze_grid_regions(image, grid_size, analysis_func):
    """
    Generic grid-based analysis to reduce code duplication.
    
    Args:
        image: Input image (grayscale or color)
        grid_size: Number of grid cells per dimension
        analysis_func: Function to apply to each grid cell
        
    Returns:
        2D array of analysis results
    """
    h, w = image.shape[:2]
    cell_h, cell_w = h // grid_size, w // grid_size
    results = np.zeros((grid_size, grid_size))
    
    for i in range(grid_size):
        for j in range(grid_size):
            cell = image[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            if cell.size > 0:
                results[i, j] = analysis_func(cell)
    
    return results

def calculate_gradient_magnitude(gray):
    """
    Calculate gradient magnitude using optimized convolution.
    Reduces duplication across shadow, edge, and highlight analysis.
    """
    grad_x = cv2.filter2D(gray.astype(np.float32), -1, SOBEL_KERNEL_X)
    grad_y = cv2.filter2D(gray.astype(np.float32), -1, SOBEL_KERNEL_Y)
    return np.sqrt(grad_x**2 + grad_y**2), grad_x, grad_y

def extract_border_regions(image, border_ratio=8):
    """
    Extract border regions for background analysis.
    Reduces duplication in background and texture analysis.
    """
    h, w = image.shape[:2]
    border_width = min(h, w) // border_ratio
    
    regions = [
        image[:border_width, :],          # top
        image[-border_width:, :],         # bottom
        image[:, :border_width],          # left  
        image[:, -border_width:]          # right
    ]
    
    return [r for r in regions if r.size > 0]

def log_to_stderr(message, color=Fore.WHITE):
    """Print colored message to stderr"""
    sys.stderr.write(color + message + Style.RESET_ALL + '\n')
    sys.stderr.flush()

# Vectorized analysis functions for ML
def analyze_shadows_ml(image):
    """Vectorized shadow analysis for ML features"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Vectorized gradient calculation
    grad_x = cv2.filter2D(gray.astype(np.float32), -1, SOBEL_KERNEL_X)
    grad_y = cv2.filter2D(gray.astype(np.float32), -1, SOBEL_KERNEL_Y)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Shadow softness
    softness = 1.0 - np.clip(np.mean(magnitude) / 100.0, 0, 1)
    
    # Direction consistency using vectorized operations
    angles = np.arctan2(grad_y, grad_x)
    significant = magnitude > np.percentile(magnitude, 75)
    if np.any(significant):
        sig_angles = angles[significant]
        consistency = np.abs(np.mean(np.exp(1j * sig_angles)))
    else:
        consistency = 0.5
    
    # Uniformity using fast convolution
    local_var = cv2.blur((gray - np.mean(gray))**2, (20, 20))
    uniformity = 1.0 - np.clip(np.std(local_var) / (np.mean(local_var) + 1e-6), 0, 1)
    
    return {
        'softness': softness,
        'direction_consistency': consistency,
        'uniformity': uniformity
    }

def analyze_highlights_ml(image):
    """Vectorized highlight analysis for ML features"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding for highlights
    threshold = np.percentile(gray, 90)
    highlights = gray > threshold
    
    # Connected components analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        highlights.astype(np.uint8), connectivity=8
    )
    
    catchlight_score = 0.0
    shape_scores = []
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 10 < area < 500:
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
            
            if aspect_ratio > 0.7:
                catchlight_score = max(catchlight_score, aspect_ratio)
            shape_scores.append(aspect_ratio)
    
    shape_regularity = np.mean(shape_scores) if shape_scores else 0.0
    
    # Vectorized distribution analysis
    h, w = gray.shape
    grid_h, grid_w = h // GRID_SIZE, w // GRID_SIZE
    distribution = []
    
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            cell = highlights[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
            distribution.append(np.sum(cell))
    
    dist_uniformity = 1.0 - np.clip(np.std(distribution) / (np.mean(distribution) + 1e-6), 0, 1)
    
    return {
        'catchlight_pattern': catchlight_score,
        'distribution_uniformity': dist_uniformity,
        'shape_regularity': shape_regularity
    }

def analyze_color_temperature_ml(image):
    """Vectorized color temperature analysis for ML features"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Vectorized temperature analysis using sliding windows
    h, w = image.shape[:2]
    window_size = min(h, w) // 8
    
    # Extract temperature values using vectorized operations
    temps = []
    for i in range(0, h - window_size, window_size // 2):
        for j in range(0, w - window_size, window_size // 2):
            window_b = b_channel[i:i+window_size, j:j+window_size]
            window_l = l_channel[i:i+window_size, j:j+window_size]
            
            if np.mean(window_l) > 30:
                temps.append(np.mean(window_b))
    
    if temps:
        consistency = 1.0 / (1.0 + np.var(temps) / 100.0)
        
        # Vectorized mixed lighting detection
        local_var = cv2.blur((b_channel - np.mean(b_channel))**2, (20, 20))
        single_source = 1.0 - np.clip(np.mean(local_var) / 50.0, 0, 1)
        
        # Gradient uniformity
        grad_mag = np.abs(np.gradient(b_channel.astype(float))).mean()
        gradient_uniformity = 1.0 - np.clip(grad_mag / 10.0, 0, 1)
    else:
        consistency = 0.5
        single_source = 0.5
        gradient_uniformity = 0.5
    
    return {
        'temperature_consistency': consistency,
        'mixed_lighting': single_source,
        'gradient_uniformity': gradient_uniformity
    }

def analyze_background_separation_ml(image):
    """Vectorized background analysis for ML features"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Fast edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Vectorized edge sharpness
    grad_mag = np.sqrt(cv2.Sobel(gray, cv2.CV_64F, 1, 0)**2 + 
                      cv2.Sobel(gray, cv2.CV_64F, 0, 1)**2)
    edge_pixels = grad_mag[edges > 0]
    edge_sharpness = np.mean(edge_pixels) / 255.0 if len(edge_pixels) > 0 else 0.5
    
    # Fast background mask using morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    background_mask = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=5) == 0
    
    if np.any(background_mask):
        bg_values = gray[background_mask]
        uniformity = 1.0 - np.clip(np.std(bg_values) / 128.0, 0, 1)
    else:
        uniformity = 0.5
    
    # Rim lighting detection
    edge_dilation = cv2.dilate(edges, kernel, iterations=2)
    rim_region = edge_dilation > 0
    
    if np.any(rim_region):
        rim_intensities = gray[rim_region]
        rim_lighting = np.percentile(rim_intensities, 90) / 255.0
    else:
        rim_lighting = 0.5
    
    return {
        'edge_sharpness': edge_sharpness,
        'background_uniformity': uniformity,
        'rim_lighting': rim_lighting
    }

# New feature extraction functions
def extract_histogram_features(image):
    """Extract histogram-based features"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate normalized histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    
    # Histogram uniformity
    hist_uniformity = 1.0 - np.std(hist) * 100
    
    # Exposure consistency
    low_clip = np.sum(hist[:5])
    high_clip = np.sum(hist[-5:])
    exposure_consistency = 1.0 - (low_clip + high_clip)
    
    # Background peak detection
    peaks = signal.find_peaks(hist, height=0.01)[0]
    if len(peaks) > 0:
        peak_heights = hist[peaks]
        bg_peak_prominence = min(np.max(peak_heights) / np.mean(hist) / 10.0, 1.0)
    else:
        bg_peak_prominence = 0.5
    
    # Dynamic range
    non_zero = np.where(hist > 0.001)[0]
    if len(non_zero) > 0:
        dynamic_range = (non_zero[-1] - non_zero[0]) / 255.0
    else:
        dynamic_range = 0.5
    
    return [hist_uniformity, exposure_consistency, bg_peak_prominence, dynamic_range]

def extract_composition_features(image):
    """Extract composition-related features"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Focus gradient analysis using Laplacian variance
    grid_h, grid_w = h // GRID_SIZE, w // GRID_SIZE
    focus_map = np.zeros((GRID_SIZE, GRID_SIZE))
    
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            cell = gray[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
            laplacian = cv2.Laplacian(cell, cv2.CV_64F)
            focus_map[i, j] = np.var(laplacian)
    
    # Find focus center and calculate gradient
    max_idx = np.unravel_index(np.argmax(focus_map), focus_map.shape)
    center_i, center_j = max_idx
    
    distances = []
    focus_values = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            dist = np.sqrt((i - center_i)**2 + (j - center_j)**2)
            distances.append(dist)
            focus_values.append(focus_map[i, j])
    
    if len(distances) > 1 and np.std(distances) > 0:
        correlation = np.corrcoef(distances, focus_values)[0, 1]
        focus_gradient = abs(correlation) if not np.isnan(correlation) else 0.5
    else:
        focus_gradient = 0.5
    
    # Rule of thirds using saliency
    saliency = cv2.filter2D(gray.astype(np.float32), -1, SOBEL_KERNEL_X)**2 + \
               cv2.filter2D(gray.astype(np.float32), -1, SOBEL_KERNEL_Y)**2
    saliency_smooth = cv2.GaussianBlur(saliency, (31, 31), 0)
    
    max_sal_idx = np.unravel_index(np.argmax(saliency_smooth), saliency_smooth.shape)
    subject_y, subject_x = max_sal_idx
    
    # Distance to rule of thirds
    thirds_x = [w//3, 2*w//3]
    thirds_y = [h//3, 2*h//3]
    
    min_dist = min([np.sqrt((subject_x - tx)**2 + (subject_y - ty)**2) 
                   for tx in thirds_x for ty in thirds_y])
    
    diagonal = np.sqrt(h**2 + w**2)
    rule_of_thirds = 1.0 - min(min_dist / (diagonal * 0.2), 1.0)
    
    # Symmetry analysis
    left = gray[:, :w//2]
    right = gray[:, w//2:2*(w//2)]
    
    if left.shape[1] == right.shape[1]:
        right_flipped = np.fliplr(right)
        symmetry = 1.0 - np.mean(np.abs(left - right_flipped)) / 255.0
    else:
        symmetry = 0.5
    
    return [focus_gradient, rule_of_thirds, symmetry]

def extract_edge_features(image):
    """Extract edge and segmentation features"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 30, 100)
    
    # Seamless background detection
    h, w = gray.shape
    border_width = min(h, w) // 10
    
    border_edges = np.concatenate([
        edges[:border_width, :].flatten(),
        edges[-border_width:, :].flatten(),
        edges[:, :border_width].flatten(),
        edges[:, -border_width:].flatten()
    ])
    
    border_edge_density = np.sum(border_edges > 0) / len(border_edges)
    seamless_bg = 1.0 - border_edge_density
    
    # Edge continuity using Hough lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
    
    if lines is not None:
        line_lengths = [np.sqrt((x2-x1)**2 + (y2-y1)**2) 
                       for x1, y1, x2, y2 in lines[:, 0]]
        avg_line_length = np.mean(line_lengths) / np.sqrt(h**2 + w**2)
        edge_continuity = min(avg_line_length * 5, 1.0)
    else:
        edge_continuity = 0.0
    
    return [seamless_bg, edge_continuity]

def analyze_lighting(analysis_image, verbose=False):
    """
    Comprehensive lighting analysis - the core foundation of studio detection.
    Now enhanced with ML-based feature extraction for improved accuracy.
    
    This function runs all four fundamental lighting analysis components:
    1. Shadow Analysis - examines shadow characteristics
    2. Highlight Analysis - examines bright spots and reflections  
    3. Color Temperature Analysis - examines lighting color consistency
    4. Background Separation Analysis - examines subject/background relationship
    
    Args:
        analysis_image: The preprocessed image to analyze
        verbose: Whether to show detailed analysis output
        
    Returns:
        dict: Contains all analysis results and combined confidence score
    """
    
    # Use ML algorithms for enhanced feature detection
    use_ml_features = True
    
    if use_ml_features and not verbose:
        # Fast path: use ML-enhanced analysis
        shadow_results = analyze_shadows_ml(analysis_image)
        highlight_results = analyze_highlights_ml(analysis_image)
        color_results = analyze_color_temperature_ml(analysis_image)
        background_results = analyze_background_separation_ml(analysis_image)
        
        # Add ML-based feature detection
        ml_lighting = ml_analyzer.analyze_lighting_clusters(analysis_image)
        ml_segmentation = ml_analyzer.analyze_background_segmentation(analysis_image)
        ml_patterns = ml_analyzer.detect_professional_patterns(analysis_image)
        
        # Combine traditional and ML features for confidence
        ml_confidence = (
            (shadow_results['softness'] * 0.15) +
            (shadow_results['direction_consistency'] * 0.1) +
            (shadow_results['uniformity'] * 0.1) +
            (highlight_results['catchlight_pattern'] * 0.1) +
            (highlight_results['distribution_uniformity'] * 0.1) +
            (color_results['temperature_consistency'] * 0.1) +
            (background_results['background_uniformity'] * 0.1) +
            (ml_lighting['lighting_uniformity'] * 0.075) +
            (ml_segmentation['separation_quality'] * 0.075) +
            (ml_patterns['frequency_signature'] * 0.05)
        )
        
        # Create display-friendly results
        shadow_display = {
            'softness': shadow_results['softness'],
            'direction_consistency': shadow_results['direction_consistency'],
            'uniformity': shadow_results['uniformity'],
            'overall_score': (shadow_results['softness'] * 0.4 + 
                            shadow_results['direction_consistency'] * 0.3 + 
                            shadow_results['uniformity'] * 0.3)
        }
        
        highlight_display = {
            'catchlight_detected': highlight_results['catchlight_pattern'] > 0.5,
            'catchlight_pattern': highlight_results['catchlight_pattern'],
            'distribution_uniformity': highlight_results['distribution_uniformity'],
            'shape_regularity': highlight_results['shape_regularity'],
            'overall_score': (highlight_results['catchlight_pattern'] * 0.3 + 
                            highlight_results['distribution_uniformity'] * 0.35 + 
                            highlight_results['shape_regularity'] * 0.35)
        }
        
        color_display = {
            'temperature_consistency': color_results['temperature_consistency'],
            'mixed_lighting': color_results['mixed_lighting'],
            'gradient_uniformity': color_results['gradient_uniformity'],
            'overall_score': (color_results['temperature_consistency'] * 0.4 + 
                            color_results['mixed_lighting'] * 0.3 + 
                            color_results['gradient_uniformity'] * 0.3)
        }
        
        background_display = {
            'edge_sharpness': background_results['edge_sharpness'],
            'background_uniformity': background_results['background_uniformity'],
            'rim_lighting': background_results['rim_lighting'],
            'overall_score': (background_results['edge_sharpness'] * 0.35 + 
                            background_results['background_uniformity'] * 0.35 + 
                            background_results['rim_lighting'] * 0.3)
        }
        
        return {
            'shadow_results': shadow_display,
            'highlight_results': highlight_display,
            'color_results': color_display,
            'background_results': background_display,
            'lighting_confidence': ml_confidence,
            'ml_enhanced': True
        }
    
    # Original verbose analysis path
    # 1. Shadow Analysis
    log_to_stderr("[1/4] Shadow Analysis", Fore.GREEN + Style.BRIGHT)
    log_to_stderr("-" * 40, Fore.GREEN)
    shadow_results = analyze_shadows(analysis_image, verbose) if not use_ml else analyze_shadows_ml(analysis_image)
    if use_ml:
        # Convert ML results to display format
        shadow_results['overall_score'] = (shadow_results['softness'] * 0.4 + 
                                          shadow_results['direction_consistency'] * 0.3 + 
                                          shadow_results['uniformity'] * 0.3)
    log_to_stderr(f"✓ Shadow softness: {shadow_results['softness']:.2f} ({format_score_interpretation(shadow_results['softness'])})", Fore.WHITE)
    log_to_stderr(f"✓ Direction consistency: {shadow_results['direction_consistency']:.2f} ({format_score_interpretation(shadow_results['direction_consistency'])})", Fore.WHITE)
    log_to_stderr(f"✓ Shadow uniformity: {shadow_results['uniformity']:.2f} ({format_score_interpretation(shadow_results['uniformity'])})", Fore.WHITE)
    log_to_stderr(f"→ Shadow score: {shadow_results['overall_score']*100:.0f}% ({format_score_interpretation(shadow_results['overall_score'])})", Fore.YELLOW + Style.BRIGHT)
    log_to_stderr("")
    
    # 2. Highlight Analysis
    log_to_stderr("[2/4] Highlight Analysis", Fore.GREEN + Style.BRIGHT)
    log_to_stderr("-" * 40, Fore.GREEN)
    highlight_results = analyze_highlights(analysis_image, verbose) if not use_ml else analyze_highlights_ml(analysis_image)
    if use_ml:
        highlight_results['catchlight_detected'] = highlight_results['catchlight_pattern'] > 0.5
        highlight_results['overall_score'] = (highlight_results['catchlight_pattern'] * 0.3 + 
                                            highlight_results['distribution_uniformity'] * 0.35 + 
                                            highlight_results['shape_regularity'] * 0.35)
    catchlight_text = "Yes (circular pattern)" if highlight_results['catchlight_detected'] else "No"
    log_to_stderr(f"✓ Catchlights detected: {catchlight_text}", Fore.WHITE)
    log_to_stderr(f"✓ Distribution uniformity: {highlight_results['distribution_uniformity']:.2f} ({format_score_interpretation(highlight_results['distribution_uniformity'])})", Fore.WHITE)
    log_to_stderr(f"✓ Shape regularity: {highlight_results['shape_regularity']:.2f} ({format_score_interpretation(highlight_results['shape_regularity'])})", Fore.WHITE)
    log_to_stderr(f"→ Highlight score: {highlight_results['overall_score']*100:.0f}% ({format_score_interpretation(highlight_results['overall_score'])})", Fore.YELLOW + Style.BRIGHT)
    log_to_stderr("")
    
    # 3. Color Temperature Analysis
    log_to_stderr("[3/4] Color Temperature Analysis", Fore.GREEN + Style.BRIGHT)
    log_to_stderr("-" * 40, Fore.GREEN)
    color_results = analyze_color_temperature(analysis_image, verbose) if not use_ml else analyze_color_temperature_ml(analysis_image)
    if use_ml:
        color_results['overall_score'] = (color_results['temperature_consistency'] * 0.4 + 
                                        color_results['mixed_lighting'] * 0.3 + 
                                        color_results['gradient_uniformity'] * 0.3)
    log_to_stderr(f"✓ Temperature consistency: {color_results['temperature_consistency']:.2f} ({format_score_interpretation(color_results['temperature_consistency'])})", Fore.WHITE)
    log_to_stderr(f"✓ Single light source: {color_results['mixed_lighting']:.2f} ({format_score_interpretation(color_results['mixed_lighting'])})", Fore.WHITE)
    log_to_stderr(f"✓ Gradient uniformity: {color_results['gradient_uniformity']:.2f} ({format_score_interpretation(color_results['gradient_uniformity'])})", Fore.WHITE)
    log_to_stderr(f"→ Color temperature score: {color_results['overall_score']*100:.0f}% ({format_score_interpretation(color_results['overall_score'])})", Fore.YELLOW + Style.BRIGHT)
    log_to_stderr("")
    
    # 4. Background Separation Analysis
    log_to_stderr("[4/4] Background Separation Analysis", Fore.GREEN + Style.BRIGHT)
    log_to_stderr("-" * 40, Fore.GREEN)
    background_results = analyze_background_separation(analysis_image, verbose) if not use_ml else analyze_background_separation_ml(analysis_image)
    if use_ml:
        background_results['overall_score'] = (background_results['edge_sharpness'] * 0.35 + 
                                             background_results['background_uniformity'] * 0.35 + 
                                             background_results['rim_lighting'] * 0.3)
    log_to_stderr(f"✓ Edge sharpness: {background_results['edge_sharpness']:.2f} ({format_score_interpretation(background_results['edge_sharpness'])})", Fore.WHITE)
    log_to_stderr(f"✓ Background uniformity: {background_results['background_uniformity']:.2f} ({format_score_interpretation(background_results['background_uniformity'])})", Fore.WHITE)
    log_to_stderr(f"✓ Rim lighting presence: {background_results['rim_lighting']:.2f} ({format_score_interpretation(background_results['rim_lighting'])})", Fore.WHITE)
    log_to_stderr(f"→ Background separation score: {background_results['overall_score']*100:.0f}% ({format_score_interpretation(background_results['overall_score'])})", Fore.YELLOW + Style.BRIGHT)
    log_to_stderr("")
    
    # Calculate lighting analysis confidence score
    if use_ml_features:
        # Use ML-enhanced features for final confidence
        ml_lighting = ml_analyzer.analyze_lighting_clusters(analysis_image)
        ml_segmentation = ml_analyzer.analyze_background_segmentation(analysis_image)
        ml_patterns = ml_analyzer.detect_professional_patterns(analysis_image)
        
        # Combine traditional lighting analysis with ML features
        traditional_confidence = (shadow_results['overall_score'] * 0.25 +
                                 highlight_results['overall_score'] * 0.25 +
                                 color_results['overall_score'] * 0.25 +
                                 background_results['overall_score'] * 0.25)
        
        ml_boost = (ml_lighting['lighting_uniformity'] * 0.3 +
                   ml_segmentation['separation_quality'] * 0.4 +
                   ml_patterns['frequency_signature'] * 0.3)
        
        lighting_confidence = traditional_confidence * 0.7 + ml_boost * 0.3
        log_to_stderr("→ Using ML-enhanced feature detection", Fore.CYAN + Style.BRIGHT)
    else:
        lighting_confidence = (shadow_results['overall_score'] * 0.25 +
                              highlight_results['overall_score'] * 0.25 +
                              color_results['overall_score'] * 0.25 +
                              background_results['overall_score'] * 0.25)
    
    return {
        'shadow_results': shadow_results,
        'highlight_results': highlight_results,
        'color_results': color_results,
        'background_results': background_results,
        'lighting_confidence': lighting_confidence,
        'ml_enhanced': use_ml
    }

def resize_image_for_analysis(image):
    """
    Resize image to optimal resolution for analysis.
    Reduces to 512px max dimension for best speed/accuracy balance.
    
    Returns:
        tuple: (resized_image, was_resized)
    """
    height, width = image.shape[:2]
    if max(height, width) > MAX_RESOLUTION:
        scale = MAX_RESOLUTION / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized, True
    return image, False

def analyze_frequency_domain(image, verbose=False):
    """
    Phase 3: Analyze frequency characteristics of the image.
    
    This phase addresses cases where lighting analysis alone is inconclusive.
    
    Studio photography typically shows:
    - Simple frequency patterns (smooth backgrounds)
    - Regular gradients (professional lighting)
    - Less high-frequency noise (controlled environment)
    
    Natural photography shows:
    - Complex frequency patterns (varied textures)
    - Irregular patterns (natural elements)
    - More high-frequency components (environmental details)
    
    Returns:
        dict: Contains frequency analysis metrics and overall score
    """
    # Convert to grayscale for frequency analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply 2D FFT to analyze frequency distribution
    f_transform = fft.fft2(gray)
    f_shift = fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    
    # Log transform for better visualization of spectrum
    log_spectrum = np.log(magnitude_spectrum + 1)
    
    # Analyze frequency complexity - studio images have simpler frequency patterns
    # High frequencies indicate texture complexity (more natural)
    # Low frequencies indicate smooth gradients (more studio)
    
    # Calculate power in different frequency bands
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create frequency masks
    low_freq_mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(low_freq_mask, (ccol, crow), 30, 1, -1)
    
    high_freq_mask = np.ones((rows, cols), np.uint8)
    cv2.circle(high_freq_mask, (ccol, crow), 100, 0, -1)
    
    # Calculate power ratios
    low_freq_power = np.sum(magnitude_spectrum * low_freq_mask)
    high_freq_power = np.sum(magnitude_spectrum * high_freq_mask)
    total_power = np.sum(magnitude_spectrum)
    
    low_freq_ratio = low_freq_power / total_power
    high_freq_ratio = high_freq_power / total_power
    
    # Studio images typically have higher low-frequency content
    frequency_complexity = 1.0 - high_freq_ratio
    frequency_complexity = np.clip(frequency_complexity, 0, 1)
    
    # Analyze background smoothness using gradient analysis
    # Studio backgrounds are typically very smooth
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Background is typically the outer regions of the image
    h, w = gray.shape
    border_mask = np.zeros((h, w), np.uint8)
    border_width = min(h, w) // 8
    border_mask[:border_width, :] = 1
    border_mask[-border_width:, :] = 1
    border_mask[:, :border_width] = 1
    border_mask[:, -border_width:] = 1
    
    background_gradient = np.mean(gradient_magnitude[border_mask == 1])
    background_smoothness = 1.0 - min(background_gradient / 50.0, 1.0)
    
    # Calculate overall frequency score
    overall_score = (frequency_complexity * 0.6 + background_smoothness * 0.4)
    
    if verbose:
        log_to_stderr(f"  • Frequency analysis:", Fore.CYAN)
        log_to_stderr(f"    - Low frequency ratio: {low_freq_ratio:.3f}", Fore.CYAN)
        log_to_stderr(f"    - High frequency ratio: {high_freq_ratio:.3f}", Fore.CYAN)
        log_to_stderr(f"    - Background gradient: {background_gradient:.2f}", Fore.CYAN)
    
    return {
        'frequency_complexity': frequency_complexity,
        'background_smoothness': background_smoothness,
        'low_freq_ratio': low_freq_ratio,
        'high_freq_ratio': high_freq_ratio,
        'overall_score': overall_score
    }

def analyze_depth_of_field(image, verbose=False):
    """
    Analyze depth of field characteristics typical of studio photography.
    
    Studio photography often features:
    - Controlled depth of field with professional lenses
    - Characteristic bokeh patterns
    - Sharp subject-to-background transitions
    - Professional lens aberration patterns
    
    Returns:
        dict: Contains depth of field analysis metrics and overall score
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use Laplacian variance to measure focus across regions
    # Divide image into grid to analyze focus distribution
    h, w = gray.shape
    grid_size = 8
    cell_h, cell_w = h // grid_size, w // grid_size
    
    focus_map = []
    for i in range(grid_size):
        row = []
        for j in range(grid_size):
            cell = gray[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            # Calculate Laplacian variance (measure of focus)
            laplacian = cv2.Laplacian(cell, cv2.CV_64F)
            focus_value = np.var(laplacian)
            row.append(focus_value)
        focus_map.append(row)
    
    focus_map = np.array(focus_map)
    
    # Find the region with highest focus (likely the subject)
    max_focus_idx = np.unravel_index(np.argmax(focus_map), focus_map.shape)
    center_i, center_j = max_focus_idx
    
    # Calculate focus falloff from center
    distances = []
    focus_values = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            distance = np.sqrt((i - center_i)**2 + (j - center_j)**2)
            distances.append(distance)
            focus_values.append(focus_map[i, j])
    
    # Measure how quickly focus falls off with distance
    # Studio photos often have sharp falloff (controlled DOF)
    if len(distances) > 1:
        focus_gradient = np.corrcoef(distances, focus_values)[0, 1]
        focus_gradient = abs(focus_gradient) if not np.isnan(focus_gradient) else 0
    else:
        focus_gradient = 0
    
    # Analyze blur characteristics in out-of-focus regions
    # Studio lenses create characteristic bokeh patterns
    blur_regions = focus_map < (np.max(focus_map) * 0.3)
    if np.any(blur_regions):
        blur_uniformity = 1.0 - np.std(focus_map[blur_regions]) / (np.mean(focus_map[blur_regions]) + 1e-6)
        blur_uniformity = np.clip(blur_uniformity, 0, 1)
    else:
        blur_uniformity = 0.5
    
    # Calculate overall DOF score
    overall_score = (focus_gradient * 0.5 + blur_uniformity * 0.5)
    overall_score = np.clip(overall_score, 0, 1)
    
    if verbose:
        log_to_stderr(f"  • Depth of field analysis:", Fore.CYAN)
        log_to_stderr(f"    - Focus gradient correlation: {focus_gradient:.3f}", Fore.CYAN)
        log_to_stderr(f"    - Blur uniformity: {blur_uniformity:.3f}", Fore.CYAN)
        log_to_stderr(f"    - Max focus at grid: ({center_i}, {center_j})", Fore.CYAN)
    
    return {
        'focus_gradient': focus_gradient,
        'blur_uniformity': blur_uniformity,
        'focus_map': focus_map.tolist(),
        'overall_score': overall_score
    }

def analyze_composition(image, verbose=False):
    """
    Analyze composition patterns typical of studio photography.
    
    Studio photographers often follow:
    - Rule of thirds positioning
    - Symmetrical compositions
    - Controlled negative space
    - Professional framing principles
    
    Returns:
        dict: Contains composition analysis metrics and overall score
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Simple saliency detection using center-surround difference
    # This approximates where the subject is located
    center_region = gray[h//4:3*h//4, w//4:3*w//4]
    center_mean = np.mean(center_region)
    
    # Calculate saliency map using gradient magnitude
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    saliency = np.sqrt(grad_x**2 + grad_y**2)
    
    # Find the most salient region (likely subject)
    saliency_smooth = gaussian_filter(saliency, sigma=min(h, w) / 20)
    max_saliency_idx = np.unravel_index(np.argmax(saliency_smooth), saliency_smooth.shape)
    subject_y, subject_x = max_saliency_idx
    
    # Check rule of thirds positioning
    # Rule of thirds lines are at 1/3 and 2/3 of image dimensions
    third_lines_x = [w//3, 2*w//3]
    third_lines_y = [h//3, 2*h//3]
    
    # Distance to nearest rule of thirds intersection
    min_distance = float('inf')
    for tx in third_lines_x:
        for ty in third_lines_y:
            distance = np.sqrt((subject_x - tx)**2 + (subject_y - ty)**2)
            min_distance = min(min_distance, distance)
    
    # Normalize distance by image diagonal
    diagonal = np.sqrt(h**2 + w**2)
    rule_of_thirds_score = 1.0 - min(min_distance / (diagonal * 0.2), 1.0)
    
    # Analyze symmetry
    # Check both horizontal and vertical symmetry
    left_half = gray[:, :w//2]
    right_half = gray[:, w//2:]
    right_half_flipped = np.fliplr(right_half)
    
    # Resize to match if needed
    min_width = min(left_half.shape[1], right_half_flipped.shape[1])
    left_half = left_half[:, :min_width]
    right_half_flipped = right_half_flipped[:, :min_width]
    
    # Calculate horizontal symmetry
    if left_half.shape == right_half_flipped.shape:
        horizontal_symmetry = 1.0 - np.mean(np.abs(left_half - right_half_flipped)) / 255.0
    else:
        horizontal_symmetry = 0.5
    
    # Calculate negative space (areas with low variance)
    # Studio photos often have clean, simple backgrounds
    variance_map = ndimage.generic_filter(gray, np.var, size=20)
    low_variance_ratio = np.sum(variance_map < np.percentile(variance_map, 25)) / variance_map.size
    negative_space_score = low_variance_ratio
    
    # Calculate overall composition score
    overall_score = (rule_of_thirds_score * 0.4 + 
                    horizontal_symmetry * 0.3 + 
                    negative_space_score * 0.3)
    
    if verbose:
        log_to_stderr(f"  • Composition analysis:", Fore.CYAN)
        log_to_stderr(f"    - Subject position: ({subject_x}, {subject_y})", Fore.CYAN)
        log_to_stderr(f"    - Rule of thirds distance: {min_distance:.1f}", Fore.CYAN)
        log_to_stderr(f"    - Horizontal symmetry: {horizontal_symmetry:.3f}", Fore.CYAN)
        log_to_stderr(f"    - Negative space ratio: {low_variance_ratio:.3f}", Fore.CYAN)
    
    return {
        'rule_of_thirds_score': rule_of_thirds_score,
        'horizontal_symmetry': horizontal_symmetry,
        'negative_space_score': negative_space_score,
        'subject_position': (int(subject_x), int(subject_y)),
        'overall_score': overall_score
    }

def analyze_texture(image, verbose=False):
    """
    Analyze texture characteristics using Gray Level Co-occurrence Matrix (GLCM).
    
    Studio photography typically features:
    - Homogeneous backgrounds with low texture complexity
    - High contrast between subject and background textures
    - Controlled texture patterns from professional lighting
    
    Returns:
        dict: Contains texture analysis metrics and overall score
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Reduce bit depth for GLCM analysis (computational efficiency)
    gray_reduced = (gray // 32).astype(np.uint8)  # 8 levels instead of 256
    
    # Analyze texture in different regions
    # Background region (border areas)
    border_width = min(h, w) // 8
    background_regions = [
        gray_reduced[:border_width, :],  # top
        gray_reduced[-border_width:, :],  # bottom
        gray_reduced[:, :border_width],  # left
        gray_reduced[:, -border_width:]  # right
    ]
    
    background_textures = []
    for region in background_regions:
        if region.size > 100:  # Ensure minimum size for analysis
            try:
                # Calculate GLCM for this region
                glcm = graycomatrix(region, distances=[1], angles=[0], levels=8, symmetric=True, normed=True)
                # Calculate texture properties
                contrast = graycoprops(glcm, 'contrast')[0, 0]
                homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                energy = graycoprops(glcm, 'energy')[0, 0]
                
                background_textures.append({
                    'contrast': contrast,
                    'homogeneity': homogeneity,
                    'energy': energy
                })
            except:
                # Handle edge cases where GLCM calculation fails
                continue
    
    # Calculate average background texture properties
    if background_textures:
        avg_bg_contrast = np.mean([t['contrast'] for t in background_textures])
        avg_bg_homogeneity = np.mean([t['homogeneity'] for t in background_textures])
        avg_bg_energy = np.mean([t['energy'] for t in background_textures])
    else:
        avg_bg_contrast = 0.5
        avg_bg_homogeneity = 0.5
        avg_bg_energy = 0.5
    
    # Analyze center region (likely subject area)
    center_region = gray_reduced[h//4:3*h//4, w//4:3*w//4]
    if center_region.size > 100:
        try:
            center_glcm = graycomatrix(center_region, distances=[1], angles=[0], levels=8, symmetric=True, normed=True)
            center_contrast = graycoprops(center_glcm, 'contrast')[0, 0]
            center_homogeneity = graycoprops(center_glcm, 'homogeneity')[0, 0]
            center_energy = graycoprops(center_glcm, 'energy')[0, 0]
        except:
            center_contrast = 0.5
            center_homogeneity = 0.5
            center_energy = 0.5
    else:
        center_contrast = 0.5
        center_homogeneity = 0.5
        center_energy = 0.5
    
    # Studio backgrounds should be homogeneous (high homogeneity, low contrast)
    background_homogeneity_score = avg_bg_homogeneity
    background_simplicity_score = 1.0 - min(avg_bg_contrast / 2.0, 1.0)
    
    # Calculate texture contrast between foreground and background
    texture_contrast_ratio = abs(center_contrast - avg_bg_contrast) / (avg_bg_contrast + 0.1)
    texture_contrast_score = min(texture_contrast_ratio / 2.0, 1.0)
    
    # Calculate overall texture score
    overall_score = (background_homogeneity_score * 0.4 + 
                    background_simplicity_score * 0.4 + 
                    texture_contrast_score * 0.2)
    
    if verbose:
        log_to_stderr(f"  • Texture analysis:", Fore.CYAN)
        log_to_stderr(f"    - Background contrast: {avg_bg_contrast:.3f}", Fore.CYAN)
        log_to_stderr(f"    - Background homogeneity: {avg_bg_homogeneity:.3f}", Fore.CYAN)
        log_to_stderr(f"    - Center contrast: {center_contrast:.3f}", Fore.CYAN)
        log_to_stderr(f"    - Texture contrast ratio: {texture_contrast_ratio:.3f}", Fore.CYAN)
    
    return {
        'background_homogeneity': background_homogeneity_score,
        'background_simplicity': background_simplicity_score,
        'texture_contrast': texture_contrast_score,
        'avg_bg_contrast': avg_bg_contrast,
        'center_contrast': center_contrast,
        'overall_score': overall_score
    }

def analyze_frequency_and_composition(analysis_image, verbose=False):
    """
    Phase 3: Comprehensive frequency domain and composition analysis.
    
    This phase is triggered when lighting analysis returns moderate confidence (0.3-0.7).
    Combines four advanced analysis techniques to improve accuracy for ambiguous cases.
    
    Args:
        analysis_image: The preprocessed image to analyze
        verbose: Whether to show detailed analysis output
        
    Returns:
        dict: Contains all Phase 3 analysis results and combined confidence score
    """
    phase3_start_time = time.time()
    
    # 1. Frequency Domain Analysis
    log_to_stderr("[1/4] Fourier Transform Analysis", Fore.BLUE + Style.BRIGHT)
    log_to_stderr("-" * 40, Fore.BLUE)
    frequency_results = analyze_frequency_domain(analysis_image, verbose)
    log_to_stderr(f"✓ Frequency complexity: {frequency_results['frequency_complexity']:.2f} ({format_score_interpretation(frequency_results['frequency_complexity'])})", Fore.WHITE)
    log_to_stderr(f"✓ Background smoothness: {frequency_results['background_smoothness']:.2f} ({format_score_interpretation(frequency_results['background_smoothness'])})", Fore.WHITE)
    log_to_stderr(f"→ Frequency score: {frequency_results['overall_score']*100:.0f}% ({format_score_interpretation(frequency_results['overall_score'])})", Fore.YELLOW + Style.BRIGHT)
    log_to_stderr("")
    
    # 2. Depth of Field Analysis
    log_to_stderr("[2/4] Depth of Field Analysis", Fore.BLUE + Style.BRIGHT)
    log_to_stderr("-" * 40, Fore.BLUE)
    dof_results = analyze_depth_of_field(analysis_image, verbose)
    log_to_stderr(f"✓ Focus gradient: {dof_results['focus_gradient']:.2f} ({format_score_interpretation(dof_results['focus_gradient'])})", Fore.WHITE)
    log_to_stderr(f"✓ Blur uniformity: {dof_results['blur_uniformity']:.2f} ({format_score_interpretation(dof_results['blur_uniformity'])})", Fore.WHITE)
    log_to_stderr(f"→ DOF score: {dof_results['overall_score']*100:.0f}% ({format_score_interpretation(dof_results['overall_score'])})", Fore.YELLOW + Style.BRIGHT)
    log_to_stderr("")
    
    # 3. Composition Analysis
    log_to_stderr("[3/4] Composition Analysis", Fore.BLUE + Style.BRIGHT)
    log_to_stderr("-" * 40, Fore.BLUE)
    composition_results = analyze_composition(analysis_image, verbose)
    log_to_stderr(f"✓ Rule of thirds: {composition_results['rule_of_thirds_score']:.2f} ({format_score_interpretation(composition_results['rule_of_thirds_score'])})", Fore.WHITE)
    log_to_stderr(f"✓ Symmetry: {composition_results['horizontal_symmetry']:.2f} ({format_score_interpretation(composition_results['horizontal_symmetry'])})", Fore.WHITE)
    log_to_stderr(f"✓ Negative space: {composition_results['negative_space_score']:.2f} ({format_score_interpretation(composition_results['negative_space_score'])})", Fore.WHITE)
    log_to_stderr(f"→ Composition score: {composition_results['overall_score']*100:.0f}% ({format_score_interpretation(composition_results['overall_score'])})", Fore.YELLOW + Style.BRIGHT)
    log_to_stderr("")
    
    # 4. Texture Analysis
    log_to_stderr("[4/4] Texture Analysis", Fore.BLUE + Style.BRIGHT)
    log_to_stderr("-" * 40, Fore.BLUE)
    texture_results = analyze_texture(analysis_image, verbose)
    log_to_stderr(f"✓ Background homogeneity: {texture_results['background_homogeneity']:.2f} ({format_score_interpretation(texture_results['background_homogeneity'])})", Fore.WHITE)
    log_to_stderr(f"✓ Background simplicity: {texture_results['background_simplicity']:.2f} ({format_score_interpretation(texture_results['background_simplicity'])})", Fore.WHITE)
    log_to_stderr(f"✓ Texture contrast: {texture_results['texture_contrast']:.2f} ({format_score_interpretation(texture_results['texture_contrast'])})", Fore.WHITE)
    log_to_stderr(f"→ Texture score: {texture_results['overall_score']*100:.0f}% ({format_score_interpretation(texture_results['overall_score'])})", Fore.YELLOW + Style.BRIGHT)
    log_to_stderr("")
    
    # Calculate Phase 3 confidence score
    phase3_confidence = (frequency_results['overall_score'] * 0.3 +
                        dof_results['overall_score'] * 0.25 +
                        composition_results['overall_score'] * 0.25 +
                        texture_results['overall_score'] * 0.2)
    
    phase3_time = time.time() - phase3_start_time
    
    log_to_stderr(f"Phase 3 analysis completed in {phase3_time*1000:.0f}ms", Fore.BLUE)
    log_to_stderr("")
    
    return {
        'frequency_results': frequency_results,
        'dof_results': dof_results,
        'composition_results': composition_results,
        'texture_results': texture_results,
        'phase3_confidence': phase3_confidence,
        'processing_time': phase3_time
    }

def analyze_shadows(image, verbose=False):
    """
    Analyze shadow characteristics in the image.
    Studio shadows are typically softer, more uniform, and directionally consistent.
    
    Returns:
        dict: Contains individual metrics and overall score
    """
    # Convert to grayscale for shadow analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Calculate gradients to detect shadow edges
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Analyze shadow softness by examining gradient transitions
    # Soft shadows have gradual transitions (lower gradient magnitudes)
    mean_gradient = np.mean(gradient_magnitude)
    std_gradient = np.std(gradient_magnitude)
    
    # Normalize to 0-1 scale (lower gradients = softer shadows = higher score)
    softness_score = 1.0 - min(mean_gradient / 100.0, 1.0)
    
    # Analyze direction consistency using gradient angles
    gradient_angles = np.arctan2(grad_y, grad_x)
    # Remove near-zero gradients to focus on actual edges
    significant_gradients = gradient_magnitude > np.percentile(gradient_magnitude, 75)
    if np.any(significant_gradients):
        significant_angles = gradient_angles[significant_gradients]
        # Calculate circular variance for direction consistency
        angle_variance = 1 - np.abs(np.mean(np.exp(1j * significant_angles)))
        direction_consistency = 1.0 - angle_variance
    else:
        direction_consistency = 0.5
    
    # Analyze shadow density uniformity using local variance
    # Studio lighting creates more uniform shadow densities
    local_variance = ndimage.generic_filter(gray, np.var, size=20)
    uniformity_score = 1.0 - (np.std(local_variance) / (np.mean(local_variance) + 1e-6))
    uniformity_score = max(0, min(1, uniformity_score))
    
    # Calculate overall shadow score
    overall_score = (softness_score * 0.4 + 
                    direction_consistency * 0.3 + 
                    uniformity_score * 0.3)
    
    if verbose:
        log_to_stderr(f"  • Shadow gradient analysis:", Fore.CYAN)
        log_to_stderr(f"    - Mean gradient: {mean_gradient:.2f}", Fore.CYAN)
        log_to_stderr(f"    - Gradient std dev: {std_gradient:.2f}", Fore.CYAN)
        log_to_stderr(f"  • Direction analysis:", Fore.CYAN)
        log_to_stderr(f"    - Angle variance: {angle_variance if 'angle_variance' in locals() else 'N/A'}", Fore.CYAN)
    
    return {
        'softness': softness_score,
        'direction_consistency': direction_consistency,
        'uniformity': uniformity_score,
        'overall_score': overall_score
    }

def analyze_highlights(image, verbose=False):
    """
    Analyze highlight characteristics in the image.
    Studio lighting creates controlled, uniform highlights with regular shapes.
    
    Returns:
        dict: Contains individual metrics and overall score
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect bright regions (potential highlights)
    threshold = np.percentile(gray, 90)
    highlights = gray > threshold
    
    # Analyze catchlights (especially important for portraits)
    # Look for small, bright, circular regions
    catchlight_detected = False
    catchlight_pattern = 0.0
    
    # Find contours of highlight regions
    contours, _ = cv2.findContours(highlights.astype(np.uint8), 
                                   cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Analyze shape regularity of highlights
        circularity_scores = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10:  # Ignore very small regions
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    circularity_scores.append(circularity)
                    
                    # Check for catchlight characteristics (small, circular, bright)
                    if 20 < area < 500 and circularity > 0.7:
                        catchlight_detected = True
                        catchlight_pattern = max(catchlight_pattern, circularity)
        
        shape_regularity = np.mean(circularity_scores) if circularity_scores else 0.0
    else:
        shape_regularity = 0.0
    
    # Analyze distribution uniformity
    # Divide image into grid and check highlight distribution
    h, w = gray.shape
    grid_size = 8
    cell_h, cell_w = h // grid_size, w // grid_size
    
    highlight_distribution = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell = highlights[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            highlight_distribution.append(np.sum(cell))
    
    # Calculate distribution uniformity (lower variance = more uniform)
    distribution_uniformity = 1.0 - (np.std(highlight_distribution) / 
                                    (np.mean(highlight_distribution) + 1e-6))
    distribution_uniformity = max(0, min(1, distribution_uniformity))
    
    # Calculate overall highlight score
    overall_score = (catchlight_pattern * 0.3 + 
                    distribution_uniformity * 0.35 + 
                    shape_regularity * 0.35)
    
    if verbose:
        log_to_stderr(f"  • Highlight detection:", Fore.CYAN)
        log_to_stderr(f"    - Number of highlight regions: {len(contours)}", Fore.CYAN)
        log_to_stderr(f"    - Threshold value: {threshold:.2f}", Fore.CYAN)
        log_to_stderr(f"  • Shape analysis:", Fore.CYAN)
        log_to_stderr(f"    - Average circularity: {shape_regularity:.3f}", Fore.CYAN)
    
    return {
        'catchlight_detected': catchlight_detected,
        'catchlight_pattern': catchlight_pattern,
        'distribution_uniformity': distribution_uniformity,
        'shape_regularity': shape_regularity,
        'overall_score': overall_score
    }

def analyze_color_temperature(image, verbose=False):
    """
    Analyze color temperature consistency across the image.
    Studio lighting typically has uniform color temperature.
    
    Returns:
        dict: Contains individual metrics and overall score
    """
    # Convert to LAB color space for better color analysis
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Extract color channels
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Analyze color temperature using the b channel (blue-yellow axis)
    # Divide image into regions
    h, w = image.shape[:2]
    region_size = min(h, w) // 8
    
    color_temps = []
    for i in range(0, h - region_size, region_size // 2):
        for j in range(0, w - region_size, region_size // 2):
            region_b = b_channel[i:i+region_size, j:j+region_size]
            # Skip very dark regions
            region_l = l_channel[i:i+region_size, j:j+region_size]
            if np.mean(region_l) > 30:
                color_temps.append(np.mean(region_b))
    
    if color_temps:
        # Calculate temperature consistency
        temp_variance = np.var(color_temps)
        temp_consistency = 1.0 / (1.0 + temp_variance / 100.0)
        
        # Detect mixed lighting sources (high local variance)
        local_variance = ndimage.generic_filter(b_channel, np.var, size=20)
        mixed_lighting_score = 1.0 - (np.mean(local_variance) / 50.0)
        mixed_lighting_score = max(0, min(1, mixed_lighting_score))
        
        # Check for gradients (natural light often has gradients)
        gradient_y = np.abs(np.diff(b_channel, axis=0))
        gradient_x = np.abs(np.diff(b_channel, axis=1))
        gradient_score = 1.0 - (np.mean(gradient_y) + np.mean(gradient_x)) / 20.0
        gradient_score = max(0, min(1, gradient_score))
    else:
        temp_consistency = 0.5
        mixed_lighting_score = 0.5
        gradient_score = 0.5
    
    # Calculate overall color temperature score
    overall_score = (temp_consistency * 0.4 + 
                    mixed_lighting_score * 0.3 + 
                    gradient_score * 0.3)
    
    if verbose:
        log_to_stderr(f"  • Color temperature analysis:", Fore.CYAN)
        log_to_stderr(f"    - Number of regions analyzed: {len(color_temps)}", Fore.CYAN)
        log_to_stderr(f"    - Temperature variance: {temp_variance if 'temp_variance' in locals() else 'N/A'}", Fore.CYAN)
        log_to_stderr(f"  • Gradient analysis:", Fore.CYAN)
        log_to_stderr(f"    - Mean gradient: {(np.mean(gradient_y) + np.mean(gradient_x)) / 2 if 'gradient_y' in locals() else 'N/A'}", Fore.CYAN)
    
    return {
        'temperature_consistency': temp_consistency,
        'mixed_lighting': mixed_lighting_score,
        'gradient_uniformity': gradient_score,
        'overall_score': overall_score
    }

def analyze_background_separation(image, verbose=False):
    """
    Analyze subject/background separation characteristics.
    Studio photos often have clear separation with rim lighting.
    
    Returns:
        dict: Contains individual metrics and overall score
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect edges using Canny edge detector
    edges = cv2.Canny(gray, 50, 150)
    
    # Analyze edge sharpness by examining gradient magnitudes
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Get gradient values at edge locations
    edge_gradients = gradient_magnitude[edges > 0]
    if len(edge_gradients) > 0:
        edge_sharpness = np.mean(edge_gradients) / 255.0
        edge_sharpness = min(edge_sharpness, 1.0)
    else:
        edge_sharpness = 0.5
    
    # Analyze background uniformity
    # Use morphological operations to estimate background regions
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=10)
    
    # Estimate background as regions far from edges
    background_mask = dilated_edges == 0
    if np.any(background_mask):
        background_values = gray[background_mask]
        background_uniformity = 1.0 - (np.std(background_values) / 128.0)
        background_uniformity = max(0, min(1, background_uniformity))
    else:
        background_uniformity = 0.5
    
    # Detect rim lighting (bright edges around subject)
    # Look for high intensity values near edges
    edge_dilation = cv2.dilate(edges, kernel, iterations=2)
    edge_region = edge_dilation > 0
    if np.any(edge_region):
        edge_intensities = gray[edge_region]
        rim_lighting_score = np.percentile(edge_intensities, 90) / 255.0
    else:
        rim_lighting_score = 0.5
    
    # Calculate overall background separation score
    overall_score = (edge_sharpness * 0.35 + 
                    background_uniformity * 0.35 + 
                    rim_lighting_score * 0.3)
    
    if verbose:
        log_to_stderr(f"  • Edge detection:", Fore.CYAN)
        log_to_stderr(f"    - Number of edge pixels: {np.sum(edges > 0)}", Fore.CYAN)
        log_to_stderr(f"    - Mean edge gradient: {np.mean(edge_gradients) if len(edge_gradients) > 0 else 'N/A'}", Fore.CYAN)
        log_to_stderr(f"  • Background analysis:", Fore.CYAN)
        log_to_stderr(f"    - Background pixels: {np.sum(background_mask)}", Fore.CYAN)
    
    return {
        'edge_sharpness': edge_sharpness,
        'background_uniformity': background_uniformity,
        'rim_lighting': rim_lighting_score,
        'overall_score': overall_score
    }

def format_score_interpretation(score):
    """Convert numeric score to human-readable interpretation"""
    if score >= 0.8:
        return "very studio-like"
    elif score >= 0.6:
        return "studio-like"
    elif score >= 0.4:
        return "mixed characteristics"
    elif score >= 0.2:
        return "natural light characteristics"
    else:
        return "very natural"

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='ML-Enhanced Studio Photography Detector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python studio_detector.py portrait.jpg
  python studio_detector.py --verbose landscape.png
  python studio_detector.py --quiet photo.bmp
  python studio_detector.py --features image.jpg  # Show ML analysis details
        """
    )
    parser.add_argument('image_path', nargs='?', help='Path to the image file')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Show more detailed analysis output')
    parser.add_argument('--quiet', '-q', action='store_true', 
                       help='Suppress all stderr output, only show JSON')
    parser.add_argument('--features', '-f', action='store_true',
                       help='Show ML feature analysis details')
    
    args = parser.parse_args()
    
    # Check if image path provided for analysis
    if not args.image_path:
        parser.print_help()
        sys.exit(1)
    
    # Don't print any logs if quiet mode
    if args.quiet:
        sys.stderr = open(os.devnull, 'w')
    
    try:
        # Load the image
        image = cv2.imread(args.image_path)
        if image is None:
            log_to_stderr(f"Error: Could not load image from '{args.image_path}'", Fore.RED)
            print(json.dumps({"error": "Could not load image", "is_studio": False, "confidence": 0.0}))
            sys.exit(1)
        
        # Print header
        log_to_stderr("=" * 50, Fore.YELLOW)
        log_to_stderr("ML-Enhanced Studio Photography Detector", Fore.YELLOW + Style.BRIGHT)
        log_to_stderr("=" * 50, Fore.YELLOW)
        log_to_stderr(f"Analyzing: {args.image_path}", Fore.WHITE)
        log_to_stderr(f"Original size: {image.shape[1]}x{image.shape[0]}", Fore.WHITE)
        
        # Resize image for optimal analysis
        analysis_image, was_resized = resize_image_for_analysis(image)
        if was_resized:
            log_to_stderr(f"Resized to: {analysis_image.shape[1]}x{analysis_image.shape[0]} for optimal analysis", Fore.CYAN)
            log_to_stderr(f"Pre-computed constants loaded in {setup_time*1000:.1f}ms", Fore.CYAN)
        
        log_to_stderr("")
        
        # Phase 1: Core Lighting Analysis (Always runs)
        log_to_stderr("=== PHASE 1: LIGHTING ANALYSIS ===", Fore.MAGENTA + Style.BRIGHT)
        log_to_stderr("")
        
        lighting_results = analyze_lighting(analysis_image, args.verbose)
        
        log_to_stderr("=== LIGHTING ANALYSIS COMPLETE ===", Fore.MAGENTA + Style.BRIGHT)
        log_to_stderr(f"→ Lighting confidence: {lighting_results['lighting_confidence']*100:.0f}% ({format_score_interpretation(lighting_results['lighting_confidence'])})", Fore.MAGENTA + Style.BRIGHT)
        log_to_stderr("")
        
        # Check if we need Phase 3 analysis
        lighting_confidence = lighting_results['lighting_confidence']
        phase3_results = None
        
        if 0.3 <= lighting_confidence <= 0.7:
            log_to_stderr("=== PHASE 3: FREQUENCY & COMPOSITION ANALYSIS ===", Fore.BLUE + Style.BRIGHT)
            log_to_stderr(f"Lighting analysis inconclusive (confidence: {lighting_confidence:.2f})", Fore.BLUE)
            log_to_stderr("Running additional frequency domain and composition analysis...", Fore.BLUE)
            log_to_stderr("")
            
            phase3_results = analyze_frequency_and_composition(analysis_image, args.verbose)
            
            log_to_stderr("=== PHASE 3 ANALYSIS COMPLETE ===", Fore.BLUE + Style.BRIGHT)
            log_to_stderr(f"→ Phase 3 confidence: {phase3_results['phase3_confidence']*100:.0f}% ({format_score_interpretation(phase3_results['phase3_confidence'])})", Fore.BLUE + Style.BRIGHT)
            log_to_stderr("")
            
            # Combine Phase 1 and Phase 3 results
            # Weight: 60% lighting analysis, 40% frequency/composition analysis
            confidence = (lighting_confidence * 0.6 + phase3_results['phase3_confidence'] * 0.4)
            
            log_to_stderr("=== COMBINED ANALYSIS RESULTS ===", Fore.MAGENTA + Style.BRIGHT)
            log_to_stderr(f"Phase 1 (Lighting): {lighting_confidence:.2f}", Fore.MAGENTA)
            log_to_stderr(f"Phase 3 (Frequency/Comp): {phase3_results['phase3_confidence']:.2f}", Fore.MAGENTA)
            log_to_stderr(f"Combined confidence: {confidence:.2f}", Fore.MAGENTA + Style.BRIGHT)
            log_to_stderr("")
        else:
            # Use lighting analysis alone for clear cases
            confidence = lighting_confidence
            if lighting_confidence < 0.3:
                log_to_stderr("=== PHASE 3: SKIPPED ===", Fore.YELLOW + Style.BRIGHT)
                log_to_stderr(f"Lighting confidence {lighting_confidence:.2f} < 0.3 - clearly natural light", Fore.YELLOW)
                log_to_stderr("Skipping frequency analysis for performance", Fore.YELLOW)
            else:
                log_to_stderr("=== PHASE 3: SKIPPED ===", Fore.YELLOW + Style.BRIGHT)
                log_to_stderr(f"Lighting confidence {lighting_confidence:.2f} > 0.7 - clearly studio light", Fore.YELLOW)
                log_to_stderr("Skipping frequency analysis for performance", Fore.YELLOW)
            log_to_stderr("")
        
        # Final confidence calculation
        final_confidence = confidence
        
        is_studio = final_confidence > 0.5
        
        # Print summary
        log_to_stderr("=" * 50, Fore.YELLOW)
        log_to_stderr("ANALYSIS COMPLETE", Fore.YELLOW + Style.BRIGHT)
        log_to_stderr("=" * 50, Fore.YELLOW)
        
        # Show ML status
        if hasattr(lighting_results, 'ml_enhanced') and lighting_results.get('ml_enhanced'):
            log_to_stderr("ML Enhancement: ACTIVE (K-means clustering, GMM segmentation)", Fore.CYAN + Style.BRIGHT)
        else:
            log_to_stderr("ML Enhancement: Using traditional heuristics", Fore.YELLOW)
        
        # Show ML feature analysis if requested
        if args.features:
            log_to_stderr("\nML Feature Analysis:", Fore.CYAN + Style.BRIGHT)
            
            # Run ML analysis
            ml_lighting = ml_analyzer.analyze_lighting_clusters(analysis_image)
            ml_segmentation = ml_analyzer.analyze_background_segmentation(analysis_image)
            ml_patterns = ml_analyzer.detect_professional_patterns(analysis_image)
            
            log_to_stderr(f"  Lighting Clusters:", Fore.CYAN)
            log_to_stderr(f"    - Lighting separation: {ml_lighting['lighting_separation']:.3f}", Fore.CYAN)
            log_to_stderr(f"    - Lighting uniformity: {ml_lighting['lighting_uniformity']:.3f}", Fore.CYAN)
            log_to_stderr(f"    - Cluster count: {ml_lighting['cluster_count']}", Fore.CYAN)
            
            log_to_stderr(f"  Background Segmentation:", Fore.CYAN)
            log_to_stderr(f"    - Separation quality: {ml_segmentation['separation_quality']:.3f}", Fore.CYAN)
            log_to_stderr(f"    - Edge alignment: {ml_segmentation['edge_alignment']:.3f}", Fore.CYAN)
            log_to_stderr(f"    - Background uniformity: {ml_segmentation['background_uniformity']:.3f}", Fore.CYAN)
            
            log_to_stderr(f"  Professional Patterns:", Fore.CYAN)
            log_to_stderr(f"    - Professional histogram: {ml_patterns['professional_histogram']:.3f}", Fore.CYAN)
            log_to_stderr(f"    - Histogram skewness: {ml_patterns['histogram_skewness']:.3f}", Fore.CYAN)
            log_to_stderr(f"    - Frequency signature: {ml_patterns['frequency_signature']:.3f}", Fore.CYAN)
        
        log_to_stderr("")
        
        # Output JSON result to stdout
        result = {
            "is_studio": bool(is_studio),  # Convert numpy bool to Python bool
            "confidence": round(float(final_confidence), 2),  # Ensure float type
            "ml_enhanced": lighting_results.get('ml_enhanced', False)
        }
        print(json.dumps(result))
        
    except Exception as e:
        log_to_stderr(f"Error: {str(e)}", Fore.RED)
        print(json.dumps({"error": str(e), "is_studio": False, "confidence": 0.0}))
        sys.exit(1)

if __name__ == "__main__":
    main()