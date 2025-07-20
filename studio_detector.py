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
import numpy as np
import cv2
from scipy import ndimage, signal
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

def log_to_stderr(message, color=Fore.WHITE):
    """Print colored message to stderr"""
    sys.stderr.write(color + message + Style.RESET_ALL + '\n')
    sys.stderr.flush()

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
        description='Analyze an image to determine if it was taken in a studio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python studio_detector.py portrait.jpg
  python studio_detector.py --verbose landscape.png
  python studio_detector.py --quiet photo.bmp
        """
    )
    parser.add_argument('image_path', nargs='?', help='Path to the image file')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Show more detailed analysis output')
    parser.add_argument('--quiet', '-q', action='store_true', 
                       help='Suppress all stderr output, only show JSON')
    
    args = parser.parse_args()
    
    # Check if image path provided
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
        log_to_stderr("=" * 40, Fore.YELLOW)
        log_to_stderr("Studio Photography Detector", Fore.YELLOW + Style.BRIGHT)
        log_to_stderr("=" * 40, Fore.YELLOW)
        log_to_stderr(f"Analyzing: {args.image_path}", Fore.WHITE)
        log_to_stderr(f"Original size: {image.shape[1]}x{image.shape[0]}", Fore.WHITE)
        
        # Resize image for optimal analysis
        analysis_image, was_resized = resize_image_for_analysis(image)
        if was_resized:
            log_to_stderr(f"Resized to: {analysis_image.shape[1]}x{analysis_image.shape[0]} for optimal analysis", Fore.CYAN)
            log_to_stderr(f"Pre-computed constants loaded in {setup_time*1000:.1f}ms", Fore.CYAN)
        
        log_to_stderr("")
        
        # Run all analysis components
        results = {}
        
        # 1. Shadow Analysis
        log_to_stderr("[1/4] Shadow Analysis", Fore.GREEN + Style.BRIGHT)
        log_to_stderr("-" * 40, Fore.GREEN)
        shadow_results = analyze_shadows(analysis_image, args.verbose)
        log_to_stderr(f"✓ Shadow softness: {shadow_results['softness']:.2f} ({format_score_interpretation(shadow_results['softness'])})", Fore.WHITE)
        log_to_stderr(f"✓ Direction consistency: {shadow_results['direction_consistency']:.2f} ({format_score_interpretation(shadow_results['direction_consistency'])})", Fore.WHITE)
        log_to_stderr(f"✓ Shadow uniformity: {shadow_results['uniformity']:.2f} ({format_score_interpretation(shadow_results['uniformity'])})", Fore.WHITE)
        log_to_stderr(f"→ Shadow score: {shadow_results['overall_score']*100:.0f}% ({format_score_interpretation(shadow_results['overall_score'])})", Fore.YELLOW + Style.BRIGHT)
        log_to_stderr("")
        
        # 2. Highlight Analysis
        log_to_stderr("[2/4] Highlight Analysis", Fore.GREEN + Style.BRIGHT)
        log_to_stderr("-" * 40, Fore.GREEN)
        highlight_results = analyze_highlights(analysis_image, args.verbose)
        catchlight_text = "Yes (circular pattern)" if highlight_results['catchlight_detected'] else "No"
        log_to_stderr(f"✓ Catchlights detected: {catchlight_text}", Fore.WHITE)
        log_to_stderr(f"✓ Distribution uniformity: {highlight_results['distribution_uniformity']:.2f} ({format_score_interpretation(highlight_results['distribution_uniformity'])})", Fore.WHITE)
        log_to_stderr(f"✓ Shape regularity: {highlight_results['shape_regularity']:.2f} ({format_score_interpretation(highlight_results['shape_regularity'])})", Fore.WHITE)
        log_to_stderr(f"→ Highlight score: {highlight_results['overall_score']*100:.0f}% ({format_score_interpretation(highlight_results['overall_score'])})", Fore.YELLOW + Style.BRIGHT)
        log_to_stderr("")
        
        # 3. Color Temperature Analysis
        log_to_stderr("[3/4] Color Temperature Analysis", Fore.GREEN + Style.BRIGHT)
        log_to_stderr("-" * 40, Fore.GREEN)
        color_results = analyze_color_temperature(analysis_image, args.verbose)
        log_to_stderr(f"✓ Temperature consistency: {color_results['temperature_consistency']:.2f} ({format_score_interpretation(color_results['temperature_consistency'])})", Fore.WHITE)
        log_to_stderr(f"✓ Single light source: {color_results['mixed_lighting']:.2f} ({format_score_interpretation(color_results['mixed_lighting'])})", Fore.WHITE)
        log_to_stderr(f"✓ Gradient uniformity: {color_results['gradient_uniformity']:.2f} ({format_score_interpretation(color_results['gradient_uniformity'])})", Fore.WHITE)
        log_to_stderr(f"→ Color temperature score: {color_results['overall_score']*100:.0f}% ({format_score_interpretation(color_results['overall_score'])})", Fore.YELLOW + Style.BRIGHT)
        log_to_stderr("")
        
        # 4. Background Separation Analysis
        log_to_stderr("[4/4] Background Separation Analysis", Fore.GREEN + Style.BRIGHT)
        log_to_stderr("-" * 40, Fore.GREEN)
        background_results = analyze_background_separation(analysis_image, args.verbose)
        log_to_stderr(f"✓ Edge sharpness: {background_results['edge_sharpness']:.2f} ({format_score_interpretation(background_results['edge_sharpness'])})", Fore.WHITE)
        log_to_stderr(f"✓ Background uniformity: {background_results['background_uniformity']:.2f} ({format_score_interpretation(background_results['background_uniformity'])})", Fore.WHITE)
        log_to_stderr(f"✓ Rim lighting presence: {background_results['rim_lighting']:.2f} ({format_score_interpretation(background_results['rim_lighting'])})", Fore.WHITE)
        log_to_stderr(f"→ Background separation score: {background_results['overall_score']*100:.0f}% ({format_score_interpretation(background_results['overall_score'])})", Fore.YELLOW + Style.BRIGHT)
        log_to_stderr("")
        
        # Calculate final confidence score
        confidence = (shadow_results['overall_score'] * 0.25 +
                     highlight_results['overall_score'] * 0.25 +
                     color_results['overall_score'] * 0.25 +
                     background_results['overall_score'] * 0.25)
        
        is_studio = confidence > 0.5
        
        # Print summary
        log_to_stderr("=" * 40, Fore.YELLOW)
        log_to_stderr("ANALYSIS COMPLETE", Fore.YELLOW + Style.BRIGHT)
        log_to_stderr("=" * 40, Fore.YELLOW)
        log_to_stderr("")
        
        # Output JSON result to stdout
        result = {
            "is_studio": bool(is_studio),  # Convert numpy bool to Python bool
            "confidence": round(float(confidence), 2)  # Ensure float type
        }
        print(json.dumps(result))
        
    except Exception as e:
        log_to_stderr(f"Error: {str(e)}", Fore.RED)
        print(json.dumps({"error": str(e), "is_studio": False, "confidence": 0.0}))
        sys.exit(1)

if __name__ == "__main__":
    main()