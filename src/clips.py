import numpy as np
import math
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
from scipy.ndimage import zoom

def three_d_ken_burns_effect(clip, depth=30, zoom_range=(1.1, 1.5), pan_range=(-0.1, 0.1)):
    """Creates a 3D Ken Burns effect with depth perception"""
    def effect(get_frame, t):
        img = Image.fromarray(get_frame(t))
        width, height = img.size
        
        # Create depth map using edge detection
        img_array = np.array(img.convert('L'))
        edges = cv2.Canny(img_array, 100, 200)
        depth_map = cv2.GaussianBlur(edges, (21, 21), 0)
        
        # Calculate zoom and pan parameters
        progress = t / clip.duration
        zoom_factor = zoom_range[0] + (zoom_range[1] - zoom_range[0]) * progress
        pan_x = pan_range[0] + (pan_range[1] - pan_range[0]) * progress
        
        # Apply 3D effect using depth map
        channels = list(img.split())
        for i, channel in enumerate(channels):
            offset = int(depth * (i - 1) * np.sin(progress * 2 * np.pi))
            if offset != 0:
                channel = ImageOps.offset(channel, offset, 0)
            channels[i] = channel
        
        # Merge channels with offset
        img = Image.merge('RGB', channels)
        
        # Apply zoom and pan
        new_size = (int(width * zoom_factor), int(height * zoom_factor))
        img = img.resize(new_size, Image.LANCZOS)
        
        # Calculate crop coordinates for pan effect
        left = int((new_size[0] - width) * (0.5 + pan_x))
        top = int((new_size[1] - height) * 0.5)
        right = left + width
        bottom = top + height
        
        img = img.crop((left, top, right, bottom))
        
        result = np.array(img)
        img.close()
        
        return result
    
    return clip.fl(effect)

def holographic_effect(clip, intensity=0.5):
    """Creates a holographic effect with scanlines and color distortion"""
    def effect(get_frame, t):
        img = Image.fromarray(get_frame(t))
        width, height = img.size
        
        # Create scanlines
        scanlines = np.zeros((height, width, 3), dtype=np.uint8)
        scanlines[::3] = [0, 255, 255]  # Cyan scanlines
        
        # Apply holographic distortion
        r, g, b = img.split()
        r = ImageOps.offset(r, int(10 * np.sin(t * 2 * np.pi)), 0)
        b = ImageOps.offset(b, int(-10 * np.sin(t * 2 * np.pi)), 0)
        
        img = Image.merge('RGB', (r, g, b))
        result = np.array(img) * (1 - intensity) + scanlines * intensity
        
        return result.astype(np.uint8)
    
    return clip.fl(effect)

def matrix_rain_effect(clip):
    """Creates a Matrix-style digital rain effect"""
    def effect(get_frame, t):
        img = np.array(Image.fromarray(get_frame(t)))
        height, width = img.shape[:2]
        
        # Create digital rain
        rain = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(0, width, 10):
            start = int((t * 1000 + i) % height)
            rain[start:start+20, i:i+2] = [0, 255, 0]  # Green rain
        
        # Blend with original image
        result = cv2.addWeighted(img, 0.8, rain, 0.2, 0)
        return result
    
    return clip.fl(effect)

def pixel_sorting_effect(clip, threshold=128):
    """Creates a pixel sorting effect"""
    def effect(get_frame, t):
        img = np.array(Image.fromarray(get_frame(t)))
        brightness = np.mean(img, axis=2)
        
        # Sort pixels based on brightness
        for row in range(img.shape[0]):
            mask = brightness[row] > threshold
            img[row][mask] = np.sort(img[row][mask], axis=0)
        
        return img
    
    return clip.fl(effect)

def double_exposure_effect(clip, blend_ratio=0.5):
    """Creates a double exposure effect"""
    def effect(get_frame, t):
        current_frame = np.array(Image.fromarray(get_frame(t)))
        offset_time = (t + 0.5) % clip.duration
        future_frame = np.array(Image.fromarray(get_frame(offset_time)))
        
        result = cv2.addWeighted(current_frame, 1-blend_ratio, future_frame, blend_ratio, 0)
        return result
    
    return clip.fl(effect)


def glitch_wave_effect(clip, amplitude=10, frequency=2):
    """Creates a glitch wave effect with RGB splitting"""
    def effect(get_frame, t):
        img = Image.fromarray(get_frame(t))
        width, height = img.size
        
        # Split channels
        r, g, b = img.split()
        
        # Apply wave distortion to each channel
        for channel in [r, g, b]:
            pixels = np.array(channel)
            for i in range(height):
                offset = int(amplitude * math.sin(2 * math.pi * frequency * (i/height + t)))
                pixels[i] = np.roll(pixels[i], offset)
            channel.paste(Image.fromarray(pixels))
        
        # Merge with offset
        result = Image.merge('RGB', (r, g, b))
        return np.array(result)
    
    return clip.fl(effect)

def kaleidoscope_effect(clip, segments=8):
    """Creates a kaleidoscope effect"""
    def effect(get_frame, t):
        img = Image.fromarray(get_frame(t))
        width, height = img.size
        center = (width//2, height//2)
        
        # Create segment
        angle = 360 / segments
        segment = img.rotate(-angle/2, expand=False)
        
        # Create kaleidoscope
        result = Image.new('RGB', (width, height))
        for i in range(segments):
            rotated = segment.rotate(angle * i)
            result.paste(rotated, (0, 0), None)
        
        return np.array(result)
    
    return clip.fl(effect)

def neon_glow_effect(clip, intensity=1.5):
    """Creates a neon glow effect"""
    def effect(get_frame, t):
        img = Image.fromarray(get_frame(t))
        
        # Enhance colors
        enhancer = ImageEnhance.Color(img)
        img_colored = enhancer.enhance(intensity)
        
        # Add bloom effect
        bloom = img_colored.filter(ImageFilter.GaussianBlur(radius=15))
        result = Image.blend(img_colored, bloom, 0.3)
        
        return np.array(result)
    
    return clip.fl(effect)

def liquid_distortion_effect(clip, scale=20):
    """Creates a liquid distortion effect"""
    def effect(get_frame, t):
        img = np.array(Image.fromarray(get_frame(t)))
        height, width = img.shape[:2]
        
        # Create displacement map
        x = np.linspace(0, 2*np.pi, width)
        y = np.linspace(0, 2*np.pi, height)
        X, Y = np.meshgrid(x, y)
        
        dx = scale * np.sin(X + t*2*np.pi)
        dy = scale * np.cos(Y + t*2*np.pi)
        
        # Apply displacement
        rows, cols = np.indices((height, width))
        new_rows = np.clip(rows + dy, 0, height-1).astype(int)
        new_cols = np.clip(cols + dx, 0, width-1).astype(int)
        
        return img[new_rows, new_cols]
    
    return clip.fl(effect)

def time_displacement_effect(clip, wave_speed=2):
    """Creates a time displacement effect"""
    def effect(get_frame, t):
        width = clip.w
        height = clip.h
        
        # Create time displacement map
        displacement = np.zeros((height, width))
        for y in range(height):
            displacement[y] = np.sin(y/30 + t*wave_speed) * 0.5
        
        # Sample frames at different times
        result = np.zeros((height, width, 3))
        for y in range(height):
            new_t = (t + displacement[y]) % clip.duration
            result[y] = get_frame(new_t)[y]
        
        return result.astype(np.uint8)
    
    return clip.fl(effect)

def rgb_split_pulse_effect(clip, max_offset=20):
    """Creates an RGB split effect with pulsing"""
    def effect(get_frame, t):
        img = Image.fromarray(get_frame(t))
        r, g, b = img.split()
        
        # Calculate pulsing offset
        offset = int(max_offset * abs(math.sin(t * 2 * math.pi)))
        
        # Offset red and blue channels
        r = ImageOps.offset(r, offset, 0)
        b = ImageOps.offset(b, -offset, 0)
        
        result = Image.merge('RGB', (r, g, b))
        return np.array(result)
    
    return clip.fl(effect)

def zoom_blur_effect(clip, max_zoom=1.5):
    """Creates a dynamic zoom blur effect"""
    def effect(get_frame, t):
        img = Image.fromarray(get_frame(t))
        width, height = img.size
        
        # Create multiple zoomed layers
        layers = []
        for i in range(10):
            zoom = 1 + (max_zoom - 1) * (i/10)
            size = (int(width * zoom), int(height * zoom))
            zoomed = img.resize(size, Image.LANCZOS)
            
            # Center crop
            left = (size[0] - width) // 2
            top = (size[1] - height) // 2
            zoomed = zoomed.crop((left, top, left + width, top + height))
            layers.append(np.array(zoomed, dtype=float))
        
        # Blend layers
        result = np.zeros_like(layers[0])
        weights = np.linspace(1, 0, len(layers))
        for layer, weight in zip(layers, weights):
            result += layer * weight
            
        return result.astype(np.uint8)
    
    return clip.fl(effect)

def pixelate_transition_effect(clip, min_size=20):
    """Creates a pixelation transition effect"""
    def effect(get_frame, t):
        img = Image.fromarray(get_frame(t))
        width, height = img.size
        
        # Calculate pixel size based on time
        progress = abs(math.sin(t * math.pi))
        pixel_size = max(1, int(min_size * progress))
        
        # Pixelate
        small = img.resize((width//pixel_size, height//pixel_size), Image.NEAREST)
        result = small.resize((width, height), Image.NEAREST)
        
        return np.array(result)
    
    return clip.fl(effect)

def dream_effect(clip, blur_radius=10):
    """Creates a dreamy, soft effect"""
    def effect(get_frame, t):
        img = Image.fromarray(get_frame(t))
        
        # Add soft blur
        blurred = img.filter(ImageFilter.GaussianBlur(blur_radius))
        
        # Enhance brightness and contrast
        enhancer = ImageEnhance.Brightness(blurred)
        brightened = enhancer.enhance(1.2)
        
        enhancer = ImageEnhance.Contrast(brightened)
        result = enhancer.enhance(0.8)
        
        return np.array(result)
    
    return clip.fl(effect)