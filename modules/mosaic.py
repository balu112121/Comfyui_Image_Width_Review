from PIL import Image, ImageDraw
import numpy as np

def apply_mosaic_to_region(img, region, block_size, strength=1.0, shape='rect'):
    x1, y1, x2, y2 = region
    if x1 >= x2 or y1 >= y2:
        return
    roi = img.crop(region)
    if roi.width == 0 or roi.height == 0:
        return
    roi_np = np.array(roi)
    roi_h, roi_w = roi_np.shape[:2]
    small_w = max(1, roi_w // block_size)
    small_h = max(1, roi_h // block_size)
    small = Image.fromarray(roi_np).resize((small_w, small_h), Image.Resampling.NEAREST)
    mosaic_roi = small.resize((roi_w, roi_h), Image.Resampling.NEAREST)
    if strength < 1.0:
        mosaic_roi = Image.blend(roi, mosaic_roi, strength)
    if shape == 'circle':
        mask = Image.new('L', (roi_w, roi_h), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, roi_w-1, roi_h-1), fill=255)
        img.paste(mosaic_roi, region, mask)
    else:
        img.paste(mosaic_roi, region)