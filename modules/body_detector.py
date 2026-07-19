import mediapipe as mp
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mediapipe")

class BodyDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def detect(self, image: Image.Image):
        img_rgb = np.array(image.convert('RGB'))
        results = self.pose.process(img_rgb)
        if not results.pose_landmarks:
            return []
        landmarks = []
        h, w = img_rgb.shape[:2]
        for lm in results.pose_landmarks.landmark:
            landmarks.append({
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': lm.visibility,
                'pixel_x': int(lm.x * w),
                'pixel_y': int(lm.y * h)
            })
        return landmarks
    
    def get_chest_boxes(self, landmarks, image_size, expand_ratio=0.3, ratio=0.35,
                        offset_x=0, offset_y=0):
        if not landmarks or len(landmarks) < 25:
            return []
        w, h = image_size
        ls = landmarks[11]
        rs = landmarks[12]
        lh = landmarks[23]
        rh = landmarks[24]
        if any(lm['visibility'] < 0.5 for lm in [ls, rs, lh, rh]):
            return []
        
        shoulder_y = (ls['pixel_y'] + rs['pixel_y']) / 2
        hip_y = (lh['pixel_y'] + rh['pixel_y']) / 2
        torso_height = hip_y - shoulder_y
        if torso_height <= 0:
            return []
        
        chest_bottom = shoulder_y + torso_height * ratio
        
        shoulder_width = abs(rs['pixel_x'] - ls['pixel_x'])
        base_x1 = min(ls['pixel_x'], rs['pixel_x']) - int(shoulder_width * 0.1)
        base_x2 = max(ls['pixel_x'], rs['pixel_x']) + int(shoulder_width * 0.1)
        base_y1 = shoulder_y - int(torso_height * 0.1)
        base_y2 = chest_bottom
        
        center_x = (base_x1 + base_x2) / 2
        center_y = (base_y1 + base_y2) / 2
        width = base_x2 - base_x1
        height = base_y2 - base_y1
        new_width = width * expand_ratio
        new_height = height * expand_ratio
        x1 = int(center_x - new_width/2)
        x2 = int(center_x + new_width/2)
        y1 = int(center_y - new_height/2)
        y2 = int(center_y + new_height/2)
        
        x1 += offset_x
        x2 += offset_x
        y1 += offset_y
        y2 += offset_y
        
        x1 = max(0, x1)
        x2 = min(w, x2)
        y1 = max(0, y1)
        y2 = min(h, y2)
        if x1 >= x2 or y1 >= y2:
            return []
        return [(x1, y1, x2, y2)]
    
    def get_groin_boxes(self, landmarks, image_size, expand_ratio=0.4,
                        offset_x=0, offset_y=0):
        if not landmarks or len(landmarks) < 25:
            return []
        w, h = image_size
        lh = landmarks[23]
        rh = landmarks[24]
        if lh['visibility'] < 0.5 or rh['visibility'] < 0.5:
            return []
        center_x = (lh['pixel_x'] + rh['pixel_x']) // 2
        center_y = (lh['pixel_y'] + rh['pixel_y']) // 2
        hip_width = abs(rh['pixel_x'] - lh['pixel_x'])
        region_size = int(hip_width * 0.8 * expand_ratio)
        
        x1 = center_x - region_size // 2
        x2 = center_x + region_size // 2
        y1 = center_y - int(region_size * 0.2)
        y2 = center_y + int(region_size * 1.0)
        
        x1 += offset_x
        x2 += offset_x
        y1 += offset_y
        y2 += offset_y
        
        x1 = max(0, x1)
        x2 = min(w, x2)
        y1 = max(0, y1)
        y2 = min(h, y2)
        if x1 >= x2 or y1 >= y2:
            return []
        return [(x1, y1, x2, y2)]