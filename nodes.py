import torch
import numpy as np
from PIL import Image
import comfy.utils

from .modules.body_detector import BodyDetector
from .modules.mosaic import apply_mosaic_to_region

class Comfyui_Image_Width_Review:
    """
    南光AI宽限图像节点
    自动检测图像中人物的胸部和阴部区域并打马赛克
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
            },
            "optional": {
                # ========== 胸部参数 ==========
                "chest_detection": (["启用", "禁用"], {"default": "启用", "display": "胸部检测"}),
                "chest_block_size": ("INT", {"default": 10, "min": 2, "max": 50, "step": 1, "display": "胸部马赛克块大小"}),
                "chest_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "胸部马赛克强度"}),
                "chest_region_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1, "display": "胸部区域扩展"}),
                "chest_ratio": ("FLOAT", {"default": 0.35, "min": 0.2, "max": 0.6, "step": 0.01, "display": "胸部下缘比例（肩到髋）"}),
                "chest_offset_x": ("INT", {"default": 0, "min": -200, "max": 200, "step": 1, "display": "胸部水平偏移（像素）"}),
                "chest_offset_y": ("INT", {"default": 0, "min": -200, "max": 200, "step": 1, "display": "胸部垂直偏移（像素）"}),
                
                # ========== 阴部参数 ==========
                "groin_detection": (["启用", "禁用"], {"default": "启用", "display": "阴部检测"}),
                "groin_block_size": ("INT", {"default": 10, "min": 2, "max": 50, "step": 1, "display": "阴部马赛克块大小"}),
                "groin_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "阴部马赛克强度"}),
                "groin_region_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1, "display": "阴部区域扩展"}),
                "groin_offset_x": ("INT", {"default": 0, "min": -200, "max": 200, "step": 1, "display": "阴部水平偏移（像素）"}),
                "groin_offset_y": ("INT", {"default": 0, "min": -200, "max": 200, "step": 1, "display": "阴部垂直偏移（像素）"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "process"
    CATEGORY = "南光AI/图像"
    
    def process(self, image,
                chest_detection="启用", chest_block_size=10, chest_strength=1.0, chest_region_scale=1.0,
                chest_ratio=0.35, chest_offset_x=0, chest_offset_y=0,
                groin_detection="启用", groin_block_size=10, groin_strength=1.0, groin_region_scale=1.0,
                groin_offset_x=0, groin_offset_y=0):
        
        if not hasattr(self, '_detector'):
            self._detector = BodyDetector()
        detector = self._detector
        
        batch_size = image.shape[0]
        result_images = []
        
        for i in range(batch_size):
            img_tensor = image[i]
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np, mode='RGB')
            
            landmarks = detector.detect(pil_img)
            regions_params = []
            
            # 胸部检测（矩形马赛克）
            if chest_detection == "启用":
                expand = 0.3 * chest_region_scale
                boxes = detector.get_chest_boxes(landmarks, pil_img.size,
                                                  expand_ratio=expand,
                                                  ratio=chest_ratio,
                                                  offset_x=chest_offset_x,
                                                  offset_y=chest_offset_y)
                for box in boxes:
                    regions_params.append((box, chest_block_size, chest_strength, 'rect'))
            
            # 阴部检测（圆形马赛克）
            if groin_detection == "启用":
                expand = 0.4 * groin_region_scale
                boxes = detector.get_groin_boxes(landmarks, pil_img.size,
                                                  expand_ratio=expand,
                                                  offset_x=groin_offset_x,
                                                  offset_y=groin_offset_y)
                for box in boxes:
                    regions_params.append((box, groin_block_size, groin_strength, 'circle'))
            
            result_img = pil_img.copy()
            for (x1, y1, x2, y2), block_size, strength, shape in regions_params:
                x1 = max(0, min(x1, result_img.width))
                x2 = max(0, min(x2, result_img.width))
                y1 = max(0, min(y1, result_img.height))
                y2 = max(0, min(y2, result_img.height))
                if x1 >= x2 or y1 >= y2:
                    continue
                region = (x1, y1, x2, y2)
                apply_mosaic_to_region(result_img, region, block_size, strength, shape)
            
            result_np = np.array(result_img).astype(np.float32) / 255.0
            result_tensor = torch.from_numpy(result_np).unsqueeze(0)
            result_images.append(result_tensor)
        
        if len(result_images) == 1:
            return (result_images[0],)
        else:
            return (torch.cat(result_images, dim=0),)


NODE_CLASS_MAPPINGS = {
    "Comfyui_Image_Width_Review": Comfyui_Image_Width_Review
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Comfyui_Image_Width_Review": "南光AI宽限图像"
}