import cv2
import numpy as np

def blend_with_lighting(room, warped_tile, mask):
    # 1. Extract luminance (lighting) from the original room
    gray = cv2.cvtColor(room, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    # 2. Adjust contrast of the lighting map so shadows aren't pitch black
    light_map = (gray * 0.7) + 0.3 
    
    # Expand to 3 channels to multiply with RGB image
    light_map = np.expand_dims(light_map, axis=2)

    # 3. Multiply the new tiles by the original lighting
    warped_tile_float = warped_tile.astype(np.float32)
    blended = (warped_tile_float * light_map)
    
    # Constrain values and convert back to image format
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # 4. Paste the blended tiles only where the floor mask is active
    result = room.copy()
    binary_mask = (mask > 0) 
    
    result[binary_mask] = blended[binary_mask]

    return result