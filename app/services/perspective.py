import cv2
import numpy as np

def create_tiled_canvas(tile_img, target_width, target_height):
    h, w = tile_img.shape[:2]
    if h == 0 or w == 0: return tile_img
    reps_x = int(np.ceil(target_width / w))
    reps_y = int(np.ceil(target_height / h))
    tiled = np.tile(tile_img, (reps_y, reps_x, 1))
    return tiled[:int(target_height), :int(target_width)]

def apply_tiles(room, mask, tile, surface_type="floor", scale=1.0):
    room_h, room_w = room.shape[:2]
    max_dim = max(room_w, room_h)
    
    # --- THE SCALE FIX ---
    # We divide the base grid by your scale slider! 
    # If scale is 3, target_tiles_across becomes 1.5 (resulting in massive luxury tiles!)
    base_tiles_across = 4.5 if surface_type == "floor" else 5.0 
    target_tiles_across = base_tiles_across / float(max(0.1, scale))
    
    tile_h, tile_w = tile.shape[:2]
    aspect_ratio = tile_h / tile_w
    
    new_tile_w = max(1, int(max_dim / target_tiles_across))
    new_tile_h = max(1, int(new_tile_w * aspect_ratio))
    tile_resized = cv2.resize(tile, (new_tile_w, new_tile_h))

    if surface_type == "floor":
        canvas_w = int(room_w * 5.0)
        canvas_h = int(room_h * 5.0)
        tiled_canvas = create_tiled_canvas(tile_resized, canvas_w, canvas_h)

        src_pts = np.float32([[0, 0], [canvas_w, 0], [canvas_w, canvas_h], [0, canvas_h]])

        horizon_y = room_h * 0.35  
        bottom_y = room_h * 3.0    
        
        top_w = room_w * 0.4      
        bottom_w = room_w * 6.0    

        dst_pts = np.float32([
            [(room_w - top_w) / 2, horizon_y],
            [((room_w - top_w) / 2) + top_w, horizon_y],
            [((room_w - bottom_w) / 2) + bottom_w, bottom_y],
            [(room_w - bottom_w) / 2, bottom_y]
        ])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_tiles = cv2.warpPerspective(tiled_canvas, M, (room_w, room_h), flags=cv2.INTER_LINEAR)

    else:
        # WALL LOGIC
        warped_tiles = create_tiled_canvas(tile_resized, room_w, room_h)

    return warped_tiles