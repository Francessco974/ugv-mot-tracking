import matplotlib.patches as patches


def _get_color_from_id(obj_id):
    """Generate color from class id"""
    hash_val = hash(str(obj_id)) & 0xFFFFFF
    return f"#{hash_val:06x}"    


def gt_visual_rectangle(left,up,width,height,bb_class):
    return patches.Rectangle(
    (left, up),   
    width,               
    height,              
    linewidth=2,        
    edgecolor=_get_color_from_id(bb_class),      
    facecolor='none',
    linestyle='--'  # Creates a standard dashed line
    )    
    
def dt_visual_rectangle(left, up, width, height, bb_class):
    return patches.Rectangle(
        (left, up),   
        width,               
        height,              
        linewidth=1,        
        edgecolor=_get_color_from_id(bb_class),      
        facecolor='none',
        
    )
    

