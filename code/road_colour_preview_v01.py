"""Extract real Y/U/V crop for JPEG preview; no invented chroma or Y changes."""
import numpy as np
from road_observer_v01 import visible_y_plane


def preview_planes(array,config,roi):
    y=visible_y_plane(array,config)
    h,w=y.shape;stride=config['stride']
    if stride%2 or len(roi)!=4 or any(type(v) is not int or v%2 for v in roi):
        raise ValueError('even YUV420 stride and crop coordinates required')
    x0,y0,x1,y1=roi
    if not 0<=x0<x1<=w or not 0<=y0<y1<=h:raise ValueError('crop outside visible image')
    packed=array.reshape((array.shape[0]*2,stride//2))
    u=packed[2*h:2*h+h//2,:w//2]
    v=packed[2*h+h//2:,:w//2]
    # Rotate all three planes identically; input array and analysis Y stay intact.
    return tuple(np.ascontiguousarray(plane[::-1,::-1]) for plane in (
        y[y0:y1,x0:x1],u[y0//2:y1//2,x0//2:x1//2],v[y0//2:y1//2,x0//2:x1//2]))
