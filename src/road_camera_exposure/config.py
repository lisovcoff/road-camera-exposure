"""Validate explicit setup before hardware imports or output creation."""
import copy
import ipaddress
import json
import math
from pathlib import Path
from types import SimpleNamespace


def load(path):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    validate(data)
    return data


def validate(c,require_reference=False):
    if not isinstance(c,dict) or c.get('schema_version')!=1:
        raise ValueError('configuration schema_version must be 1')
    if c.get('driver')!='picamera2-imx708':
        raise ValueError('bundled runner supports picamera2-imx708 only; see camera adapter guide')
    for key,low,high in [('camera_index',0,8),('duration_s',10,14400),('port',1024,65535),
                         ('session_bytes',1_000_000,100_000_000),('validation_bytes',140_000_000,250_000_000),
                         ('reserve_bytes',1_000_000_000,10**12),('preview_quality',40,95)]:
        if type(c.get(key)) is not int or not low<=c[key]<=high:raise ValueError(key+' outside supported bounds')
    ip=ipaddress.ip_address(c.get('bind',''))
    if ip.version!=4 or not ip.is_private or ip.is_unspecified:
        raise ValueError('bind must be an explicit private or loopback IPv4, not 0.0.0.0')
    r=c.get('roi')
    if (not isinstance(r,list) or len(r)!=4 or any(type(x) is not int or x%2 for x in r)
        or not (0<=r[0]<r[2]<=1920 and 0<=r[1]<r[3]<=1080 and
                64<=r[2]-r[0]<=640 and 64<=r[3]-r[1]<=360)):
        raise ValueError('roi must be even main-image [x0,y0,x1,y1], width 64..640, height 64..360')
    t=c.get('tracking_rectangle')
    if (not isinstance(t,list) or len(t)!=4 or any(type(x) is not int for x in t)
        or not (8<=t[0]<t[2]<=r[2]-r[0]-8 and 8<=t[1]<t[3]<=r[3]-r[1]-8)
        or t[2]-t[0]<32 or t[3]-t[1]<32):
        raise ValueError('tracking_rectangle must be an interior rectangle relative to the crop')
    if type(c.get('rotate_preview')) is not bool:raise ValueError('rotate_preview must be boolean')
    lens=c.get('lens')
    if type(lens) not in (int,float) or not math.isfinite(lens) or not 0<=lens<=2:
        raise ValueError('lens is a manual candidate in 0..2, not a focus calibration')
    gains=c.get('colour_gains')
    if gains is None:
        if require_reference:raise ValueError('capture a reference first: colour_gains are missing')
    elif not isinstance(gains,list) or len(gains)!=2 or any(type(x) not in (int,float) or not math.isfinite(x) or not .1<=x<=8 for x in gains):
        raise ValueError('colour_gains must be two measured finite values in .1..8')
    return c


def runtime_args(c,output,bind=None):
    data=copy.deepcopy(c)
    if bind is not None:data['bind']=bind
    validate(data,require_reference=True)
    data['output']=Path(output)
    data['duration']=data.pop('duration_s')
    return SimpleNamespace(**data)
