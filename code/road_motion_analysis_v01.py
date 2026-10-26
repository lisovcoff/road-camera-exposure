"""Offline affine registration and apparent line-width diagnostics; no camera I/O.

Registration quality must be inspected. Line width is not sensor MTF or
recognition accuracy. Current affine model does not compensate rolling shutter.
"""
import numpy as np


def sample(image,x,y):
    a=np.asarray(image,dtype=float)
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    x0=np.floor(x).astype(int);y0=np.floor(y).astype(int)
    valid=(x0>=0)&(y0>=0)&(x0+1<a.shape[1])&(y0+1<a.shape[0])
    xx=np.clip(x0,0,a.shape[1]-2);yy=np.clip(y0,0,a.shape[0]-2)
    dx=x-x0;dy=y-y0
    result=(a[yy,xx]*(1-dx)*(1-dy)+a[yy,xx+1]*dx*(1-dy)+
            a[yy+1,xx]*(1-dx)*dy+a[yy+1,xx+1]*dx*dy)
    return result,valid


def smooth3(image):
    p=np.pad(np.asarray(image,dtype=float),1,mode='edge')
    return sum(p[y:y+image.shape[0],x:x+image.shape[1]] for y in range(3) for x in range(3))/9


def register_affine(reference,image,rectangle,initial=None):
    """Robust forward-additive LK, joint linear intensity scale/offset.

    Maps coordinates relative to reference rectangle center into current image.
    Returns a fit, including poor fits; caller MUST apply quality rejection.
    """
    r=np.asarray(reference,dtype=float);a=np.asarray(image,dtype=float)
    if r.ndim!=2 or r.shape!=a.shape or min(r.shape)<16 or not np.isfinite(r).all() or not np.isfinite(a).all():
        raise ValueError('finite same-size 2D images required')
    x0,y0,x1,y1=rectangle
    if not (1<=x0<x1<r.shape[1]-1 and 1<=y0<y1<r.shape[0]-1):raise ValueError('interior reference rectangle required')
    center=np.array([(x0+x1)/2,(y0+y1)/2])
    warp=np.array([[1.,0.,center[0]],[0.,1.,center[1]]]) if initial is None else np.array(initial,dtype=float).copy()
    if warp.shape!=(2,3) or not np.isfinite(warp).all():raise ValueError('finite 2x3 initial affine required')
    gain,offset=1.,0.
    for stride in (4,2,1):
        ar=smooth3(a)[::stride,::stride];rr=smooth3(r)[::stride,::stride]
        gy,gx=np.gradient(ar)
        yy,xx=np.mgrid[y0:y1:3,x0:x1:3]
        coords=np.stack((xx.ravel()-center[0],yy.ravel()-center[1],np.ones(xx.size)))
        target,_=sample(rr,xx.ravel()/stride,yy.ravel()/stride)
        for _ in range(18):
            q=warp@coords
            values,valid=sample(ar,q[0]/stride,q[1]/stride)
            if valid.mean()<.9:break
            ix,_=sample(gx,q[0]/stride,q[1]/stride);iy,_=sample(gy,q[0]/stride,q[1]/stride)
            ix/=stride;iy/=stride
            residual=values-gain*target-offset
            threshold=max(4.,float(np.median(abs(residual[valid])))*2.5)
            weight=np.sqrt(np.minimum(1.,threshold/np.maximum(abs(residual),1e-9))*valid)
            jac=np.column_stack((ix*coords[0],ix*coords[1],ix,
                                 iy*coords[0],iy*coords[1],iy,-target,-np.ones(len(target))))
            delta=np.linalg.lstsq(jac*weight[:,None],residual*weight,rcond=1e-6)[0]
            # Trust region prevents a failed fit from jumping across the scene.
            if max(abs(delta[2]),abs(delta[5]))>20:delta*=20/max(abs(delta[2]),abs(delta[5]))
            warp-=delta[:6].reshape(2,3);gain-=delta[6];offset-=delta[7]
            if np.linalg.norm(delta[:6])<.002:break
    q=warp@coords;values,valid=sample(a,q[0],q[1]);target,_=sample(r,xx.ravel(),yy.ravel())
    corr=0. if valid.sum()<20 or np.std(values[valid])<1e-8 or np.std(target[valid])<1e-8 else float(np.corrcoef(values[valid],target[valid])[0,1])
    determinant=float(np.linalg.det(warp[:,:2]));angle=float(np.degrees(np.arctan2(warp[1,0],warp[0,0])))
    return {'warp':warp,'correlation':corr,'valid_fraction':float(valid.mean()),
            'determinant':determinant,'angle_deg':angle,
            'accepted':bool(corr>=.75 and valid.mean()>=.98 and .65<determinant<1.4 and abs(angle)<40)}


def feature_profile(reference,image,warp,rectangle,*,line_window=(392,438),y_range=(80,175)):
    """Reference-defined thin dark line; average perpendicular profiles.

    Per-row center follows the reference line. No per-frame re-centering which
    could hide motion broadening. FWHM is in reference-coordinate pixels.
    """
    x0,y0,x1,y1=rectangle;center=np.array([(x0+x1)/2,(y0+y1)/2])
    ys=np.arange(*y_range);xs=np.arange(*line_window)
    minima=[]
    ref=smooth3(reference)
    for y in ys:minima.append(xs[np.argmin(ref[y,xs])])
    slope,intercept=np.polyfit(ys,minima,1)
    if np.std(np.array(minima)-(slope*ys+intercept))>2.:
        return {'valid':False,'reason':'reference line not stable'}
    offsets=np.arange(-16,16.01,.25)
    # Horizontal rather than perpendicular width; reference slope is reported.
    xx=(slope*ys+intercept)[:,None]+offsets[None,:]
    yy=np.broadcast_to(ys[:,None],xx.shape)
    coords=np.stack((xx.ravel()-center[0],yy.ravel()-center[1],np.ones(xx.size)))
    q=warp@coords;values,valid=sample(image,q[0],q[1])
    if valid.mean()<.99:return {'valid':False,'reason':'line outside crop'}
    profile=values.reshape(xx.shape).mean(axis=0)
    baseline=float(np.median(profile[abs(offsets)>12]));minimum=float(np.min(profile[abs(offsets)<5]))
    depth=baseline-minimum
    if depth<8:return {'valid':False,'reason':'insufficient line contrast','depth_y':depth}
    inside=profile<baseline-depth/2
    k=int(np.argmin(np.where(abs(offsets)<5,profile,np.inf)));left=right=k
    while left>0 and inside[left-1]:left-=1
    while right<len(offsets)-1 and inside[right+1]:right+=1
    if left==0 or right==len(offsets)-1:return {'valid':False,'reason':'line width clipped'}
    return {'valid':True,'width_fwhm_px':float(offsets[right]-offsets[left]+.25),
            'depth_y':depth,'background_y':baseline,'relative_contrast':depth/max(baseline,1),
            'reference_line_slope':float(slope),'profile':profile.tolist(),'offsets_px':offsets.tolist()}


def matched_velocity_rows(tracks,feature_relative=(-40.,0.),min_correlation=.85):
    """Central difference at first-pixel exposure midpoint; reject each gap >50ms.

    Speed is transverse reference-x pixels/s for the measured line, consistent
    with its horizontal FWHM. It is not vehicle speed or a physical distance.
    """
    output=[];point=np.array([*feature_relative,1.],dtype=float)
    for i in range(1,len(tracks)-1):
        a,b,c=tracks[i-1:i+2]
        if not all(r['accepted'] and r['correlation']>=min_correlation for r in (a,b,c)) or not b['line']['valid']:
            continue
        t=[r['sensor_ns']-r['exposure_us']*500 for r in (a,b,c)]
        if not 0<t[1]-t[0]<=50_000_000 or not 0<t[2]-t[1]<=50_000_000:continue
        warp=np.asarray(b['warp'],dtype=float)
        if not (.85<b['determinant']<1.15 and abs(b['angle_deg'])<20):continue
        velocity=(np.asarray(c['warp'])@point-np.asarray(a['warp'])@point)/((t[2]-t[0])/1e9)
        reference_velocity=np.linalg.solve(warp[:,:2],velocity)
        speed=abs(reference_velocity[0]-b['line']['reference_line_slope']*reference_velocity[1])
        output.append({'index':b['index'],'speed_px_s':float(speed),
            'width_fwhm_px':b['line']['width_fwhm_px'],'depth_y':b['line']['depth_y'],
            'relative_contrast':b['line']['relative_contrast'],'angle_deg':b['angle_deg'],
            'determinant':b['determinant'],'correlation':b['correlation']})
    return output
