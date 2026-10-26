"""Post-ISP static patch diagnostics. Does not rank algorithms or infer sensor SNR."""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np


def analyze(directory):
    root=Path(directory)
    setup=json.loads((root/'setup.json').read_text(encoding='utf-8'))
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    log=root/'observations.jsonl'
    if log.stat().st_size>3_000_000:
        raise ValueError('trial log exceeds supported size')
    rows=[json.loads(line) for line in log.read_text(encoding='utf-8').splitlines()]
    if not rows or len(rows)>301 or summary['error'] or summary['cleanup_errors'] or len(rows)!=summary['samples']:
        raise ValueError('incomplete, failed or oversized trial')
    md=[r['actual_metadata'] for r in rows]
    result={'mode':setup['trial']['mode'],'samples':len(rows),'sensor_noise_measured':False,
            'proof_of_detection':False,'quality_ranking':None,
            'actual':{key:{'min':min(m[key] for m in md),'median':statistics.median(m[key] for m in md),
                          'max':max(m[key] for m in md)}
                      for key in ('ExposureTime','AnalogueGain','DigitalGain','FrameDuration')},
            'patches':{},'warnings':['Post-ISP Y variation: affected by ISP temporal/spatial denoise.',
                                    'Static scene and correct flat target/background ROIs must be checked visually.']}
    pair_rows=[r for r in rows if r['patches']]
    if any(r.get('readback')!='reported_modes' for r in rows):
        result['warnings'].append('Control mode metadata partly unreported; active AE is indirect evidence only.')
    eligible=len(pair_rows)>=4
    if not eligible:result['warnings'].append('Fewer than four patch pairs.')
    if max(m['ExposureTime'] for m in md)-min(m['ExposureTime'] for m in md)>42 or (
            max(m['AnalogueGain'] for m in md)-min(m['AnalogueGain'] for m in md)>.05):
        eligible=False;result['warnings'].append('Exposure/gain varied; noise proxy is not stationary.')
    for name,roi in [('target',setup['trial']['target_roi']),('background',setup['trial']['background_roi'])]:
        arrays=[]
        for row in pair_rows:
            filename=row['patches'][name]
            if not isinstance(filename,str) or Path(filename).name!=filename:
                raise ValueError('unsafe patch filename')
            path=root/filename
            if path.stat().st_size>20_000:raise ValueError('oversized patch')
            a=np.load(path,allow_pickle=False)
            if a.dtype!=np.uint8 or a.shape!=(roi[3]-roi[1],roi[2]-roi[0]):
                raise ValueError('patch shape/type differs from recorded ROI')
            arrays.append(a.astype(float))
        if arrays:
            stack=np.stack(arrays);means=stack.mean(axis=(1,2));drift=float(np.ptp(means))
            # Difference of consecutive disjoint pairs removes stationary texture.
            difference=stack[1::2][:len(stack)//2]-stack[::2][:len(stack)//2]
            variation=None if not len(difference) else float(np.sqrt(np.mean(np.var(difference,axis=(1,2)))/2))
            result['patches'][name]={'mean_y':float(np.mean(means)),'mean_drift_y':drift,
                'temporal_difference_std_y':variation,'patch_frames':len(arrays)}
            if drift>2:
                eligible=False;result['warnings'].append(name+': brightness drift >2 Y; inspect scene/light changes.')
    result['stationarity_checks_passed']=eligible
    if len(result['patches'])==2:
        t,b=result['patches']['target'],result['patches']['background']
        contrast=abs(t['mean_y']-b['mean_y'])
        noise=[t['temporal_difference_std_y'],b['temporal_difference_std_y']]
        denom=None if any(x is None for x in noise) else float(np.hypot(*noise))
        result['contrast_delta_y']=contrast
        result['contrast_variation_proxy']=(contrast/denom if eligible and denom is not None and denom>0 else None)
        if denom==0:result['warnings'].append('Zero observed variation is not proof of zero sensor noise.')
    result['comparison_signature']={key:setup[key] for key in ('actual_camera_configuration','versions')}
    result['comparison_signature']['fixed_geometry_colour']={key:setup['trial'][key]
        for key in ('target_roi','background_roi','colour_gains','lens')}
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directories',type=Path,nargs='+');p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(argv);results=[analyze(d) for d in a.directories]
    compatible=all(r['comparison_signature']==results[0]['comparison_signature'] for r in results)
    with a.output.open('x',encoding='utf-8') as out:
        json.dump({'compatible_configuration':compatible,'results':results,'winner':None,
            'note':'Compatibility does not prove same light, scene, noise or motion.'},out,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'runs':len(results),'compatible_configuration':compatible,'winner':None}))


if __name__=='__main__':main()
