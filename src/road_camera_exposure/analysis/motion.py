"""Affine-track captured trial_* folders; describe speed-binned line widths.

One run per mode is feasibility evidence, not independent repeated trials.
Never converts adjacent video frames into independent statistical repeats.
"""
import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

from ..motion_analysis import register_affine,feature_profile,matched_velocity_rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--template-roi',type=int,nargs=4,required=True)
    p.add_argument('--line-window',type=int,nargs=2,required=True);p.add_argument('--line-y',type=int,nargs=2,required=True)
    p.add_argument('--feature-relative',type=float,nargs=2,required=True)
    p.add_argument('--tracking-cache',type=Path,help='Optional prior fits, timestamps must match captures; cache provenance must be audited')
    a=p.parse_args(argv);a.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for directory in sorted(a.root.glob('trial_*')):
        raw=[json.loads(l) for l in (directory/'observations.jsonl').read_text(encoding='utf-8').splitlines()]
        summary=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
        setup=json.loads((directory/'setup.json').read_text(encoding='utf-8'))
        if summary['error'] or summary['cleanup_errors'] or len(raw)!=summary['frames']:raise ValueError('failed/incomplete capture')
        cached=None if a.tracking_cache is None else a.tracking_cache/(directory.name+'_tracking_001.json')
        if cached is not None:
            tracks=json.loads(cached.read_text(encoding='utf-8'))
            if len(tracks)!=len(raw) or any(t['sensor_ns']!=r['actual_metadata']['SensorTimestamp'] or t['exposure_us']!=r['actual_metadata']['ExposureTime'] for t,r in zip(tracks,raw)):
                raise ValueError('cached tracking does not match captured frame timestamps')
        else:
            ref=np.load(directory/raw[0]['file'],allow_pickle=False);warp=None;tracks=[]
            for i,row in enumerate(raw):
                image=np.load(directory/row['file'],allow_pickle=False)
                fit=register_affine(ref,image,a.template_roi,initial=warp)
                if fit['accepted']:warp=fit['warp']
                line=feature_profile(ref,image,fit['warp'],a.template_roi,line_window=a.line_window,y_range=a.line_y) if fit['accepted'] else {'valid':False}
                tracks.append({'index':i,'sensor_ns':row['actual_metadata']['SensorTimestamp'],
                    'exposure_us':row['actual_metadata']['ExposureTime'],
                    **{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in fit.items()},'line':line})
        (a.output/(directory.name+'_tracks.json')).write_text(json.dumps(tracks),encoding='utf-8')
        rows=matched_velocity_rows(tracks,a.feature_relative)
        with (a.output/(directory.name+'_eligible.csv')).open('x',newline='') as out:
            if rows:
                w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        bins=[]
        for low,high in [(0,20),(50,150),(150,250),(250,350)]:
            selected=[r for r in rows if low<=r['speed_px_s']<high]
            if selected:
                bins.append({'speed_bin_px_s':[low,high],'frames':len(selected),
                    'width_median_px':float(np.median([r['width_fwhm_px'] for r in selected])),
                    'width_iqr_px':np.percentile([r['width_fwhm_px'] for r in selected],[25,75]).tolist(),
                    'contrast_median':float(np.median([r['relative_contrast'] for r in selected]))})
        results.append({'trial':directory.name,'mode':setup['motion_spec']['trial']['mode'],
            'frames':len(raw),'tracking_accepted':sum(t['accepted'] for t in tracks),'eligible':len(rows),
            'gaps_over_50ms':sum((b['actual_metadata']['SensorTimestamp']-a['actual_metadata']['SensorTimestamp'])>50_000_000 for a,b in zip(raw,raw[1:])),'bins':bins})
        print(directory.name,len(rows),'eligible',flush=True)
    report={'results':results,'independent_repeats_per_mode':1,'global_superiority_proven':False,
        'scope':'apparent line width in this sequential moving-card feasibility test; not recall or sensor MTF',
        'parameters':vars(a),'affine_source_sha256':hashlib.sha256(inspect.getsource(register_affine).encode()).hexdigest()}
    with (a.output/'report.json').open('x',encoding='utf-8') as out:json.dump(report,out,default=str,ensure_ascii=False,indent=2)


if __name__=='__main__':main()
