"""Camera-free sensitivity study. Synthetic electrons; no IMX708 predictions."""
import argparse
from dataclasses import asdict
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from road_motion_budget_v01 import exposure_budget, diagnostic_target_cnr


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for velocity in (0.,100.,500.,1000.,2000.):
        budget=exposure_budget([[velocity,0]],blur_limit_px=1.,uncertainty_px_s=0.,
            now_ns=100,measured_ns=90,max_age_ns=20,frame_duration_us=33333,
            min_exposure_us=100,sensor_max_us=32680,readback_tolerance_us=42,
            geometry_calibrated=False)
        for flux in (.1,100.,1000.):
            for policy,t in [('long_exposure_reference',32680),('motion_budget',budget.max_requested_exposure_us)]:
                cnr=diagnostic_target_cnr(electrons_per_ms=flux,exposure_us=t,
                    read_noise_e=3.,target_contrast=.5,target_width_px=4.,velocity_px_s=velocity)
                rows.append(dict(velocity_px_s=velocity,synthetic_electrons_per_ms=flux,
                    policy=policy,exposure_us=t,blur_px=velocity*t/1e6,cnr_proxy=cnr,
                    illustrative_joint_pass=velocity*t/1e6<=1 and cnr>=5))
    with (args.output/'synthetic.csv').open('x',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (args.output/'scope.json').write_text(json.dumps(dict(
        synthetic=True,hardware_access=False,camera_calibrated=False,
        reference='constant 32.68ms only; NOT simulation of native AE',
        thresholds='illustrative blur<=1px AND CNR proxy>=5; not approved detection criteria',
        omitted=['ISP','optics','saturation','rolling shutter','detector','motion uncertainty'],
        rows=len(rows)),indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    t=np.geomspace(100,32680,200)
    for velocity in (0.,500.,2000.):
        cnr=[diagnostic_target_cnr(electrons_per_ms=100.,exposure_us=float(x),
             read_noise_e=3.,target_contrast=.5,target_width_px=4.,velocity_px_s=velocity) for x in t]
        axes[0].plot(t/1000,cnr,label=f'{velocity:g} px/s')
    axes[0].set(xlabel='Exposure (ms)',ylabel='Synthetic contrast/noise proxy',
                title='Shorter exposure can help OR hurt')
    axes[0].legend();axes[0].grid(alpha=.3)
    for policy in ('long_exposure_reference','motion_budget'):
        r=[r for r in rows if r['policy']==policy and r['synthetic_electrons_per_ms']==100.]
        axes[1].plot([r['velocity_px_s'] for r in r],[r['blur_px'] for r in r],marker='o',label=policy)
    axes[1].axhline(1,color='gray',linestyle='--',label='Illustrative 1 px limit')
    axes[1].set(xlabel='Image velocity (px/s)',ylabel='Linearized blur (px)',title='Blur constraint alone is insufficient')
    axes[1].legend();axes[1].grid(alpha=.3)
    fig.suptitle('SYNTHETIC MODEL — not hardware evidence')
    fig.savefig(args.output/'motion_tradeoff.png',dpi=160)
    print(json.dumps({'rows':len(rows),'synthetic_only':True,'output':str(args.output)}))


if __name__=='__main__':
    main()
