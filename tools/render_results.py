"""Render public descriptive plots from the included aggregates, no camera/data download."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('assets/results/measurements.json'))
    p.add_argument('--output',type=Path,required=True,help='New directory; never overwrite published evidence')
    a=p.parse_args();data=json.loads(a.input.read_text(encoding='utf-8'))
    a.output.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    modes=data['modes'];x=np.arange(len(modes));widths=[r['width_median_px'] for r in modes]
    err=np.array([[w-r['width_iqr_px'][0],r['width_iqr_px'][1]-w] for w,r in zip(widths,modes)]).T
    fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained')
    bars=ax.bar(x,widths,color=['#408bc9','#d68b32','#8794a4','#2ba46b'],yerr=err,capsize=5)
    for b,r in zip(bars,modes):ax.text(b.get_x()+b.get_width()/2,b.get_height()+1.05,f"{b.get_height():.2f} px\n{r['eligible_frames_in_band']} frames",ha='center')
    ax.set_xticks(x,['Native Short','Native Normal','Fixed 4 ms','Adaptive'])
    ax.set_ylim(0,8.8);ax.set_ylabel('Median apparent line FWHM (reference pixels)')
    ax.set_title('Moving target at 150–250 px/s · one independent trial per mode\nWhiskers show frame IQR, not confidence intervals')
    ax.grid(axis='y',alpha=.25)
    for suffix in ('png','svg'):fig.savefig(a.output/('line-width.'+suffix),dpi=150)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3),layout='constrained')
    value=data['resources']['isolated_algorithm']['one_core_percent']
    ax.barh(['Tracker + shutter policy'],[value],color='#2ba46b');ax.set_xlim(0,100)
    ax.set_xlabel('One CPU core (%)');ax.set_title('Pi 5 · isolated algorithm, 8.3 evaluations/s\nCapture/ISP, HTTP and image/video codecs excluded')
    ax.text(value+2,0,f'{value:.1f}%',va='center');ax.grid(axis='x',alpha=.25)
    for suffix in ('png','svg'):fig.savefig(a.output/('cpu-cost.'+suffix),dpi=150)
    plt.close(fig)


if __name__=='__main__':main()
