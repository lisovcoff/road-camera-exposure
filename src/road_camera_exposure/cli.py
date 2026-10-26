"""User entry point. Hardware access is explicit; core commands work offline."""
import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import load,runtime_args


def main(argv=None):
    p=argparse.ArgumentParser(prog='roadcam',description='Motion-aware exposure laboratory toolkit')
    p.add_argument('--version',action='version',version=__version__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('inspect',help='List Picamera2 cameras; no capture or controls')
    sub.add_parser('simulate',help='Run deterministic synthetic feedback, no camera')
    sub.add_parser('analyze-motion',help='Offline captured-motion analysis; see --help')
    sub.add_parser('analyze-static',help='Offline static patch analysis; see --help')
    for command in ('run','reference'):
        q=sub.add_parser(command,help='Colour target demo' if command=='run' else 'Measure AWB and save a colour reference')
        q.add_argument('--config',required=True,type=Path);q.add_argument('--output',type=Path)
        q.add_argument('--run-camera',action='store_true',help='Explicitly open/configure the camera')
        if command=='run':
            q.add_argument('--allow-lab',action='store_true',help='Acknowledge uncalibrated target demonstration')
            q.add_argument('--bind',help='Override private IPv4 for LAN preview')
    # Dispatch analysis argv intact, including subcommand --help.
    args_list=list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] in ('analyze-motion','analyze-static','simulate'):
        command=args_list.pop(0)
        if command=='analyze-motion':
            from .analysis.motion import main as task
        elif command=='analyze-static':
            from .analysis.static import main as task
        else:
            from .simulation import main as task
        return task(args_list) or 0
    a=p.parse_args(args_list)
    try:
        if a.command=='inspect':
            from picamera2 import Picamera2
            print(json.dumps(Picamera2.global_camera_info(),default=str,indent=2));return 0
        config=load(a.config)
        if not a.run_camera:
            print(json.dumps({'dry_run':True,'camera_started':False,'command':a.command,
                              'driver':config['driver'],'colour_reference_available':config['colour_gains'] is not None}))
            return 0
        if a.output is None:p.error('new --output directory required with --run-camera')
        if a.command=='reference':
            from .adapters.reference import capture
            capture(config,a.output)
            print('Saved reference.jpg, metadata.json and config.json in '+str(a.output));return 0
        if not a.allow_lab:p.error('--allow-lab required: target demonstration is not a calibrated road system')
        runtime=runtime_args(config,a.output,a.bind)
        from .adapters.picamera2 import run
        run(runtime);return 0
    except ModuleNotFoundError as error:
        print('Missing dependency: '+str(error)+'. See docs/installation.md.',file=sys.stderr);return 2
    except (ValueError,RuntimeError,OSError,TimeoutError) as error:
        print(str(error),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
