"""Receive and decode a bounded HTTP MJPEG sample; no image/video files saved."""
import argparse
import io
import json
import time
import urllib.request
from PIL import Image


def probe(url, count=60):
    total = 0
    arrivals = []
    sizes = set()
    with urllib.request.urlopen(url.rstrip('/')+'/stream.mjpg', timeout=5) as response:
        for _ in range(count):
            if response.readline(100) != b'--FRAME\r\n':
                raise ValueError('invalid boundary')
            headers = {}
            for _ in range(8):
                line = response.readline(200)
                if line == b'\r\n':
                    break
                key, value = line.decode('ascii').split(':',1)
                headers[key.lower()] = value.strip()
            length = int(headers['content-length'])
            if not 0 < length <= 3_000_000 or total+length > 60_000_000:
                raise ValueError('sample byte bound exceeded')
            frame = response.read(length)
            if len(frame) != length or response.read(2) != b'\r\n':
                raise ValueError('truncated frame')
            arrivals.append(time.monotonic())
            with Image.open(io.BytesIO(frame)) as image:
                image.load()
                sizes.add(image.size)
            total += length
    duration = arrivals[-1]-arrivals[0]
    return {'decoded_frames':len(arrivals),'dimensions':sorted(sizes),'bytes':total,
            'receive_fps':(len(arrivals)-1)/duration,'receive_span_s':duration,
            'largest_arrival_gap_ms':max(b-a for a,b in zip(arrivals,arrivals[1:]))*1000,
            'note':'client receive+decode sample, not browser rendering or end-to-end optical latency'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('url')
    args = parser.parse_args()
    print(json.dumps(probe(args.url)))
