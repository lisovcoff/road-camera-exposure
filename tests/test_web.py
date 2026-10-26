import json
from pathlib import Path
import sys
import threading
import time
import unittest
import urllib.request
import urllib.error

from road_camera_exposure.web import LiveState, create_server
from road_camera_exposure.core import ExposureLimits


class LiveTests(unittest.TestCase):
    def test_latest_frame_bounded_and_invalid_rejected(self):
        state = LiveState()
        for i in range(100):
            state.publish(b'\xff\xd8'+str(i).encode()+b'\xff\xd9')
        self.assertEqual(state.frame, b'\xff\xd899\xff\xd9')
        self.assertEqual(state.sequence, 100)
        with self.assertRaises(ValueError):
            state.publish(b'not jpeg')
        self.assertEqual(state.sequence, 100)


    def test_http_stream_status_stale_and_shutdown(self):
        state = LiveState()
        server = create_server(('127.0.0.1',0),state)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = 'http://127.0.0.1:'+str(server.server_port)
        try:
            with urllib.request.urlopen(url+'/status.json') as response:
                self.assertEqual(json.load(response)['encoded_frames'],0)
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(url+'/snapshot.jpg')
            self.assertEqual(caught.exception.code,503)
            caught.exception.close()
            state.publish(b'\xff\xd8test\xff\xd9')
            with urllib.request.urlopen(url+'/stream.mjpg',timeout=3) as response:
                self.assertEqual(response.readline(),b'--FRAME\r\n')
                self.assertEqual(response.readline(),b'Content-Type: image/jpeg\r\n')
                self.assertIn(b'8',response.readline())
                self.assertEqual(response.readline(),b'\r\n')
                self.assertEqual(response.read(8),state.frame)
            with state.condition:
                state.last_frame = time.monotonic()-5
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(url+'/snapshot.jpg')
            self.assertEqual(caught.exception.code,503)
            caught.exception.close()
        finally:
            state.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())


if __name__ == '__main__':
    unittest.main()
