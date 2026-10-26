"""Local demonstration UI. Latest JPEG only; bounded clients, no video recording."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = '''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Road Camera · Live</title>
<style>body{background:#111923;color:#edf3fa;font:17px system-ui;margin:24px}main{max-width:1200px;margin:auto}img{width:100%;background:#000}button{padding:10px;margin:8px}pre{white-space:pre-wrap}#health{color:#ffc970}.stats{padding:16px;background:#202d3c;border-radius:12px}</style>
<main><h1>Камера · автоматическая экспозиция</h1><p id="health">Подключение…</p>
<img id="video" src="/stream.mjpg" alt="Трансляция камеры">
<button onclick="rot=!rot;video.style.transform=rot?'rotate(180deg)':''">Повернуть изображение</button>
<button onclick="video.requestFullscreen()">На весь экран</button>
<div class="stats" id="stats"></div><p>Область замера — центральные 70% ширины и 50% высоты. Фокус фиксирован, пока не откалиброван.
Без света невозможно сохранить детали; длинная выдержка увеличивает смаз движения. Видео не записывается.</p></main>
<script>let rot=false;const video=document.getElementById('video');
async function poll(){try{const r=await fetch('/status.json',{cache:'no-store'});if(!r.ok)throw Error(r.status);const s=await r.json();
document.getElementById('health').textContent=s.error?'ОСТАНОВЛЕНО: '+s.error:s.age_s>2?'Нет свежих данных камеры':'Прямой эфир · '+s.profile;
const m=s.actual_metadata||{},d=s.decision||{};
document.getElementById('stats').textContent=`Выдержка: ${((m.ExposureTime||0)/1000).toFixed(2)} мс | Усиление: ${(m.AnalogueGain||0).toFixed(2)}× | Цифровое: ${(m.DigitalGain||0).toFixed(2)}× | Яркость Y: ${(s.meter_y||0).toFixed(1)} | Поток кодировщика: ${(s.encoded_fps||0).toFixed(1)} кадр/с | Состояние: ${(d.statuses||[]).join(', ')} | LensPosition: ${m.LensPosition??'—'} | Возраст данных: ${s.age_s.toFixed(1)} с`;
}catch(e){document.getElementById('health').textContent='Связь потеряна — изображение могло остановиться';}setTimeout(poll,500);}poll();</script></html>'''.encode('utf-8')

class LiveState:
    def __init__(self):
        self.condition = threading.Condition()
        self.frame = None
        self.sequence = 0
        self.closed = False
        self.record = {}
        self.updated = time.monotonic()
        self.started = None
        self.last_frame = None
        self.slots = threading.BoundedSemaphore(3)

    def publish(self, frame):
        data = bytes(frame)
        if not data.startswith(b'\xff\xd8') or not data.endswith(b'\xff\xd9') or len(data) > 3_000_000:
            raise ValueError('invalid or oversized JPEG')
        with self.condition:
            now = time.monotonic()
            if self.started is None:
                self.started = now
            self.last_frame = now
            self.frame = data
            self.sequence += 1
            self.condition.notify_all()

    def update(self, record):
        with self.condition:
            self.record = dict(record)
            self.updated = time.monotonic()

    def status(self):
        with self.condition:
            now = time.monotonic()
            span = 0 if self.started is None else self.last_frame-self.started
            return dict(self.record, age_s=now-self.updated,
                        video_age_s=None if self.last_frame is None else now-self.last_frame,
                        encoded_frames=self.sequence,
                        encoded_fps=(self.sequence-1)/span if span > 0 else 0)

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


def create_server(address, state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def setup(self):
            super().setup()
            self.connection.settimeout(3)

        def response(self, content, mime, code=200):
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            try:
                if self.path == '/':
                    self.response(PAGE, 'text/html; charset=utf-8')
                elif self.path == '/status.json':
                    self.response(json.dumps(state.status()).encode(), 'application/json')
                elif self.path == '/snapshot.jpg':
                    with state.condition:
                        frame = state.frame
                        fresh = state.last_frame is not None and time.monotonic()-state.last_frame < 2
                    self.response(frame if frame and fresh else b'No fresh frame', 'image/jpeg' if frame and fresh else 'text/plain', 200 if frame and fresh else 503)
                elif self.path == '/stream.mjpg':
                    if not state.slots.acquire(blocking=False):
                        self.response(b'Three viewers maximum', 'text/plain', 503)
                        return
                    try:
                        self.send_response(200)
                        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
                        self.send_header('Cache-Control', 'no-store')
                        self.end_headers()
                        last = 0
                        while True:
                            with state.condition:
                                ready = state.condition.wait_for(lambda: state.sequence != last or state.closed, timeout=3)
                                if not ready or state.closed:
                                    break
                                frame, last = state.frame, state.sequence
                            self.wfile.write(b'--FRAME\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(frame)).encode()+b'\r\n\r\n'+frame+b'\r\n')
                    finally:
                        state.slots.release()
                else:
                    self.response(b'Not found', 'text/plain', 404)
            except (OSError, TimeoutError):
                pass

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True
        # Bound all handlers, including idle or slow non-stream clients.
        clients = threading.BoundedSemaphore(8)

        def process_request(self, request, client_address):
            if not self.clients.acquire(blocking=False):
                self.shutdown_request(request)
                return
            try:
                super().process_request(request, client_address)
            except BaseException:
                self.clients.release()
                raise

        def process_request_thread(self, request, client_address):
            try:
                super().process_request_thread(request, client_address)
            finally:
                self.clients.release()

    return Server(address, Handler)
