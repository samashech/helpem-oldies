"""Status page for the family, reachable only from this computer."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from .app import App
from .safety import Reason

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Elder Help</title><style>
:root{--bg:#f6f4ef;--ink:#23211c;--muted:#6d695e;--line:#d9d5ca;--warn:#a33a1f;--panel:#fffdf8}
@media (prefers-color-scheme:dark){:root{--bg:#191814;--ink:#efece3;--muted:#a5a093;--line:#39362e;--warn:#ef8a6c;--panel:#22211b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.45 system-ui,sans-serif}
main{max-width:1000px;margin:auto;padding:20px 16px;display:grid;gap:20px;grid-template-columns:minmax(0,3fr) minmax(0,2fr)}
@media (max-width:760px){main{grid-template-columns:minmax(0,1fr)}}
h1{font-size:15px;font-weight:600;margin:0 0 4px;color:var(--muted)}#activity{font-size:40px;font-weight:700;margin:0;text-transform:capitalize}
#phase{margin:4px 0 16px}#phase.alert,#banner{color:var(--warn);font-weight:600}
img{width:100%;aspect-ratio:4/3;object-fit:cover;background:#000;border-radius:6px;display:block}
button{font:inherit;padding:10px 14px;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:6px;cursor:pointer;margin:0 8px 8px 0}
ol{list-style:none;padding:0;margin:12px 0 0;font-size:14px}li{padding:6px 0;border-top:1px solid var(--line);overflow-wrap:anywhere}time{color:var(--muted);margin-right:8px}
</style></head><body><main><section><img src="/stream" alt="Live camera view with body outline"></section><section>
<h1>Right now</h1><p id="activity">...</p><p id="phase"></p><p id="banner"></p>
<button onclick="post('/ok')">I'm okay, cancel the check</button><button onclick="post('/check')">Check on them now</button>
<h1 style="margin-top:16px">Recent events</h1><ol id="events"></ol></section></main><script>
const $=id=>document.getElementById(id);
const post=p=>fetch(p,{method:'POST'});
async function tick(){try{const s=await (await fetch('/state')).json();
$('activity').textContent=(s.activity==='upright'?'upper body only':s.activity)+(s.object?' · using '+s.object:'');
$('phase').textContent=s.phase==='watching'?'Watching':s.phase+': '+(s.reason||'').replaceAll('_',' ');
$('phase').className=s.phase==='watching'?'':'alert';
const b=[];if(!s.alerts_live)b.push('Practice mode: calls and texts are not being sent.');if(!s.camera_ok)b.push('Camera is not working.');if(s.view==='head_and_shoulders'||s.view==='upper_body')b.push('The camera can only see part of the body, so it cannot detect walking or someone lying on the floor. Move it further back and higher up.');$('banner').textContent=b.join(' ');
$('events').replaceChildren(...s.events.slice(0,12).map(e=>{const li=document.createElement('li'),t=document.createElement('time');t.textContent=e.time.slice(11);
const {time,event,...rest}=e;li.append(t,event.replaceAll('_',' ')+' '+Object.values(rest).filter(v=>v!==null&&!(Array.isArray(v)&&!v.length)).join(', '));return li}))
}catch(e){$('banner').textContent='Lost connection to the monitor.'}}
tick();setInterval(tick,1000)</script></body></html>"""


def serve(app: App, port: int) -> ThreadingHTTPServer:
    allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        def _trusted(self) -> bool:
            # Stops other websites in the browser from reading the camera or cancelling a check.
            origin = self.headers.get("Origin")
            return self.headers.get("Host") in allowed and (
                origin is None or origin.removeprefix("http://") in allowed
            )

        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._trusted():
                return self._send(b"forbidden", "text/plain", 403)
            if self.path == "/":
                return self._send(PAGE.encode(), "text/html; charset=utf-8")
            if self.path == "/state":
                return self._send(json.dumps(app.state(), ensure_ascii=False).encode(), "application/json")
            if self.path == "/stream":
                return self._stream()
            self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            if not self._trusted():
                return self._send(b"forbidden", "text/plain", 403)
            if self.path == "/ok":
                app.stand_down()
            elif self.path == "/check":
                app.monitor.request_check(Reason.MANUAL)
            else:
                return self._send(b"not found", "text/plain", 404)
            self._send(b"ok", "text/plain")

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while not app.stop.is_set():
                    frame = app.latest_frame
                    if frame is not None:
                        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        if ok:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
