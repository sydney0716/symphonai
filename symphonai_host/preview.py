"""Loopback-only development preview proxy and isolated element picker."""

from __future__ import annotations

import gzip
import http.client
import json
import os
import re
import secrets
import select
import socket
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from symphonai_api.identity import new_id
from symphonai_api.secrets import mask_secrets


PICKER = r"""(() => {
  const token = __TOKEN__;
  const control = document.createElement('div');
  const shadow = control.attachShadow({mode: 'closed'});
  control.style.cssText = 'position:fixed;right:16px;bottom:16px;z-index:2147483647';
  shadow.innerHTML = '<style>button,input{font:14px system-ui;padding:8px}form{display:none;background:white;padding:8px;border:1px solid #888;color:black}p{font:12px system-ui;max-width:260px}</style><button type="button">Select</button><form><input aria-label="Comment on element" placeholder="How should this change?" maxlength="2000"><button>Add</button><p></p></form>';
  document.documentElement.append(control);
  const button = shadow.querySelector('button'), form = shadow.querySelector('form');
  const input = shadow.querySelector('input'), status = shadow.querySelector('p');
  let selecting = false, hovered = null, original = null, selected = null;
  function clear() {
    if (hovered) {
      if (original === null) hovered.removeAttribute('style');
      else hovered.setAttribute('style', original);
    }
    hovered = null;
  }
  function cancel() { selecting = false; clear(); selected = null; form.style.display = 'none'; button.textContent = 'Select'; }
  button.onclick = () => { cancel(); selecting = true; button.textContent = 'Click an element'; };
  document.addEventListener('mouseover', (event) => {
    if (!selecting || event.target === control) return;
    clear(); hovered = event.target; original = hovered.getAttribute('style'); hovered.style.outline = '2px solid #2457c5';
  }, true);
  document.addEventListener('click', (event) => {
    if (!selecting || event.target === control) return;
    event.preventDefault(); event.stopImmediatePropagation();
    selected = event.target; clear(); selecting = false;
    const rect = selected.getBoundingClientRect();
    control.style.right = 'auto'; control.style.bottom = 'auto';
    control.style.left = Math.max(8, Math.min(rect.left, innerWidth - 320)) + 'px';
    control.style.top = Math.max(8, Math.min(rect.bottom + 8, innerHeight - 120)) + 'px';
    form.style.display = 'block'; input.value = ''; status.textContent = ''; input.focus();
  }, true);
  document.addEventListener('keydown', (event) => { if (event.key === 'Escape') cancel(); }, true);
  function selector(element) {
    const parts = [];
    for (let node = element; node && node.nodeType === 1; node = node.parentElement) {
      if (node.id) {
        const id = '#' + CSS.escape(node.id);
        if (document.querySelectorAll(id).length === 1) { parts.unshift(id); break; }
      }
      let part = node.localName;
      if (node.parentElement) {
        const siblings = [...node.parentElement.children].filter(item => item.localName === node.localName);
        if (siblings.length > 1) part += ':nth-of-type(' + (siblings.indexOf(node) + 1) + ')';
      }
      parts.unshift(part);
    }
    return parts.join(' > ');
  }
  function source(element) {
    for (let node = element; node; node = node.parentElement) {
      for (const key of Object.keys(node)) {
        if (key.startsWith('__reactFiber$') || key.startsWith('__reactInternalInstance$')) {
          for (let fiber = node[key]; fiber; fiber = fiber.return) {
            const info = fiber._debugSource;
            if (info?.fileName) return info.fileName + (info.lineNumber ? ':' + info.lineNumber : '');
          }
        }
      }
      if (node._debugSource?.fileName) return node._debugSource.fileName;
      const vue = node.__vueParentComponent?.type?.__file;
      if (vue) return vue;
      const svelte = node.__svelte_meta?.loc;
      if (svelte?.file) return svelte.file + (svelte.line ? ':' + svelte.line : '');
      for (const name of ['data-source', 'data-source-file', 'data-file', 'data-loc']) {
        if (node.getAttribute(name)) return node.getAttribute(name);
      }
    }
    return '';
  }
  form.onsubmit = async (event) => {
    event.preventDefault();
    if (!selected || !input.value.trim()) return;
    const mark = {comment: input.value, url: location.href, selector: selector(selected),
      html: selected.outerHTML.slice(0, 2000), text: (selected.innerText || selected.textContent || '').slice(0, 2000),
      source_hint: source(selected)};
    try {
      const reply = await fetch('/preview/mark', {method:'POST', headers:{'Content-Type':'application/json','X-Preview-Token':token}, body:JSON.stringify(mark)});
      if (!reply.ok) throw new Error('Could not add mark');
      cancel();
    } catch { status.textContent = 'Could not add mark. Try again.'; }
  };
})();"""


def target_url(url):
    try:
        parsed = urlsplit(url)
        if (parsed.scheme != 'http' or parsed.hostname not in ('localhost', '127.0.0.1', '::1')
                or parsed.username is not None or parsed.password is not None or parsed.fragment):
            raise ValueError()
        port = parsed.port if parsed.port is not None else 80
        if port < 1:
            raise ValueError()
    except (TypeError, ValueError):
        raise ValueError('Preview requires an http URL on localhost, 127.0.0.1 or [::1]') from None
    return parsed, port


def source_guess(policy, text, html):
    classes = re.findall(r'class=["\']([^"\']+)', html)
    needles = [item for item in (text.strip()[:120], *(word for group in classes for word in group.split())) if len(item) >= 3]
    if not needles:
        return ''
    best, score = '', 0
    scanned = 0
    for folder, directories, files in os.walk(policy.repo_root, followlinks=False):
        directories[:] = [name for name in directories if name not in ('.git', 'node_modules', '.venv', 'dist', 'build', '__pycache__')
                          and policy.check_read(Path(folder) / name).allowed]
        for name in files:
            path = Path(folder) / name
            if path.suffix.lower() not in ('.html', '.htm', '.jsx', '.tsx', '.js', '.ts', '.vue', '.svelte', '.css'):
                continue
            if not policy.check_read(path).allowed:
                continue
            scanned += 1
            if scanned > 5000:
                return best
            try:
                if path.stat().st_size > 200000:
                    continue
                body = path.read_text(encoding='utf-8')
            except (OSError, UnicodeError):
                continue
            matched = sum(len(needle) for needle in needles if needle in body)
            if matched > score:
                best, score = str(path.relative_to(policy.repo_root)), matched
    return best


class Preview:
    def __init__(self, url, policy, add_mark):
        self.target, self.port = target_url(url)
        self.policy, self.add_mark = policy, add_mark
        self.token = secrets.token_urlsafe(32)
        self._closed = threading.Event()
        self._sockets = set()
        self._lock = threading.Lock()
        preview = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *_args):
                pass

            def _reply(self, status, value):
                data = json.dumps(value).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _own_origin(self):
                hosts = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
                host = self.headers.get('Host', '')
                origins = self.headers.get_all('Origin', [])
                if (len(self.headers.get_all('Host', [])) != 1 or host not in hosts
                        or (origins and origins != [f'http://{host}'])):
                    self.close_connection = True
                    self._reply(403, {'error': 'Preview origin required'})
                    return False
                return True

            def do_POST(self):
                if urlsplit(self.path).path == '/preview/mark':
                    if not self._own_origin():
                        return
                    try:
                        length = int(self.headers.get('Content-Length', '0'))
                        if not 0 < length <= 16000:
                            raise ValueError('Invalid mark size')
                        mark = preview.mark(self.headers.get('X-Preview-Token', ''), json.loads(self.rfile.read(length)))
                        self._reply(200, mark)
                    except PermissionError:
                        self._reply(403, {'error': 'Invalid preview token'})
                    except (ValueError, TypeError):
                        self._reply(400, {'error': 'Invalid preview mark'})
                    return
                self.proxy()

            def proxy(self):
                if not self._own_origin():
                    return
                if not self.path.startswith('/') or self.path.startswith('//'):
                    self._reply(400, {'error': 'Origin-relative paths required'})
                    return
                upstream = None
                started = False
                upgrade = False
                try:
                    # Never resolve an arbitrary hostname, even for localhost.
                    addresses = ['::1'] if preview.target.hostname == '::1' else ['127.0.0.1']
                    if preview.target.hostname == 'localhost':
                        addresses.append('::1')
                    for address in addresses:
                        try:
                            upstream = socket.create_connection((address, preview.port), timeout=15)
                            break
                        except OSError:
                            continue
                    if upstream is None:
                        raise OSError('Local server unavailable')
                    with preview._lock:
                        if preview._closed.is_set():
                            raise OSError('Preview stopped')
                        preview._sockets.update((upstream, self.connection))
                    upgrade = self.headers.get('Upgrade', '').lower() == 'websocket'
                    lines = [f'{self.command} {self.path} HTTP/1.1']
                    for name, value in self.headers.items():
                        if name.lower() in ('host', 'accept-encoding', 'connection', 'proxy-connection'):
                            continue
                        lines.append(f'{name}: {value}')
                    lines.extend((f'Host: {preview.target.netloc}', 'Accept-Encoding: identity',
                                  'Connection: Upgrade' if upgrade else 'Connection: close', '', ''))
                    upstream.sendall('\r\n'.join(lines).encode('latin-1'))
                    if self.headers.get('Transfer-Encoding', '').lower() == 'chunked':
                        while True:
                            line = self.rfile.readline()
                            upstream.sendall(line)
                            size = int(line.split(b';', 1)[0], 16)
                            if size:
                                upstream.sendall(self.rfile.read(size + 2))
                            else:
                                while True:
                                    trailer = self.rfile.readline()
                                    upstream.sendall(trailer)
                                    if trailer == b'\r\n':
                                        break
                                break
                    else:
                        remaining = int(self.headers.get('Content-Length', '0'))
                        while remaining:
                            block = self.rfile.read(min(65536, remaining))
                            if not block:
                                raise OSError('Incomplete request')
                            upstream.sendall(block)
                            remaining -= len(block)
                    if upgrade:
                        upstream.settimeout(None)
                        self.connection.settimeout(None)
                        self.wfile.flush()
                        while not preview._closed.is_set():
                            ready, _, _ = select.select([upstream, self.connection], [], [], 1)
                            for source in ready:
                                block = source.recv(65536)
                                if not block:
                                    return
                                (self.connection if source is upstream else upstream).sendall(block)
                        return
                    response = http.client.HTTPResponse(upstream)
                    response.begin()
                    html = 'text/html' in response.getheader('Content-Type', '').lower() and self.command != 'HEAD'
                    data = None
                    if html:
                        data = response.read()
                        encoding = response.getheader('Content-Encoding', '').lower()
                        if encoding == 'gzip':
                            data = gzip.decompress(data)
                        elif encoding == 'deflate':
                            data = zlib.decompress(data)
                        elif encoding not in ('', 'identity'):
                            raise ValueError('Unsupported HTML encoding')
                        script = ('<script>' + PICKER.replace('__TOKEN__', json.dumps(preview.token)) + '</script>').encode()
                        index = data.lower().rfind(b'</body>')
                        data = data[:index] + script + data[index:] if index >= 0 else data + script
                    self.send_response_only(response.status, response.reason)
                    for name, value in response.getheaders():
                        if name.lower() in ('connection', 'transfer-encoding'):
                            continue
                        if html and name.lower() in ('content-length', 'content-encoding', 'content-security-policy', 'etag'):
                            continue
                        self.send_header(name, value)
                    self.send_header('Connection', 'close')
                    if html:
                        self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    started = True
                    if self.command != 'HEAD':
                        if html:
                            self.wfile.write(data)
                        else:
                            while block := response.read1(65536):
                                self.wfile.write(block)
                                self.wfile.flush()
                except (OSError, ValueError, zlib.error, http.client.HTTPException):
                    if not started and not upgrade:
                        try:
                            self._reply(502, {'error': 'Local preview server unavailable'})
                        except OSError:
                            pass
                    self.close_connection = True
                finally:
                    self.close_connection = True
                    if upstream is not None:
                        with preview._lock:
                            preview._sockets.discard(upstream)
                            preview._sockets.discard(self.connection)
                        upstream.close()

            do_GET = proxy
            do_HEAD = proxy
            do_PUT = proxy
            do_PATCH = proxy
            do_DELETE = proxy
            do_OPTIONS = proxy

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.origin = f'http://127.0.0.1:{self.server.server_port}'
        self.url = self.origin + (self.target.path or '/') + ('?' + self.target.query if self.target.query else '')
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True, name='symphonai-preview')
        self.thread.start()

    def mark(self, token, value):
        if self._closed.is_set() or not isinstance(token, str) or not secrets.compare_digest(token, self.token):
            raise PermissionError('Invalid preview token')
        if not isinstance(value, dict):
            raise ValueError('Invalid preview mark')
        limits = {'comment': 2000, 'url': 4000, 'selector': 2000, 'html': 2000, 'text': 2000, 'source_hint': 1000}
        mark = {}
        for field, limit in limits.items():
            text = value.get(field, '')
            if not isinstance(text, str) or len(text) > limit:
                raise ValueError('Invalid preview mark field')
            mark[field] = mask_secrets(text)
        if not all(mark[field].strip() for field in ('comment', 'url', 'selector', 'html')):
            raise ValueError('Missing preview mark field')
        location = urlsplit(mark['url'])
        if (location.scheme, location.netloc) not in (('http', urlsplit(self.origin).netloc), ('http', self.target.netloc)):
            raise ValueError('Mark URL is outside this preview')
        mark['source_guess'] = not bool(mark['source_hint'])
        if mark['source_guess']:
            mark['source_hint'] = source_guess(self.policy, mark['text'], mark['html'])
        mark['id'] = new_id('mark')
        if self._closed.is_set():
            raise PermissionError('Preview stopped')
        self.add_mark(mark)
        return mark

    def close(self):
        self._closed.set()
        self.server.shutdown()
        self.server.server_close()
        with self._lock:
            for connection in self._sockets:
                connection.close()
            self._sockets.clear()
        self.thread.join(timeout=2)
