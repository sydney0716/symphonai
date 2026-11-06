"""MCP OAuth discovery, PKCE and atomic storage in the guarded credentials folder."""
from __future__ import annotations

from symphonai_api.providers.base import read_json

import base64
import hashlib
import json
import math
import re
import secrets
import threading
import time
from urllib.parse import quote, urlencode, urlsplit, urlunsplit
from urllib.request import Request, build_opener

from symphonai_api.web import RedirectBlocker, valid_service_url
from symphonai_api.paths import ensure_private_directory, symphonai_home, write_atomic


class AuthError(ValueError):
    """An authentication failure whose message contains no server response or token."""



def _endpoint(url: str) -> str:
    if not valid_service_url(url):
        raise AuthError('OAuth requires HTTPS endpoints (HTTP is allowed on loopback only)')
    return url


def _json_request(url, data=None, *, form=False):
    try:
        _endpoint(url)
        body = None if data is None else (urlencode(data).encode() if form else json.dumps(data).encode())
        headers = {'Accept': 'application/json'}
        if body is not None:
            headers['Content-Type'] = 'application/x-www-form-urlencoded' if form else 'application/json'
        result = read_json(build_opener(RedirectBlocker()), Request(url, data=body, headers=headers), 1_000_000)
        return result
    except AuthError:
        raise
    except Exception:
        raise AuthError('OAuth request failed') from None


class TokenStore:
    def __init__(self, server: str):
        self.path = symphonai_home() / 'credentials' / f'mcp-{quote(server, safe="")}.json'

    def load(self):
        try:
            result = json.loads(self.path.read_text())
            return result if isinstance(result, dict) else {}
        except (OSError, ValueError, UnicodeError):
            return {}

    def save(self, record):
        try:
            ensure_private_directory(self.path.parent)
            self.path.parent.chmod(0o700)
            write_atomic(self.path, json.dumps(record), mode=0o600)
        except OSError:
            raise AuthError('Could not store MCP credentials') from None


class OAuthSession:
    def __init__(self, server: str, resource: str):
        self.server, self.resource = server, resource
        self.store = TokenStore(server)
        self.record = self.store.load()
        if self.record.get('resource') != resource:
            self.record = {}
        self.challenge = ''
        self._lock = threading.RLock()

    def discover(self):
        match = re.search(r'resource_metadata\s*=\s*"([^"]+)"', self.challenge, re.I)
        parsed = urlsplit(self.resource)
        metadata_url = match.group(1) if match else urlunsplit((parsed.scheme, parsed.netloc, '/.well-known/oauth-protected-resource' + parsed.path.rstrip('/'), '', ''))
        resource = _json_request(metadata_url)
        if resource.get('resource', self.resource).rstrip('/') != self.resource.rstrip('/'):
            raise AuthError('OAuth resource metadata does not match this MCP server')
        issuers = resource.get('authorization_servers', [])
        if not issuers or not isinstance(issuers[0], str):
            raise AuthError('OAuth metadata has no authorization server')
        issuer = _endpoint(issuers[0]).rstrip('/')
        parsed = urlsplit(issuer)
        urls = [urlunsplit((parsed.scheme, parsed.netloc, '/.well-known/oauth-authorization-server' + parsed.path, '', '')),
                issuer + '/.well-known/openid-configuration']
        for url in urls:
            try:
                metadata = _json_request(url)
                break
            except AuthError:
                metadata = None
        if metadata is None or metadata.get('issuer', issuer).rstrip('/') != issuer:
            raise AuthError('Could not discover the OAuth authorization server')
        for key in ('authorization_endpoint', 'token_endpoint'):
            _endpoint(metadata.get(key, ''))
        return metadata, resource

    def begin(self, redirect_uri):
        metadata, resource = self.discover()
        registration = metadata.get('registration_endpoint')
        if not registration:
            raise AuthError('Authorization server does not support dynamic client registration')
        client = _json_request(registration, {'client_name': 'SymphonAI', 'redirect_uris': [redirect_uri],
            'grant_types': ['authorization_code', 'refresh_token'], 'response_types': ['code'],
            'token_endpoint_auth_method': 'none'})
        client_id = client.get('client_id')
        if not isinstance(client_id, str) or not client_id:
            raise AuthError('OAuth registration returned no client identifier')
        verifier, state = secrets.token_urlsafe(48), secrets.token_urlsafe(32)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b'=').decode()
        fields = {'response_type': 'code', 'client_id': client_id, 'redirect_uri': redirect_uri,
                  'state': state, 'code_challenge': challenge, 'code_challenge_method': 'S256', 'resource': self.resource}
        scopes = resource.get('scopes_supported', metadata.get('scopes_supported', []))
        if scopes:
            fields['scope'] = ' '.join(scopes)
        flow = {'state': state, 'expires': time.monotonic() + 600, 'verifier': verifier,
                'redirect_uri': redirect_uri, 'client_id': client_id, 'token_endpoint': metadata['token_endpoint']}
        separator = '&' if '?' in metadata['authorization_endpoint'] else '?'
        return metadata['authorization_endpoint'] + separator + urlencode(fields), flow

    def finish(self, flow, code):
        if not isinstance(code, str) or not code:
            raise AuthError('OAuth callback omitted its authorization code')
        token = _json_request(flow['token_endpoint'], {'grant_type': 'authorization_code', 'code': code,
            'redirect_uri': flow['redirect_uri'], 'client_id': flow['client_id'],
            'code_verifier': flow['verifier'], 'resource': self.resource}, form=True)
        self._save_token(token, {'resource': self.resource, 'client_id': flow['client_id'], 'token_endpoint': flow['token_endpoint']})

    def _save_token(self, token, record):
        access = token.get('access_token')
        token_type = token.get('token_type', 'Bearer')
        if not isinstance(access, str) or not access or not isinstance(token_type, str) or token_type.lower() != 'bearer':
            raise AuthError('OAuth returned an invalid access token')
        updated = {**record, 'access_token': access}
        if token.get('refresh_token'):
            updated['refresh_token'] = token['refresh_token']
        try:
            lifetime = float(token.get('expires_in', 3600))
            if not math.isfinite(lifetime) or lifetime < 0:
                raise ValueError()
            updated['expires_at'] = time.time() + lifetime
        except (ValueError, TypeError):
            raise AuthError('OAuth returned an invalid token expiry') from None
        self.store.save(updated)
        self.record = updated

    def access_token(self, *, force=False):
        with self._lock:
            if not self.record:
                return None
            try:
                expiry = float(self.record.get('expires_at', 0))
                if not math.isfinite(expiry) or not isinstance(self.record.get('access_token'), str):
                    raise ValueError()
            except (TypeError, ValueError):
                self.record = {}
                raise AuthError('MCP sign-in required') from None
            if force or expiry <= time.time() + 10:
                refresh = self.record.get('refresh_token')
                if not refresh:
                    raise AuthError('MCP sign-in required')
                try:
                    token = _json_request(self.record['token_endpoint'], {'grant_type': 'refresh_token',
                        'refresh_token': refresh, 'client_id': self.record['client_id'], 'resource': self.resource}, form=True)
                    self._save_token(token, self.record)
                except (AuthError, KeyError):
                    self.record = {}
                    raise AuthError('MCP sign-in required: refresh failed') from None
            return self.record.get('access_token')
