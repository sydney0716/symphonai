"""Core access boundaries through public policies, tools and agent calls."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import urllib.error
from dataclasses import replace
from unittest import mock

from scripts.checks.harness import check, skip
from scripts.checks.workspace import handoff_spec, workspace
from symphonai_api.agent_loop import ApiAgent
from symphonai_api.agent_file import load_agent_file
from symphonai_api.config import CapabilityCeiling
from symphonai_api.extensions import load_extensions
from symphonai_api.leader import DispatchSubagentTool
from symphonai_api.models import Message, ModelResponse, ModelRequest, Role, ToolCall
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.base import ProviderError
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_api.providers.openai_provider import OpenAIProvider
from symphonai_api.tools import shell
from symphonai_host.broker import EventBroker
from symphonai_host.run import HostRun


def execute(ws, name, arguments, policy=None):
    return ws.tools[name].execute(ToolCall('access-call', name, arguments), policy or ws.policy)


def shell_policy(ws, **changes):
    return replace(ws.policy, shell_enabled=True, shell_allowlist=[()], mode='allow',
                   allow_uncheckable_shell=True, **changes)


def run_shell(ws, argv, policy=None):
    return execute(ws, 'run_shell', {'argv': argv}, policy or shell_policy(ws))


def seatbelt():
    if sys.platform != 'darwin' or not shell.sandbox_available():
        skip('macOS Seatbelt is unavailable')


def configured_policy(ws, home):
    run = HostRun(FakeModelProvider(), shell_policy(ws), EventBroker(),
                  sessions_root=ws.outside/'sessions',
                  extensions=load_extensions(repo_root=ws.root, home=home))
    try:
        return replace(run.policy, mode='allow')
    finally:
        run.close()


@check('access.owner_limits_hold_before_approval')
def owner_limits_hold_before_approval():
    with workspace() as ws:
        docs = ws.root/'docs'; docs.mkdir()
        requests = []

        def approve(request):
            requests.append(request)
            return True

        base = replace(ws.policy, shell_enabled=True, shell_allowlist=[()],
                       fetch_enabled=True, approval_callback=approve)
        for mode in ('plan', 'ask', 'auto', 'allow'):
            policy = replace(base, mode=mode)
            bounded = CapabilityCeiling(allowed_write_scope=(docs,)).bound(policy)
            for copied in (bounded, replace(bounded), policy.narrowed(bounded), bounded.narrowed(policy),
                           CapabilityCeiling().bound(bounded)):
                assert not copied.check_write('src/a.py').allowed
            assert not CapabilityCeiling(shell_enabled=False).bound(policy).check_shell(['echo', 'hello']).allowed
            assert not CapabilityCeiling(fetch_enabled=False).bound(policy).check_fetch('https://8.8.8.8').allowed
            assert not CapabilityCeiling(shell_allowlist=(('git', 'status'),)).bound(policy).check_shell(['echo', 'hello']).allowed
            assert not CapabilityCeiling(fetch_allowlist=('allowed.example',)).bound(policy).check_fetch('https://8.8.8.8').allowed
        assert requests == []

        agent = ws.root/'worker.toml'
        agent.write_text('prompt = "Work"\n[model]\nprovider = "fake"\n[policy]\nshell_enabled = false\nfetch_enabled = false\n')
        disabled = load_agent_file(agent, repo_root=ws.root).policy_ceiling
        for mode in ('ask', 'auto'):
            policy = replace(base.narrowed(disabled), mode=mode)
            for copied in (policy, replace(policy), CapabilityCeiling().bound(policy)):
                assert not copied.check_shell(['echo', 'hello']).allowed
                assert not copied.check_fetch('https://8.8.8.8').allowed
        assert requests == []

        agent.write_text('prompt = "Work"\n[model]\nprovider = "fake"\n[policy]\n')
        omitted = load_agent_file(agent, repo_root=ws.root).policy_ceiling
        assert replace(base.narrowed(omitted), mode='ask').check_shell(['echo', 'hello']).allowed
        assert len(requests) == 1
        requests.clear()
        soft = replace(base, mode='ask', allowed_write_scope=[docs])
        assert soft.check_write('src/a.py').allowed
        assert len(requests) == 1 and "outside this agent's write scope" in requests[0].details
        requests.clear()
        assert not replace(soft, mode='allow').check_write('src/a.py').allowed
        assert requests == []

        diffs = []
        for mode in ('ask', 'auto'):
            path = ws.root/'diff.txt'; path.write_text('before\n')
            policy = replace(base, mode=mode)
            assert execute(ws, 'read_file', {'path': 'diff.txt'}, policy).ok
            assert execute(ws, 'edit_file', {'path': 'diff.txt', 'old_string': 'before', 'new_string': 'after'}, policy).ok
            assert len(requests) == 1 and requests[0].diff
            diffs.append(requests.pop().diff)
        assert diffs[0] == diffs[1]


@check('access.file_writes_stay_in_scope')
def file_writes_stay_in_scope():
    with workspace() as ws:
        allowed = ws.root/'allowed'; allowed.mkdir()
        denied = ws.root/'denied.txt'; denied.write_text('before')
        outside = ws.outside/'outside.txt'; outside.write_text('before')
        link = allowed/'linked.txt'; link.symlink_to(outside)
        policy = replace(ws.policy, allowed_write_scope=[allowed])
        # Prime edit ledgers through real reads, so an unread-file guard cannot hide a scope defect.
        assert execute(ws, 'read_file', {'path': str(denied)}).ok
        assert execute(ws, 'read_file', {'path': str(outside)}, PermissionPolicy(ws.outside)).ok
        for path in (str(denied), os.path.relpath(outside, ws.root), str(link)):
            for name, arguments in (
                ('write_file', {'path': path, 'content': 'after'}),
                ('edit_file', {'path': path, 'old_string': 'before', 'new_string': 'after'}),
                ('multi_edit_file', {'path': path, 'edits': [{'old_string': 'before', 'new_string': 'after'}]}),
            ):
                assert not execute(ws, name, arguments, policy).ok, (name, path)
                assert denied.read_text() == outside.read_text() == 'before'
        for name in ('write_file', 'edit_file', 'multi_edit_file'):
            target = allowed/(name+'.txt'); target.write_text('before')
            assert execute(ws, 'read_file', {'path': str(target)}, policy).ok
            arguments = {'path': str(target), 'content': 'after'} if name == 'write_file' else (
                {'path': str(target), 'old_string': 'before', 'new_string': 'after'} if name == 'edit_file'
                else {'path': str(target), 'edits': [{'old_string': 'before', 'new_string': 'after'}]}
            )
            assert execute(ws, name, arguments, policy).ok and target.read_text() == 'after'


def forbidden_fixture(ws):
    paths = ['.env', '.ENV', '.env.local', 'cert.pem', 'cert.PEM', 'private.key', '.ssh/credentials']
    for relative in paths:
        path = ws.root/relative; path.parent.mkdir(exist_ok=True)
        path.write_text('SECRET_NEEDLE')
    (ws.root/'ordinary.txt').write_text('VISIBLE_NEEDLE')
    return paths


@check('access.file_tools_refuse_forbidden_files')
def file_tools_refuse_forbidden_files():
    with workspace() as ws:
        for path in forbidden_fixture(ws):
            assert not execute(ws, 'read_file', {'path': path}).ok, path
            assert not execute(ws, 'write_file', {'path': path, 'content': 'changed'}).ok, path


@check('access.discovery_omits_forbidden_files')
def discovery_omits_forbidden_files():
    with workspace() as ws:
        paths = forbidden_fixture(ws)
        for name, args in [('list_files', {'path': '.'}), ('glob', {'pattern': '**/*'}), ('grep', {'pattern': 'NEEDLE'})]:
            result = execute(ws, name, args)
            assert result.ok, (name, result.error)
            assert 'ordinary.txt' in result.content, name
            assert 'SECRET_NEEDLE' not in result.content
            if name != 'list_files':
                assert not any(path in result.content for path in paths), name
        for path in paths:
            result = execute(ws, 'read_file', {'path': path})
            assert not result.ok and 'SECRET_NEEDLE' not in result.content


def forbidden_shell_read(method):
    seatbelt()
    with workspace() as ws:
        secret = ws.root/'private.key'; secret.write_text('FIXTURE_PRIVATE_VALUE')
        if method == 'cat': argv = ['cat', str(secret)]
        elif method == 'copy': argv = ['cp', str(secret), str(ws.root/'copied.txt')]
        elif method == 'symlink':
            (ws.root/'alias').symlink_to(secret)
            argv = ['cat', str(ws.root/'alias')]
        else:
            script = ws.root/'probe.py'
            script.write_text("from pathlib import Path\nprint(Path('private.key').read_text())\n")
            argv = [sys.executable, str(script)]
        result = run_shell(ws, argv)
        text = (result.error or '') + result.content
        assert not result.ok and 'FIXTURE_PRIVATE_VALUE' not in text
        assert 'private.key' in text and '[sandbox] readable' in text


@check('access.seatbelt_blocks_cat')
def seatbelt_blocks_cat(): forbidden_shell_read('cat')


@check('access.seatbelt_blocks_copy')
def seatbelt_blocks_copy(): forbidden_shell_read('copy')


@check('access.seatbelt_blocks_symlink_reads')
def seatbelt_blocks_symlink_reads(): forbidden_shell_read('symlink')


@check('access.seatbelt_blocks_script_reads')
def seatbelt_blocks_script_reads(): forbidden_shell_read('script')


@check('access.seatbelt_keeps_ordinary_git_and_venv_reads')
def seatbelt_keeps_ordinary_git_and_venv_reads():
    seatbelt()
    with workspace() as ws:
        (ws.root/'.git').mkdir(); (ws.root/'.git'/'config').write_text('git fixture')
        (ws.root/'.venv').mkdir(); (ws.root/'.venv'/'cacert.pem').write_text('certificate fixture')
        for path, text in [('existing.txt', 'hello from disk'), ('.git/config', 'git fixture'), ('.venv/cacert.pem', 'certificate fixture')]:
            result = run_shell(ws, ['cat', path])
            assert result.ok and text in result.content, (path, result.error)


@check('access.sandbox_readable_lifts_only_its_pattern')
def sandbox_readable_lifts_only_its_pattern():
    seatbelt()
    with workspace() as ws:
        home = ws.outside/'home'; (home/'.symphonai').mkdir(parents=True)
        (home/'.symphonai'/'config.toml').write_text('[sandbox]\nreadable = [".env"]\n')
        (ws.root/'.env').write_text('allowed fixture')
        (ws.root/'.env.local').write_text('still forbidden fixture')
        policy = configured_policy(ws, home)
        allowed = run_shell(ws, ['cat', '.env'], policy)
        denied = run_shell(ws, ['cat', '.env.local'], policy)
        assert allowed.ok and 'allowed fixture' in allowed.content
        assert not denied.ok and 'still forbidden fixture' not in denied.content


@check('access.bwrap_masks_secrets_but_not_git_or_venv')
def bwrap_masks_secrets_but_not_git_or_venv():
    with workspace() as ws:
        paths = forbidden_fixture(ws)
        (ws.root/'.git').mkdir(); (ws.root/'.git'/'config').write_text('git')
        (ws.root/'.venv').mkdir(); (ws.root/'.venv'/'cacert.pem').write_text('cert')
        with mock.patch.object(sys, 'platform', 'linux'), mock.patch.object(shell, 'sandbox_available', return_value=True):
            argv = shell.sandbox_command(shell_policy(ws), ['echo', 'ok'], ws.root)
        hidden = {argv[i+2] for i in range(len(argv)-2) if argv[i:i+2] == ['--ro-bind', '/dev/null']}
        directories = {argv[i+1] for i in range(len(argv)-1) if argv[i] == '--tmpfs'}
        forbidden_names = {Path(path).name.casefold() for path in paths}
        actual_files = {str(path) for path in ws.root.iterdir() if path.is_file() and path.name.casefold() in forbidden_names}
        assert actual_files <= hidden
        assert str(ws.root/'.ssh') in directories
        assert str(ws.root/'.git'/'config') not in hidden
        assert str(ws.root/'.venv'/'cacert.pem') not in hidden
        assert '--unshare-net' in argv and argv[-2:] == ['echo', 'ok']


@check('access.sandbox_cannot_write_outside_project')
def sandbox_cannot_write_outside_project():
    seatbelt()
    with workspace() as ws, tempfile.TemporaryDirectory(prefix='.access-check-', dir=Path(__file__).resolve().parents[2]) as outside:
        target = Path(outside)/'marker'
        result = run_shell(ws, [sys.executable, '-c', f'from pathlib import Path; Path({str(target)!r}).touch()'])
        assert not result.ok and not target.exists()


@check('access.sandbox_network_requires_permission')
def sandbox_network_requires_permission():
    seatbelt()
    with workspace() as ws, socket.socket() as server:
        server.bind(('127.0.0.1', 0)); server.listen()
        port = server.getsockname()[1]
        loopback = f'import socket; socket.create_connection(("127.0.0.1",{port}),timeout=0.3).close()'
        reserved = """import socket, json, time
started = time.monotonic()
try:
    socket.create_connection(('192.0.2.1', 9), timeout=0.3).close()
    result = {'connected': True, 'errno': None}
except OSError as exc:
    result = {'connected': False, 'errno': exc.errno}
result['elapsed'] = time.monotonic() - started
print(json.dumps(result))
"""
        for allowed in (False, True):
            policy = shell_policy(ws, sandbox_network=allowed)
            assert run_shell(ws, [sys.executable, '-c', loopback], policy).ok
            result = run_shell(ws, [sys.executable, '-c', reserved], policy)
            assert result.ok, result.error
            observed = json.loads(result.content)
            denied = observed['errno'] in (1, 13)
            assert denied is not allowed, (allowed, observed)
            assert not observed['connected']
            if not allowed:
                assert observed['elapsed'] < 0.3, observed


@check('access.plan_allows_reads_and_refuses_mutations')
def plan_allows_reads_and_refuses_mutations():
    with workspace() as ws:
        policy = replace(shell_policy(ws), mode='plan')
        assert execute(ws, 'read_file', {'path': 'existing.txt'}, policy).ok
        for name, args in [('write_file', {'path': 'new.txt', 'content': 'bad'}),
                           ('edit_file', {'path': 'existing.txt', 'old_string': 'hello', 'new_string': 'bad'}),
                           ('multi_edit_file', {'path': 'existing.txt', 'edits': [{'old_string': 'hello', 'new_string': 'bad'}]}),
                           ('run_shell', {'argv': ['touch', 'new.txt']})]:
            assert not execute(ws, name, args, policy).ok, name
        assert not (ws.root/'new.txt').exists()
        assert (ws.root/'existing.txt').read_text() == 'hello from disk'


@check('access.ask_shell_requires_approval')
def ask_shell_requires_approval():
    for approved in (False, True):
        with workspace() as ws:
            requests = []
            def approval(request): requests.append(request); return approved
            policy = replace(shell_policy(ws), mode='ask', approval_callback=approval)
            result = run_shell(ws, ['touch', 'approved.txt'], policy)
            assert result.ok is approved and bool(requests)
            assert (ws.root/'approved.txt').exists() is approved


@check('access.handoff_validation_requires_approval')
def handoff_validation_requires_approval():
    for approved in (False, True):
        with workspace() as ws:
            requests = []
            def approval(request): requests.append(request); return approved
            policy = replace(shell_policy(ws), mode='ask', approval_callback=approval)
            dispatch = DispatchSubagentTool(FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, 'done'))]), policy)
            result = dispatch.execute(ToolCall('handoff', 'dispatch_subagent', {
                'subagent_name': 'worker', 'task': handoff_spec('inspect', 'touch validated.txt'),
            }), policy)
            assert result.ok is approved
            assert any(request.operation == 'run_shell' for request in requests)
            assert (ws.root/'validated.txt').exists() is approved


@check('access.always_deny_commands_do_not_execute')
def always_deny_commands_do_not_execute():
    with workspace() as ws:
        target = ws.root/'marker'; target.touch()
        for argv in (['rm', str(target)], ['sudo', '--version'], ['curl', '--version']):
            assert not run_shell(ws, argv).ok, argv[0]
        assert target.exists()


@check('access.only_user_config_can_lift_always_deny')
def only_user_config_can_lift_always_deny():
    for user_scope in (False, True):
        with workspace() as ws:
            home = ws.outside/'home'; (home/'.symphonai').mkdir(parents=True)
            folder = home/'.symphonai' if user_scope else ws.root/'.symphonai'
            folder.mkdir(exist_ok=True)
            (folder/'config.toml').write_text('[always_deny]\nlift = ["rm"]\n')
            policy = configured_policy(ws, home)
            target = ws.root/'removable'; target.touch()
            result = run_shell(ws, ['rm', str(target)], policy)
            assert result.ok is user_scope and target.exists() is not user_scope


@check('access.tool_results_mask_secret_values_and_key_shapes')
def tool_results_mask_secret_values_and_key_shapes():
    with workspace() as ws:
        secret = 'CORE_FIXTURE_SECRET_VALUE_12345'; key = 'sk-'+'A'*30; value = 'fake_key_value_12345'
        (ws.root/'ordinary.txt').write_text(f'normal {secret} {key} api_key={value}')
        responses = [ModelResponse(Message(Role.ASSISTANT, tool_calls=[ToolCall('read', 'read_file', {'path': 'ordinary.txt'})])), ModelResponse(Message(Role.ASSISTANT, 'done'))]
        with mock.patch.dict(os.environ, {'CORE_ACCESS_SECRET': secret}):
            result = ApiAgent(FakeModelProvider(responses), ws.tools, ws.policy).run([Message(Role.USER, 'read')])
        text = next(m.tool_result.content for m in result.messages if m.tool_result is not None)
        assert 'normal' in text and all(value not in text for value in (secret, key, value))


@check('access.provider_errors_mask_the_api_key')
def provider_errors_mask_the_api_key():
    key = 'core-provider-key-not-real-12345'
    error = urllib.error.HTTPError('https://fixture.invalid', 400, 'bad', {}, io.BytesIO(json.dumps({'error': {'message': key}}).encode()))
    with mock.patch.dict(os.environ, {'OPENAI_API_KEY': key}), mock.patch('urllib.request.urlopen', side_effect=error):
        try: OpenAIProvider().create_response(ModelRequest(messages=[Message(Role.USER, 'request')]))
        except ProviderError as exc: assert key not in str(exc)
        else: raise AssertionError('scripted provider failure did not fail')


@check('access.agent_shell_has_no_secret_named_variables')
def agent_shell_has_no_secret_named_variables():
    with workspace() as ws, mock.patch.dict(os.environ, {'CORE_KEY': 'fixture', 'CORE_SECRET': 'fixture', 'CORE_TOKEN': 'fixture'}):
        code = 'import os,json; print(json.dumps([k for k in os.environ if any(s in k.upper() for s in ("KEY","SECRET","TOKEN"))]))'
        result = run_shell(ws, [sys.executable, '-c', code])
        assert result.ok and json.loads(result.content) == []


@check('access.patch_preflights_every_path')
def patch_preflights_every_path():
    with workspace() as ws:
        first = ws.root/'first.txt'
        first.write_text('before\n')
        assert execute(ws, 'read_file', {'path': 'first.txt'}).ok
        for denied in ('.env', str(ws.outside/'outside.txt')):
            patch = ('*** Begin Patch\n*** Update File: first.txt\n@@\n-before\n+after\n'
                     f'*** Add File: {denied}\n+blocked\n*** End Patch')
            result = execute(ws, 'apply_patch', {'input': patch})
            assert not result.ok and denied in result.error
            assert first.read_text() == 'before\n'
            assert not (ws.outside/'outside.txt').exists()


@check('access.mcp_credentials_are_private_to_the_host')
def mcp_credentials_are_private_to_the_host():
    with workspace() as ws:
        default_home = ws.root / 'default-home'
        configured = ws.root / 'configured-home'
        with mock.patch('pathlib.Path.home', return_value=default_home), mock.patch.dict(os.environ, {'SYMPHONAI_HOME': str(configured)}):
            targets = [default_home / '.symphonai' / 'credentials', configured / 'credentials']
            policy = shell_policy(ws, forbidden_patterns=(), sandbox_readable=('credentials',), shell_sandbox=True)
            for directory in targets:
                directory.mkdir(parents=True)
                path = directory / 'mcp-scripted.json'
                path.write_text('fixture-secret-do-not-read')
                assert not execute(ws, 'read_file', {'path': str(path)}, policy).ok
                assert not execute(ws, 'write_file', {'path': str(path), 'content': 'replacement'}, policy).ok
            if not shell.sandbox_available():
                skip('platform shell sandbox is unavailable')
            for directory in targets:
                result = run_shell(ws, ['cat', str(directory / 'mcp-scripted.json')], policy)
                assert not result.ok
                assert 'fixture-secret-do-not-read' not in result.content


@check('access.preview_tokens_only_add_local_preview_marks')
def preview_tokens_only_add_local_preview_marks():
    import http.client
    import re
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import urlsplit
    from symphonai_host.server import HostServer
    from symphonai_host.projects import ProjectRuntime

    class Page(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = b'<html><body><button class="special-button">Button</button></body></html>'
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with workspace() as ws:
        (ws.root / 'button.jsx').write_text('<button className="special-button">Button</button>')
        page = ThreadingHTTPServer(('127.0.0.1', 0), Page)
        thread = threading.Thread(target=page.serve_forever, daemon=True)
        thread.start()
        broker = EventBroker()
        run = HostRun(FakeModelProvider(), ws.policy, broker, sessions_root=ws.outside / 'sessions',
                      memory_extraction_enabled=False)
        host = HostServer(FakeModelProvider(), ProjectRuntime(ws.root, run), broker=broker, home=ws.outside / 'home')
        host.start()

        def request(path, body=None, token=None, preview_token=None):
            connection = http.client.HTTPConnection('127.0.0.1', host.port, timeout=5)
            headers = {'Content-Type': 'application/json'}
            if token:
                headers['Authorization'] = 'Bearer ' + token
            if preview_token:
                headers['X-Preview-Token'] = preview_token
            connection.request('GET' if body is None else 'POST', path,
                               body=None if body is None else json.dumps(body), headers=headers)
            response = connection.getresponse()
            status, raw = response.status, response.read()
            connection.close()
            try:
                value = json.loads(raw)
            except ValueError:
                value = None
            return status, value

        try:
            for url in ('http://example.com', 'https://localhost', 'http://127.0.0.1.example.com', 'http://user@localhost'):
                status, _ = request('/preview/start', {'url': url}, host.token)
                assert status == 400, url
            status, reply = request('/preview/start', {'url': f'http://127.0.0.1:{page.server_port}'}, host.token)
            assert status == 200
            preview_url = reply['preview_url']
            proxy = urlsplit(preview_url)
            connection = http.client.HTTPConnection(proxy.hostname, proxy.port, timeout=5)
            connection.request('GET', '/')
            response = connection.getresponse()
            html = response.read().decode()
            connection.close()
            preview_token = re.search(r'const token = "([^"]+)"', html)[1]
            mark = {'comment': 'Make it blue', 'url': preview_url, 'selector': 'button.special-button',
                    'html': '<button class="special-button">Button</button>', 'text': 'Button', 'source_hint': ''}
            assert request('/preview/mark', mark)[0] == 403
            assert request('/preview/mark', mark, preview_token='invalid')[0] == 403
            status, added = request('/preview/mark', mark, preview_token=preview_token)
            assert status == 200 and added['source_guess'] and added['source_hint'] == 'button.jsx'
            assert request('/preview/marks', token=host.token)[1]['marks'][0]['id'] == added['id']
            for path, body in (('/preview/marks', None), ('/settings', None), ('/preview/stop', {}),
                               ('/preview/start', {'url': preview_url}), ('/mode', {'mode': 'allow'})):
                assert request(path, body, preview_token, preview_token)[0] in (401, 403), path
            assert request('/preview/mark', {'remove': [added['id']]}, preview_token=preview_token)[0] in (401, 403)
            assert request('/preview/mark', {'remove': [added['id']], 'session_id': None}, host.token)[0] == 200
            assert not request('/preview/marks', token=host.token)[1]['marks']
            request('/preview/stop', {}, host.token)
            assert request('/preview/mark', mark, preview_token=preview_token)[0] == 403
        finally:
            host.close()
            page.shutdown()
            page.server_close()
            thread.join(timeout=2)


@check('access.auto_fixed_rules_refuse_before_review')
def auto_fixed_rules_refuse_before_review():
    from symphonai_api.permissions import PermissionDecision
    from symphonai_host.approvals import ApprovalBroker
    from symphonai_host.auto_review import review_call

    reviewer = FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, '{"decision":"approve","reason":"fixture"}'))])
    requests = []

    def review(request):
        requests.append(request)
        review_call(reviewer, None, request, user_text='fixture', agent_text='', repo_root=ws.root,
                    on_usage=lambda *args: None)
        return PermissionDecision.allow()

    with workspace() as ws:
        broker = ApprovalBroker(lambda approval: False, review=review)
        policy = replace(ws.policy, mode='auto', shell_enabled=True, shell_allowlist=[()], approval_callback=broker.callback)
        for name, arguments in (
            ('run_shell', {'argv': ['rm', '-rf', '/']}),
            ('write_file', {'path': '.env', 'content': 'must not write'}),
            ('write_file', {'path': str(ws.outside / 'outside.txt'), 'content': 'must not write'}),
        ):
            result = execute(ws, name, arguments, policy)
            assert not result.ok, name
        assert requests == [] and reviewer.call_count == 0
        assert not broker.pending()


@check('access.shell_strings_are_judged_as_the_shell_runs_them')
def shell_strings_are_judged_as_the_shell_runs_them():
    from symphonai_api.permissions import DenialReason, PermissionDecision
    from symphonai_host.approvals import ApprovalBroker

    with workspace() as ws:
        policy = PermissionPolicy(repo_root=ws.root, allowed_write_scope=[ws.root],
                                  shell_enabled=True, shell_allowlist=[()],
                                  shell_sandbox=False, sandbox_network=False, fetch_enabled=True)
        denied = (
            ('plan', "echo ok # it's\nrm f.txt\n# that's it"),
            ('plan', "GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.fsmonitor GIT_CONFIG_VALUE_0='touch pwned' git status"),
            ('plan', 'env A=value git status'),
            ('plan', 'cat < .env'),
            ('allow', 'cat 0< .env'),
            ('allow', 'git -C . reset --hard'),
            ('allow', 'A=value git -C . reset --hard'),
            ('allow', 'env A=value git -c x=y commit'),
            ('allow', 'git --no-pager -c a=b commit'),
            ('allow', 'git -C . push'),
            ('allow', 'find . -delete'),
        )
        for mode, text in denied:
            assert not replace(policy, mode=mode).check_shell([shell.shell_executable(), '-c', text]).allowed, text
        for text in ('git status', 'git -C sub log -5', 'grep -n x file',
                     'cat file # note', 'ls | head', 'echo a#b', "echo '# literal'"):
            assert replace(policy, mode='plan').check_shell([shell.shell_executable(), '-c', text]).allowed, text
        broker = ApprovalBroker(lambda request: False, shell_grants=(('git', 'status'),))
        asking = replace(policy, mode='ask', approval_callback=broker.callback)
        assert asking.check_shell([shell.shell_executable(), '-c', 'git status']).allowed
        for text in ('A=value git status', 'env A=value git status'):
            assert not asking.check_shell([shell.shell_executable(), '-c', text]).allowed, text
        assert replace(policy, mode='plan').check_shell(
            [shell.shell_executable(), '-c', 'cat < .env']).denial == DenialReason.FORBIDDEN_PATTERN


@check('access.fetch_and_mcp_never_reach_another_host')
def fetch_and_mcp_never_reach_another_host():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from symphonai_api.mcp import McpClient, McpError, McpServerSpec
    from symphonai_api.permissions import DenialReason

    with workspace() as ws:
        policy = replace(ws.policy, fetch_enabled=True, mode='ask',
                         approval_callback=lambda request: True)
        for host in ('2130706433', '127.1', '0x7f000001', '0177.0.0.1',
                     '2852039166', '127.0.0.1', 'localhost', '169.254.169.254',
                     '[::ffff:127.0.0.1]'):
            assert policy.check_fetch(f'http://{host}/').denial == DenialReason.BLOCKED_HOST, host
        received = []

        class Destination(BaseHTTPRequestHandler):
            def do_GET(self):
                received.append(self.headers.get('Authorization'))
                self.send_response(200)
                self.end_headers()
            do_POST = do_GET
            def log_message(self, *args):
                pass

        destination = ThreadingHTTPServer(('127.0.0.1', 0), Destination)
        target = f'http://127.0.0.1:{destination.server_port}/'

        class Redirect(Destination):
            def do_POST(self):
                self.send_response(302)
                self.send_header('Location', target)
                self.end_headers()

        source = ThreadingHTTPServer(('127.0.0.1', 0), Redirect)
        threads = [threading.Thread(target=server.serve_forever, daemon=True)
                   for server in (source, destination)]
        for thread in threads:
            thread.start()
        client = McpClient(McpServerSpec('redirect', (), enabled=True,
                           url=f'http://127.0.0.1:{source.server_port}/',
                           bearer_token_env_var='SYMPHONAI_CHECK_MCP_TOKEN'), cwd=ws.root)
        try:
            with mock.patch.dict(os.environ, {'SYMPHONAI_CHECK_MCP_TOKEN': 'fixture-token'}):
                try:
                    client.start()
                except McpError as exc:
                    assert target.rstrip('/') in str(exc), str(exc)
                else:
                    raise AssertionError('cross-origin MCP redirect was accepted')
            assert received == [], received
        finally:
            client.close()
            for server in (source, destination):
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join(timeout=2)


@check('access.sandbox_blocks_writes_to_git_config_and_secrets')
def sandbox_blocks_writes_to_git_config_and_secrets():
    import shutil
    import subprocess

    if not shell.sandbox_available():
        skip('platform shell sandbox is unavailable')
    git = shutil.which('git')
    if git is None:
        skip('git is unavailable')
    with tempfile.TemporaryDirectory() as fixture:
        base = Path(fixture).resolve()
        roots = [base/'repo', base/'extra']
        environment = shell.shell_environment()
        environment.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
                           GIT_OPTIONAL_LOCKS='0')
        for root in roots:
            root.mkdir()
            subprocess.run([git, 'init', '-q', str(root)], env=environment,
                           check=True, capture_output=True, timeout=10)
            (root/'.git/hooks/post-commit').write_text('fixture hook\n')
            (root/'.symphonai').mkdir()
            (root/'.symphonai/config.toml').write_text('# fixture config\n')
            (root/'.env').write_text('FIXTURE_ONLY=value\n')
            (root/'private.pem').write_text('fixture private value\n')
            (root/'readable.key').write_text('fixture readable value\n')
            (root/'.ssh').mkdir()
            (root/'.ssh/marker.txt').write_text('fixture directory value\n')
            (root/'src').mkdir()
        policy = PermissionPolicy(repo_root=roots[0], allowed_write_scope=roots,
                                  sandbox_write_scope=roots, shell_sandbox=True,
                                  sandbox_readable=('*.key',))

        def run(root, command):
            argv = shell.sandbox_command(policy, [shell.shell_executable(), '-c', command], root)
            return subprocess.run(argv, cwd=root, env=environment, capture_output=True,
                                  text=True, timeout=10)

        def unchanged(root, command, path):
            before = path.stat()
            result = run(root, command)
            assert result.returncode != 0, (command, result.stdout, result.stderr)
            after = path.stat()
            assert (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (
                after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), path

        for root in roots:
            for command, path in (
                ('printf x > .git/hooks/post-commit', root/'.git/hooks/post-commit'),
                ('git config core.fsmonitor x', root/'.git/config'),
                ('echo x > .symphonai/config.toml', root/'.symphonai/config.toml'),
                ('echo x > .env', root/'.env'),
                ('echo x > private.pem', root/'private.pem'),
                ('echo x > readable.key', root/'readable.key'),
                ('echo x > .ssh/marker.txt', root/'.ssh/marker.txt'),
            ):
                unchanged(root, command, path)
            for command in ('git status --porcelain', 'git diff', 'echo x > src/a.txt',
                            'echo x > src/new.txt'):
                result = run(root, command)
                assert result.returncode == 0, (command, result.stderr)
            assert (root/'src/new.txt').read_text() == 'x\n'
            assert 'fixture readable value' in run(root, 'cat readable.key').stdout
            assert 'fixture private value' not in run(root, 'cat private.pem').stdout
            assert 'fixture directory value' not in run(root, 'cat .ssh/marker.txt').stdout
            if sys.platform == 'darwin':
                assert run(root, 'echo x > new.key').returncode != 0
                assert not (root/'new.key').exists()

        # A gitdir outside the writable root remains protected even though temp is writable.
        target = base/'gitdir'
        (roots[0]/'.git').rename(target)
        (roots[0]/'.git').write_text('gitdir: ../gitdir\n')
        unchanged(roots[0], 'git config core.fsmonitor x', target/'config')
        unchanged(roots[0], 'echo x > ../gitdir/hooks/post-commit', target/'hooks/post-commit')
        unchanged(roots[0], 'echo x > .git', roots[0]/'.git')
        assert run(roots[0], 'git status --porcelain').returncode == 0
        temporary = base/'ordinary-temp.txt'
        assert run(roots[0], f'echo x > "{temporary}"').returncode == 0
        assert temporary.read_text() == 'x\n'
