use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::process::{Child, ChildStderr, ChildStdout, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};
use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindowBuilder, WindowEvent};

const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);
const TARGET_TRIPLE: &str = env!("TARGET_TRIPLE");

#[derive(Deserialize, Serialize)]
struct Handshake {
    port: u16,
    token: String,
}

struct Sidecar {
    child: Child,
    _stdout: ChildStdout,
    _stderr: ChildStderr,
}

impl Sidecar {
    fn shutdown(&mut self) {
        terminate_child(&mut self.child);
    }
}

impl Drop for Sidecar {
    fn drop(&mut self) {
        self.shutdown();
    }
}

struct SidecarState(Mutex<Option<Sidecar>>);

impl SidecarState {
    fn shutdown(&self) {
        if let Some(mut sidecar) = self.0.lock().expect("sidecar lock poisoned").take() {
            sidecar.shutdown();
        }
    }
}

fn terminate_child(child: &mut Child) {
    if matches!(child.try_wait(), Ok(Some(_))) {
        return;
    }
    unsafe {
        libc::kill(child.id() as i32, libc::SIGTERM);
    }
    let deadline = Instant::now() + SHUTDOWN_GRACE;
    while Instant::now() < deadline {
        if matches!(child.try_wait(), Ok(Some(_))) {
            return;
        }
        thread::sleep(Duration::from_millis(50));
    }
    let _ = child.kill();
    let _ = child.wait();
}

fn sidecar_path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let directory_name = format!("SymphonAI-host-{TARGET_TRIPLE}");
    let directory = app
        .path()
        .resource_dir()
        .map_err(|_| "could not locate packaged resources".to_string())?
        .join(&directory_name);
    let executable = directory.join(&directory_name);
    if !directory.join("_internal").is_dir() || !executable.is_file() {
        return Err("packaged host bundle is incomplete".to_string());
    }
    Ok(executable)
}

fn launch_sidecar(path: PathBuf) -> Result<(Handshake, Sidecar), String> {
    let mut child = Command::new(path)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|_| "could not launch packaged host".to_string())?;
    let mut stdout = match child.stdout.take() {
        Some(stdout) => stdout,
        None => {
            terminate_child(&mut child);
            return Err("packaged host stdout was not piped".to_string());
        }
    };
    let mut stderr = match child.stderr.take() {
        Some(stderr) => stderr,
        None => {
            terminate_child(&mut child);
            return Err("packaged host stderr was not piped".to_string());
        }
    };
    let mut line = String::new();
    let read = match BufReader::new(&mut stdout).read_line(&mut line) {
        Ok(read) => read,
        Err(_) => {
            terminate_child(&mut child);
            return Err("could not read the host handshake".to_string());
        }
    };
    if read == 0 {
        let _ = child.wait();
        let mut stderr_text = String::new();
        let _ = stderr.read_to_string(&mut stderr_text);
        let stderr_text = stderr_text.trim();
        return Err(if stderr_text.is_empty() {
            "host exited before printing a handshake".to_string()
        } else {
            format!("host exited before handshake: {stderr_text}")
        });
    }

    let value: serde_json::Value = match serde_json::from_str(&line) {
        Ok(value) => value,
        Err(_) => {
            terminate_child(&mut child);
            return Err("host handshake was not valid JSON".to_string());
        }
    };
    let port = match value.get("port").and_then(serde_json::Value::as_u64) {
        Some(port) if (1..=u16::MAX as u64).contains(&port) => port as u16,
        _ => {
            terminate_child(&mut child);
            return Err("host handshake had an invalid port".to_string());
        }
    };
    let token = match value.get("token").and_then(serde_json::Value::as_str) {
        Some(token) if !token.is_empty() => token.to_string(),
        _ => {
            terminate_child(&mut child);
            return Err("host handshake had an invalid token".to_string());
        }
    };
    Ok((
        Handshake { port, token },
        Sidecar {
            child,
            _stdout: stdout,
            _stderr: stderr,
        },
    ))
}

fn shutdown(app: &tauri::AppHandle) {
    app.state::<SidecarState>().shutdown();
}

fn main() {
    let app = tauri::Builder::default()
        .setup(|app| {
            let path = sidecar_path(app.handle()).map_err(std::io::Error::other)?;
            let (handshake, sidecar) =
                launch_sidecar(path).map_err(std::io::Error::other)?;
            let serialized = serde_json::to_string(&handshake)
                .map_err(|_| std::io::Error::other("could not prepare host bridge"))?;
            app.manage(SidecarState(Mutex::new(Some(sidecar))));
            let initialization = format!(
                "Object.defineProperty(window, '__symphonaiShell', {{value: Object.freeze({serialized})}}); Object.defineProperty(window, '__symphonai', {{value: window.__symphonaiShell}});"
            );
            WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
                .title("SymphonAI")
                .initialization_script(&initialization)
                .build()
                .map_err(|_| std::io::Error::other("could not create application window"))?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("could not build SymphonAI shell");

    let exit_handle = app.handle().clone();
    ctrlc::set_handler(move || exit_handle.exit(0)).expect("could not install exit handler");
    app.run(|handle, event| match event {
        RunEvent::WindowEvent {
            event: WindowEvent::CloseRequested { .. },
            ..
        }
        | RunEvent::ExitRequested { .. }
        | RunEvent::Exit => shutdown(handle),
        _ => {}
    });
}
