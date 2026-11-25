//! One tracing setup for every binary.
//!
//! Filter with `VICTORY_LOG` (falls back to `RUST_LOG`), e.g.
//! `VICTORY_LOG=victory_loader=trace,info`. Debug and trace output exist
//! only in builds with the `diagnostics` feature; a plain release build
//! logs info and above whatever the filter asks for.
//!
//! `VICTORY_LOG_ASYNC=1` moves the stderr writes off the calling thread: each
//! event is formatted on the caller, then a background thread writes it out.
//! Keep the guard from [`init`] alive, and call [`flush`] before any
//! `process::exit`, so queued lines reach stderr.

use std::io::{self, Write};
use std::sync::mpsc::{self, Sender};
use std::sync::{Mutex, OnceLock};
use std::thread::JoinHandle;

use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// The most verbose level this build can log.
const CAP: LevelFilter = if cfg!(feature = "diagnostics") {
    LevelFilter::TRACE
} else {
    LevelFilter::INFO
};

/// Queue to the writer thread; `None` once [`flush`] has closed it.
static ASYNC_TX: OnceLock<Mutex<Option<Sender<Vec<u8>>>>> = OnceLock::new();
static ASYNC_WORKER: Mutex<Option<JoinHandle<()>>> = Mutex::new(None);

/// Dropping this drains the async log queue. Hold it for the whole of `main`.
pub struct LogGuard;

impl Drop for LogGuard {
    fn drop(&mut self) {
        flush();
    }
}

/// Closes the async log queue and waits until every queued line is written.
/// No-op unless `VICTORY_LOG_ASYNC=1`; later events fall back to direct stderr.
pub fn flush() {
    if let Some(tx) = ASYNC_TX.get()
        && let Ok(mut tx) = tx.lock()
    {
        tx.take();
    }
    if let Ok(mut worker) = ASYNC_WORKER.lock()
        && let Some(worker) = worker.take()
    {
        let _ = worker.join();
    }
}

/// One event's formatted bytes, sent to the writer thread as one message on drop.
pub struct AsyncLine(Vec<u8>);

impl Write for AsyncLine {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Drop for AsyncLine {
    fn drop(&mut self) {
        let line = std::mem::take(&mut self.0);
        let tx = ASYNC_TX.get().and_then(|tx| tx.lock().ok()?.clone());
        let line = match tx {
            Some(tx) => match tx.send(line) {
                Ok(()) => return,
                Err(mpsc::SendError(line)) => line,
            },
            // Already flushed: write here.
            None => line,
        };
        let _ = io::stderr().write_all(&line);
    }
}

/// The sink for each event: the async queue when enabled, else stderr.
fn sink() -> Box<dyn Write> {
    if ASYNC_TX.get().is_some() {
        Box::new(AsyncLine(Vec::new()))
    } else {
        Box::new(io::stderr())
    }
}

fn start_async_writer() {
    let (tx, rx) = mpsc::channel::<Vec<u8>>();
    let worker = std::thread::Builder::new()
        .name("log-writer".into())
        .spawn(move || {
            let mut err = io::stderr().lock();
            for line in rx {
                let _ = err.write_all(&line);
            }
            let _ = err.flush();
        })
        .ok();
    let _ = ASYNC_TX.set(Mutex::new(Some(tx)));
    if let Ok(mut slot) = ASYNC_WORKER.lock() {
        *slot = worker;
    }
}

pub fn init(default_level: &str) -> LogGuard {
    let filter = std::env::var("VICTORY_LOG")
        .or_else(|_| std::env::var("RUST_LOG"))
        .ok()
        .and_then(|v| EnvFilter::try_new(v).ok())
        .unwrap_or_else(|| EnvFilter::new(default_level));
    if std::env::var("VICTORY_LOG_ASYNC").is_ok_and(|v| v == "1") {
        start_async_writer();
    }
    let _ = tracing_subscriber::registry()
        .with(CAP)
        .with(filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(true)
                .with_writer(sink),
        )
        .try_init();
    LogGuard
}
