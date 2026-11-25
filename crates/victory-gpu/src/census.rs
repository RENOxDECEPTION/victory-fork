//! `VICTORY_CENSUS=1`: a census of the title buffers whose host copies are in
//! the card's memory (`crate::shadow::cards`), written to the log every
//! `VICTORY_CENSUS_SECONDS` (10). Logging only: nothing here changes what
//! is copied, when, or where.
//!
//! Per copy: the title range, the `XMemVirtualAlloc` reservation it lies in
//! with the flags the title asked for (the runtime drops write-combining,
//! uncached and the GPU access bits; they are kept here), how much of the
//! range is physical-page views and how much of that shows a page at more
//! than one address, and per frame the bytes sent up to the card and the
//! bytes fetched back from it. A copy the title only writes (sent up,
//! nothing fetched), in memory it asked write-combining, at one address, is
//! the kind that could live in the card's memory with the CPU writing it
//! through the BAR instead of a copy.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Whether the census is on (`VICTORY_CENSUS=1`).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CENSUS").is_some_and(|v| v == "1"))
}

fn period() -> Duration {
    static S: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    Duration::from_secs(*S.get_or_init(|| {
        std::env::var("VICTORY_CENSUS_SECONDS").ok().and_then(|v| v.trim().parse().ok()).filter(|&s| s > 0).unwrap_or(10)
    }))
}

/// A reservation of `XMemVirtualAlloc`, as the title asked for it.
struct Reservation {
    len: usize,
    kind: u32,
    flags: u64,
    protect: u32,
}

static RESERVATIONS: Mutex<BTreeMap<usize, Reservation>> = Mutex::new(BTreeMap::new());

/// The runtime reserved `[at, at + len)` for the title (`XMemVirtualAlloc`
/// at an address of its choice), with the title's own `kind`, XMem `flags`
/// and console `protect` (before the runtime cleans them).
pub fn title_memory_reserved(at: usize, len: usize, kind: u32, flags: u64, protect: u32) {
    if !(on() || skip_wc_back() || own_wc()) || at == 0 || len == 0 {
        return;
    }
    RESERVATIONS.lock().unwrap_or_else(|e| e.into_inner()).insert(at, Reservation { len, kind, flags, protect });
}

/// `VICTORY_SKIP_WC_BACK=1`, an EXPERIMENT, not a fix: what shaders wrote
/// to a card copy of title memory the title reserved write-combining is not
/// fetched back into title memory. It tests whether the title's CPU reads
/// that memory at all (the census found 97 of 100 MB a frame fetched back
/// from it). Expect artifacts where the title's CPU writes part of a
/// 64 KiB piece the card wrote: the next upload sends the stale bytes there.
pub(crate) fn skip_wc_back() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = skip_wc_compare() || std::env::var_os("VICTORY_SKIP_WC_BACK").is_some_and(|v| v == "1");
        if on {
            tracing::warn!("VICTORY_SKIP_WC_BACK=1: shader writes to write-combining title memory are NOT copied back (experiment)");
        }
        on
    })
}

/// `VICTORY_OWN=wc` (off by default): what shaders wrote to a card copy of
/// title memory the title reserved write-combining is left on the card, as
/// with [`skip_wc_back`], but the copy then owns those 64 KiB pieces
/// (`crate::shadow`'s `Owned`): the CPU's writes to such a piece go up as
/// the runs of bytes they changed, never as the whole piece, so the stale
/// title bytes beside them do not land over the card's; a reader of ours
/// that takes title memory there (the title's buffer copies, fills, texture
/// refreshes, another copy's refresh) has the piece fetched and merged
/// first, and the piece is not left on the card again; a piece whose page
/// shows at more than one address, or that lies in a shader-visible
/// descriptor heap, always comes back as before. `VICTORY_SKIP_WC_BACK=1`
/// is left as it was, to compare the two; set one of them, not both (with
/// both, `VICTORY_SKIP_WC_BACK` decides and nothing is owned).
///
/// What it still assumes: that the title's CPU does not read what shaders
/// wrote to write-combining memory (no page is guarded for it), and that
/// the kernel (file writes, DirectStorage) does not read it either.
pub(crate) fn own_wc() -> bool {
    #[cfg(test)]
    if let Some(forced) = OWN_FORCED.with(|f| f.get()) {
        return forced;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("VICTORY_OWN").is_ok_and(|v| v.trim().eq_ignore_ascii_case("wc"));
        if on {
            tracing::warn!("VICTORY_OWN=wc: shader writes to write-combining title memory stay on the card; the CPU's writes there go up by changed runs (experiment)");
        }
        on
    })
}

#[cfg(test)]
thread_local! {
    /// Tests: [`own_wc`]'s answer on this thread, whatever the environment.
    static OWN_FORCED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Tests: makes [`own_wc`] answer `on` on this thread (`None`: from the
/// environment again).
#[cfg(test)]
pub(crate) fn force_own(on: Option<bool>) {
    OWN_FORCED.with(|f| f.set(on));
}

/// With [`own_wc`]: how many changed runs one refresh of an owned piece
/// may send up (`VICTORY_OWN_RUNS`, 8 by default, at least 1). A piece
/// that needs more is written by the CPU and the card both, finely (each
/// run is a card copy of its own, and gap bytes between runs may hold the
/// card's bytes, so runs are never joined): its runs go up this once, and
/// then it is fetched at the next submission and comes back as before
/// from then on.
pub(crate) fn own_runs_cap() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_OWN_RUNS").ok().and_then(|v| v.trim().parse::<usize>().ok()).map_or(8, |n| n.max(1)))
}

/// With [`own_wc`], `VICTORY_OWN_ALIASED=1` (off by default; to measure
/// what leaving aliased pieces out costs): a piece whose page is shown at
/// more than one address is owned too, if every address it is shown at
/// lies in a write-combining reservation.
///
/// What can go wrong with it: the copy-back was what made the other
/// addresses see what shaders wrote. Title memory at every address of the
/// page stays stale; another copy of ours over another address (a refresh,
/// a texture, a buffer copy there) takes the stale bytes and the fetch for
/// readers does not see it (pages are matched by the owning address
/// only); and if shaders write the page through copies at two addresses,
/// the two card copies drift apart with nothing bringing them together.
pub(crate) fn own_aliased() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_OWN_ALIASED").is_some_and(|v| v == "1"))
}

/// The counters of [`own_wc`], since the last line ([`own_say`]).
#[derive(Clone, Copy)]
pub(crate) enum OwnCount {
    /// Pieces left on the card, and their bytes.
    Left,
    LeftBytes,
    /// CPU writes to owned pieces sent up by changed runs: runs and bytes.
    Runs,
    RunBytes,
    /// Owned pieces refreshed by changed runs, not uploaded whole, and
    /// the bytes of them that did not go up.
    Pieces,
    Avoided,
    /// Owned pieces that needed more than [`own_runs_cap`] runs: no longer
    /// owned.
    Mixed,
    /// Aliased pieces owned all the same ([`own_aliased`]).
    AliasedOwned,
    /// Pieces fetched for a reader of ours, and their bytes; the settles
    /// of flights that took.
    Fetched,
    FetchedBytes,
    Settles,
    /// Pieces that came back as before: aliased, read by a reader of ours,
    /// in a shader-visible descriptor heap.
    Aliased,
    Read,
    Heap,
}

#[cfg(feature = "diagnostics")]
static OWN_COUNTS: [std::sync::atomic::AtomicU64; 14] = [const { std::sync::atomic::AtomicU64::new(0) }; 14];

/// With [`own_wc`] (diagnostics): per copy, by the title address it starts
/// at, since the last line: its bytes, whether it is a window, the owned
/// pieces refreshed by runs, the runs, their bytes and the longest.
#[cfg(feature = "diagnostics")]
static OWN_RUNS_BY: Mutex<BTreeMap<u64, (u64, bool, u64, u64, u64, u64)>> = Mutex::new(BTreeMap::new());

/// With [`own_wc`] (diagnostics): an owned piece of the copy at `start`
/// (`len` bytes, a window or not) was refreshed with `runs` changed runs of
/// `bytes` in all, the longest `longest`. Said in [`own_say`], the copies
/// with the most runs first: which memory the CPU writes finely beside
/// what shaders wrote.
pub(crate) fn own_runs_in(start: u64, len: u64, window: bool, runs: u64, bytes: u64, longest: u64) {
    #[cfg(feature = "diagnostics")]
    {
        let mut by = OWN_RUNS_BY.lock().unwrap_or_else(|e| e.into_inner());
        let e = by.entry(start).or_insert((len, window, 0, 0, 0, 0));
        e.2 += 1;
        e.3 += runs;
        e.4 += bytes;
        e.5 = e.5.max(longest);
    }
    #[cfg(not(feature = "diagnostics"))]
    let _ = (start, len, window, runs, bytes, longest);
}

/// Adds `n` to an [`OwnCount`] (diagnostics).
pub(crate) fn own_count(c: OwnCount, n: u64) {
    #[cfg(feature = "diagnostics")]
    OWN_COUNTS[c as usize].fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    #[cfg(not(feature = "diagnostics"))]
    let _ = (c, n);
}

/// The [`OwnCount`]s of the last five seconds, said once in five seconds
/// at most (diagnostics; called at every submission with [`own_wc`]).
pub(crate) fn own_say() {
    #[cfg(feature = "diagnostics")]
    {
        use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
        static EPOCH: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        static SAID: AtomicU64 = AtomicU64::new(0);
        let now = EPOCH.get_or_init(Instant::now).elapsed().as_secs() / 5;
        if SAID.swap(now, Relaxed) == now {
            return;
        }
        let c: Vec<u64> = OWN_COUNTS.iter().map(|c| c.swap(0, Relaxed)).collect();
        let by = std::mem::take(&mut *OWN_RUNS_BY.lock().unwrap_or_else(|e| e.into_inner()));
        let mut by: Vec<_> = by.into_iter().collect();
        by.sort_unstable_by_key(|(_, v)| std::cmp::Reverse(v.3));
        let most: Vec<String> = by
            .iter()
            .take(6)
            .map(|(at, v)| {
                format!(
                    "{at:#x}+{:#x}{}: {} pieces, {} runs, {} bytes, longest {}",
                    v.0,
                    if v.1 { " (window)" } else { "" },
                    v.2,
                    v.3,
                    v.4,
                    v.5
                )
            })
            .collect();
        tracing::info!(
            "VICTORY_OWN=wc in 5 s: {} pieces ({} MB) left on the card; {} changed runs ({} KB) sent up to owned pieces, {} whole-piece uploads avoided ({} MB not sent); {} pieces no longer owned for more than {} runs; {} pieces ({} KB) fetched for readers of ours, {} settles for them; came back as before: {} aliased ({} aliased owned with VICTORY_OWN_ALIASED=1), {} read by us, {} in descriptor heaps; the copies with the most runs: {}",
            c[0], c[1] >> 20, c[2], c[3] >> 10, c[4], c[5] >> 20, c[6], own_runs_cap(), c[8], c[9] >> 10, c[10], c[11], c[7], c[12], c[13],
            if most.is_empty() { "-".to_owned() } else { most.join("; ") }
        );
    }
}

/// `VICTORY_SKIP_WC_COMPARE=1`, the second half of the same EXPERIMENT
/// (it implies [`skip_wc_back`]): the card does not hash or compare the
/// declared writes of copies of write-combining title memory at all, since
/// nothing of them is fetched back. The CPU's changes are still sent up.
///
/// UNSAFE, do not use for play: one tester's RTX 5080 lost its device about
/// 75 s into the world with it (on `f85101b` with `VICTORY_ASYNC_SUBMIT=1
/// VICTORY_QUEUE2=1`); other runs, on a 5080 and a 3080 Ti, had none, and
/// the cause is not known. `VICTORY_SKIP_WC_BACK=1` alone has had no device
/// loss in any reported run.
pub(crate) fn skip_wc_compare() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_SKIP_WC_COMPARE").is_some_and(|v| v == "1");
        if on {
            tracing::warn!(
                "VICTORY_SKIP_WC_COMPARE=1: copies of write-combining title memory are not hashed or compared on the card \
                 (UNSAFE experiment: a device removal was reported with it; use VICTORY_SKIP_WC_BACK=1 alone)"
            );
        }
        on
    })
}

/// Every declared title range must lie wholly inside a single WC reservation.
/// A card copy (especially an aligned loose window) can cover several.
pub(crate) fn all_wc(ranges: &[(usize, usize)]) -> bool {
    let reservations = RESERVATIONS.lock().unwrap_or_else(|e| e.into_inner());
    all_wc_in(&reservations, ranges)
}

fn all_wc_in(reservations: &BTreeMap<usize, Reservation>, ranges: &[(usize, usize)]) -> bool {
    !ranges.is_empty() && ranges.iter().all(|&(from, to)| {
        // Seeking from the exclusive end also detects a new reservation
        // beginning inside the range. An allocation ending at `to` fits.
        from < to && reservations.range(..to).next_back().is_some_and(|(&base, r)| {
            base <= from && r.protect & 0x400 != 0 && base.checked_add(r.len).is_some_and(|end| to <= end)
        })
    })
}

/// Declared ranges left out of the compare by [`skip_wc_compare`].
pub(crate) static UNCOMPARED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Pieces left out by [`skip_wc_back`]: count and bytes.
pub(crate) static SKIPPED: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// A flagged piece is clipped to its declaration. Keep every piece of a
/// mixed declaration so the normal read-back covers it in full.
pub(crate) fn skip_back(declared: (usize, usize), bytes: u64) -> bool {
    if !skip_wc_back() {
        return false;
    }
    let wc = all_wc(&[declared]);
    if wc {
        use std::sync::atomic::Ordering;
        let n = SKIPPED[0].fetch_add(1, Ordering::Relaxed) + 1;
        let b = SKIPPED[1].fetch_add(bytes, Ordering::Relaxed) + bytes;
        if n.is_power_of_two() {
            tracing::info!("VICTORY_SKIP_WC_BACK: {n} pieces ({} MB) of write-combining memory left on the card so far", b >> 20);
        }
    }
    wc
}

#[cfg(test)]
mod wc_tests {
    use super::{Reservation, all_wc_in};
    use std::collections::BTreeMap;

    fn reservation(len: usize, protect: u32) -> Reservation {
        Reservation { len, kind: 0, flags: 0, protect }
    }

    #[test]
    fn a_write_in_the_ordinary_half_of_a_card_window_still_comes_back() {
        const MIB: usize = 1 << 20;
        let window = 8 * MIB;
        let middle = window + 2 * MIB;
        let map = BTreeMap::from([(window, reservation(2 * MIB, 0x404)), (middle, reservation(2 * MIB, 0x04))]);
        let wc = (middle - 65536, middle);
        let ordinary = (middle, middle + 65536);
        assert!(all_wc_in(&map, &[wc]));
        assert!(!all_wc_in(&map, &[ordinary]));
        assert!(!all_wc_in(&map, &[wc, ordinary]));
        assert!(!all_wc_in(&map, &[(wc.0, ordinary.1)]));
    }

    #[test]
    fn a_declaration_must_fit_one_known_reservation() {
        let map = BTreeMap::from([
            (0x10000, reservation(0x10000, 0x404)),
            (0x20000, reservation(0x10000, 0x404)),
            (usize::MAX - 8, reservation(16, 0x404)),
        ]);
        assert!(all_wc_in(&map, &[(0x10000, 0x20000), (0x20000, 0x30000)]));
        assert!(!all_wc_in(&map, &[(0x1ffff, 0x20001)]));
        assert!(!all_wc_in(&map, &[(0x30000, 0x30001)]));
        assert!(!all_wc_in(&map, &[(0x10000, 0x10000)]));
        assert!(!all_wc_in(&map, &[(usize::MAX - 8, 1)]));
        assert!(!all_wc_in(&map, &[(usize::MAX - 8, usize::MAX)]));
    }
}

/// Host resources made, by where: count and bytes ([`made`]).
const KINDS: [&str; 9] = ["read-back", "upload", "shared", "uav", "texture", "reserved texture", "depth twin", "placed in title memory", "host list"];
static MADE: [std::sync::atomic::AtomicU64; 18] = [const { std::sync::atomic::AtomicU64::new(0) }; 18];
static MADE_LAST: Mutex<[u64; 18]> = Mutex::new([0; 18]);

/// With the census on: a host resource of kind `kind` ([`KINDS`]) and
/// `bytes` (0 when not known) was asked of the device.
pub(crate) fn made(kind: usize, bytes: u64) {
    if on() && kind < KINDS.len() {
        use std::sync::atomic::Ordering;
        MADE[kind * 2].fetch_add(1, Ordering::Relaxed);
        MADE[kind * 2 + 1].fetch_add(bytes, Ordering::Relaxed);
    }
}

/// Counters of one copy, by the title address it starts at.
#[derive(Default, Clone, Copy)]
struct Copy {
    len: u64,
    made_frame: u64,
    sent: u64,
    sent_ranges: u64,
    compared: u64,
    fetched: u64,
    fetched_pieces: u64,
}

struct Census {
    copies: BTreeMap<u64, Copy>,
    /// The counters at the last report, to print per-frame rates.
    last: BTreeMap<u64, Copy>,
    last_at: Option<(Instant, u64)>,
}

static CENSUS: Mutex<Census> = Mutex::new(Census { copies: BTreeMap::new(), last: BTreeMap::new(), last_at: None });

fn census() -> std::sync::MutexGuard<'static, Census> {
    CENSUS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A copy of the title buffer at `start` of `len` bytes was made in the
/// card's memory.
pub(crate) fn copy_made(start: u64, len: u64) {
    if !on() {
        return;
    }
    let frame = crate::frame::current();
    let mut c = census();
    let e = c.copies.entry(start).or_default();
    e.len = len;
    e.made_frame = frame;
}

/// `bytes` of the copy at `start` were sent up to the card in `ranges` ranges.
pub(crate) fn sent(start: u64, bytes: u64, ranges: u64) {
    if !on() {
        return;
    }
    {
        let mut c = census();
        let e = c.copies.entry(start).or_default();
        e.sent += bytes;
        e.sent_ranges += ranges;
    }
    report_if_due();
}

/// `pieces` 64 KiB pieces of the copy at `start` were compared on the card.
pub(crate) fn compared(start: u64, pieces: u64) {
    if on() {
        census().copies.entry(start).or_default().compared += pieces;
    }
}

/// One span of `bytes` from the card copy at `start` was received for merging.
/// For an in-flight slot this counts its usable span, after excluding any bytes
/// already received from a newer fetch. Slot padding and bytes written back to
/// title memory are separate from this count.
pub(crate) fn fetched(start: u64, bytes: u64) {
    if !on() {
        return;
    }
    {
        let mut c = census();
        let e = c.copies.entry(start).or_default();
        e.fetched += bytes;
        e.fetched_pieces += 1;
    }
    report_if_due();
}

/// The console protection bits the runtime drops, spelled out.
fn protect_words(p: u32) -> String {
    let mut w = Vec::new();
    if p & 0x400 != 0 {
        w.push("WC");
    }
    if p & 0x200 != 0 {
        w.push("uncached");
    }
    if p & 0x100 != 0 {
        w.push("guard");
    }
    let gpu = p & 0x7_f800;
    let mut s = w.join("+");
    if gpu != 0 {
        if !s.is_empty() {
            s.push('+');
        }
        s.push_str(&format!("gpu{gpu:#x}"));
    }
    if s.is_empty() { "-".to_owned() } else { s }
}

fn report_if_due() {
    let frame = crate::frame::current();
    let now = Instant::now();
    let mut c = match CENSUS.try_lock() {
        Ok(c) => c,
        Err(_) => return,
    };
    let Some((then, then_frame)) = c.last_at else {
        c.last_at = Some((now, frame));
        return;
    };
    if now.duration_since(then) < period() {
        return;
    }
    let frames = frame.saturating_sub(then_frame).max(1) as f64;
    let secs = now.duration_since(then).as_secs_f64();
    c.last_at = Some((now, frame));
    let copies = c.copies.clone();
    let last = std::mem::replace(&mut c.last, copies.clone());
    drop(c);

    let reservations = RESERVATIONS.lock().unwrap_or_else(|e| e.into_inner());
    let mb = |b: u64| b as f64 / 1e6;
    struct Row {
        start: u64,
        len: u64,
        sent: f64,
        ranges: f64,
        compared: f64,
        fetched: f64,
        pieces: f64,
        res: String,
        words: String,
        wc: bool,
        in_views: usize,
        multi: usize,
    }
    let mut rows: Vec<Row> = copies
        .iter()
        .map(|(&start, now_)| {
            let was = last.get(&start).copied().unwrap_or_default();
            let res = reservations
                .range(..=start as usize)
                .next_back()
                .filter(|(base, r)| *base + r.len > start as usize);
            let (res_s, words, wc) = match res {
                Some((base, r)) => (
                    format!("{base:#x}+{}MB kind {:#x} flags {:#x} protect {:#x}", r.len >> 20, r.kind, r.flags, r.protect),
                    protect_words(r.protect),
                    r.protect & 0x400 != 0,
                ),
                None => ("not in a recorded reservation".to_owned(), "?".to_owned(), false),
            };
            let (in_views, multi) = crate::inplace::view_census(start as usize, now_.len as usize);
            Row {
                start,
                len: now_.len,
                sent: mb(now_.sent - was.sent) / frames,
                ranges: (now_.sent_ranges - was.sent_ranges) as f64 / frames,
                compared: (now_.compared - was.compared) as f64 / frames,
                fetched: mb(now_.fetched - was.fetched) / frames,
                pieces: (now_.fetched_pieces - was.fetched_pieces) as f64 / frames,
                res: res_s,
                words,
                wc,
                in_views,
                multi,
            }
        })
        .collect();
    drop(reservations);
    rows.sort_by(|a, b| (b.fetched + b.sent).total_cmp(&(a.fetched + a.sent)));

    let total = |f: &dyn Fn(&Row) -> f64| rows.iter().map(f).sum::<f64>();
    let only_written = rows.iter().filter(|r| r.sent > 0.0 && r.fetched == 0.0).count();
    let fetched_wc = total(&|r| if r.wc { r.fetched } else { 0.0 });
    let fetched_multi = total(&|r| if r.multi > 0 { r.fetched } else { 0.0 });
    tracing::info!(
        target: "census",
        "census over {secs:.1} s, {frames:.0} frames: {} copies on the card, {:.0} MB; per frame {:.1} MB sent up, {:.1} MB fetched back \
         ({:.1} MB of it from write-combining memory, {:.1} MB from copies with multi-address pages), {:.0} pieces compared; \
         {only_written} copies only sent up (CPU writes, nothing fetched); {} copies with multi-address pages",
        rows.len(),
        mb(rows.iter().map(|r| r.len).sum()),
        total(&|r| r.sent),
        total(&|r| r.fetched),
        fetched_wc,
        fetched_multi,
        total(&|r| r.compared),
        rows.iter().filter(|r| r.multi > 0).count(),
    );
    {
        use std::sync::atomic::Ordering;
        let mut last = MADE_LAST.lock().unwrap_or_else(|e| e.into_inner());
        let now_made: [u64; 18] = std::array::from_fn(|i| MADE[i].load(Ordering::Relaxed));
        let parts: Vec<String> = KINDS
            .iter()
            .enumerate()
            .map(|(k, name)| {
                let n = (now_made[k * 2] - last[k * 2]) as f64 / frames;
                let b = (now_made[k * 2 + 1] - last[k * 2 + 1]) as f64 / frames / 1e6;
                format!("{name} {n:.1} ({b:.1} MB)")
            })
            .collect();
        let total: u64 = (0..KINDS.len()).map(|k| now_made[k * 2] - last[k * 2]).sum();
        *last = now_made;
        tracing::info!(
            target: "census",
            "census resources made per frame: {:.1} in all; {}",
            total as f64 / frames,
            parts.join(", ")
        );
    }
    for r in rows.iter().take(48) {
        tracing::info!(
            target: "census",
            "census copy {:#x}+{:.1}MB: per frame sent {:.2} MB in {:.1} ranges, compared {:.0} pieces, fetched {:.2} MB in {:.1} pieces; \
             views {:.1} MB, multi-address {:.1} MB; {} [{}]",
            r.start,
            mb(r.len),
            r.sent,
            r.ranges,
            r.compared,
            r.fetched,
            r.pieces,
            mb(r.in_views as u64),
            mb(r.multi as u64),
            r.words,
            r.res
        );
    }
}
