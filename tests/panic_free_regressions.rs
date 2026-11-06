//! Inputs the `panic_free_decode` fuzz targets found, decoded as the
//! targets decode them (one Annex B packet, flush, drain), with one and
//! with four threads. The contract is no panic at all, including one a
//! pool worker contains: libFuzzer counts every panic, and a contained
//! one still costs the picture.
//!
//! * `crash-94b9739e`: a pred_weight_table weight far outside -128..=127
//!   overflowed weighted prediction's 32-bit arithmetic (§7.4.3.2 bounds
//!   it; the parser now refuses the slice).
//! * `crash-a50aa2fc`: an IDR picture, then an SPS declaring another
//!   picture size and P slices without an IDR: the frame-threaded
//!   sample stage waited for the reference rows by the new picture's
//!   height while motion compensation read the reference clamped to its
//!   own, taller planes, past the rows it had waited for.

use std::sync::atomic::{AtomicUsize, Ordering};

use oxideav_core::{CodecId, Decoder, ExecutionContext, Packet, TimeBase};
use oxideav_h264::h264_decoder::H264CodecDecoder;

static PANICS: AtomicUsize = AtomicUsize::new(0);

const FIXTURES: &[(&str, &[u8])] = &[
    (
        "94b9739e (weight out of range)",
        include_bytes!("fixtures/fuzz_regressions/crash-94b9739e101ac81ebe9c26e4b9f8f2c23bc93103"),
    ),
    (
        "a50aa2fc (reference of another size)",
        include_bytes!("fixtures/fuzz_regressions/crash-a50aa2fc3bfbe44ec328b2db3773f198b6ec9545"),
    ),
];

fn decode(data: &[u8], threads: usize) {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    dec.set_execution_context(&ExecutionContext { threads });
    let pkt = Packet::new(0, TimeBase::new(1, 30), data.to_vec());
    let _ = dec.send_packet(&pkt);
    let _ = dec.flush();
    for _ in 0..64 {
        if dec.receive_frame().is_err() {
            break;
        }
    }
}

#[test]
fn fuzz_found_inputs_decode_without_any_panic() {
    // Counts every panic on any thread, contained or not; this file
    // holds this one test so no other test's panics are counted.
    std::panic::set_hook(Box::new(|_| {
        PANICS.fetch_add(1, Ordering::SeqCst);
    }));
    let mut failures = Vec::new();
    for (name, data) in FIXTURES {
        for threads in [1, 4] {
            let before = PANICS.load(Ordering::SeqCst);
            // A panic on this thread is counted by the hook as well.
            let _ = std::panic::catch_unwind(|| decode(data, threads));
            let panics = PANICS.load(Ordering::SeqCst) - before;
            if panics > 0 {
                failures.push(format!(
                    "{name} with {threads} thread(s): {panics} panic(s)"
                ));
            }
        }
    }
    let _ = std::panic::take_hook();
    assert!(failures.is_empty(), "{failures:#?}");
}
