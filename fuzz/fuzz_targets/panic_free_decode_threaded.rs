#![no_main]

//! [`panic_free_decode`](panic_free_decode.rs)'s Annex B path with the
//! decoder given four threads, so malformed input also runs through
//! slice parsing on the pool and pictures reconstructed row by row while
//! later pictures wait on their rows. The contract is the same: every
//! call returns a `Result`, with no panic, no abort and no thread left
//! waiting (run it with `-ignore_timeouts=0` so a hang counts as a
//! failure).

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, Decoder, ExecutionContext, Packet, TimeBase};
use oxideav_h264::h264_decoder::H264CodecDecoder;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    dec.set_execution_context(&ExecutionContext { threads: 4 });
    let pkt = Packet::new(0, TimeBase::new(1, 30), data.to_vec());
    let _ = dec.send_packet(&pkt);
    let _ = dec.flush();
    for _ in 0..64 {
        if dec.receive_frame().is_err() {
            break;
        }
    }
});
