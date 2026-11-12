//! §8.4.2.2 — fractional-sample interpolation on a clipped sample
//! window.
//!
//! The motion-compensation kernels used by the reconstruction path.
//! Each kernel first gathers the integer samples a block can touch —
//! for a `w`x`h` luma block the `(w + 5)`x`(h + 5)` neighbourhood of
//! eqs. 8-239 / 8-240 (Table 8-11 reaches 2 samples left / above and 3
//! right / below), for chroma the `(w + 1)`x`(h + 1)` neighbourhood of
//! eqs. 8-262..8-269 — with the spec's `Clip3` coordinate clamping
//! applied while gathering. The filters (eqs. 8-241..8-261 and 8-270)
//! then run on the window without any further bounds handling, so the
//! result equals the per-sample clamped derivation for every block
//! position, including blocks reaching outside the picture.
//!
//! The source plane is read through [`PlaneSrc`], either one flat
//! row-major plane or a plane published as per-macroblock-row bands by
//! a picture still being decoded on another thread (frame threading);
//! both give the same samples.
//!
//! Clean-room: derived only from ITU-T Rec. H.264 (08/2024) §8.4.2.2.

use crate::picture::{PicSample, Sample};
use std::sync::OnceLock;

/// Largest window a kernel gathers: a 16x16 luma block plus the
/// 6-tap filter margins (2 before, 3 after) in each direction.
const WIN: usize = 21;

/// A reference sample plane, addressed by row.
#[derive(Clone, Copy)]
pub(crate) enum PlaneSrc<'a, S = Sample> {
    /// A flat row-major plane: row `y` starts at `data[y * stride]`.
    /// A field of a frame is a flat plane with a doubled stride.
    Flat {
        data: &'a [S],
        stride: usize,
        width: usize,
        height: usize,
    },
    /// A plane published in bands of `band_rows` rows (each band a
    /// tightly packed `band_rows * width` buffer). Every band a block
    /// reads must already be published.
    #[cfg_attr(not(test), allow(dead_code))]
    Banded {
        bands: &'a [OnceLock<Box<[S]>>],
        band_rows: usize,
        width: usize,
        height: usize,
    },
}

impl<'a, S: PicSample> PlaneSrc<'a, S> {
    #[inline]
    pub(crate) fn width(&self) -> usize {
        match *self {
            PlaneSrc::Flat { width, .. } | PlaneSrc::Banded { width, .. } => width,
        }
    }

    #[inline]
    pub(crate) fn height(&self) -> usize {
        match *self {
            PlaneSrc::Flat { height, .. } | PlaneSrc::Banded { height, .. } => height,
        }
    }

    /// Row `y` (`y < height`), `width` samples long.
    #[inline]
    fn row(&self, y: usize) -> &'a [S] {
        match *self {
            PlaneSrc::Flat {
                data,
                stride,
                width,
                ..
            } => &data[y * stride..y * stride + width],
            PlaneSrc::Banded {
                bands,
                band_rows,
                width,
                ..
            } => {
                let band = bands[y / band_rows]
                    .get()
                    .expect("reference band read before it was published");
                let o = (y % band_rows) * width;
                &band[o..o + width]
            }
        }
    }
}

/// A working sample of a gathered window: `i32`, or `i16` for 8-bit
/// samples, whose filters run twice as many samples per vector.
trait Lane: Copy + Default {
    fn of<S: PicSample>(s: S) -> Self;
}

impl Lane for i32 {
    #[inline(always)]
    fn of<S: PicSample>(s: S) -> Self {
        s.into()
    }
}

impl Lane for i16 {
    /// Stored samples are at most 14 bits, so they fit.
    #[inline(always)]
    fn of<S: PicSample>(s: S) -> Self {
        Into::<i32>::into(s) as i16
    }
}

/// Whether `S` is the 8-bit stored sample, for which the luma 6-tap
/// stage and the chroma filter fit 16-bit lanes.
#[inline(always)]
fn is_byte<S: PicSample>() -> bool {
    core::mem::size_of::<S>() == 1
}

/// Gather the `ww`x`wh` window whose top-left integer sample is
/// `(x0, y0)` into `out` (row stride `ostride`), clamping every
/// coordinate into the plane (eqs. 8-239 / 8-240, 8-262..8-269).
#[inline]
#[allow(clippy::too_many_arguments)]
fn gather<S: PicSample, L: Lane>(
    src: &PlaneSrc<'_, S>,
    x0: i32,
    y0: i32,
    ww: usize,
    wh: usize,
    out: &mut [L],
    ostride: usize,
) {
    let width = src.width() as i32;
    let ymax = src.height() as i32 - 1;
    // Interior window of a flat plane (nearly every block): no
    // coordinate clamping, and the row length is dispatched once for the
    // whole window instead of once per row.
    if let PlaneSrc::Flat { data, stride, .. } = *src
        && x0 >= 0
        && y0 >= 0
        && x0 + ww as i32 <= width
        && y0 + wh as i32 <= ymax + 1
    {
        let base = y0 as usize * stride + x0 as usize;
        match ww {
            2 => rows_fixed::<S, L, 2>(data, base, stride, wh, out, ostride),
            3 => rows_fixed::<S, L, 3>(data, base, stride, wh, out, ostride),
            4 => rows_fixed::<S, L, 4>(data, base, stride, wh, out, ostride),
            5 => rows_fixed::<S, L, 5>(data, base, stride, wh, out, ostride),
            8 => rows_fixed::<S, L, 8>(data, base, stride, wh, out, ostride),
            9 => rows_fixed::<S, L, 9>(data, base, stride, wh, out, ostride),
            13 => rows_fixed::<S, L, 13>(data, base, stride, wh, out, ostride),
            16 => rows_fixed::<S, L, 16>(data, base, stride, wh, out, ostride),
            17 => rows_fixed::<S, L, 17>(data, base, stride, wh, out, ostride),
            21 => rows_fixed::<S, L, 21>(data, base, stride, wh, out, ostride),
            _ => {
                for r in 0..wh {
                    let s = &data[base + r * stride..base + r * stride + ww];
                    widen(&mut out[r * ostride..r * ostride + ww], s);
                }
            }
        }
        return;
    }
    let inside_x = x0 >= 0 && x0 + ww as i32 <= width;
    for r in 0..wh {
        let y = (y0 + r as i32).clamp(0, ymax) as usize;
        let row = src.row(y);
        let o = &mut out[r * ostride..r * ostride + ww];
        if inside_x {
            copy_short(o, &row[x0 as usize..x0 as usize + ww]);
        } else {
            for (c, slot) in o.iter_mut().enumerate() {
                *slot = L::of(row[(x0 + c as i32).clamp(0, width - 1) as usize]);
            }
        }
    }
}

/// Widen stored samples into working samples.
#[inline(always)]
fn widen<S: PicSample, L: Lane>(dst: &mut [L], src: &[S]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = L::of(s);
    }
}

/// Widen `wh` rows of `N` stored samples, the first starting at
/// `data[base]` and each `stride` after the previous, into `out` (row
/// stride `ostride`). `N` is a compile-time constant so each row is a
/// few widening vector loads and stores.
#[inline(always)]
fn rows_fixed<S: PicSample, L: Lane, const N: usize>(
    data: &[S],
    base: usize,
    stride: usize,
    wh: usize,
    out: &mut [L],
    ostride: usize,
) {
    for r in 0..wh {
        let s: &[S; N] = data[base + r * stride..][..N]
            .try_into()
            .expect("row length");
        let d: &mut [L; N] = (&mut out[r * ostride..][..N])
            .try_into()
            .expect("row length");
        *d = s.map(L::of);
    }
}

/// Widen the short stored-sample rows of a block window into `dst`.
/// The row lengths a kernel uses are dispatched to fixed-size loops,
/// which compile to a few widening vector loads and stores (a generic
/// loop per row costs more than the copy itself at these lengths).
#[inline(always)]
fn copy_short<S: PicSample, L: Lane>(dst: &mut [L], src: &[S]) {
    #[inline(always)]
    fn fixed<S: PicSample, L: Lane, const N: usize>(dst: &mut [L], src: &[S]) {
        let d: &mut [L; N] = (&mut dst[..N]).try_into().expect("row length");
        let s: &[S; N] = (&src[..N]).try_into().expect("row length");
        for (d, &s) in d.iter_mut().zip(s) {
            *d = L::of(s);
        }
    }
    match dst.len() {
        2 => fixed::<S, L, 2>(dst, src),
        3 => fixed::<S, L, 3>(dst, src),
        4 => fixed::<S, L, 4>(dst, src),
        5 => fixed::<S, L, 5>(dst, src),
        8 => fixed::<S, L, 8>(dst, src),
        9 => fixed::<S, L, 9>(dst, src),
        13 => fixed::<S, L, 13>(dst, src),
        16 => fixed::<S, L, 16>(dst, src),
        17 => fixed::<S, L, 17>(dst, src),
        21 => fixed::<S, L, 21>(dst, src),
        n => widen(&mut dst[..n], &src[..n]),
    }
}

#[inline(always)]
fn tap6(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32) -> i32 {
    a - 5 * b + 20 * c + 20 * d - 5 * e + f
}

#[inline(always)]
fn clip1(v: i32, max_v: i32) -> i32 {
    // Max-then-min: `i32::clamp` asserts its bounds are ordered, and that
    // panic branch keeps the row loops scalar.
    v.max(0).min(max_v)
}

/// §8.4.2.2.1 — predict a `w`x`h` luma block (`w`, `h` ≤ 16) at the
/// integer position `(int_x, int_y)` with quarter-sample offset
/// `(x_frac, y_frac)` (each 0..=3), writing `dst[y * dst_stride + x]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn luma<S: PicSample>(
    src: &PlaneSrc<'_, S>,
    int_x: i32,
    int_y: i32,
    x_frac: u8,
    y_frac: u8,
    w: usize,
    h: usize,
    max_v: i32,
    dst: &mut [i32],
    dst_stride: usize,
) {
    debug_assert!(w <= 16 && h <= 16 && x_frac < 4 && y_frac < 4);
    if x_frac == 0 && y_frac == 0 {
        // Table 8-12 — G: the integer samples themselves.
        gather(src, int_x, int_y, w, h, dst, dst_stride);
        return;
    }
    let ws = w + 5;
    if is_byte::<S>() && matches!(w, 8 | 16) {
        // 8-bit samples: the window and the 6-tap stage in 16-bit lanes
        // (at width 4, 32-bit lanes already fill a vector).
        let mut win = [0i16; WIN * WIN];
        gather(src, int_x - 2, int_y - 2, ws, h + 5, &mut win, ws);
        match w {
            8 => luma_fixed16::<8, 13>(&win, x_frac, y_frac, h, max_v, dst, dst_stride),
            _ => luma_fixed16::<16, 21>(&win, x_frac, y_frac, h, max_v, dst, dst_stride),
        }
        return;
    }
    let mut win = [0i32; WIN * WIN];
    gather(src, int_x - 2, int_y - 2, ws, h + 5, &mut win, ws);
    // Luma partitions are 4, 8 or 16 samples wide: a fixed width gives
    // each row loop a fixed trip count and no per-sample branching, so
    // it compiles to vector arithmetic.
    match w {
        4 => return luma_fixed::<4, 9>(&win, x_frac, y_frac, h, max_v, dst, dst_stride),
        8 => return luma_fixed::<8, 13>(&win, x_frac, y_frac, h, max_v, dst, dst_stride),
        16 => return luma_fixed::<16, 21>(&win, x_frac, y_frac, h, max_v, dst, dst_stride),
        _ => {}
    }
    // G at (x, y) of the block is win[(y + 2) * ws + x + 2].
    let g = |x: usize, y: usize| win[(y + 2) * ws + x + 2];
    // eq. 8-241 — b1 at (x, y + dy) (dy = 0: b, dy = 1: s).
    let b1 = |x: usize, yy: usize| {
        let r = &win[yy * ws + x..yy * ws + x + 6];
        tap6(r[0], r[1], r[2], r[3], r[4], r[5])
    };
    // eq. 8-242 — h1 at (x + dx, y) (dx = 0: h, dx = 1: m), `xx` the
    // window column of the sample.
    let h1 = |xx: usize, y: usize| {
        let c = y * ws + xx;
        tap6(
            win[c],
            win[c + ws],
            win[c + 2 * ws],
            win[c + 3 * ws],
            win[c + 4 * ws],
            win[c + 5 * ws],
        )
    };
    let half = |v1: i32| clip1((v1 + 16) >> 5, max_v);
    match (x_frac, y_frac) {
        // a / b / c — horizontal half sample, averaged with G or H.
        (_, 0) => {
            for y in 0..h {
                let drow = &mut dst[y * dst_stride..y * dst_stride + w];
                for (x, d) in drow.iter_mut().enumerate() {
                    let b = half(b1(x, y + 2));
                    *d = match x_frac {
                        1 => (g(x, y) + b + 1) >> 1,
                        2 => b,
                        _ => (g(x + 1, y) + b + 1) >> 1,
                    };
                }
            }
        }
        // d / h / n — vertical half sample, averaged with G or M.
        (0, _) => {
            for y in 0..h {
                let drow = &mut dst[y * dst_stride..y * dst_stride + w];
                for (x, d) in drow.iter_mut().enumerate() {
                    let hh = half(h1(x + 2, y));
                    *d = match y_frac {
                        1 => (g(x, y) + hh + 1) >> 1,
                        2 => hh,
                        _ => (g(x, y + 1) + hh + 1) >> 1,
                    };
                }
            }
        }
        // f / j / q — j from the b1 intermediates (eq. 8-246),
        // averaged with b or s.
        (2, _) => {
            let mut mid = [0i32; WIN * 16];
            for yy in 0..h + 5 {
                for x in 0..w {
                    mid[yy * w + x] = b1(x, yy);
                }
            }
            for y in 0..h {
                let drow = &mut dst[y * dst_stride..y * dst_stride + w];
                for (x, d) in drow.iter_mut().enumerate() {
                    let c = y * w + x;
                    let j1 = tap6(
                        mid[c],
                        mid[c + w],
                        mid[c + 2 * w],
                        mid[c + 3 * w],
                        mid[c + 4 * w],
                        mid[c + 5 * w],
                    );
                    let j = clip1((j1 + 512) >> 10, max_v);
                    *d = match y_frac {
                        1 => (half(mid[c + 2 * w]) + j + 1) >> 1,
                        2 => j,
                        _ => (half(mid[c + 3 * w]) + j + 1) >> 1,
                    };
                }
            }
        }
        // i / k — j from the h1 intermediates (eq. 8-245), averaged
        // with h or m.
        (_, 2) => {
            let mut mid = [0i32; 16 * WIN];
            for y in 0..h {
                for xx in 0..ws {
                    mid[y * ws + xx] = h1(xx, y);
                }
            }
            let side = if x_frac == 1 { 2 } else { 3 };
            for y in 0..h {
                let drow = &mut dst[y * dst_stride..y * dst_stride + w];
                let r = &mid[y * ws..y * ws + ws];
                for (x, d) in drow.iter_mut().enumerate() {
                    let j1 = tap6(r[x], r[x + 1], r[x + 2], r[x + 3], r[x + 4], r[x + 5]);
                    let j = clip1((j1 + 512) >> 10, max_v);
                    *d = (half(r[x + side]) + j + 1) >> 1;
                }
            }
        }
        // e / g / p / r — diagonal averages of b or s with h or m.
        _ => {
            let by = if y_frac == 1 { 2 } else { 3 };
            let hx = if x_frac == 1 { 2 } else { 3 };
            for y in 0..h {
                let drow = &mut dst[y * dst_stride..y * dst_stride + w];
                for (x, d) in drow.iter_mut().enumerate() {
                    let b = half(b1(x, y + by));
                    let hh = half(h1(x + hx, y));
                    *d = (b + hh + 1) >> 1;
                }
            }
        }
    }
}

/// The 6-tap filter (eqs. 8-241 / 8-242) over `W` consecutive outputs,
/// output `x` from `r[x..x + 6]`.
#[inline(always)]
fn tap6_row<const W: usize>(r: &[i32]) -> [i32; W] {
    let r = &r[..W + 5];
    let mut out = [0i32; W];
    for (x, o) in out.iter_mut().enumerate() {
        *o = tap6(r[x], r[x + 1], r[x + 2], r[x + 3], r[x + 4], r[x + 5]);
    }
    out
}

/// The 6-tap filter down six rows of `W` samples each.
#[inline(always)]
fn tap6_cols<const W: usize>(rows: [&[i32]; 6]) -> [i32; W] {
    let [a, b, c, d, e, f] = rows.map(|r| &r[..W]);
    let mut out = [0i32; W];
    for (x, o) in out.iter_mut().enumerate() {
        *o = tap6(a[x], b[x], c[x], d[x], e[x], f[x]);
    }
    out
}

/// [`luma`] for a block `W` samples wide, on its gathered window `win`
/// (row stride `WS` = `W` + 5). Each row is computed as whole arrays,
/// with the fractional position chosen outside the sample loops.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn luma_fixed<const W: usize, const WS: usize>(
    win: &[i32],
    x_frac: u8,
    y_frac: u8,
    h: usize,
    max_v: i32,
    dst: &mut [i32],
    dst_stride: usize,
) {
    debug_assert_eq!(WS, W + 5);
    let row = |yy: usize| -> &[i32] { &win[yy * WS..yy * WS + WS] };
    let half = |v: i32| clip1((v + 16) >> 5, max_v);
    let avg = |a: i32, b: i32| (a + b + 1) >> 1;
    let out = |dst: &mut [i32], y: usize, v: [i32; W]| {
        dst[y * dst_stride..y * dst_stride + W].copy_from_slice(&v);
    };
    // Vertical 6-tap of window columns `x0..x0 + W` at block row `y`.
    let h1 = |x0: usize, y: usize| -> [i32; W] {
        tap6_cols::<W>([0, 1, 2, 3, 4, 5].map(|k| &row(y + k)[x0..]))
    };
    match (x_frac, y_frac) {
        // a / b / c — horizontal half sample, averaged with G or H.
        (_, 0) => {
            for y in 0..h {
                let r = row(y + 2);
                let b = tap6_row::<W>(r).map(half);
                let v: [i32; W] = match x_frac {
                    2 => b,
                    1 => std::array::from_fn(|x| avg(r[x + 2], b[x])),
                    _ => std::array::from_fn(|x| avg(r[x + 3], b[x])),
                };
                out(dst, y, v);
            }
        }
        // d / h / n — vertical half sample, averaged with G or M.
        (0, _) => {
            for y in 0..h {
                let hh = h1(2, y).map(half);
                let v: [i32; W] = match y_frac {
                    2 => hh,
                    1 => {
                        let g = &row(y + 2)[2..];
                        std::array::from_fn(|x| avg(g[x], hh[x]))
                    }
                    _ => {
                        let g = &row(y + 3)[2..];
                        std::array::from_fn(|x| avg(g[x], hh[x]))
                    }
                };
                out(dst, y, v);
            }
        }
        // f / j / q — j from the b1 intermediates (eq. 8-246), averaged
        // with b or s.
        (2, _) => {
            let mut mid = [[0i32; W]; WIN];
            for (yy, m) in mid.iter_mut().enumerate().take(h + 5) {
                *m = tap6_row::<W>(row(yy));
            }
            for y in 0..h {
                let j1 = tap6_cols::<W>([0, 1, 2, 3, 4, 5].map(|k| &mid[y + k][..]));
                let j = j1.map(|v| clip1((v + 512) >> 10, max_v));
                let side = &mid[y + if y_frac == 1 { 2 } else { 3 }];
                let v: [i32; W] = match y_frac {
                    2 => j,
                    _ => std::array::from_fn(|x| avg(half(side[x]), j[x])),
                };
                out(dst, y, v);
            }
        }
        // i / k — j from the h1 intermediates (eq. 8-245), averaged
        // with h or m.
        (_, 2) => {
            let side = if x_frac == 1 { 2 } else { 3 };
            for y in 0..h {
                let m: [i32; WS] = tap6_cols::<WS>([0, 1, 2, 3, 4, 5].map(|k| row(y + k)));
                let j = tap6_row::<W>(&m).map(|v| clip1((v + 512) >> 10, max_v));
                let s = &m[side..];
                out(dst, y, std::array::from_fn(|x| avg(half(s[x]), j[x])));
            }
        }
        // e / g / p / r — diagonal averages of b or s with h or m.
        _ => {
            let by = if y_frac == 1 { 2 } else { 3 };
            let hx = if x_frac == 1 { 2 } else { 3 };
            for y in 0..h {
                let b = tap6_row::<W>(row(y + by)).map(half);
                let hh = h1(hx, y).map(half);
                out(dst, y, std::array::from_fn(|x| avg(b[x], hh[x])));
            }
        }
    }
}

/// eqs. 8-241 / 8-242 on 8-bit samples in 16 bits: an intermediate
/// `b1` / `h1` lies in -2550..=10710.
#[inline(always)]
fn tap6_16(a: i16, b: i16, c: i16, d: i16, e: i16, f: i16) -> i16 {
    a - 5 * b + 20 * c + 20 * d - 5 * e + f
}

/// [`tap6_row`] on 16-bit samples. Written as plain loops over fixed-size
/// arrays, which always inline and vectorize (`std::array::from_fn` and
/// `map` here were left out of line, one call per row).
#[inline(always)]
fn tap6_row16<const W: usize>(r: &[i16]) -> [i16; W] {
    let r = &r[..W + 5];
    let mut out = [0i16; W];
    for (x, o) in out.iter_mut().enumerate() {
        *o = tap6_16(r[x], r[x + 1], r[x + 2], r[x + 3], r[x + 4], r[x + 5]);
    }
    out
}

/// [`tap6_cols`] on 16-bit samples.
#[inline(always)]
fn tap6_cols16<const W: usize>(rows: [&[i16]; 6]) -> [i16; W] {
    let [a, b, c, d, e, f] = rows.map(|r| &r[..W]);
    let mut out = [0i16; W];
    for (x, o) in out.iter_mut().enumerate() {
        *o = tap6_16(a[x], b[x], c[x], d[x], e[x], f[x]);
    }
    out
}

/// The second 6-tap stage (eqs. 8-245 / 8-246) along a row of 16-bit
/// intermediates, in 32 bits (`j1` reaches about 4.8 * 10^5), then
/// eq. 8-247's rounding, shift and Clip1.
#[inline(always)]
fn center_row16<const W: usize>(r: &[i16], max_v: i32) -> [i16; W] {
    let r = &r[..W + 5];
    let mut out = [0i16; W];
    for (x, o) in out.iter_mut().enumerate() {
        let j1 = tap6(
            r[x].into(),
            r[x + 1].into(),
            r[x + 2].into(),
            r[x + 3].into(),
            r[x + 4].into(),
            r[x + 5].into(),
        );
        *o = clip1((j1 + 512) >> 10, max_v) as i16;
    }
    out
}

/// [`center_row16`] down six rows of intermediates.
#[inline(always)]
fn center_cols16<const W: usize>(rows: [&[i16]; 6], max_v: i32) -> [i16; W] {
    let [a, b, c, d, e, f] = rows.map(|r| &r[..W]);
    let mut out = [0i16; W];
    for (x, o) in out.iter_mut().enumerate() {
        let j1 = tap6(
            a[x].into(),
            b[x].into(),
            c[x].into(),
            d[x].into(),
            e[x].into(),
            f[x].into(),
        );
        *o = clip1((j1 + 512) >> 10, max_v) as i16;
    }
    out
}

/// The half-sample value of a first-stage intermediate (eqs. 8-243,
/// 8-244: `Clip1((v + 16) >> 5)`), for each of `W` samples in place.
#[inline(always)]
fn half16<const W: usize>(v: &mut [i16; W], max: i16) {
    for s in v.iter_mut() {
        *s = ((*s + 16) >> 5).max(0).min(max);
    }
}

/// `(a + b + 1) >> 1` (eqs. 8-250..8-261) into `a`, for `W` samples.
#[inline(always)]
fn avg16<const W: usize>(a: &mut [i16; W], b: &[i16]) {
    let b = &b[..W];
    for (a, &b) in a.iter_mut().zip(b) {
        *a = (*a + b + 1) >> 1;
    }
}

/// One output row of `W` samples, widened into the 32-bit prediction.
#[inline(always)]
fn store_row16<const W: usize>(dst: &mut [i32], v: &[i16; W]) {
    let d: &mut [i32; W] = (&mut dst[..W]).try_into().expect("row length");
    for (d, &v) in d.iter_mut().zip(v) {
        *d = v.into();
    }
}

/// [`luma_fixed`] for 8-bit samples in 16-bit lanes: the window, the
/// first 6-tap stage, the half-sample clip and the averages fit 16 bits
/// (a first-stage intermediate is in -2550..=10710); the second stage of
/// the center positions widens to 32 bits. Equal to [`luma_fixed`] on any
/// window of 8-bit samples.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn luma_fixed16<const W: usize, const WS: usize>(
    win: &[i16],
    x_frac: u8,
    y_frac: u8,
    h: usize,
    max_v: i32,
    dst: &mut [i32],
    dst_stride: usize,
) {
    debug_assert_eq!(WS, W + 5);
    // Clip1 of an 8-bit block: its bound fits 16 bits.
    let max16 = max_v.min(i32::from(i16::MAX)) as i16;
    let row = |yy: usize| -> &[i16] { &win[yy * WS..yy * WS + WS] };
    // Vertical 6-tap of window columns `x0..x0 + W` at block row `y`.
    #[inline(always)]
    fn h1<const W: usize, const WS: usize>(win: &[i16], x0: usize, y: usize) -> [i16; W] {
        let r = |k: usize| &win[(y + k) * WS + x0..];
        tap6_cols16::<W>([r(0), r(1), r(2), r(3), r(4), r(5)])
    }
    match (x_frac, y_frac) {
        // a / b / c — horizontal half sample, averaged with G or H.
        (_, 0) => {
            for y in 0..h {
                let r = row(y + 2);
                let mut v = tap6_row16::<W>(r);
                half16(&mut v, max16);
                match x_frac {
                    2 => {}
                    1 => avg16(&mut v, &r[2..]),
                    _ => avg16(&mut v, &r[3..]),
                }
                store_row16(&mut dst[y * dst_stride..], &v);
            }
        }
        // d / h / n — vertical half sample, averaged with G or M.
        (0, _) => {
            for y in 0..h {
                let mut v = h1::<W, WS>(win, 2, y);
                half16(&mut v, max16);
                match y_frac {
                    2 => {}
                    1 => avg16(&mut v, &row(y + 2)[2..]),
                    _ => avg16(&mut v, &row(y + 3)[2..]),
                }
                store_row16(&mut dst[y * dst_stride..], &v);
            }
        }
        // f / j / q — j from the b1 intermediates (eq. 8-246), averaged
        // with b or s.
        (2, _) => {
            let mut mid = [[0i16; W]; WIN];
            for (yy, m) in mid.iter_mut().enumerate().take(h + 5) {
                *m = tap6_row16::<W>(row(yy));
            }
            for y in 0..h {
                let m = |k: usize| &mid[y + k][..];
                let mut v = center_cols16::<W>([m(0), m(1), m(2), m(3), m(4), m(5)], max_v);
                if y_frac != 2 {
                    let mut side = mid[y + if y_frac == 1 { 2 } else { 3 }];
                    half16(&mut side, max16);
                    avg16(&mut v, &side);
                }
                store_row16(&mut dst[y * dst_stride..], &v);
            }
        }
        // i / k — j from the h1 intermediates (eq. 8-245), averaged
        // with h or m.
        (_, 2) => {
            let side = if x_frac == 1 { 2 } else { 3 };
            for y in 0..h {
                let r = |k: usize| row(y + k);
                let m: [i16; WS] = tap6_cols16::<WS>([r(0), r(1), r(2), r(3), r(4), r(5)]);
                let mut v = center_row16::<W>(&m, max_v);
                let mut s: [i16; W] = m[side..side + W].try_into().expect("row length");
                half16(&mut s, max16);
                avg16(&mut v, &s);
                store_row16(&mut dst[y * dst_stride..], &v);
            }
        }
        // e / g / p / r — diagonal averages of b or s with h or m.
        _ => {
            let by = if y_frac == 1 { 2 } else { 3 };
            let hx = if x_frac == 1 { 2 } else { 3 };
            for y in 0..h {
                let mut v = tap6_row16::<W>(row(y + by));
                half16(&mut v, max16);
                let mut hh = h1::<W, WS>(win, hx, y);
                half16(&mut hh, max16);
                avg16(&mut v, &hh);
                store_row16(&mut dst[y * dst_stride..], &v);
            }
        }
    }
}

/// eq. 8-270 for a chroma block `W` samples wide (window row stride
/// `WS` = `W` + 1) of 8-bit samples, in 16-bit lanes: the weights sum to
/// 64, so a weighted sum is at most 64 * 255 + 32.
#[inline(always)]
fn chroma_fixed16<const W: usize, const WS: usize>(
    win: &[i16],
    x_frac: u8,
    y_frac: u8,
    h: usize,
    dst: &mut [i32],
    dst_stride: usize,
) {
    debug_assert_eq!(WS, W + 1);
    let xf = i16::from(x_frac);
    let yf = i16::from(y_frac);
    let c00 = (8 - xf) * (8 - yf);
    let c10 = xf * (8 - yf);
    let c01 = (8 - xf) * yf;
    let c11 = xf * yf;
    for y in 0..h {
        let ra: &[i16; WS] = win[y * WS..][..WS].try_into().expect("row length");
        let rc: &[i16; WS] = win[(y + 1) * WS..][..WS].try_into().expect("row length");
        let mut v = [0i16; W];
        for (x, o) in v.iter_mut().enumerate() {
            *o = (c00 * ra[x] + c10 * ra[x + 1] + c01 * rc[x] + c11 * rc[x + 1] + 32) >> 6;
        }
        store_row16(&mut dst[y * dst_stride..], &v);
    }
}

/// §8.4.2.2.2 — predict a `w`x`h` chroma block (`w`, `h` ≤ 16) at the
/// integer position `(int_x, int_y)` with eighth-sample offset
/// `(x_frac, y_frac)` (each 0..=7), eq. 8-270.
#[allow(clippy::too_many_arguments)]
pub(crate) fn chroma<S: PicSample>(
    src: &PlaneSrc<'_, S>,
    int_x: i32,
    int_y: i32,
    x_frac: u8,
    y_frac: u8,
    w: usize,
    h: usize,
    dst: &mut [i32],
    dst_stride: usize,
) {
    debug_assert!(w <= 16 && h <= 16 && x_frac < 8 && y_frac < 8);
    if x_frac == 0 && y_frac == 0 {
        gather(src, int_x, int_y, w, h, dst, dst_stride);
        return;
    }
    let ws = w + 1;
    if is_byte::<S>() && matches!(w, 2 | 4 | 8) {
        // 8-bit samples: the window and the filter in 16-bit lanes.
        let mut win = [0i16; 17 * 17];
        gather(src, int_x, int_y, ws, h + 1, &mut win, ws);
        match w {
            2 => chroma_fixed16::<2, 3>(&win, x_frac, y_frac, h, dst, dst_stride),
            4 => chroma_fixed16::<4, 5>(&win, x_frac, y_frac, h, dst, dst_stride),
            _ => chroma_fixed16::<8, 9>(&win, x_frac, y_frac, h, dst, dst_stride),
        }
        return;
    }
    let mut win = [0i32; 17 * 17];
    gather(src, int_x, int_y, ws, h + 1, &mut win, ws);
    let xf = x_frac as i32;
    let yf = y_frac as i32;
    let c00 = (8 - xf) * (8 - yf);
    let c10 = xf * (8 - yf);
    let c01 = (8 - xf) * yf;
    let c11 = xf * yf;
    for y in 0..h {
        let ra = &win[y * ws..y * ws + ws];
        let rc = &win[(y + 1) * ws..(y + 1) * ws + ws];
        let drow = &mut dst[y * dst_stride..y * dst_stride + w];
        for (x, d) in drow.iter_mut().enumerate() {
            *d = (c00 * ra[x] + c10 * ra[x + 1] + c01 * rc[x] + c11 * rc[x + 1] + 32) >> 6;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(w: usize, h: usize, seed: u32) -> Vec<i32> {
        let mut s = seed;
        (0..w * h)
            .map(|_| {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((s >> 16) & 0xFF) as i32
            })
            .collect()
    }

    /// The window kernels equal the reference per-sample clamped
    /// interpolators for every fractional position, block size and a
    /// sweep of positions straddling every picture edge — through both
    /// the flat and the banded plane views, with the samples stored as
    /// `S` (bytes take the 16-bit-lane kernels, 16-bit samples the 32-bit
    /// ones).
    fn check_against_reference<S: PicSample>() {
        let (pw, ph) = (40usize, 36usize);
        let data = plane(pw, ph, 7);
        // The stored form of the same samples, for the window kernels.
        let stored: Vec<S> = data.iter().map(|&s| S::from_clipped(s)).collect();
        let bands: Vec<OnceLock<Box<[S]>>> = stored
            .chunks(pw * 4)
            .map(|c| {
                let l = OnceLock::new();
                let _ = l.set(c.to_vec().into_boxed_slice());
                l
            })
            .collect();
        let srcs = [
            PlaneSrc::Flat {
                data: &stored,
                stride: pw,
                width: pw,
                height: ph,
            },
            PlaneSrc::Banded {
                bands: &bands,
                band_rows: 4,
                width: pw,
                height: ph,
            },
        ];
        let sizes = [
            (4usize, 4usize),
            (8, 4),
            (4, 8),
            (8, 8),
            (16, 8),
            (8, 16),
            (16, 16),
        ];
        for src in &srcs {
            for &(w, h) in &sizes {
                for int_y in [-25i32, -3, 0, 5, 20, 33, 50] {
                    for int_x in [-30i32, -2, 0, 7, 24, 37, 60] {
                        for xf in 0..4u8 {
                            for yf in 0..4u8 {
                                let mut a = vec![0i32; w * h];
                                let mut b = vec![0i32; w * h];
                                luma(src, int_x, int_y, xf, yf, w, h, 255, &mut a, w);
                                crate::simd::scalar::interpolate_luma(
                                    &data, pw, pw, ph, int_x, int_y, xf, yf, w as u32, h as u32, 8,
                                    &mut b, w,
                                )
                                .unwrap();
                                assert_eq!(
                                    a, b,
                                    "luma {w}x{h} at ({int_x},{int_y}) frac ({xf},{yf})"
                                );
                            }
                        }
                        // 4:2:0 and 4:2:2 chroma shapes of this partition.
                        for (cw, chh) in [(w / 2, h / 2), (w / 2, h)] {
                            for xf in 0..8u8 {
                                for yf in 0..8u8 {
                                    let mut a = vec![0i32; cw * chh];
                                    let mut b = vec![0i32; cw * chh];
                                    chroma(src, int_x, int_y, xf, yf, cw, chh, &mut a, cw);
                                    crate::simd::scalar::interpolate_chroma(
                                        &data, pw, pw, ph, int_x, int_y, xf, yf, cw as u32,
                                        chh as u32, 8, &mut b, cw,
                                    )
                                    .unwrap();
                                    assert_eq!(
                                        a, b,
                                        "chroma {cw}x{chh} at ({int_x},{int_y}) frac ({xf},{yf})"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn luma_and_chroma_match_reference_interpolators() {
        check_against_reference::<Sample>();
    }

    /// 8-bit samples take the 16-bit-lane kernels, which must give the
    /// same samples.
    #[test]
    fn byte_samples_match_reference_interpolators() {
        check_against_reference::<u8>();
    }

    /// The 16-bit lanes at their extremes: windows of all 0 and all 255
    /// and a checkerboard of the two put each first-stage intermediate at
    /// the ends of its range (-2550 and 10710), and the center positions'
    /// second stage at its ends; the byte kernels equal the 32-bit ones.
    #[test]
    fn byte_kernels_match_wide_kernels_at_the_range_ends() {
        let (pw, ph) = (24usize, 24usize);
        let patterns: [fn(usize, usize) -> i32; 4] = [
            |_, _| 0,
            |_, _| 255,
            |x, y| if (x + y) % 2 == 0 { 255 } else { 0 },
            |x, y| if (x / 2 + y / 2) % 2 == 0 { 255 } else { 0 },
        ];
        for pattern in patterns {
            let data: Vec<i32> = (0..pw * ph).map(|i| pattern(i % pw, i / pw)).collect();
            let narrow: Vec<u8> = data.iter().map(|&s| s as u8).collect();
            let wide: Vec<Sample> = data.iter().map(|&s| s as Sample).collect();
            let flat = |d| PlaneSrc::Flat {
                data: d,
                stride: pw,
                width: pw,
                height: ph,
            };
            let (n, wd) = (
                flat(&narrow[..]),
                PlaneSrc::Flat {
                    data: &wide[..],
                    stride: pw,
                    width: pw,
                    height: ph,
                },
            );
            for (w, h) in [(4usize, 4usize), (8, 8), (16, 16), (16, 8)] {
                for (x, y) in [(3i32, 3i32), (4, 5)] {
                    for xf in 0..4u8 {
                        for yf in 0..4u8 {
                            let mut a = vec![0i32; w * h];
                            let mut b = vec![0i32; w * h];
                            luma(&n, x, y, xf, yf, w, h, 255, &mut a, w);
                            luma(&wd, x, y, xf, yf, w, h, 255, &mut b, w);
                            assert_eq!(a, b, "luma {w}x{h} frac ({xf},{yf})");
                        }
                    }
                    for xf in 0..8u8 {
                        for yf in 0..8u8 {
                            let (cw, chh) = (w / 2, h / 2);
                            let mut a = vec![0i32; cw * chh];
                            let mut b = vec![0i32; cw * chh];
                            chroma(&n, x, y, xf, yf, cw, chh, &mut a, cw);
                            chroma(&wd, x, y, xf, yf, cw, chh, &mut b, cw);
                            assert_eq!(a, b, "chroma {cw}x{chh} frac ({xf},{yf})");
                        }
                    }
                }
            }
        }
    }
}
