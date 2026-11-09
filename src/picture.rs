//! Picture sample buffer for a decoded frame/field.
//!
//! Holds luma + chroma sample planes sized from the active SPS. The
//! stored sample type is a parameter, [`PicSample`]: `u16` holds every
//! bit depth the spec allows (up to 14), `u8` holds 8-bit pictures in
//! half the memory traffic; the actual value range is
//! `0..(1 << bit_depth)`. Arithmetic on samples happens in `i32`: the
//! accessors widen on read and [`to_sample`] narrows on write.
//!
//! Spec references:
//! * §6.2 — ChromaArrayType and sample-array layout.
//! * §7.4.2.1.1 — picture-size derivations (PicWidthInSamplesL,
//!   PicHeightInSamplesL, MbWidthC, MbHeightC).
//! * §5.7 — Clip3 / Clip1 used when sampling outside the picture
//!   (returns the nearest edge sample, per §8.4.2.2.1).
//!
//! Clean-room: derived only from ITU-T Rec. H.264 (08/2024).

/// A stored picture sample. `u16` holds every bit depth of §7.4.2.1.1
/// (`BitDepthY` / `BitDepthC` ≤ 14); `u8` holds pictures whose luma
/// and chroma are both 8-bit in half the memory traffic. Both are
/// narrower than the `i32` the arithmetic uses.
pub trait PicSample:
    Copy + Default + Eq + Ord + core::fmt::Debug + Send + Sync + 'static + Into<i32>
{
    /// The largest stored value.
    const STORED_MAX: i32;

    /// Narrow a reconstructed sample (already clipped to its bit depth
    /// by the Clip1 of the producing process) to its stored form. Every
    /// producer clips first, so the saturation is a guard, never a
    /// change of value; debug builds assert it.
    fn from_clipped(v: i32) -> Self;

    /// Clip a row of `i32` samples to `0..=max` (Clip1, `max` at most
    /// `i16::MAX` and at most [`Self::STORED_MAX`]) and store them: one clamp
    /// per sample, so a producer need not clip first.
    fn store_clipped(dst: &mut [Self], src: &[i32], max: i32);

    /// Append a row of samples of an 8-bit plane to an output buffer,
    /// one byte each.
    fn extend_8bit(out: &mut Vec<u8>, row: &[Self]);

    /// The picture inside `any` when it stores this sample type.
    fn picture(any: &AnyPicture) -> Option<&Picture<Self>>;

    /// The picture inside `any` when it stores this sample type.
    fn picture_mut(any: &mut AnyPicture) -> Option<&mut Picture<Self>>;

    /// Frame threading: the published bands of `planes` when they store
    /// this sample type.
    #[doc(hidden)]
    fn bands(
        planes: &crate::frame_thread::SharedBands,
    ) -> Option<&crate::frame_thread::BandSet<Self>>;
}

/// The wide stored sample, for pictures of more than 8 bits per sample
/// (and the default sample type of [`Picture`]).
pub type Sample = u16;

impl PicSample for u16 {
    const STORED_MAX: i32 = u16::MAX as i32;

    #[inline]
    fn extend_8bit(out: &mut Vec<u8>, row: &[Self]) {
        out.extend(row.iter().map(|&s| s.min(255) as u8));
    }

    #[inline]
    fn picture(any: &AnyPicture) -> Option<&Picture<Self>> {
        match any {
            AnyPicture::Wide(p) => Some(p),
            AnyPicture::Narrow(_) => None,
        }
    }

    #[inline]
    fn picture_mut(any: &mut AnyPicture) -> Option<&mut Picture<Self>> {
        match any {
            AnyPicture::Wide(p) => Some(p),
            AnyPicture::Narrow(_) => None,
        }
    }

    #[inline]
    fn bands(
        planes: &crate::frame_thread::SharedBands,
    ) -> Option<&crate::frame_thread::BandSet<Self>> {
        match planes {
            crate::frame_thread::SharedBands::Wide(b) => Some(b),
            crate::frame_thread::SharedBands::Narrow(_) => None,
        }
    }

    #[inline(always)]
    fn from_clipped(v: i32) -> Self {
        debug_assert!(
            (0..=Self::STORED_MAX).contains(&v),
            "sample {v} outside the stored range"
        );
        v.clamp(0, Self::STORED_MAX) as u16
    }

    #[inline]
    fn store_clipped(dst: &mut [Self], src: &[i32], max: i32) {
        debug_assert_eq!(dst.len(), src.len());
        debug_assert!((0..=i32::from(i16::MAX)).contains(&max));
        // Saturating to i16 first, then clamping in 16 bits, equals the
        // 32-bit clamp for any `max` that fits i16 (every bit depth does)
        // and maps onto 16-bit vector packs and min / max that x86-64's
        // baseline SSE2 has (it lacks 32-bit min / max).
        let max = max as i16;
        for (d, &s) in dst.iter_mut().zip(src) {
            let s = s.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
            *d = s.clamp(0, max) as u16;
        }
    }
}

impl PicSample for u8 {
    const STORED_MAX: i32 = u8::MAX as i32;

    #[inline]
    fn extend_8bit(out: &mut Vec<u8>, row: &[Self]) {
        out.extend_from_slice(row);
    }

    #[inline]
    fn picture(any: &AnyPicture) -> Option<&Picture<Self>> {
        match any {
            AnyPicture::Narrow(p) => Some(p),
            AnyPicture::Wide(_) => None,
        }
    }

    #[inline]
    fn picture_mut(any: &mut AnyPicture) -> Option<&mut Picture<Self>> {
        match any {
            AnyPicture::Narrow(p) => Some(p),
            AnyPicture::Wide(_) => None,
        }
    }

    #[inline]
    fn bands(
        planes: &crate::frame_thread::SharedBands,
    ) -> Option<&crate::frame_thread::BandSet<Self>> {
        match planes {
            crate::frame_thread::SharedBands::Narrow(b) => Some(b),
            crate::frame_thread::SharedBands::Wide(_) => None,
        }
    }

    #[inline(always)]
    fn from_clipped(v: i32) -> Self {
        debug_assert!(
            (0..=Self::STORED_MAX).contains(&v),
            "sample {v} outside the stored range"
        );
        v.clamp(0, Self::STORED_MAX) as u8
    }

    #[inline]
    fn store_clipped(dst: &mut [Self], src: &[i32], max: i32) {
        debug_assert_eq!(dst.len(), src.len());
        debug_assert!((0..=Self::STORED_MAX).contains(&max));
        // As for u16: saturate to i16, clamp in 16 bits (SSE2 packs and
        // min / max), then narrow; the clamp leaves values in 0..=255.
        let max = max as i16;
        for (d, &s) in dst.iter_mut().zip(src) {
            let s = s.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
            *d = s.clamp(0, max) as u8;
        }
    }
}

/// Narrow a reconstructed sample to its stored form
/// ([`PicSample::from_clipped`]).
#[inline(always)]
pub fn to_sample<S: PicSample>(v: i32) -> S {
    S::from_clipped(v)
}

/// Narrow a row of clipped `i32` samples into stored samples
/// ([`to_sample`] for each; `dst` and `src` have equal lengths).
#[inline]
pub fn store_samples<S: PicSample>(dst: &mut [S], src: &[i32]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = S::from_clipped(s);
    }
}

/// Clip a row of `i32` samples to `0..=max` and store them
/// ([`PicSample::store_clipped`]).
#[inline]
pub fn store_clipped<S: PicSample>(dst: &mut [S], src: &[i32], max: i32) {
    S::store_clipped(dst, src, max);
}

/// Widen a row of stored samples into `i32` working samples.
#[inline]
pub fn load_samples<S: PicSample>(dst: &mut [i32], src: &[S]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = s.into();
    }
}

/// §8.4.1.2.1 Table 8-7 — `PicCodingStruct( X )` of a decoded picture:
/// FLD (coded with `field_pic_flag` = 1), FRM (frame,
/// `mb_adaptive_frame_field_flag` = 0) or AFRM (MBAFF frame).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PicCodingStruct {
    #[default]
    Frm,
    Fld,
    Afrm,
}

/// A decoded picture sample buffer.
///
/// `luma` is a row-major `i32` buffer of size
/// `width_in_samples * height_in_samples`. Chroma planes `cb` / `cr`
/// are sized from `chroma_array_type` (and are empty for monochrome).
///
/// POC and frame_num are set by the caller after reconstruction.
#[derive(Debug, Clone)]
pub struct Picture<S = Sample> {
    pub width_in_samples: u32,
    pub height_in_samples: u32,
    /// §6.2 — 0 = monochrome (or separate_colour_plane_flag),
    /// 1 = 4:2:0, 2 = 4:2:2, 3 = 4:4:4.
    pub chroma_array_type: u32,
    pub bit_depth_luma: u32,
    pub bit_depth_chroma: u32,
    pub luma: Vec<S>,
    pub cb: Vec<S>,
    pub cr: Vec<S>,
    /// §8.2.5.2 — set on the placeholder pictures synthesised to fill a
    /// gap in `frame_num`. Their samples are "not available for
    /// prediction of other pictures": a conforming bitstream never
    /// samples a non-existing frame, so motion compensation refuses to
    /// (reconstructing from invented placeholder samples would silently
    /// fabricate picture content).
    pub non_existing: bool,
    /// Set by caller after reconstruction.
    pub pic_order_cnt: i32,
    pub frame_num: u32,
    /// §8.4.1.2.3 — co-located motion data for temporal-direct
    /// derivation. When populated, `mv_l0_grid[mb_addr * 16 + blk4]`
    /// holds the list-0 MV (in 1/4-pel units) of the 4x4 block, and
    /// `ref_idx_l0_grid[mb_addr * 4 + blk8]` holds the list-0 refIdx
    /// (−1 when the list is not used). `is_intra_grid[mb_addr]` flags
    /// intra-coded macroblocks (colZeroFlag wants to treat intra as
    /// "L0 unavailable").
    ///
    /// The grids are sized as:
    ///   - `mv_*_grid`: `width_in_mbs * height_in_mbs * 16`
    ///   - `ref_idx_*_grid`: `width_in_mbs * height_in_mbs * 4`
    ///   - `is_intra_grid`: `width_in_mbs * height_in_mbs`
    ///
    /// Absent for I-only pictures or for pictures whose motion data is
    /// not needed by later B slices.
    pub mb_width_in_picture: u32,
    pub mv_l0_grid: Vec<(i16, i16)>,
    pub mv_l1_grid: Vec<(i16, i16)>,
    pub ref_idx_l0_grid: Vec<i8>,
    pub ref_idx_l1_grid: Vec<i8>,
    pub is_intra_grid: Vec<bool>,
    /// §8.4.1.2.3 — per-slice snapshot of POCs in the slice's
    /// RefPicList0 that was used when this picture was decoded.
    /// Needed by a later B-slice's temporal-direct derivation to map
    /// this picture's colocated block's refIdxL0 back to a concrete
    /// picture identity (its POC), which is then matched against the
    /// current slice's RefPicList0 to produce refIdxL0 for the current
    /// block (MapColToList0 of eq. 8-191).
    ///
    /// Snapshotted at reconstruction time per slice; if the picture
    /// comprises multiple slices we keep the first non-empty value
    /// (all slices of a coded picture share reference pictures modulo
    /// RPLM — near-enough for I/P pictures used as colocated).
    pub ref_list_0_pocs: Vec<i32>,
    pub ref_list_1_pocs: Vec<i32>,
    /// §8.4.1.2.3 — whether the picture at position `k` in
    /// `ref_list_0_pocs` / `ref_list_1_pocs` is a long-term reference.
    /// Parallel arrays to the `_pocs` arrays above. Needed for the
    /// eq. 8-195 short-circuit.
    pub ref_list_0_longterm: Vec<bool>,
    pub ref_list_1_longterm: Vec<bool>,
    /// §8.4.1.2.1 Table 8-7 — how this picture was coded (FLD / FRM /
    /// AFRM). Drives the Table 8-8 mbAddrCol / yM / vertMvScale rows
    /// when this picture serves as `colPic`.
    pub coding_struct: PicCodingStruct,
    /// §7.4.3 — for a coded FIELD picture, its parity (`true` =
    /// bottom). Meaningless for frames.
    pub is_bottom_field: bool,
    /// §8.2.1 — TopFieldOrderCnt / BottomFieldOrderCnt of the coded
    /// picture (for a coded field only its own parity is meaningful).
    /// Needed for the §8.4.1.2.3 per-field tb/td distances when the
    /// current macroblock is a field macroblock of an MBAFF frame.
    pub top_field_order_cnt: i32,
    pub bottom_field_order_cnt: i32,
    /// §6.4.12.2 / Table 8-8 — per-MB `mb_field_decoding_flag`
    /// snapshot for AFRM pictures (`fieldDecodingFlagX` of the
    /// co-located macroblock). Empty for FLD / FRM pictures.
    pub mb_field_flags: Vec<bool>,
    /// §8.4.1.2.1 Table 8-6 — set on a [`Picture::field_view`] of a
    /// stored FRAME (the view's parity). When `RefPicList1[ 0 ]` of a
    /// field slice resolves to such a view, colPic is "the frame
    /// containing RefPicList1[ 0 ]": the view carries the FRAME's
    /// motion grids / coding struct / reference-list snapshots so the
    /// temporal-direct derivation can address it in frame coordinates.
    pub view_of_frame_parity: Option<u8>,
    /// §8.4.1.2.3 MapColToList0 — picture-identity snapshot of the
    /// reference lists active when this picture was decoded: per-entry
    /// DPB storage key, field parity (`None` for frame/pair units) and
    /// the frame-level UNIT key containing the entry (for a coded
    /// field of a complementary pair: the pair's unit key; for frame
    /// units: the entry's own key). Parallel to `ref_list_*_pocs`.
    pub ref_list_0_keys: Vec<u32>,
    pub ref_list_0_parities: Vec<Option<u8>>,
    pub ref_list_0_unit_keys: Vec<u32>,
    pub ref_list_1_keys: Vec<u32>,
    pub ref_list_1_parities: Vec<Option<u8>>,
    pub ref_list_1_unit_keys: Vec<u32>,
}

impl Picture<Sample> {
    /// Allocate a zero-filled wide-sample picture ([`Picture::zeroed`]).
    pub fn new(
        width_in_samples: u32,
        height_in_samples: u32,
        chroma_array_type: u32,
        bit_depth_luma: u32,
        bit_depth_chroma: u32,
    ) -> Self {
        Self::zeroed(
            width_in_samples,
            height_in_samples,
            chroma_array_type,
            bit_depth_luma,
            bit_depth_chroma,
        )
    }
}

impl<S: PicSample> Picture<S> {
    /// Allocate a zero-filled picture of the requested geometry.
    pub fn zeroed(
        width_in_samples: u32,
        height_in_samples: u32,
        chroma_array_type: u32,
        bit_depth_luma: u32,
        bit_depth_chroma: u32,
    ) -> Self {
        let luma_len = (width_in_samples as usize) * (height_in_samples as usize);
        // §6.2 / Table 6-1 — MbWidthC / MbHeightC.
        //   ChromaArrayType == 0: no chroma.
        //   ChromaArrayType == 1 (4:2:0): MbWidthC=8, MbHeightC=8 → half W/H.
        //   ChromaArrayType == 2 (4:2:2): MbWidthC=8, MbHeightC=16 → half W, full H.
        //   ChromaArrayType == 3 (4:4:4): MbWidthC=16, MbHeightC=16 → full W, full H.
        let (cw, ch) = chroma_dims(chroma_array_type, width_in_samples, height_in_samples);
        let chroma_len = (cw as usize) * (ch as usize);
        Self {
            width_in_samples,
            height_in_samples,
            chroma_array_type,
            bit_depth_luma,
            bit_depth_chroma,
            luma: vec![S::default(); luma_len],
            cb: vec![S::default(); chroma_len],
            cr: vec![S::default(); chroma_len],
            non_existing: false,
            pic_order_cnt: 0,
            frame_num: 0,
            mb_width_in_picture: 0,
            mv_l0_grid: Vec::new(),
            mv_l1_grid: Vec::new(),
            ref_idx_l0_grid: Vec::new(),
            ref_idx_l1_grid: Vec::new(),
            is_intra_grid: Vec::new(),
            ref_list_0_pocs: Vec::new(),
            ref_list_1_pocs: Vec::new(),
            ref_list_0_longterm: Vec::new(),
            ref_list_1_longterm: Vec::new(),
            coding_struct: PicCodingStruct::default(),
            is_bottom_field: false,
            top_field_order_cnt: 0,
            bottom_field_order_cnt: 0,
            mb_field_flags: Vec::new(),
            view_of_frame_parity: None,
            ref_list_0_keys: Vec::new(),
            ref_list_0_parities: Vec::new(),
            ref_list_0_unit_keys: Vec::new(),
            ref_list_1_keys: Vec::new(),
            ref_list_1_parities: Vec::new(),
            ref_list_1_unit_keys: Vec::new(),
        }
    }

    /// [`Picture::new`] of `old`'s geometry, reusing its allocations: the
    /// sample planes, and the motion grids emptied as `new` leaves them.
    /// The planes keep `old`'s samples unless `zero` is set; a caller may
    /// leave it unset only when every sample of the new picture is
    /// written before anything reads it.
    pub fn recycled(old: Picture<S>, zero: bool) -> Self {
        let mut p = Self::zeroed(
            0,
            0,
            old.chroma_array_type,
            old.bit_depth_luma,
            old.bit_depth_chroma,
        );
        p.width_in_samples = old.width_in_samples;
        p.height_in_samples = old.height_in_samples;
        p.luma = old.luma;
        p.cb = old.cb;
        p.cr = old.cr;
        if zero {
            p.luma.fill(S::default());
            p.cb.fill(S::default());
            p.cr.fill(S::default());
        }
        fn emptied<T>(mut v: Vec<T>) -> Vec<T> {
            v.clear();
            v
        }
        p.mv_l0_grid = emptied(old.mv_l0_grid);
        p.mv_l1_grid = emptied(old.mv_l1_grid);
        p.ref_idx_l0_grid = emptied(old.ref_idx_l0_grid);
        p.ref_idx_l1_grid = emptied(old.ref_idx_l1_grid);
        p.is_intra_grid = emptied(old.is_intra_grid);
        p.mb_field_flags = emptied(old.mb_field_flags);
        p
    }

    /// True when `self` has the geometry and sample format given.
    pub fn has_shape(
        &self,
        width_in_samples: u32,
        height_in_samples: u32,
        chroma_array_type: u32,
        bit_depth_luma: u32,
        bit_depth_chroma: u32,
    ) -> bool {
        let (cw, ch) = chroma_dims(chroma_array_type, width_in_samples, height_in_samples);
        self.width_in_samples == width_in_samples
            && self.height_in_samples == height_in_samples
            && self.chroma_array_type == chroma_array_type
            && self.bit_depth_luma == bit_depth_luma
            && self.bit_depth_chroma == bit_depth_chroma
            && self.luma.len() == (width_in_samples as usize) * (height_in_samples as usize)
            && self.cb.len() == (cw as usize) * (ch as usize)
            && self.cr.len() == self.cb.len()
    }

    /// §8.4.2.1 / §8.2.4.2.5 — extract one parity field of this FRAME
    /// picture as a standalone half-height picture (top field = even
    /// sample rows of every plane, bottom = odd rows; the chroma planes
    /// of every interlace-capable format have even height and split by
    /// the same row parity). Used when a coded FIELD picture references
    /// a field of a picture that was stored as a frame. The FRAME's
    /// motion / colocated grids, per-MB field flags, coding struct and
    /// reference-list identity snapshots are carried on the view (in
    /// FRAME addressing) with `view_of_frame_parity` marking it, so
    /// the §8.4.1.2.1 Table 8-6 "colPic = the frame containing
    /// RefPicList1[ 0 ]" row can address the containing frame. The
    /// caller stamps `pic_order_cnt` with the FIELD's own order count.
    pub fn field_view(&self, bottom: bool) -> Picture<S> {
        let mut out = Self::zeroed(
            self.width_in_samples,
            self.height_in_samples / 2,
            self.chroma_array_type,
            self.bit_depth_luma,
            self.bit_depth_chroma,
        );
        let take_rows = |src: &[S], width: usize, height: usize| -> Vec<S> {
            let mut v = Vec::with_capacity(width * height / 2);
            for row in (usize::from(bottom)..height).step_by(2) {
                v.extend_from_slice(&src[row * width..(row + 1) * width]);
            }
            v
        };
        out.luma = take_rows(
            &self.luma,
            self.width_in_samples as usize,
            self.height_in_samples as usize,
        );
        let cw = self.chroma_width() as usize;
        let ch = self.chroma_height() as usize;
        if ch > 0 {
            out.cb = take_rows(&self.cb, cw, ch);
            out.cr = take_rows(&self.cr, cw, ch);
        }
        out.non_existing = self.non_existing;
        out.frame_num = self.frame_num;
        out.pic_order_cnt = self.pic_order_cnt;
        // §8.4.1.2.1 — carry the FRAME's colocated-motion state so the
        // view can serve as "the frame containing RefPicList1[ 0 ]"
        // (Table 8-6, field_pic_flag == 1 row 1).
        out.coding_struct = self.coding_struct;
        out.top_field_order_cnt = self.top_field_order_cnt;
        out.bottom_field_order_cnt = self.bottom_field_order_cnt;
        out.mb_width_in_picture = self.mb_width_in_picture;
        out.mv_l0_grid = self.mv_l0_grid.clone();
        out.mv_l1_grid = self.mv_l1_grid.clone();
        out.ref_idx_l0_grid = self.ref_idx_l0_grid.clone();
        out.ref_idx_l1_grid = self.ref_idx_l1_grid.clone();
        out.is_intra_grid = self.is_intra_grid.clone();
        out.mb_field_flags = self.mb_field_flags.clone();
        out.ref_list_0_pocs = self.ref_list_0_pocs.clone();
        out.ref_list_1_pocs = self.ref_list_1_pocs.clone();
        out.ref_list_0_longterm = self.ref_list_0_longterm.clone();
        out.ref_list_1_longterm = self.ref_list_1_longterm.clone();
        out.ref_list_0_keys = self.ref_list_0_keys.clone();
        out.ref_list_0_parities = self.ref_list_0_parities.clone();
        out.ref_list_0_unit_keys = self.ref_list_0_unit_keys.clone();
        out.ref_list_1_keys = self.ref_list_1_keys.clone();
        out.ref_list_1_parities = self.ref_list_1_parities.clone();
        out.ref_list_1_unit_keys = self.ref_list_1_unit_keys.clone();
        out.view_of_frame_parity = Some(u8::from(bottom));
        out
    }

    /// §6.2 — chroma plane width in samples.
    pub fn chroma_width(&self) -> u32 {
        chroma_dims(
            self.chroma_array_type,
            self.width_in_samples,
            self.height_in_samples,
        )
        .0
    }

    /// §6.2 — chroma plane height in samples.
    pub fn chroma_height(&self) -> u32 {
        chroma_dims(
            self.chroma_array_type,
            self.width_in_samples,
            self.height_in_samples,
        )
        .1
    }

    /// Clamped luma sample access — coordinates outside the picture are
    /// clipped to the picture edge per §8.4.2.2.1 / Clip3 of §5.7. This
    /// matches the neighbour-sampling rule used by intra prediction when
    /// a neighbour is available but the sample index goes off-picture.
    #[inline]
    pub fn luma_at(&self, x: i32, y: i32) -> i32 {
        if self.width_in_samples == 0 || self.height_in_samples == 0 {
            return 0;
        }
        let xi = clip3_i32(0, self.width_in_samples as i32 - 1, x) as usize;
        let yi = clip3_i32(0, self.height_in_samples as i32 - 1, y) as usize;
        self.luma[yi * (self.width_in_samples as usize) + xi].into()
    }

    /// Clamped chroma Cb access (§6.2).
    #[inline]
    pub fn cb_at(&self, x: i32, y: i32) -> i32 {
        let cw = self.chroma_width();
        let ch = self.chroma_height();
        if cw == 0 || ch == 0 {
            return 0;
        }
        let xi = clip3_i32(0, cw as i32 - 1, x) as usize;
        let yi = clip3_i32(0, ch as i32 - 1, y) as usize;
        self.cb[yi * (cw as usize) + xi].into()
    }

    /// Clamped chroma Cr access.
    #[inline]
    pub fn cr_at(&self, x: i32, y: i32) -> i32 {
        let cw = self.chroma_width();
        let ch = self.chroma_height();
        if cw == 0 || ch == 0 {
            return 0;
        }
        let xi = clip3_i32(0, cw as i32 - 1, x) as usize;
        let yi = clip3_i32(0, ch as i32 - 1, y) as usize;
        self.cr[yi * (cw as usize) + xi].into()
    }

    /// Write a luma sample. Out-of-bounds writes are silently ignored
    /// (callers should not produce such writes — this is a hard guard
    /// for robustness during reconstruction edge handling).
    #[inline]
    pub fn set_luma(&mut self, x: i32, y: i32, v: i32) {
        if x < 0 || y < 0 {
            return;
        }
        let xi = x as u32;
        let yi = y as u32;
        if xi >= self.width_in_samples || yi >= self.height_in_samples {
            return;
        }
        let idx = (yi as usize) * (self.width_in_samples as usize) + (xi as usize);
        self.luma[idx] = to_sample(v);
    }

    /// Write a Cb sample.
    #[inline]
    pub fn set_cb(&mut self, x: i32, y: i32, v: i32) {
        if x < 0 || y < 0 {
            return;
        }
        let cw = self.chroma_width();
        let ch = self.chroma_height();
        let xi = x as u32;
        let yi = y as u32;
        if xi >= cw || yi >= ch {
            return;
        }
        let idx = (yi as usize) * (cw as usize) + (xi as usize);
        self.cb[idx] = to_sample(v);
    }

    /// Write a Cr sample.
    #[inline]
    pub fn set_cr(&mut self, x: i32, y: i32, v: i32) {
        if x < 0 || y < 0 {
            return;
        }
        let cw = self.chroma_width();
        let ch = self.chroma_height();
        let xi = x as u32;
        let yi = y as u32;
        if xi >= cw || yi >= ch {
            return;
        }
        let idx = (yi as usize) * (cw as usize) + (xi as usize);
        self.cr[idx] = to_sample(v);
    }

    /// Fetch the colocated L0 MV + refIdx for the 4x4 block at
    /// (`mb_addr`, `blk4`). Returns `None` if motion data was not
    /// populated for this picture (e.g. I-only pic) or `mb_addr` /
    /// `blk4` is out of range.
    ///
    /// `blk4` uses the Figure 6-10 raster index (0..=15).
    pub fn colocated_l0(&self, mb_addr: u32, blk4: usize) -> Option<((i16, i16), i8, bool)> {
        if self.mv_l0_grid.is_empty() {
            return None;
        }
        let mv_idx = (mb_addr as usize).checked_mul(16)?.checked_add(blk4)?;
        let mv = *self.mv_l0_grid.get(mv_idx)?;
        let ref_idx = *self
            .ref_idx_l0_grid
            .get((mb_addr as usize) * 4 + (blk4 / 4))?;
        let is_intra = self
            .is_intra_grid
            .get(mb_addr as usize)
            .copied()
            .unwrap_or(false);
        Some((mv, ref_idx, is_intra))
    }

    /// Fetch the colocated L1 MV + refIdx. Used when the colocated
    /// block has predFlagL0 == 0 and the caller needs to fall back on
    /// its L1 MV per §8.4.1.2.3.
    pub fn colocated_l1(&self, mb_addr: u32, blk4: usize) -> Option<((i16, i16), i8, bool)> {
        if self.mv_l1_grid.is_empty() {
            return None;
        }
        let mv_idx = (mb_addr as usize).checked_mul(16)?.checked_add(blk4)?;
        let mv = *self.mv_l1_grid.get(mv_idx)?;
        let ref_idx = *self
            .ref_idx_l1_grid
            .get((mb_addr as usize) * 4 + (blk4 / 4))?;
        let is_intra = self
            .is_intra_grid
            .get(mb_addr as usize)
            .copied()
            .unwrap_or(false);
        Some((mv, ref_idx, is_intra))
    }
}

/// §6.2 / Table 6-1 — chroma plane (width, height) given
/// `chroma_array_type` and the luma frame dimensions.
/// Whether a picture of these bit depths stores its samples as `u8`
/// ([`AnyPicture::Narrow`]): both luma and chroma 8-bit.
#[inline]
pub fn stores_narrow(bit_depth_luma: u32, bit_depth_chroma: u32) -> bool {
    bit_depth_luma == 8 && bit_depth_chroma == 8
}

/// A decoded picture of either stored sample type. The decoder keeps
/// pictures in this form; reconstruction runs on the [`Picture`]
/// inside, generic over its sample type. A sequence's bit depths are
/// fixed by its SPS (§7.4.2.1.1), so every picture a slice references
/// has the current picture's variant.
#[derive(Debug, Clone)]
pub enum AnyPicture {
    /// Both bit depths 8: `u8` samples.
    Narrow(Picture<u8>),
    /// Any bit depth above 8: `u16` samples.
    Wide(Picture<u16>),
}

/// Evaluate `$body` with `$p` bound to the [`Picture`] inside an
/// [`AnyPicture`] (or a reference to one), whichever its sample type.
macro_rules! with_pic {
    ($any:expr, $p:ident => $body:expr) => {
        match $any {
            $crate::picture::AnyPicture::Narrow($p) => $body,
            $crate::picture::AnyPicture::Wide($p) => $body,
        }
    };
}
pub(crate) use with_pic;

impl From<Picture<u8>> for AnyPicture {
    fn from(p: Picture<u8>) -> Self {
        AnyPicture::Narrow(p)
    }
}

impl From<Picture<u16>> for AnyPicture {
    fn from(p: Picture<u16>) -> Self {
        AnyPicture::Wide(p)
    }
}

impl AnyPicture {
    /// Allocate a zero-filled picture, `u8` samples when both bit depths
    /// are 8 ([`stores_narrow`]), `u16` otherwise.
    pub fn zeroed(
        width_in_samples: u32,
        height_in_samples: u32,
        chroma_array_type: u32,
        bit_depth_luma: u32,
        bit_depth_chroma: u32,
    ) -> Self {
        if stores_narrow(bit_depth_luma, bit_depth_chroma) {
            AnyPicture::Narrow(Picture::zeroed(
                width_in_samples,
                height_in_samples,
                chroma_array_type,
                bit_depth_luma,
                bit_depth_chroma,
            ))
        } else {
            AnyPicture::Wide(Picture::zeroed(
                width_in_samples,
                height_in_samples,
                chroma_array_type,
                bit_depth_luma,
                bit_depth_chroma,
            ))
        }
    }

    /// An empty picture (no samples, no metadata), as a placeholder.
    pub fn empty() -> Self {
        AnyPicture::Narrow(Picture::zeroed(0, 0, 0, 8, 8))
    }

    /// [`Picture::recycled`] on whichever picture is inside.
    pub fn recycled(old: AnyPicture, zero: bool) -> Self {
        match old {
            AnyPicture::Narrow(p) => AnyPicture::Narrow(Picture::recycled(p, zero)),
            AnyPicture::Wide(p) => AnyPicture::Wide(Picture::recycled(p, zero)),
        }
    }

    /// [`Picture::has_shape`], and the sample type those bit depths store.
    pub fn has_shape(
        &self,
        width_in_samples: u32,
        height_in_samples: u32,
        chroma_array_type: u32,
        bit_depth_luma: u32,
        bit_depth_chroma: u32,
    ) -> bool {
        matches!(self, AnyPicture::Narrow(_)) == stores_narrow(bit_depth_luma, bit_depth_chroma)
            && with_pic!(self, p => p.has_shape(
                width_in_samples,
                height_in_samples,
                chroma_array_type,
                bit_depth_luma,
                bit_depth_chroma,
            ))
    }

    /// [`Picture::field_view`] of whichever picture is inside.
    pub fn field_view(&self, bottom: bool) -> AnyPicture {
        match self {
            AnyPicture::Narrow(p) => AnyPicture::Narrow(p.field_view(bottom)),
            AnyPicture::Wide(p) => AnyPicture::Wide(p.field_view(bottom)),
        }
    }

    /// Whether the picture holds no luma samples (a metadata-only shell).
    pub fn has_no_samples(&self) -> bool {
        with_pic!(self, p => p.luma.is_empty())
    }

    pub fn width_in_samples(&self) -> u32 {
        with_pic!(self, p => p.width_in_samples)
    }

    pub fn height_in_samples(&self) -> u32 {
        with_pic!(self, p => p.height_in_samples)
    }

    pub fn chroma_array_type(&self) -> u32 {
        with_pic!(self, p => p.chroma_array_type)
    }

    pub fn pic_order_cnt(&self) -> i32 {
        with_pic!(self, p => p.pic_order_cnt)
    }

    pub fn frame_num(&self) -> u32 {
        with_pic!(self, p => p.frame_num)
    }

    pub fn luma_at(&self, x: i32, y: i32) -> i32 {
        with_pic!(self, p => p.luma_at(x, y))
    }

    pub fn cb_at(&self, x: i32, y: i32) -> i32 {
        with_pic!(self, p => p.cb_at(x, y))
    }

    pub fn cr_at(&self, x: i32, y: i32) -> i32 {
        with_pic!(self, p => p.cr_at(x, y))
    }
}

fn chroma_dims(chroma_array_type: u32, w: u32, h: u32) -> (u32, u32) {
    match chroma_array_type {
        0 => (0, 0),
        1 => (w / 2, h / 2),
        2 => (w / 2, h),
        3 => (w, h),
        _ => (0, 0),
    }
}

/// §5.7 — `Clip3(x, y, z) = min(y, max(x, z))`. Inlined here so that
/// Picture stays a self-contained module.
#[inline]
fn clip3_i32(x: i32, y: i32, z: i32) -> i32 {
    if z < x {
        x
    } else if z > y {
        y
    } else {
        z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A recycled picture is a new picture of the same geometry: zeroed
    /// planes when asked, the old samples otherwise, and every other field
    /// (motion grids included) as `new` leaves it.
    #[test]
    fn recycled_equals_new() {
        let mut old = Picture::new(32, 16, 1, 8, 8);
        old.set_luma(3, 4, 200);
        old.pic_order_cnt = 7;
        old.frame_num = 3;
        old.non_existing = true;
        old.mv_l0_grid = vec![(1, 2); 32];
        old.is_intra_grid = vec![true; 2];
        old.ref_list_0_pocs = vec![4];
        let zeroed = Picture::recycled(old.clone(), true);
        assert_eq!(
            format!("{zeroed:?}"),
            format!("{:?}", Picture::new(32, 16, 1, 8, 8))
        );
        assert!(zeroed.mv_l0_grid.capacity() >= 32, "grid allocation kept");
        let kept = Picture::recycled(old, false);
        assert_eq!(kept.luma_at(3, 4), 200);
        let mut expect = Picture::new(32, 16, 1, 8, 8);
        expect.set_luma(3, 4, 200);
        assert_eq!(format!("{kept:?}"), format!("{expect:?}"));
        assert!(kept.has_shape(32, 16, 1, 8, 8));
        assert!(!kept.has_shape(32, 16, 2, 8, 8));
        assert!(!kept.has_shape(16, 16, 1, 8, 8));
    }

    #[test]
    fn allocation_monochrome() {
        let p = Picture::new(32, 16, 0, 8, 8);
        assert_eq!(p.luma.len(), 32 * 16);
        assert!(p.cb.is_empty());
        assert!(p.cr.is_empty());
        assert_eq!(p.chroma_width(), 0);
        assert_eq!(p.chroma_height(), 0);
    }

    #[test]
    fn allocation_yuv420() {
        let p = Picture::new(32, 16, 1, 8, 8);
        assert_eq!(p.luma.len(), 32 * 16);
        assert_eq!(p.cb.len(), 16 * 8);
        assert_eq!(p.cr.len(), 16 * 8);
        assert_eq!(p.chroma_width(), 16);
        assert_eq!(p.chroma_height(), 8);
    }

    #[test]
    fn allocation_yuv422() {
        let p = Picture::new(32, 16, 2, 8, 8);
        assert_eq!(p.luma.len(), 32 * 16);
        assert_eq!(p.cb.len(), 16 * 16);
        assert_eq!(p.cr.len(), 16 * 16);
        assert_eq!(p.chroma_width(), 16);
        assert_eq!(p.chroma_height(), 16);
    }

    #[test]
    fn allocation_yuv444() {
        let p = Picture::new(32, 16, 3, 10, 10);
        assert_eq!(p.luma.len(), 32 * 16);
        assert_eq!(p.cb.len(), 32 * 16);
        assert_eq!(p.cr.len(), 32 * 16);
        assert_eq!(p.chroma_width(), 32);
        assert_eq!(p.chroma_height(), 16);
    }

    #[test]
    fn luma_set_and_get() {
        let mut p = Picture::new(16, 16, 1, 8, 8);
        p.set_luma(3, 4, 123);
        assert_eq!(p.luma_at(3, 4), 123);
        // Other samples untouched.
        assert_eq!(p.luma_at(0, 0), 0);
    }

    #[test]
    fn luma_edge_clamp() {
        let mut p = Picture::new(4, 4, 1, 8, 8);
        // Corner plant so we can detect the clamp.
        p.set_luma(0, 0, 42);
        p.set_luma(3, 3, 77);
        // Off the top-left clamps to (0, 0).
        assert_eq!(p.luma_at(-1, -1), 42);
        // Off the bottom-right clamps to (3, 3).
        assert_eq!(p.luma_at(10, 10), 77);
        // Off the top clamps to row 0.
        p.set_luma(2, 0, 55);
        assert_eq!(p.luma_at(2, -3), 55);
    }

    #[test]
    fn chroma_set_and_get() {
        let mut p = Picture::new(16, 16, 1, 8, 8);
        p.set_cb(2, 3, 10);
        p.set_cr(4, 5, 20);
        assert_eq!(p.cb_at(2, 3), 10);
        assert_eq!(p.cr_at(4, 5), 20);
    }

    #[test]
    fn chroma_edge_clamp() {
        let mut p = Picture::new(8, 8, 1, 8, 8);
        p.set_cb(0, 0, 1);
        p.set_cb(3, 3, 2);
        assert_eq!(p.cb_at(-1, -1), 1);
        assert_eq!(p.cb_at(5, 5), 2);
    }

    #[test]
    fn set_out_of_bounds_silently_ignored() {
        let mut p = Picture::new(4, 4, 1, 8, 8);
        // Should not panic.
        p.set_luma(100, 100, 42);
        p.set_luma(-1, -1, 42);
        assert_eq!(p.luma, vec![0; 16]);
    }

    #[test]
    fn eight_bit_pictures_store_bytes_and_deeper_ones_u16() {
        assert!(stores_narrow(8, 8));
        for (y, c) in [(8, 9), (9, 8), (10, 10), (8, 14), (14, 14)] {
            assert!(!stores_narrow(y, c), "{y}/{c}-bit");
        }
        assert!(matches!(
            AnyPicture::zeroed(32, 16, 1, 8, 8),
            AnyPicture::Narrow(_)
        ));
        assert!(matches!(
            AnyPicture::zeroed(32, 16, 1, 8, 10),
            AnyPicture::Wide(_)
        ));
        assert!(matches!(
            AnyPicture::zeroed(32, 16, 1, 10, 8),
            AnyPicture::Wide(_)
        ));
    }

    #[test]
    fn any_picture_keeps_its_sample_type_through_recycling_and_views() {
        let narrow = AnyPicture::zeroed(32, 16, 1, 8, 8);
        let wide = AnyPicture::zeroed(32, 16, 1, 10, 10);
        // A spare picture is reused only for the sample type the new
        // picture's bit depths store.
        assert!(narrow.has_shape(32, 16, 1, 8, 8));
        assert!(!narrow.has_shape(32, 16, 1, 10, 10));
        assert!(wide.has_shape(32, 16, 1, 10, 10));
        assert!(!wide.has_shape(32, 16, 1, 8, 8));
        // A wide picture whose recorded depths are 8-bit (as a test or an
        // encoder helper may build) is still not a narrow one's spare.
        assert!(!AnyPicture::Wide(Picture::new(32, 16, 1, 8, 8)).has_shape(32, 16, 1, 8, 8));
        assert!(matches!(
            AnyPicture::recycled(narrow.clone(), true),
            AnyPicture::Narrow(_)
        ));
        assert!(matches!(
            AnyPicture::recycled(wide.clone(), false),
            AnyPicture::Wide(_)
        ));
        assert!(matches!(narrow.field_view(true), AnyPicture::Narrow(_)));
        assert!(matches!(wide.field_view(false), AnyPicture::Wide(_)));
        // The selectors hand out only the matching variant.
        assert!(u8::picture(&narrow).is_some() && u16::picture(&narrow).is_none());
        assert!(u16::picture(&wide).is_some() && u8::picture(&wide).is_none());
    }

    #[test]
    fn narrow_and_wide_pictures_hold_the_same_samples() {
        let mut n = Picture::<u8>::zeroed(8, 8, 1, 8, 8);
        let mut w = Picture::<u16>::zeroed(8, 8, 1, 8, 8);
        for (i, v) in [0, 1, 127, 128, 254, 255].into_iter().enumerate() {
            let (x, y) = (i as i32, (i as i32 * 3) % 8);
            n.set_luma(x, y, v);
            w.set_luma(x, y, v);
            n.set_cb(y % 4, x % 4, 255 - v);
            w.set_cb(y % 4, x % 4, 255 - v);
            assert_eq!(n.luma_at(x, y), v);
            assert_eq!(n.luma_at(x, y), w.luma_at(x, y));
            assert_eq!(n.cb_at(y % 4, x % 4), w.cb_at(y % 4, x % 4));
        }
        // Clip-and-store of a row: identical values at every 8-bit max.
        let src = [-70000, -1, 0, 1, 128, 254, 255, 256, 1000, 70000];
        let mut dn = [0u8; 10];
        let mut dw = [0u16; 10];
        store_clipped(&mut dn, &src, 255);
        store_clipped(&mut dw, &src, 255);
        assert_eq!(dn.map(u16::from), dw);
        assert_eq!(dn, [0, 0, 0, 1, 128, 254, 255, 255, 255, 255]);
        // The 8-bit output pack of either type gives the same bytes.
        let (mut on, mut ow) = (Vec::new(), Vec::new());
        u8::extend_8bit(&mut on, &dn);
        u16::extend_8bit(&mut ow, &dw);
        assert_eq!(on, ow);
    }
}
