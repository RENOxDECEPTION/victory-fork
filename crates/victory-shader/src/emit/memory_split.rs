//! Wide title-memory accesses split where they cross a 64 KiB page of the directory (a child
//! module of `lower`, sharing the emitter's state): each page maps to its own host view and
//! byte offset, so a raw access may only touch the part of the range in the page it looked up.

use super::{Attr, Bin, Lower, Term, V, pred};

impl<'a> Lower<'a> {
    /// Transfer up to four consecutive dwords in title memory. The directory
    /// maps adjacent guest pages independently, so a DXIL raw access may
    /// only touch the portion of the range in the page it looked up.
    pub(super) fn memory_transfer(
        &mut self,
        lo: V,
        hi: V,
        guard: Option<V>,
        count: u8,
        mut transfer: impl FnMut(&mut Self, V, V, std::ops::Range<u8>) -> Vec<V>,
    ) -> Vec<V> {
        let (handle, byte) = self.memory(lo, hi, guard);
        if count == 1 {
            return transfer(self, handle, byte, 0..count);
        }

        let head = self.cur;
        let page_mask = self.s.m.i32(0xFFFF);
        let within = self.s.m.bin(head, Bin::And, lo, page_mask);
        let limit = self.s.m.i32(0x10000 - 4 * u32::from(count));
        let fits = self.s.m.cmp(head, pred::IULE, within, limit);
        let (whole, split, join) = (
            self.s.m.new_block(),
            self.s.m.new_block(),
            self.s.m.new_block(),
        );
        self.s.m.terminate(head, Term::CondBr(fits, whole, split));
        self.cur = whole;
        let values = transfer(self, handle, byte, 0..count);
        self.s.m.terminate(whole, Term::Br(join));
        let mut incoming: Vec<_> = values.into_iter().map(|v| vec![(v, whole)]).collect();

        self.cur = split;
        // In this branch the last dword is in the next page. Look up that
        // page's start, including the carry when the low address wraps.
        let (end_lo, end_hi) = self.address_plus(lo, hi, 4 * i32::from(count - 1));
        let page_base = self.s.m.i32(!0xFFFF);
        let next_lo = self.s.m.bin(split, Bin::And, end_lo, page_base);
        let (next_handle, next_byte) = self.memory(next_lo, end_hi, guard);
        // DXIL's component mask is constant. Each split position has one
        // masked raw access on each page.
        for first in (1..count).rev() {
            let rest = if first > 1 {
                let (case, rest) = (self.s.m.new_block(), self.s.m.new_block());
                let boundary = self.s.m.i32(0x10000 - 4 * u32::from(first));
                let fits = self.s.m.cmp(self.cur, pred::IULE, within, boundary);
                self.s.m.terminate(self.cur, Term::CondBr(fits, case, rest));
                self.cur = case;
                rest
            } else {
                join
            };
            let mut values = transfer(self, handle, byte, 0..first);
            values.extend(transfer(self, next_handle, next_byte, first..count));
            for (edges, value) in incoming.iter_mut().zip(values) {
                edges.push((value, self.cur));
            }
            self.s.m.terminate(self.cur, Term::Br(join));
            self.cur = rest;
        }
        incoming
            .into_iter()
            .map(|edges| {
                let phi = self.s.m.phi(join, self.s.t.i32);
                self.s.m.set_phi(phi, edges);
                phi
            })
            .collect()
    }

    pub(super) fn memory_load_words(&mut self, handle: V, byte: V, count: u8) -> Vec<V> {
        let c = self.cur;
        let iu = self.s.m.undef(self.s.t.i32);
        let (mask, align) = (self.s.m.i8((1u8 << count) - 1), self.s.m.i32(4));
        let r = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[handle, byte, iu, mask, align],
        );
        (0..count)
            .map(|k| {
                let x = self.s.m.extract(c, r, u32::from(k));
                self.as_int(x)
            })
            .collect()
    }

    pub(super) fn memory_store_words(&mut self, handle: V, byte: V, words: &[V]) -> Vec<V> {
        let iu = self.s.m.undef(self.s.t.i32);
        let mut args = [iu; 9];
        args[0] = handle;
        args[1] = byte;
        args[3..3 + words.len()].copy_from_slice(words);
        args[7] = self.s.m.i8((1u8 << words.len()) - 1);
        args[8] = self.s.m.i32(4);
        self.s.op(
            self.cur,
            "rawBufferStore.i32",
            Attr::NoUnwind,
            self.s.t.void,
            140,
            &args,
        );
        Vec::new()
    }
}
