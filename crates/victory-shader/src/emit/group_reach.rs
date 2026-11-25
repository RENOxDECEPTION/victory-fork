//! The blocks every thread of a thread group reaches, for a console wave64 compute program
//! whose thread group holds several console waves, each run as two host waves of 32.
//!
//! The two host waves of a console wave exchange through LDS between group barriers, and a
//! group barrier waits for every thread of the group: a block where they meet must be one
//! that every thread of the group gets to, as often as every other, or the group stands
//! until Windows resets the card. The lowering's blocks every lane runs (no per-lane branch
//! around them) are that for a group of one console wave. With several, a branch on a scalar
//! is taken by each console wave on its own, and two console waves go different ways when the
//! scalar depends on the wave:
//!
//! - on its lanes: EXEC, a lane mask read as a number (a bit count, the first set lane), a
//!   lane's value read into a scalar, anything a vector instruction writes;
//! - on the wave itself: the thread-group info and scratch offset SGPRs, a hardware register
//!   read, a clock;
//! - on a branch that does: what a block under such a branch writes, and what a join after
//!   it takes from the way the wave came.
//!
//! The group's ids and user data are the group's, and so is what the program computes from
//! them with scalar instructions, and what it loads from scalar memory at such addresses.
//! shortcut: that a scalar load reads the same in every wave takes the console compiler's
//! word for it, which emits scalar loads for read-only data only (constants, descriptors).
//!
//! A value moved from register to register, or joined only from copies of one value
//! ([`crate::ir::copies`]), is the value it copies wherever it stands: the compiler parks the
//! scalars it needs later in other registers inside a branch and takes them back before the
//! join, which does not make them the wave's.

use std::collections::HashSet;

use crate::gfx10::{Fields, Operand};
use crate::ir::{Program, Reg, Value};
use crate::native::ComputeIo;
use crate::structure::{Atom, LaneExit, Structured};

/// Every branch (by its block) with its region: the blocks reachable from it that do not
/// post-dominate it, which not every way through the branch runs. None with an indirect exit.
pub(super) fn branch_regions(st: &Structured) -> Option<Vec<(usize, Vec<bool>)>> {
    let n = st.exits.len();
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b, e) in st.exits.iter().enumerate() {
        match *e {
            LaneExit::Jump(t) => succ[b].push(t),
            LaneExit::Branch { taken, not_taken, .. } => {
                succ[b].push(taken);
                succ[b].push(not_taken);
            }
            LaneExit::End => {}
            LaneExit::Indirect => return None,
        }
    }
    // post-dominators (bitsets; a virtual exit after every End)
    let words = n.div_ceil(64);
    let full = vec![u64::MAX; words];
    let mut pdom: Vec<Vec<u64>> = (0..n)
        .map(|b| {
            if succ[b].is_empty() {
                let mut v = vec![0; words];
                v[b / 64] |= 1 << (b % 64);
                v
            } else {
                full.clone()
            }
        })
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..n).rev() {
            if succ[b].is_empty() {
                continue;
            }
            let mut v = full.clone();
            for &t in &succ[b] {
                for w in 0..words {
                    v[w] &= pdom[t][w];
                }
            }
            v[b / 64] |= 1 << (b % 64);
            if v != pdom[b] {
                pdom[b] = v;
                changed = true;
            }
        }
    }
    let mut regions = Vec::new();
    for (d, e) in st.exits.iter().enumerate() {
        if !matches!(e, LaneExit::Branch { .. }) {
            continue;
        }
        let mut seen = vec![false; n];
        let mut stack: Vec<usize> = succ[d].clone();
        while let Some(b) = stack.pop() {
            if seen[b] || pdom[d][b / 64] >> (b % 64) & 1 == 1 {
                continue;
            }
            seen[b] = true;
            stack.extend(succ[b].iter().copied());
        }
        regions.push((d, seen));
    }
    Some(regions)
}

/// Per block of `p`: whether every console wave of a thread group runs it alike (no branch
/// around it that two console waves may take differently, see the module). None with an
/// indirect exit.
pub(super) fn group_blocks(p: &Program, st: &Structured, cs: &ComputeIo) -> Option<Vec<bool>> {
    let n = p.blocks.len();
    let regions = branch_regions(st)?;
    let copy = crate::ir::copies(p);
    let root = |v: Value| crate::ir::copy_root(&copy, v);
    let (ids, _) = cs.system_sgprs();
    let group_input = |c: u8| c < cs.user_sgprs() || ids.contains(&Some(c));
    // the values of the wave (by root), and the blocks under a branch on one
    let mut wave: HashSet<Value> = p
        .inputs
        .iter()
        .filter(|(r, _)| !matches!(*r, Reg::S(c) if group_input(c)))
        .map(|&(_, v)| v)
        .collect();
    let mut under = vec![false; n];
    let mut split = vec![false; n];
    let scalar = |r: Reg| matches!(r, Reg::S(_) | Reg::Scc);
    let alike = |i: &crate::ir::Inst| {
        let name = i.name();
        match i.fields {
            Fields::Sop { .. } => !["s_getreg", "s_get_waveid", "s_sendmsg", "s_memtime", "s_memrealtime", "s_movrel"]
                .iter()
                .any(|x| name.starts_with(x)),
            Fields::Smem { .. } => name.starts_with("s_load") || name.starts_with("s_buffer_load"),
            Fields::Sopp { .. } => true,
            _ => false,
        }
    };
    loop {
        let mut changed = false;
        let is_wave = |wave: &HashSet<Value>, v: Value| wave.contains(&root(v));
        for (b, block) in p.blocks.iter().enumerate() {
            for phi in &block.phis {
                if !scalar(phi.reg) || root(phi.value) != phi.value || wave.contains(&phi.value) {
                    continue;
                }
                let first = phi.args.first().map(|&a| root(a));
                let joins = phi.args.iter().any(|&a| Some(root(a)) != first)
                    && block.preds.iter().any(|&q| under[q] || split[q]);
                if under[b] || joins || phi.args.iter().any(|&a| is_wave(&wave, a)) {
                    changed |= wave.insert(phi.value);
                }
            }
            for inst in &block.insts {
                let of_wave = under[b] || !alike(inst) || inst.reads.iter().any(|r| is_wave(&wave, r.1));
                for w in inst.writes.iter().filter(|w| scalar(w.0)) {
                    if of_wave && root(w.1) == w.1 {
                        changed |= wave.insert(w.1);
                    }
                }
            }
        }
        for (d, region) in &regions {
            let LaneExit::Branch { when, .. } = st.exits[*d] else { continue };
            let by_wave = st.wide.contains(d)
                || st.bools.support(when).iter().any(|a| match *a {
                    Atom::Uniform(v) => is_wave(&wave, v),
                    _ => true,
                });
            if by_wave && !split[*d] {
                split[*d] = true;
                changed = true;
            }
            if split[*d] {
                // (and the branch's own block where a way through the region comes back to it:
                // the test of a loop, which a wave runs once a round)
                let to_d = |b: usize| match st.exits[b] {
                    LaneExit::Jump(t) => t == *d,
                    LaneExit::Branch { taken, not_taken, .. } => taken == *d || not_taken == *d,
                    LaneExit::End | LaneExit::Indirect => false,
                };
                let back = (0..n).any(|b| (region[b] || b == *d) && to_d(b));
                for b in 0..n {
                    if (region[b] || back && b == *d) && !under[b] {
                        under[b] = true;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    Some(under.iter().map(|u| !u).collect())
}

/// `all` (the blocks every lane of a console wave runs) narrowed to those every console wave
/// of the group runs alike, for a group of several console waves; as it is for one.
pub(super) fn narrow(all: Vec<bool>, p: &Program, st: &Structured, cs: &ComputeIo) -> Vec<bool> {
    if cs.threads.iter().product::<u32>() <= 64 {
        return all;
    }
    match group_blocks(p, st, cs) {
        Some(group) => all.iter().zip(group).map(|(&a, g)| a && g).collect(),
        None => vec![false; all.len()],
    }
}

/// First-lane reads (by the value they write) that need nothing from the other host wave,
/// where no block every thread of the group reaches is at hand to exchange in: a read under a
/// guard (some lanes of the block, e.g. the one lane a wave elects to update a shared record)
/// of a value every lane of that guard holds alike, whose result is used only under the same
/// guard, in the same block.
///
/// A host wave with a lane of the guard then reads the console's value from its own lanes.
/// One without reads something else, which nothing it runs looks at: its threads are all
/// outside the guard, the vector instructions that read the result are masked to the guard,
/// and the scalars made from it are read by nothing else (no branch, no other block, no
/// scalar memory access).
///
/// Alike in the guard's lanes: written in the block under the same guard, from scalars that
/// are not lane masks, constants and such values (a vector instruction, lane by lane), or
/// loaded from memory at an address made of them.
pub(super) fn own_first_reads(p: &Program, st: &Structured) -> HashSet<Value> {
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let mut out = HashSet::new();
    for (b, block) in p.blocks.iter().enumerate() {
        let guard = |i: usize| st.guards.get(b).and_then(|g| g.get(i)).copied().flatten();
        for (r, read) in block.insts.iter().enumerate() {
            if read.name() != "v_readfirstlane_b32" {
                continue;
            }
            let Some(g) = guard(r) else { continue };
            let Fields::Valu(v) = &read.fields else { continue };
            let Some(&Operand::Vector { reg, .. }) = v.src.first() else { continue };
            let Some(x) = read.read(Reg::V(reg)) else { continue };
            // the values alike in the guard's lanes, up to the read
            let mut alike: HashSet<Value> = HashSet::new();
            for (i, inst) in block.insts[..r].iter().enumerate() {
                if guard(i) != Some(g) {
                    continue;
                }
                let name = inst.name();
                // a value of a source register: one alike in the guard's lanes
                let of = |reg: u8| {
                    let r = inst.read(Reg::V(reg));
                    r.is_some_and(|x| alike.contains(&x))
                };
                let from = match &inst.fields {
                    // lane by lane from its listed sources only (no accumulator read from the
                    // destination, no other lane's value)
                    Fields::Valu(v)
                        if v.dpp.is_none()
                            && v.sdwa.is_none()
                            && ["v_mov_b32", "v_cvt_", "v_lshlrev_b32", "v_lshrrev_b32", "v_add_nc_u32", "v_and_b32", "v_or_b32", "v_mul_lo_u32"]
                                .iter()
                                .any(|n| name.starts_with(n)) =>
                    {
                        v.src.iter().all(|op| match *op {
                            Operand::Vector { reg, dwords: 1 } => of(reg),
                            Operand::Scalar { code, .. } => inst.read(Reg::S(code)).is_none_or(|x| !is_mask(x)),
                            Operand::Const(_) | Operand::Literal(_) => true,
                            _ => false,
                        })
                    }
                    // shortcut: the lanes of the guard load at the same address, and the
                    // memory is taken to be the same when each host wave gets there (the
                    // console compiler's elected-lane updates read a record only that lane
                    // writes)
                    Fields::Buffer { addr, .. } if name.starts_with("buffer_load") => match *addr {
                        Operand::Vector { reg, dwords } => (reg..reg.saturating_add(dwords)).all(of),
                        _ => true,
                    },
                    _ => false,
                };
                let masks = inst.reads.iter().any(|&(reg, val)| matches!(reg, Reg::S(c) if c != 126 && c != 127) && is_mask(val));
                if from && !masks {
                    alike.extend(inst.writes.iter().filter(|w| matches!(w.0, Reg::V(_))).map(|w| w.1));
                }
            }            if !alike.contains(&x) {
                continue;
            }
            // what is made from the result: scalars, and vector writes under the guard
            let mut made: HashSet<Value> = read.writes.iter().map(|w| w.1).collect();
            let mut used_ok = true;
            for (k, inst) in block.insts.iter().enumerate().skip(r + 1) {
                if !inst.reads.iter().any(|rd| made.contains(&rd.1)) {
                    continue;
                }
                let exec = inst.writes.iter().any(|w| matches!(w.0, Reg::S(126 | 127)));
                match inst.fields {
                    Fields::Sop { .. } if !exec => made.extend(inst.writes.iter().map(|w| w.1)),
                    // (not a lane write, which EXEC does not mask)
                    Fields::Valu(_) | Fields::Buffer { .. } | Fields::Image { .. }
                        if guard(k) == Some(g)
                            && !inst.name().contains("lane")
                            && inst.writes.iter().all(|w| matches!(w.0, Reg::V(_))) => {}
                    _ => {
                        used_ok = false;
                        break;
                    }
                }
            }
            let elsewhere = p.blocks.iter().enumerate().any(|(c, other)| {
                other.phis.iter().any(|phi| phi.args.iter().any(|a| made.contains(a)))
                    || c != b && other.insts.iter().any(|i| i.reads.iter().any(|rd| made.contains(&rd.1)))
            });
            if used_ok && !elsewhere {
                out.extend(read.writes.iter().map(|w| w.1));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const S_ENDPGM: u32 = 0xBF81_0000;
    const MARK: u32 = 0x7E02_0281; // v_mov_b32 v1, 1

    fn program(words: &[u32]) -> (Program, Structured) {
        let mut w = words.to_vec();
        w.push(S_ENDPGM);
        let code: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
        let p = crate::ir::build(&code, true, 256).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        (p, st)
    }

    /// 5 user SGPRs, two console waves to a group.
    const CS: ComputeIo = ComputeIo { rsrc2: 5 << 1, threads: [128, 1, 1] };

    /// Whether every console wave runs the block holding `v_mov_b32 v1, 1` alike.
    fn marked_alike(words: &[u32]) -> bool {
        let (p, st) = program(words);
        let group = group_blocks(&p, &st, &CS).expect("no indirect exit");
        let b = p.blocks.iter().position(|b| b.insts.iter().any(|i| i.machine.words[0] == MARK)).expect("the marked block");
        group[b]
    }

    #[test]
    fn a_branch_on_a_count_of_the_waves_lanes_splits_the_group() {
        // if (countbits(EXEC) != 0) { mark }
        assert!(!marked_alike(&[
            0xBE92_107E, // s_bcnt1_i32_b64 s18, exec
            0xBF07_8012, // s_cmp_lg_u32 s18, 0
            0xBF84_0001, // s_cbranch_scc0 past the mark
            MARK,
        ]));
        // if (s1 != 0) { mark }: user data, the group's
        assert!(marked_alike(&[
            0xBF07_8001, // s_cmp_lg_u32 s1, 0
            0xBF84_0001, // s_cbranch_scc0 past the mark
            MARK,
        ]));
        // if (first lane's v0 != 0) { mark }
        assert!(!marked_alike(&[
            0x7E24_0500, // v_readfirstlane_b32 s18, v0
            0xBF07_8012, // s_cmp_lg_u32 s18, 0
            0xBF84_0001, // s_cbranch_scc0 past the mark
            MARK,
        ]));
    }

    #[test]
    fn a_join_after_a_branch_of_the_wave_is_the_waves_unless_it_joins_copies() {
        // s9 = 0; if (countbits(EXEC) != 0) s9 = 1; if (s9 == 0) { mark }
        assert!(!marked_alike(&[
            0xBE89_0380, // s_mov_b32 s9, 0
            0xBE92_107E, // s_bcnt1_i32_b64 s18, exec
            0xBF07_8012, // s_cmp_lg_u32 s18, 0
            0xBF84_0001, // s_cbranch_scc0 past the next
            0xBE89_0381, // s_mov_b32 s9, 1
            0xBF06_8009, // s_cmp_eq_u32 s9, 0
            0xBF84_0001, // s_cbranch_scc0 past the mark
            MARK,
        ]));
        // s9 = s1; if (countbits(EXEC) != 0) { park s9, use the register, take it back };
        // if (s9 == 0) { mark }: s9 is s1 either way
        assert!(marked_alike(&[
            0xBE89_0301, // s_mov_b32 s9, s1
            0xBE92_107E, // s_bcnt1_i32_b64 s18, exec
            0xBF07_8012, // s_cmp_lg_u32 s18, 0
            0xBF84_0003, // s_cbranch_scc0 past the next three
            0xBE94_0309, // s_mov_b32 s20, s9
            0xBE89_0385, // s_mov_b32 s9, 5
            0xBE89_0314, // s_mov_b32 s9, s20
            0xBF06_8009, // s_cmp_eq_u32 s9, 0
            0xBF84_0001, // s_cbranch_scc0 past the mark
            MARK,
        ]));
    }

    #[test]
    fn the_test_of_a_loop_the_waves_leave_apart_is_the_waves() {
        // s9 = 0; do { mark; s9++ } while (s9 != countbits(EXEC)): the block runs once a round
        let mut w = [
            0xBE89_0380, // s_mov_b32 s9, 0
            0xBE92_107E, // s_bcnt1_i32_b64 s18, exec
            MARK,
            0x8109_8109, // s_add_i32 s9, s9, 1
            0xBF07_1209, // s_cmp_lg_u32 s9, s18
            0xBF85_FFFC, // s_cbranch_scc1 back to the mark
        ];
        assert!(!marked_alike(&w));
        // ... while (s9 != s1): user data, the same rounds in every wave
        w[4] = 0xBF07_0109; // s_cmp_lg_u32 s9, s1
        assert!(marked_alike(&w));
        // for (s9 = 0; s9 != countbits(EXEC); s9++) with the test and the mark at the head
        let mut w = [
            0xBE89_0380, // s_mov_b32 s9, 0
            0xBE92_107E, // s_bcnt1_i32_b64 s18, exec
            MARK,
            0xBF07_1209, // s_cmp_lg_u32 s9, s18
            0xBF84_0002, // s_cbranch_scc0 past the loop
            0x8109_8109, // s_add_i32 s9, s9, 1
            0xBF82_FFFB, // s_branch back to the head
        ];
        assert!(!marked_alike(&w));
        w[3] = 0xBF07_0109; // s_cmp_lg_u32 s9, s1
        assert!(marked_alike(&w));
    }

    #[test]
    fn a_group_of_one_console_wave_keeps_its_blocks() {
        let (p, st) = program(&[0xBE92_107E, 0xBF07_8012, 0xBF84_0001, MARK]);
        let one = ComputeIo { threads: [64, 1, 1], ..CS };
        let all = vec![true; p.blocks.len()];
        assert_eq!(narrow(all.clone(), &p, &st, &one), all);
        assert!(narrow(all, &p, &st, &CS).contains(&false));
    }

    /// The lane a console wave elects (its first lane left) makes v6 (`make`), reads `read`
    /// into s28 as its first lane (`v_readfirstlane_b32 s28`), sets bit 14 and keeps it in v5;
    /// `after` follows in the same block, `later` after EXEC is back.
    fn elected(make: u32, read: u32, after: &[u32], later: &[u32]) -> (Program, Structured) {
        let mut w = vec![
            0xD765_0003,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v3, -1, 0
            0xD766_0003,
            0x0002_06C1, // v_mbcnt_hi_u32_b32 v3, -1, v3
            0xBE99_147E, // s_ff1_i32_b64 s25, exec
            0xBE9A_047E, // s_mov_b64 s[26:27], exec
            0x7DA4_0619, // v_cmpx_eq_u32 exec, s25, v3
            make,
            read,
            0x881C_FF1C, // s_or_b32 s28, s28, lit
            1 << 14,
            0x7E0A_021C, // v_mov_b32 v5, s28
        ];
        w.extend_from_slice(after);
        w.push(0xBEFE_041A); // s_mov_b64 exec, s[26:27]
        w.extend_from_slice(later);
        program(&w)
    }

    fn own(p: &Program, st: &Structured) -> bool {
        let found = own_first_reads(p, st);
        let read = p.blocks.iter().flat_map(|b| &b.insts).find(|i| i.name() == "v_readfirstlane_b32").unwrap();
        found.contains(&read.writes[0].1)
    }

    #[test]
    fn the_elected_lanes_first_lane_read_needs_no_other_half() {
        const S3: u32 = 0x7E0C_0203; // v_mov_b32 v6, s3
        const OF_V6: u32 = 0x7E38_0506; // v_readfirstlane_b32 s28, v6: s3, in the elected lane
        let (p, st) = elected(S3, OF_V6, &[], &[]);
        assert!(own(&p, &st));
        // v0 differs lane by lane: the first lane's is the console's only
        let (p, st) = elected(S3, 0x7E38_0500, &[], &[]);
        assert!(!own(&p, &st));
        // v6 + s3 in place (v_add_nc_u32 v6, s3, v6): v6 was the lane's own before
        let (p, st) = elected(0x4A0C_0C03, OF_V6, &[], &[]);
        assert!(!own(&p, &st));
        // the result steers a branch (s_cmp_eq_u32 s28, 0; s_cbranch_scc1 over a move)
        let (p, st) = elected(S3, OF_V6, &[0xBF06_801C, 0xBF85_0001, 0x7E10_0280], &[]);
        assert!(!own(&p, &st));
        // ... or is read where every lane is back (v_mov_b32 v7, s28)
        let (p, st) = elected(S3, OF_V6, &[], &[0x7E0E_021C]);
        assert!(!own(&p, &st));
        // ... or written into a lane, which EXEC does not mask (v_writelane_b32 v9, s28, 0, of
        // a v9 read as a vector after: v_mov_b32 v10, v9)
        let (p, st) = elected(S3, OF_V6, &[0xD761_0009, 0x0001_001C], &[0x7E14_0309]);
        assert!(!own(&p, &st));
    }
}