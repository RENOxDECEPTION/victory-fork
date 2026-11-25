//! Lane-branch regions that both read and write LDS and still keep the console's order when a
//! console wave runs as two host waves ([`super::lower`]'s `lds_meetings`, which puts group
//! barriers on such a region's edges but cannot order two accesses inside it).
//!
//! Inside the region the two host waves run unordered, where the console ran the wave's lanes in
//! step. That changes nothing when:
//! - **one lane** at most enters the region: it then runs in one host wave, in step with itself
//!   as on the console ([`one_lane`]);
//! - **its writes are atomics only**, which leave the same words in any order, and its plain
//!   reads are of words none of those atomics touch, which the region's edges have settled
//!   ([`atomics_only`]).

use std::collections::{HashMap, HashSet};

use crate::gfx10::{Fields, Operand};
use crate::ir::{Inst, Program, Reg, Value};
use crate::structure::{Atom, Bool, Structured};

/// Whether the region `blocks` keeps the console's order as two host waves (see the module).
/// `threads`: the group's size; `all`: the blocks every lane runs.
pub(super) fn order_free(p: &Program, st: &Structured, all: &[bool], blocks: &[usize], threads: [u32; 3]) -> bool {
    if blocks.iter().any(|&b| p.blocks[b].insts.iter().any(|i| i.name() == "s_barrier")) {
        return false;
    }
    one_lane(p, st, all, blocks, threads) || atomics_only(p, st, blocks)
}

/// Where each value is written: (block, instruction).
fn defs(p: &Program) -> HashMap<Value, (usize, usize)> {
    let mut def = HashMap::new();
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            for &(_, v) in &inst.writes {
                def.insert(v, (b, i));
            }
        }
    }
    def
}

/// Whether at most one lane enters the region: every block of it entered from outside holds,
/// in every lane that enters it, a compare `x == u` of a uniform `u` with an `x` that differs
/// between any two lanes of the wave (the thread's x index, moved by uniform amounts).
fn one_lane(p: &Program, st: &Structured, all: &[bool], blocks: &[usize], threads: [u32; 3]) -> bool {
    // the x index differs between the lanes of a console wave: each wave is a run of 64 flat
    // indices, inside one row when rows are whole waves
    let [x, y, z] = threads;
    if !(x % 64 == 0 || (y == 1 && z == 1)) {
        return false;
    }
    let Some(&(_, tid)) = p.inputs.iter().find(|i| i.0 == Reg::V(0)) else {
        return false;
    };
    let uni = super::lower::lane_uniform(p, st, all);
    let def = defs(p);
    // a compare whose bit is set in one lane at most
    let single = |atom: Atom| -> bool {
        let Atom::Lane(c) = atom else { return false };
        let Some(&(b, i)) = def.get(&c) else { return false };
        let inst = &p.blocks[b].insts[i];
        if !matches!(inst.name(), "v_cmp_eq_u32" | "v_cmp_eq_i32" | "v_cmpx_eq_u32" | "v_cmpx_eq_i32") {
            return false;
        }
        let Fields::Valu(valu) = &inst.fields else { return false };
        if valu.sdwa.is_some() || valu.dpp.is_some() || valu.src.len() != 2 {
            return false;
        }
        let side = |k: usize| match valu.src[k] {
            Operand::Vector { reg, .. } => {
                inst.read(Reg::V(reg)).is_some_and(|v| distinct(p, &def, &uni, tid, v, 8))
            }
            _ => false,
        };
        (side(0) && uniform(inst, &valu.src[1], &uni)) || (side(1) && uniform(inst, &valu.src[0], &uni))
    };
    let inside: HashSet<usize> = blocks.iter().copied().collect();
    blocks.iter().filter(|&&b| p.blocks[b].preds.iter().any(|q| !inside.contains(q))).all(|&b| {
        implies(st, st.reach[b], &single, &mut HashMap::new())
    })
}

/// A source operand of `inst` that is the same in every lane.
fn uniform(inst: &Inst, op: &Operand, uni: &HashSet<Value>) -> bool {
    match *op {
        Operand::Const(_) | Operand::Literal(_) | Operand::Scalar { .. } => true,
        Operand::Vector { reg, .. } => inst.read(Reg::V(reg)).is_some_and(|v| uni.contains(&v)),
        _ => false,
    }
}

/// Whether `v` differs between any two lanes: the x index `tid`, or a bijection of such a value
/// by uniform ones (add, subtract, xor; `(u << k) + v`, `u * k + v`), `depth` steps at most.
fn distinct(p: &Program, def: &HashMap<Value, (usize, usize)>, uni: &HashSet<Value>, tid: Value, v: Value, depth: u32) -> bool {
    if v == tid {
        return true;
    }
    let Some(&(b, i)) = def.get(&v) else { return false };
    let inst = &p.blocks[b].insts[i];
    let Fields::Valu(valu) = &inst.fields else { return false };
    if depth == 0 || valu.dpp.is_some() || valu.sdwa.is_some() || valu.neg != 0 || valu.abs != 0 || valu.clamp || valu.omod != 0 {
        return false;
    }
    let addend = match inst.name() {
        "v_add_nc_u32" | "v_add_u32" | "v_sub_nc_u32" | "v_subrev_nc_u32" | "v_xor_b32" => None,
        // the varying operand only as the addend
        "v_lshl_add_u32" | "v_mad_u32_u24" => Some(2),
        _ => return false,
    };
    let varying: Vec<usize> = (0..valu.src.len()).filter(|&k| !uniform(inst, &valu.src[k], uni)).collect();
    let [k] = varying[..] else { return false };
    if addend.is_some_and(|a| a != k) {
        return false;
    }
    let Operand::Vector { reg, .. } = valu.src[k] else { return false };
    inst.read(Reg::V(reg)).is_some_and(|w| distinct(p, def, uni, tid, w, depth - 1))
}

/// Whether every lane where `b` holds has some atom `ok` accepts set.
fn implies(st: &Structured, b: Bool, ok: &dyn Fn(Atom) -> bool, memo: &mut HashMap<Bool, bool>) -> bool {
    if b == Bool::FALSE {
        return true;
    }
    if b == Bool::TRUE {
        return false;
    }
    if let Some(&r) = memo.get(&b) {
        return r;
    }
    let Some((atom, lo, hi)) = st.bools.node(b) else { return false };
    // where the atom holds the lane is fine; elsewhere the rest must do
    let r = implies(st, lo, ok, memo) && (ok(atom) || implies(st, hi, ok, memo));
    memo.insert(b, r);
    r
}

/// Whether every LDS write of the region is an atomic that leaves the same word in any order,
/// and every plain read is at a constant address none of them touch. Of the atomics that
/// return the old value, only add and subtract: a lane then gets a different, still disjoint,
/// run of what it reserves. The ISA leaves the order a wave's lanes apply an LDS atomic in
/// undefined, so the console does not promise one either.
fn atomics_only(p: &Program, st: &Structured, blocks: &[usize]) -> bool {
    let def = defs(p);
    // a VGPR value a `v_mov_b32` of a constant wrote, unguarded or under the guard of `at`
    let constant = |v: Option<Value>, at: (usize, usize)| -> Option<u32> {
        let (b, i) = *def.get(&v?)?;
        let inst = &p.blocks[b].insts[i];
        if inst.name() != "v_mov_b32" {
            return None;
        }
        let guard = st.guards.get(b).and_then(|g| g.get(i)).copied().flatten();
        let same = guard.is_none() || (b == at.0 && guard == st.guards[at.0][at.1]);
        let Fields::Valu(valu) = &inst.fields else { return None };
        if !same || valu.dpp.is_some() || valu.sdwa.is_some() {
            return None;
        }
        match *valu.src.first()? {
            Operand::Const(c @ 128..=192) => Some(u32::from(c) - 128),
            Operand::Literal(x) => Some(x),
            _ => None,
        }
    };
    // byte ranges: plain reads, atomics
    let mut reads: Vec<(u32, u32)> = Vec::new();
    let mut atomics: Vec<(u32, u32)> = Vec::new();
    for &b in blocks {
        for (i, inst) in p.blocks[b].insts.iter().enumerate() {
            let name = inst.name();
            let Fields::Ds { addr, offset0, offset1, gds, .. } = &inst.fields else {
                if name.starts_with("ds_") {
                    return false;
                }
                continue;
            };
            if name.starts_with("ds_swizzle") || name.contains("permute") {
                continue;
            }
            if *gds {
                return false;
            }
            let Some(Operand::Vector { reg, .. }) = *addr else { return false };
            let Some(base) = constant(inst.read(Reg::V(reg)), (b, i)) else { return false };
            let (o0, o1) = (u32::from(*offset0), u32::from(*offset1));
            let one = base.wrapping_add(o0 | o1 << 8);
            match name {
                "ds_read_b32" => reads.push((one, 4)),
                "ds_read_b64" => reads.push((one, 8)),
                "ds_read2_b32" => reads.extend([(base.wrapping_add(o0 * 4), 4), (base.wrapping_add(o1 * 4), 4)]),
                "ds_add_u32" | "ds_sub_u32" | "ds_min_u32" | "ds_max_u32" | "ds_min_i32" | "ds_max_i32"
                | "ds_and_b32" | "ds_or_b32" | "ds_xor_b32" | "ds_add_rtn_u32" | "ds_sub_rtn_u32" => {
                    atomics.push((one, 4));
                }
                _ => return false,
            }
        }
    }
    let overlap = |a: (u32, u32), b: (u32, u32)| {
        u64::from(a.0) < u64::from(b.0) + u64::from(b.1) && u64::from(b.0) < u64::from(a.0) + u64::from(a.1)
    };
    !atomics.is_empty() && reads.iter().all(|&r| atomics.iter().all(|&a| !overlap(r, a)))
}
