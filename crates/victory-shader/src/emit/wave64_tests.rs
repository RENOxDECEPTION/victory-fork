//! A console wave64 compute program on a host of 32-lane waves: a wave reduction inside a
//! per-lane branch, with that branch taken wave-wide ([`crate::structure::structure_wide`])
//! so that the two host waves can exchange their halves inside it; and programs in the
//! paired form (one host wave for each console wave, two lanes a thread: `super::paired`),
//! whose LDS, lane reads and masks cross the halves without barriers, checked against what
//! the console's 64 lanes compute. Lowered, validated by `dxil.dll` when present, and run on
//! the host GPU when it takes waves of 32 (WARP's are 4 lanes: neither form runs there).

use std::collections::HashSet;

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// 64 threads in one console wave. Each thread t holds the bit `1 << (t >> 1)`; the threads
/// below s0 (a root constant) OR theirs over the wave inside `if (t < s0) { ... }` and store
/// the result to dword 0 of the raw UAV.
/// `skip`: the branch past the body when no thread is below s0: `s_cbranch_execz`, or
/// `s_cbranch_scc0` on the SCC `s_and_saveexec` sets ("some lane is left").
fn program(skip: u32) -> NativeProgram {
    let mut words = vec![
        0x7E0A_0281, // v_mov_b32 v5, 1
        0x2C0C_0081, // v_lshrrev_b32 v6, 1, v0
        0x340E_0B06, // v_lshlrev_b32 v7, v6, v5
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        skip | 23,   // the branch to the restore of EXEC
    ];
    words.extend_from_slice(&super::wave::WAVE_OR_V7_TO_S7);
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    ]);
    native(&words)
}

/// A wave64 compute program of 64 threads (one console wave; the thread index in v0) with 4
/// root constants in s0..s3 and a raw UAV's descriptor in s[8:11], then `body`.
fn native(body: &[u32]) -> NativeProgram {
    native_of(body, true, 64)
}

/// The same for a console of 32-lane waves when `wave64` is not set, with `threads` threads.
fn native_of(body: &[u32], wave64: bool, threads: u32) -> NativeProgram {
    native_with(body, wave64, threads, 0)
}

/// [`native_of`] with `lds` bytes of LDS (512-byte granules).
fn native_with(body: &[u32], wave64: bool, threads: u32, lds: u32) -> NativeProgram {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
    ];
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1 | lds.div_ceil(512) << 15, // 5 user SGPRs: s0..s4
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// `program()` for a host of 32-lane waves, with the branches closing `wide` taken wave-wide.
fn lowered(native: &NativeProgram, p: &ir::Program, wide: &HashSet<usize>) -> Result<HostShader, ShaderError> {
    lowered_on(native, p, wide, 32)
}

/// [`lowered`] for a host whose waves have at most `host_wave` lanes.
fn lowered_on(
    native: &NativeProgram,
    p: &ir::Program,
    wide: &HashSet<usize>,
    host_wave: u32,
) -> Result<HostShader, ShaderError> {
    let st = structure::structure_wide(p, wide).expect("structure");
    let res = resource::track(p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(p, native).expect("io::raise");
    let was = lower::TEST_HOST_WAVE.replace(Some(host_wave));
    let lowered = lower::lower(&lower::Parts {
        native,
        p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    });
    lower::TEST_HOST_WAVE.set(was);
    let lowered = lowered?;
    Ok(HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
        half_views: lowered.half_views,
    })
}

/// 64 threads in one console wave: thread t holds the 16-bit float with bits 0x3C00 + 37 t
/// (1.0 and up, with odd mantissas, so that additions round); all of them store the wave's sum
/// (`WaveActiveSum` of halves) to dword 0. With the console's sum (its tree) and the terms'
/// sum added in a row, which differs from it.
fn half_sum_program() -> (NativeProgram, u16, u16) {
    let mut words = vec![
        0x340C_0085, // v_lshlrev_b32 v6, 5, v0
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0x4A0E_0F06, // v_add_nc_u32 v7, v6, v7
        0x4A0E_0F00, // v_add_nc_u32 v7, v0, v7
        0x4A0E_0EFF,
        0x0000_3C00, // v_add_nc_u32 v7, 0x3c00, v7
    ];
    // (the wave OR's words with v_add_f16 for v_or_b32: opcode 50 for 28)
    words.extend(super::wave::WAVE_OR_V7_TO_S7.iter().map(|&w| if w >> 25 == 28 { w & 0x01FF_FFFF | 50 << 25 } else { w }));
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ]);
    // the console's sum: lanes 2k and 2k+1, then pairs, ... each addition rounded to 16 bits
    // (to nearest, ties to even); values here are normal numbers
    let wide = |h: u16| f32::from_bits((u32::from(h >> 10) + 112) << 23 | u32::from(h & 0x3FF) << 13);
    let narrow = |f: f32| {
        let b = f.to_bits();
        let h = ((b >> 23) - 112) << 10 | (b >> 13 & 0x3FF);
        let rest = b & 0x1FFF;
        (h + u32::from(rest > 0x1000 || rest == 0x1000 && h & 1 == 1)) as u16
    };
    let add = |a: u16, b: u16| narrow(wide(a) + wide(b));
    let terms: Vec<u16> = (0..64u16).map(|t| 0x3C00 + 37 * t).collect();
    let mut tree = terms.clone();
    while tree.len() > 1 {
        tree = tree.chunks(2).map(|c| add(c[0], c[1])).collect();
    }
    let in_a_row = terms.iter().copied().reduce(add).unwrap();
    (native(&words), tree[0], in_a_row)
}

#[test]
fn a_sum_of_halves_over_the_wave_is_the_consoles_tree_bit_for_bit() {
    let (native, tree, in_a_row) = half_sum_program();
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert_eq!(p.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.name() == "v_add_f16").count(), 6);
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    assert_ne!(tree, in_a_row, "these terms tell the tree from another order");
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program (waves of 32?): {e}; not run");
            return;
        }
    };
    let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(1)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    // (the high half is lane 0's term's: zero)
    assert_eq!(out[0][0], u32::from(tree), "tree {tree:#x}, in a row {in_a_row:#x}");
    eprintln!("ran on the hardware device: the sum {tree:#x} of the console's tree");
}

/// The console's 16-bit wave sum where the lanes at or past s0 have left the program (a
/// thread past the end of the data): on the console they add +0, and after each step of the
/// tree they hold their run's partial sum all the same. On a host of 64-lane waves (AMD) the
/// lanes that are here read each partner run's sum from a lane of it that is here.
#[test]
fn a_sum_of_halves_where_lanes_have_left_is_the_consoles_tree_bit_for_bit() {
    let mut words = vec![
        0x7DA8_0000, // v_cmpx_gt_u32 exec, s0, v0: the lanes below s0 stay
        0, // s_cbranch_execz to the end (filled in below)
        0x340C_0085, // v_lshlrev_b32 v6, 5, v0
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0x4A0E_0F06, // v_add_nc_u32 v7, v6, v7
        0x4A0E_0F00, // v_add_nc_u32 v7, v0, v7
        0x4A0E_0EFF,
        0x0000_3C00, // v_add_nc_u32 v7, 0x3c00, v7
    ];
    words.extend(super::wave::WAVE_OR_V7_TO_S7.iter().map(|&w| if w >> 25 == 28 { w & 0x01FF_FFFF | 50 << 25 } else { w }));
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ]);
    // past every word after it, to the program's end
    words[1] = 0xBF88_0000 | (words.len() as u32 - 2);
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered_on(&native, &p, &HashSet::new(), 64).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let wide = |h: u16| {
        if h == 0 {
            0.0
        } else {
            f32::from_bits((u32::from(h >> 10) + 112) << 23 | u32::from(h & 0x3FF) << 13)
        }
    };
    let narrow = |f: f32| {
        if f == 0.0 {
            return 0;
        }
        let b = f.to_bits();
        let h = ((b >> 23) - 112) << 10 | (b >> 13 & 0x3FF);
        let rest = b & 0x1FFF;
        (h + u32::from(rest > 0x1000 || rest == 0x1000 && h & 1 == 1)) as u16
    };
    let add = |a: u16, b: u16| narrow(wide(a) + wide(b));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = false;
    for stay in [64u32, 63, 40, 33, 32, 17, 5, 1] {
        let terms: Vec<u16> = (0..64u32)
            .map(|t| if t < stay { (0x3C00 + 37 * t) as u16 } else { 0 })
            .collect();
        let mut tree = terms.clone();
        while tree.len() > 1 {
            tree = tree.chunks(2).map(|c| add(c[0], c[1])).collect();
        }
        let Some(got) = on_the_card(&shader, [stay, 0]) else {
            return;
        };
        ran = true;
        // (the high half is lane 0's term's: zero)
        assert_eq!(got, u32::from(tree[0]), "{stay} lanes stay: got {got:#x}, the console's tree {:#x}", tree[0]);
    }
    if ran {
        eprintln!("ran on the hardware device: the console's tree with lanes gone, in every case");
    }
}

/// One `v_add_f16` whose exact result lies between two halves: 1 + 3/1024 (0x3C03) plus 2
/// (0x4000) is 3 + 1.5 steps of that range, which the console rounds up to 0x4202. The
/// host's `f32tof16` may give 0x4201 (toward zero); with `VICTORY_F16_NEAREST=1` the
/// translation gives the console's whatever the host does.
#[test]
fn a_16_bit_sum_rounds_to_nearest_when_asked_to() {
    let words = [
        0x7E0E_02FF,
        0x0000_3C03, // v_mov_b32 v7, 0x3c03
        0x7E0C_02FF,
        0x0000_4000, // v_mov_b32 v6, 0x4000
        0x6400_0D07, // v_add_f16 v0, v7, v6
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ];
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert!(p.blocks.iter().flat_map(|b| &b.insts).any(|i| i.name() == "v_add_f16"));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for nearest in [false, true] {
        let was = lower::TEST_F16_NEAREST.replace(Some(nearest));
        let shader = lowered(&native, &p, &HashSet::new());
        lower::TEST_F16_NEAREST.set(was);
        let shader = shader.expect("lower");
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            let mut copy = shader.dxil.clone();
            v.validate(&mut copy).expect("DXIL validation failed");
        }
        let Ok(host) = Host::new(false) else {
            eprintln!("no SM 6.6 hardware device; not run");
            return;
        };
        let Ok((root, pso)) = host.compute_pipeline(&shader) else {
            eprintln!("the hardware device does not take this program; not run");
            return;
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(1)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        let got = out[0][0] & 0xFFFF;
        eprintln!("v_add_f16 0x3c03 + 0x4000 on the hardware device, nearest {nearest}: {got:#x}");
        if nearest {
            assert_eq!(got, 0x4202);
        } else {
            assert!(got == 0x4201 || got == 0x4202, "{got:#x}");
        }
    }
}

#[test]
fn a_wave_or_inside_a_lane_branch_needs_the_branch_wave_wide() {
    for (skip, name) in [(0xBF88_0000, "s_cbranch_execz"), (0xBF84_0000, "s_cbranch_scc0")] {
        a_wave_or_inside(skip, name);
    }
}

fn a_wave_or_inside(skip: u32, name: &str) {
    let native = program(skip);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let branch = (0..p.blocks.len())
        .find(|&b| p.blocks[b].insts.last().is_some_and(|i| i.name() == name))
        .expect("the branch");
    // per lane, the two host waves cannot meet inside the branch
    lower::take_needs();
    let per_lane = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&per_lane, Err(ShaderError::Unsupported { what, .. }) if *what == "a wave reduction where not every lane runs"),
        "{:?}",
        per_lane.err()
    );
    // ... and the lowering names the block, around which that branch is the one to widen
    let needs = lower::take_needs();
    let st = structure::structure(&p).expect("structure");
    assert_eq!(lower::branches_around(&st, &needs), Some(vec![branch]));
    // (taken, past the body, when every lane's bit is clear)
    assert_eq!(st.wave_any[branch], Some(false));
    // wave-wide, every lane runs the reduction (EXEC stays a guard on what it does there)
    let wide = HashSet::from([branch]);
    let st = structure::structure_wide(&p, &wide).expect("structure");
    let inside = match p.blocks[branch].exit {
        ir::Exit::Branch { not_taken, .. } => not_taken,
        _ => unreachable!(),
    };
    assert!(st.guards[inside].iter().any(Option::is_some));
    let shader = lowered(&native, &p, &wide).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    // On a GPU that takes waves of 32 (two host waves for the console's one): the OR is of
    // both halves, whichever half's lanes take the branch.
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program (waves of 32?): {e}; not run");
            return;
        }
    };
    // (threads below s0 take the branch; thread t gives bit t >> 1)
    for (below, expected) in [(64u32, u32::MAX), (40, 0x000F_FFFF), (33, 0x0001_FFFF), (10, 0x1F), (1, 1)] {
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            if let (RootType::Constants, 0, 0) = (u.root, u.slot, u.dword) {
                ud[u.index as usize] = below;
            }
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(1)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        assert_eq!(out[0][0], expected, "{name}: {below} threads in the branch");
    }
    eprintln!("ran on the hardware device ({name}): the OR of both halves in every case");
}

/// Runs `shader` on the hardware device with the root constants `s0`, `s1` and gives dword 0
/// of the UAV; None (and a line) without a device that takes it.
fn on_the_card(shader: &HostShader, constants: [u32; 2]) -> Option<u32> {
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return None;
    };
    let Ok((root, pso)) = host.compute_pipeline(shader) else {
        eprintln!("the hardware device does not take this program; not run");
        return None;
    };
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if let (RootType::Constants, 0, d @ 0..=1) = (u.root, u.slot, u.dword) {
            ud[u.index as usize] = constants[d as usize];
        }
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(1)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(out[0][0])
}

/// A 32-lane console wave. A register holds a lane mask on one path (the lanes below s0)
/// and a number on the other, and is read as a number after the join, as the trace programs
/// keep their lanes' masks and their node numbers on one stack
/// (`deferred_tiled:CS_LaserLightShadowRaytracing`, s46 at 0x104, 0x75c and 0x188). The
/// mask arrives as the ballot of its lanes.
#[test]
fn a_lane_mask_joined_with_a_number_is_read_as_the_number_it_is() {
    let words = [
        0x7D88_0000, // v_cmp_gt_u32 vcc_lo, s0, v0
        0xBE94_036A, // s_mov_b32 s20, vcc_lo
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0xBF85_0001, // s_cbranch_scc1 over the next
        0xBE94_0385, // s_mov_b32 s20, 5
        0x7E00_0214, // v_mov_b32 v0, s20
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ];
    let native = native_of(&words, false, 32);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (below, other, expected) in [(32u32, 0u32, u32::MAX), (10, 0, 0x3FF), (0, 0, 0), (10, 1, 5)] {
        let Some(got) = on_the_card(&shader, [below, other]) else { return };
        assert_eq!(got, expected, "{below} lanes in the mask, s1 = {other}");
    }
    eprintln!("ran on the hardware device: the mask's bits, or the number");
}

/// A 32-lane console wave. `if (t < s0) { ... SCC = (s1 == 7) }` and a branch on SCC after
/// the join: the skip brings the SCC of `s_and_saveexec` ("some lane is left") along
/// (`rtdirectionalshadows:CS_RTShadowTrace32HQ_TiledFarOnly`, 0x360-0x39c). On the console
/// the skip is taken by the wave, so with some lane inside every lane sees the inner test.
/// A lane that skipped on its own would see "some lane is left" instead: refused per lane,
/// right with the branch taken wave-wide. Lane 31 stores what it saw.
#[test]
fn scc_of_a_mask_operation_joined_with_a_scalar_test_needs_the_skip_wave_wide() {
    let words = [
        0x7D88_0000, // v_cmp_gt_u32 vcc_lo, s0, v0
        0xBE8D_3C6A, // s_and_saveexec_b32 s13, vcc_lo
        0xBF88_0001, // s_cbranch_execz over the next
        0xBF06_8701, // s_cmp_eq_u32 s1, 7
        0xBEFE_030D, // s_mov_b32 exec_lo, s13
        0xBF85_0002, // s_cbranch_scc1 to "seven"
        0x7E0A_0281, // v_mov_b32 v5, 1
        0xBF82_0001, // s_branch over the next
        0x7E0A_0282, // seven: v_mov_b32 v5, 2
        0x7DA4_009F, // v_cmpx_eq_u32 exec_lo, 31, v0
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16 | 5 << 8, // buffer_store_dword v5, off, s[8:11], 0
    ];
    let native = native_of(&words, false, 32);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let branch = (0..p.blocks.len())
        .find(|&b| p.blocks[b].insts.last().is_some_and(|i| i.name() == "s_cbranch_execz"))
        .expect("the skip");
    lower::take_needs();
    let per_lane = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&per_lane, Err(ShaderError::Unsupported { what, .. }) if *what == "a lane mask read as data"),
        "{:?}",
        per_lane.err()
    );
    let st = structure::structure(&p).expect("structure");
    assert_eq!(lower::branches_around(&st, &lower::take_needs()), Some(vec![branch]));
    let shader = lowered(&native, &p, &HashSet::from([branch])).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // (lanes below s0 inside, s1) -> what lane 31 stores: 2 if SCC was set at the branch
    for (below, s1, expected) in [(0u32, 7u32, 1u32), (10, 7, 2), (10, 3, 1), (32, 3, 1), (32, 7, 2)] {
        let Some(got) = on_the_card(&shader, [below, s1]) else { return };
        assert_eq!(got, expected, "{below} lanes inside, s1 = {s1}");
    }
    eprintln!("ran on the hardware device: the inner test's SCC for every lane once some lane is inside");
}

/// Edge threading (`ir::thread_edges`) can resolve an unsupported register
/// error on a wave64 compute shader that still needs the 32-lane retry passes
/// (`without_skips` / `local_waves`). Calling `emit_once` after threading lets
/// those retries run instead of failing or emitting `WaveSize(64)`.
#[test]
fn threaded_wave64_compute_still_runs_wave32_retries() {
    // Uniform triangle on s0 == 0:
    //   fork: v1 = 0; if (s0 == 0) jump join (ttmp0 uninitialized on this edge)
    //   arm:  ttmp0 = 5; v1 = 1; fallthrough join
    // join: if (v1 == 0) branch S_ENDPGM
    // body: if (t < s1) { v_readlane_b32 s7, v0, 0 } (needs without_skips on wave32);
    //       store ttmp0 + s7 to UAV.
    let words = [
        0x7E02_0280, // 0: v_mov_b32 v1, 0
        0xBF06_8000, // 1: s_cmp_eq_u32 s0, 0
        0xBF85_0002, // 2: s_cbranch_scc1 +2 -> word 5 (join)
        0xBEEC_0385, // 3: s_mov_b32 ttmp0 (108), 5
        0x7E02_0281, // 4: v_mov_b32 v1, 1
        0x7D84_0280, // 5: join: v_cmp_eq_u32 vcc, 0, v1
        0xBF87_000B, // 6: s_cbranch_vccnz +11 -> S_ENDPGM (word 18)
        0xBE87_0380, // 7: body: s_mov_b32 s7, 0
        0x7D88_0001, // 8: v_cmp_gt_u32 vcc, s1, v0
        0xBE8C_246A, // 9: s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0002, // 10: s_cbranch_execz +2 -> word 13
        0xD760_0007, // 11: v_readlane_b32 s7, v0, 0 (2 dwords)
        0x0001_0100, // 12:
        0xBEFE_040C, // 13: s_mov_b64 exec, s[12:13]
        0x8006_076C, // 14: s_add_u32 s6, ttmp0 (108), s7
        0x7E04_0206, // 15: v_mov_b32 v2, s6
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16 | 2 << 8, // 16-17: buffer_store_dword v2, off, s[8:11], 0
    ];
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let edges = super::threading::constant_edges(&p, &st);
    assert!(!edges.is_empty(), "test program must trigger constant-edge threading");
    let io = io::raise(&p, &native).expect("io");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("res");
    let empty_bytes = crate::dxbc::build(&[]);
    let empty = crate::Container::parse(&empty_bytes).unwrap();
    // Confirm the threaded program alone still requires the wave32 retry path.
    let tp = ir::thread_edges(&p, &edges).expect("thread_edges");
    let tst = structure::structure(&tp).expect("threaded structure");
    let tio = io::raise(&tp, &native).expect("threaded io");
    let tres = resource::track(&tp, &native.layout, native.stage, Some(&tst)).expect("threaded res");
    let (plain_threaded, shader) = lower::TEST_HOST_WAVE.with(|cell| {
        let was = cell.replace(Some(32));
        let plain = super::emit_program(&empty, &native, &tp, &tst, &tio, &tres);
        let full = super::emit(&empty, &native, &p, &st, &io, &res);
        cell.set(was);
        (plain, full)
    });
    assert!(
        matches!(
            &plain_threaded,
            Ok((_, Some(64))) | Err(ShaderError::Unsupported { .. })
        ),
        "expected plain threaded emit_program to need wave32 retries, got {:?}",
        plain_threaded.as_ref().map(|(_, w)| w)
    );
    let shader = shader.expect("threaded wave64 compute shader should succeed via wave32 retries");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
}


/// A 64-lane `v_add_f64` whole-wave sum lowered on a 32-lane host:
/// - for a group of 32 threads (`e.1 == true`), one host wave holds the whole group so no
///   cross-half exchange is needed;
/// - inside a per-lane branch under `with_local_waves_any` (`self.local == true`), each
///   host wave takes its own 32-lane sum instead of failing with
///   `"a wave reduction where not every lane runs"`.
#[test]
fn wave64_f64_sum_on_wave32_host_respects_small_groups_and_local_waves() {
    let vop1 = |op: u16, vdst: u8, src0: u16| {
        (0x3Fu32 << 25) | (u32::from(vdst) << 17) | (u32::from(op) << 9) | u32::from(src0)
    };
    let vop3 = |op: u16, vdst: u8, src0: u16, src1: u16, src2: u16| {
        [
            (0x35u32 << 26) | (u32::from(op) << 16) | u32::from(vdst),
            u32::from(src0) | (u32::from(src1) << 9) | (u32::from(src2) << 18),
        ]
    };
    let v_mov_dpp8 = |vdst: u8, vsrc: u8, sel: u32| [vop1(0x001, vdst, 0xE9), u32::from(vsrc) | (sel << 8)];
    let v_mov_row_mirror = |vdst: u8, vsrc: u8| {
        [
            vop1(0x001, vdst, 0xFA),
            u32::from(vsrc) | (0x140 << 8) | (1 << 19) | (0xF << 24) | (0xF << 28),
        ]
    };
    let vgpr = |n: u8| 256u16 + u16::from(n);
    let sgpr = |n: u8| u16::from(n);

    // Whole-wave f64 sum of per-lane (s0, s1) inside `if (v0 < s2)`.
    let mut words = vec![
        vop1(0x001, 6, sgpr(0)), // v_mov_b32 v6, s0
        vop1(0x001, 7, sgpr(1)), // v_mov_b32 v7, s1
        0x7D88_0002,             // v_cmp_gt_u32 vcc, s2, v0
        0xBE92_246A,             // s_and_saveexec_b64 s[18:19], vcc
    ];
    let skip_at = words.len();
    words.push(0xBF88_0000); // s_cbranch_execz (patched below)
    let body_start = words.len();
    words.push(0xBE8C_25C1); // s_or_saveexec_b64 s[12:13], -1
    words.extend_from_slice(&vop3(0x101, 2, 0x80, vgpr(6), sgpr(12))); // v_cndmask_b32 v2, 0, v6, s[12:13]
    words.extend_from_slice(&vop3(0x101, 3, 0x80, vgpr(7), sgpr(12))); // v_cndmask_b32 v3, 0, v7, s[12:13]
    for sel in [14570689u32, 9658386, 2340] {
        words.extend_from_slice(&v_mov_dpp8(4, 2, sel));
        words.extend_from_slice(&v_mov_dpp8(5, 3, sel));
        words.extend_from_slice(&vop3(0x164, 2, vgpr(4), vgpr(2), 0)); // v_add_f64
    }
    words.extend_from_slice(&v_mov_row_mirror(4, 2));
    words.extend_from_slice(&v_mov_row_mirror(5, 3));
    words.extend_from_slice(&vop3(0x164, 2, vgpr(4), vgpr(2), 0));
    words.extend_from_slice(&vop3(0x378, 4, vgpr(2), 0x80, 0x80)); // v_permlanex16_b32
    words.extend_from_slice(&vop3(0x378, 5, vgpr(3), 0x80, 0x80));
    words.extend_from_slice(&vop3(0x164, 2, vgpr(4), vgpr(2), 0));
    words.extend_from_slice(&vop3(0x360, 14, vgpr(2), 0x80, 0)); // v_readlane_b32 s14, v2, 0
    words.extend_from_slice(&vop3(0x360, 15, vgpr(3), 0x80, 0)); // v_readlane_b32 s15, v3, 0
    words.extend_from_slice(&vop3(0x164, 2, sgpr(14), vgpr(2), 0)); // v_add_f64 v[2:3], s[14:15], v[2:3]
    words.extend_from_slice(&vop3(0x360, 16, vgpr(2), 0xA0, 0)); // v_readlane_b32 s16, v2, 32
    words.extend_from_slice(&vop3(0x360, 17, vgpr(3), 0xA0, 0)); // v_readlane_b32 s17, v3, 32
    words.push(0xBEFE_040C); // s_mov_b64 exec, s[12:13]
    words.push(vop1(0x00F, 0, sgpr(16))); // v_cvt_f32_f64 v0, s[16:17]
    words.extend_from_slice(&[0x38 << 26 | 0x1C << 18, 0x80 << 24 | 2 << 16]); // buffer_store_dword v0, off, s[8:11], 0
    let body_words = (words.len() - body_start) as u32;
    words[skip_at] = 0xBF88_0000 | body_words;
    words.push(0xBEFE_0412); // s_mov_b64 exec, s[18:19]

    // 1. 64-thread group: fails without local waves, succeeds under `with_local_waves_any`.
    let n64 = native_of(&words, true, 64);
    let p64 = ir::build(&n64.code, n64.wave64, n64.vgprs).expect("ir::build");
    let err = lowered(&n64, &p64, &HashSet::new());
    assert!(
        matches!(&err, Err(ShaderError::Unsupported { what, .. }) if *what == "a wave reduction where not every lane runs"),
        "expected divergent reduction error without local waves, got {:?}",
        err.err(),
    );
    let shader64 = lower::with_local_waves_any(|| lowered(&n64, &p64, &HashSet::new())).expect("local waves f64 sum");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader64.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed (64-thread local)");
    }

    // 2. 32-thread group (wave64 on wave32 host, `e.1 == true`): no other half to exchange with.
    let n32 = native_of(&words, true, 32);
    let p32 = ir::build(&n32.code, n32.wave64, n32.vgprs).expect("ir::build");
    let shader32 = lowered(&n32, &p32, &HashSet::new()).expect("32-thread wave64 f64 sum");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader32.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed (32-thread)");
    }

    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else { return };
    let Ok((root, pso)) = host.compute_pipeline(&shader32) else { return };
    let b = 1.25f64.to_bits();
    let inputs = [b as u32, (b >> 32) as u32, 4u32, 0u32];
    let mut ud = vec![0u32; shader32.user_data_dwords().max(1) as usize];
    for u in &shader32.user_data {
        ud[u.index as usize] = match (u.root, u.slot) {
            (RootType::Constants, 0) => inputs[u.dword as usize],
            (RootType::Table, 1) => 0,
            other => panic!("unexpected user data {other:?}"),
        };
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(1)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .expect("dispatch");
    assert_eq!(out[0][0], 5.0f32.to_bits(), "32-thread wave64 f64 sum of 4 lanes * 1.25");
}

/// One console wave of 64 threads with 512 bytes of LDS: thread 0's lane zeroes the flag at
/// LDS 0, the threads below s0 raise it (`ds_write_b32` of 1) and read it back inside their
/// branch, and every thread reads it after the branch and stores it to the UAV. `second`: a
/// second write in the branch after the read, of that constant.
fn flag_program(second: Option<u32>) -> NativeProgram {
    let mut body = vec![
        0x7E12_0280, // v_mov_b32 v9, 0
    ];
    body.extend_from_slice(&ds(DS_WRITE_B32, 0, 9, 9)); // ds_write_b32 v9, v9
    let mut branch = vec![0x7E04_0281]; // v_mov_b32 v2, 1
    branch.extend_from_slice(&ds(DS_WRITE_B32, 0, 9, 2)); // ds_write_b32 v9, v2
    // (a scalar exit from the branch makes it blocks of their own, which only its lanes run)
    let mut rest = ds(DS_READ_B32, 3, 9, 0).to_vec(); // ds_read_b32 v3, v9
    if let Some(k) = second {
        rest.push(0x7E0C_02FF); // v_mov_b32 v6, lit
        rest.push(k);
        rest.extend_from_slice(&ds(DS_WRITE_B32, 0, 9, 6)); // ds_write_b32 v9, v6
    }
    branch.extend_from_slice(&[
        0xBF07_8001,                       // s_cmp_lg_u32 s1, 0
        0xBF85_0000 | rest.len() as u32, // s_cbranch_scc1 past the rest
    ]);
    branch.extend_from_slice(&rest);
    body.extend_from_slice(&[
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0000 | branch.len() as u32, // s_cbranch_execz past the branch
    ]);
    body.extend_from_slice(&branch);
    body.push(0xBEFE_040C); // s_mov_b64 exec, s[12:13]
    body.extend_from_slice(&ds(DS_READ_B32, 5, 9, 0)); // ds_read_b32 v5, v9
    body.extend_from_slice(&[
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16 | 5 << 8, // buffer_store_dword v5, off, s[8:11], 0
    ]);
    let mut native = native(&body);
    if let StageIo::Cs(cs) = &mut native.io {
        cs.rsrc2 |= 1 << 15; // 512 bytes of LDS
    }
    native
}

/// DS `op vdst, addr, data0` at offset 0.
const fn ds(op: u8, vdst: u8, addr: u8, data0: u8) -> [u32; 2] {
    [0x36 << 26 | (op as u32) << 18, (vdst as u32) << 24 | (data0 as u32) << 8 | addr as u32]
}

const DS_WRITE_B32: u8 = 0x0D;
const DS_READ_B32: u8 = 0x36;

/// The wave size `native` lowers to on a host of 32-lane waves.
fn flag_wave(native: &NativeProgram) -> Option<u32> {
    let p = crate::spill::build(native).expect("build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, native).expect("io::raise");
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let lowered = lower::lower(&lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    });
    lower::TEST_HOST_WAVE.set(was);
    lowered.expect("lower").wave_size
}

#[test]
fn a_branch_that_only_raises_an_lds_flag_runs_in_waves_of_32() {
    // a read that misses the other host wave's write changes nothing the branch leaves behind
    assert_eq!(flag_wave(&flag_program(None)), Some(32));
    // after the read the branch writes the flag again, with the same constant
    assert_eq!(flag_wave(&flag_program(Some(1))), Some(32));
}

#[test]
fn a_branch_that_writes_lds_two_different_values_keeps_64_lanes() {
    // a second value after the read: what the flag ends up holding depends on what was read
    assert_eq!(flag_wave(&flag_program(Some(2))), Some(64));
}

/// A block-wide scan over the console waves of a 512-thread group, the shape of the
/// ray-tracing structure builder's scans: lane 63 of each console wave stores the wave's
/// total (64) to shared memory, the group meets at `s_barrier`, and each lane then sums the
/// totals of the waves below its own in a loop that its whole console wave leaves together
/// (its exit compares the wave's number), carrying the sum in a scalar through
/// `v_readfirstlane` inside the loop, where only some lanes of the group run. On a host of
/// 32-lane waves that first-lane read needs no exchange: the value is a scalar plus a
/// shared-memory read at a scalar address whose memory has settled, the same in both halves
/// of a console wave (`lane_uniform`, `lds_settled`). Every thread stores the sum,
/// 64 x its wave's number.
#[test]
fn a_scan_over_console_waves_sums_each_waves_predecessors() {
    const fn ds(op: u32, vdst: u32, addr: u32, data0: u32) -> [u32; 2] {
        [0x36 << 26 | op << 18, vdst << 24 | data0 << 8 | addr]
    }
    let mut words = vec![
        0x2C0E_0086, // v_lshrrev_b32 v7, 6, v0: the console wave's number
        0x3410_0E82, // v_lshlrev_b32 v8, 2, v7: its total's address
        0x7E0C_02C0, // v_mov_b32 v6, 64: the total
        0xD765_000F,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v15, -1, 0
        0xD766_000F,
        0x0002_1EC1, // v_mbcnt_hi_u32_b32 v15, -1, v15: the lane in the console wave
        0xBE96_047E, // s_mov_b64 s[22:23], exec
        0x7DA4_1EBF, // v_cmpx_eq_u32 exec, 63, v15
    ];
    words.extend(ds(0x0D, 0, 8, 6)); // ds_write_b32 v8, v6
    words.extend_from_slice(&[
        0xBEFE_0416, // s_mov_b64 exec, s[22:23]
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBF8A_0000, // s_barrier
        0xBE94_047E, // s_mov_b64 s[20:21], exec
        0x7E18_0280, // v_mov_b32 v12, 0: the sum (wave 0 adds nothing)
        0x7DAA_0E80, // v_cmpx_ne_u32 exec, 0, v7: the waves above the first
        0xBF88_0000, // s_cbranch_execz (the end, patched below)
        0xBE98_047E, // s_mov_b64 s[24:25], exec: the lanes still looping
        0xBE90_0380, // s_mov_b32 s16, 0: the waves added
        0xBE92_0380, // s_mov_b32 s18, 0: the sum so far
        0xBF82_0000, // s_branch (the body, patched below)
    ]);
    let skip = words.len() - 5;
    let enter = words.len() - 1;
    let latch = words.len();
    words.extend_from_slice(&[
        0x87FE_181A, // s_and_b64 exec, s[26:27], s[24:25]: the lanes still looping
        0x7E24_050C, // v_readfirstlane_b32 s18, v12: the sum so far
    ]);
    let body = words.len();
    words.extend_from_slice(&[
        0x8F11_8210, // s_lshl_b32 s17, s16, 2
        0x8110_8110, // s_add_i32 s16, s16, 1
        0x7E14_0211, // v_mov_b32 v10, s17
    ]);
    words.extend(ds(0x36, 11, 10, 0)); // ds_read_b32 v11, v10
    words.extend_from_slice(&[
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x4A18_1612, // v_add_nc_u32 v12, s18, v11
        0xBE9A_047E, // s_mov_b64 s[26:27], exec
        0x7DA4_0E10, // v_cmpx_eq_u32 exec, s16, v7: the lanes whose wave's count is reached
        0x8A98_7E18, // s_andn2_b64 s[24:25], s[24:25], exec: leave
        0xBF85_0000, // s_cbranch_scc1 (the latch, patched below)
    ]);
    let again = words.len() - 1;
    let end = words.len();
    words.extend_from_slice(&[
        0xBEFE_0414, // s_mov_b64 exec, s[20:21]
        0x341C_0082, // v_lshlrev_b32 v14, 2, v0
        0xE070_1000,
        0x8002_0C0E, // buffer_store_dword v12, v14, s[8:11], 0 offen
    ]);
    // branch offsets in dwords from the instruction after the branch
    let rel = |from: usize, to: usize| (to as i32 - (from as i32 + 1)) as u16 as u32;
    words[skip] |= rel(skip, end);
    words[enter] |= rel(enter, body);
    words[again] |= rel(again, latch);
    let mut native = native_of(&words, true, 512);
    if let StageIo::Cs(cs) = &mut native.io {
        cs.rsrc2 |= 1 << 15; // 512 bytes of shared memory
    }
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(x) => x,
        Err(e) => panic!("the hardware device refuses the scan: {e}"),
    };
    let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(512)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    let want: Vec<u32> = (0..512u32).map(|t| 64 * (t >> 6)).collect();
    let wrong: Vec<(usize, u32, u32)> = out[0]
        .iter()
        .zip(&want)
        .enumerate()
        .filter(|(_, (got, want))| got != want)
        .map(|(t, (&got, &want))| (t, got, want))
        .take(8)
        .collect();
    assert!(wrong.is_empty(), "threads (index, got, want): {wrong:?}");
    eprintln!("ran on the hardware device: every thread has its waves' predecessors' sum");
}

/// One console wave of 64 threads with 512 bytes of LDS and two lane-branch regions that both
/// read and write it ([`super::lds_region`]):
/// - thread 0 alone (`v0 == 0`) writes 16 to LDS 0, reads it back, and writes 21 to LDS 4
///   and 0 to the counter at LDS 8;
/// - the threads below s0 read LDS 4 and take a slot of the counter (`ds_add_rtn_u32`).
///
/// Every thread then stores what it read (dword t), its slot (64 + t) and the counter after
/// both regions (128 + t). `read_at`: the byte the second region reads (4; 8 is the counter).
fn lds_regions_program(read_at: u32) -> NativeProgram {
    let at = |w: [u32; 2], offset: u32| [w[0] | offset, w[1]];
    let mut words = vec![
        0x7E12_0280, // v_mov_b32 v9, 0
        0x7E08_0280, // v_mov_b32 v4, 0
        0x7E0A_02C1, // v_mov_b32 v5, -1
        0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_000A, // s_cbranch_execz past the 10 dwords of the branch
        0x7E04_0290, // v_mov_b32 v2, 16
    ];
    words.extend_from_slice(&ds(DS_WRITE_B32, 0, 9, 2)); // ds_write_b32 v9, v2
    words.extend_from_slice(&ds(DS_READ_B32, 3, 9, 0)); // ds_read_b32 v3, v9
    words.push(0x4A06_0685); // v_add_nc_u32 v3, 5, v3
    words.extend_from_slice(&at(ds(DS_WRITE_B32, 0, 9, 3), 4)); // ds_write_b32 v9, v3 offset:4
    words.extend_from_slice(&at(ds(DS_WRITE_B32, 0, 9, 9), 8)); // ds_write_b32 v9, v9 offset:8
    words.extend_from_slice(&[
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0005, // s_cbranch_execz past the 5 dwords of the branch
    ]);
    words.extend_from_slice(&at(ds(DS_READ_B32, 4, 9, 0), read_at)); // ds_read_b32 v4, v9 offset:4
    words.push(0x7E0E_0281); // v_mov_b32 v7, 1
    words.extend_from_slice(&at(ds(DS_ADD_RTN_U32, 5, 9, 7), 8)); // ds_add_rtn_u32 v5, v9, v7 offset:8
    words.push(0xBEFE_040C); // s_mov_b64 exec, s[12:13]
    words.extend_from_slice(&at(ds(DS_READ_B32, 6, 9, 0), 8)); // ds_read_b32 v6, v9 offset:8
    words.extend_from_slice(&[
        0x341C_0082, // v_lshlrev_b32 v14, 2, v0
        0xE070_1000,
        0x8002_040E, // buffer_store_dword v4, v14, s[8:11], 0 offen
        0xE070_1100,
        0x8002_050E, // buffer_store_dword v5, v14, s[8:11], 0 offen offset:256
        0xE070_1200,
        0x8002_060E, // buffer_store_dword v6, v14, s[8:11], 0 offen offset:512
    ]);
    let mut native = native(&words);
    if let StageIo::Cs(cs) = &mut native.io {
        cs.rsrc2 |= 1 << 15; // 512 bytes of LDS
    }
    native
}

const DS_ADD_RTN_U32: u8 = 0x20;

#[test]
fn lds_regions_of_one_lane_or_atomics_run_in_waves_of_32() {
    let native = lds_regions_program(4);
    assert_eq!(flag_wave(&native), Some(32));
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(x) => x,
        Err(e) => panic!("the hardware device refuses the program: {e}"),
    };
    let below = 50u32;
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if let (RootType::Constants, 0, 0) = (u.root, u.slot, u.dword) {
            ud[u.index as usize] = below;
        }
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(192)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    let out = &out[0];
    let read: Vec<u32> = (0..64).map(|t| if t < below { 21 } else { 0 }).collect();
    assert_eq!(&out[..64], &read[..], "what each thread read of LDS 4");
    let mut slots: Vec<u32> = out[64..64 + below as usize].to_vec();
    slots.sort_unstable();
    assert_eq!(slots, (0..below).collect::<Vec<_>>(), "the slots the threads below s0 took");
    assert!(out[64 + below as usize..128].iter().all(|&s| s == u32::MAX), "the others took none");
    assert!(out[128..192].iter().all(|&n| n == below), "the counter after both regions: {:?}", &out[128..192]);
    eprintln!("ran on the hardware device: every thread read 21 and took one slot of {below}");
}

#[test]
fn an_lds_region_reading_what_its_atomics_change_keeps_64_lanes() {
    // the threads below s0 read the counter they add to: what each reads depends on which
    // host wave added first
    assert_eq!(flag_wave(&lds_regions_program(8)), Some(64));
}

/// The thread group of two console waves (128 threads, the thread index in v0) that the
/// title's large tiled programs have: an early-out too far for a branch (`s_cbranch_execnz`
/// over `s_getpc_b64` / `s_add_u32` / `s_addc_u32` / `s_setpc_b64` to the end) for the threads
/// at or past s0; then a test of whether every lane of the console wave left has `t != s1`
/// (a bit count of the lanes that do not), which steers a branch the console wave takes on
/// its own (`s_cbranch_vccz` on that count broadcast as a mask); then, if some lane of the
/// console wave has `t == s2`, its first lane alone sets bit 14 of dword s3 (a load, a
/// first-lane read of it, the store).
///
/// The two console waves of a group go different ways at both branches, and one of them may
/// have no lane left at all, so a group barrier inside either would wait for threads that
/// never come. The exchanges stay where every thread of the group gets to (the count, the
/// test for some lane), and the first-lane read inside needs none (`group_reach`). Each
/// thread below s0 stores 1 where its console wave passed the test, else 2.
/// `read`: the first-lane read (`v_readfirstlane_b32 s28, v5`, the loaded dword).
fn two_wave_program(read: u32) -> NativeProgram {
    let mut w: Vec<u32> = vec![
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0: the threads below s0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF89_0000, // s_cbranch_execnz to the body (patched)
        0xBE8E_1F00, // s_getpc_b64 s[14:15]
        0x800E_FF0E, // s_add_u32 s14, s14, lit (patched): the end
        0,
        0x820F_800F, // s_addc_u32 s15, s15, 0
        0xBE80_200E, // s_setpc_b64 s[14:15]
    ];
    let enter = 2;
    let getpc = 3;
    let far = 5;
    let body = w.len();
    w.extend_from_slice(&[
        0x7D8A_0001, // v_cmp_ne_u32 vcc, s1, v0
        0x8790_7E6A, // s_and_b64 s[16:17], vcc, exec
        0x8990_7E10, // s_xor_b64 s[16:17], s[16:17], exec: the lanes left with t == s1
        0xBE92_1010, // s_bcnt1_i32_b64 s18, s[16:17]
        0x8514_8180, // s_cselect_b32 s20, 0, 1: 1 when there are none
        0x7E04_0280, // v_mov_b32 v2, 0
        0x7D8A_0414, // v_cmp_ne_u32 vcc, s20, v2
        0xBE96_046A, // s_mov_b64 s[22:23], vcc
        0x87EA_167E, // s_and_b64 vcc, exec, s[22:23]
        0xBF86_0000, // s_cbranch_vccz to the other arm (patched)
        0x7E02_0281, // v_mov_b32 v1, 1
        0xBF82_0000, // s_branch to the join (patched)
    ]);
    let vccz = w.len() - 3;
    let over = w.len() - 1;
    let other = w.len();
    w.push(0x7E02_0282); // v_mov_b32 v1, 2
    let join = w.len();
    w.extend_from_slice(&[
        0x341C_0082, // v_lshlrev_b32 v14, 2, v0
        0xE070_1000,
        0x8002_010E, // buffer_store_dword v1, v14, s[8:11], 0 offen
        0x7D84_0002, // v_cmp_eq_u32 vcc, s2, v0
        0xBE98_106A, // s_bcnt1_i32_b64 s24, vcc: SCC, some lane left has t == s2
        0xBF84_0000, // s_cbranch_scc0 past the update (patched)
        0xD765_0003,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v3, -1, 0
        0xD766_0003,
        0x0002_06C1, // v_mbcnt_hi_u32_b32 v3, -1, v3: the lane
        0xBE99_147E, // s_ff1_i32_b64 s25, exec: the first lane left
        0xBE9A_047E, // s_mov_b64 s[26:27], exec
        0x7DA4_0619, // v_cmpx_eq_u32 exec, s25, v3
        0xBF88_0000, // s_cbranch_execz past the update (patched)
        0x7E0C_0203, // v_mov_b32 v6, s3
        0x340C_0C82, // v_lshlrev_b32 v6, 2, v6
        0xE030_1000,
        0x8002_0506, // buffer_load_dword v5, v6, s[8:11], 0 offen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        read,
        0x881C_FF1C, // s_or_b32 s28, s28, lit
        1 << 14,
        0x7E0A_021C, // v_mov_b32 v5, s28
        0xE070_1000,
        0x8002_0506, // buffer_store_dword v5, v6, s[8:11], 0 offen
    ]);
    let scc0 = join + 5;
    let execz = join + 13;
    let restore = w.len();
    w.push(0xBEFE_041A); // s_mov_b64 exec, s[26:27]
    let past = w.len();
    w.push(0xBEFE_040C); // s_mov_b64 exec, s[12:13]
    // branch offsets in dwords from the instruction after the branch
    let rel = |from: usize, to: usize| (to as i32 - (from as i32 + 1)) as u16 as u32;
    w[enter] |= rel(enter, body);
    w[vccz] |= rel(vccz, other);
    w[over] |= rel(over, join);
    w[scc0] |= rel(scc0, past);
    w[execz] |= rel(execz, restore);
    // the end: the `s_endpgm` after the body (3 prologue dwords before `w`), from the
    // address after `s_getpc_b64`
    let end = 3 + w.len();
    w[far] = (4 * (end - (3 + getpc + 1))) as u32;
    native_of(&w, true, 128)
}

#[test]
fn barriers_stay_where_every_console_wave_of_the_group_gets_to() {
    const OF_V5: u32 = 0x7E38_0505; // v_readfirstlane_b32 s28, v5
    let native = two_wave_program(OF_V5);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let names: Vec<&str> = p.blocks.iter().flat_map(|b| &b.insts).map(|i| i.name()).collect();
    for n in ["s_setpc_b64", "s_cbranch_execnz", "s_bcnt1_i32_b64", "s_cbranch_vccz", "v_readfirstlane_b32", "buffer_load_dword"] {
        assert!(names.contains(&n), "{n} in {names:?}");
    }
    // the far jump is a jump to the end, which the early-out goes to
    assert!(p.blocks.iter().any(|b| b.insts.last().is_some_and(|i| i.name() == "s_setpc_b64")
        && matches!(b.exit, ir::Exit::Jump(t) if p.blocks[t].insts.iter().all(|i| i.name() == "s_endpgm"))));
    // As it is, the program's every exchange is inside the early-out's lane branch.
    let refused = lowered(&native, &p, &HashSet::new()).expect_err("exchanges under the early-out");
    assert!(
        matches!(refused, ShaderError::Unsupported { what, .. } if lower::needs_every_lane(what)),
        "{refused:?}"
    );
    // Without the early-out and the skip around the update, it runs in waves of 32. Not with a
    // first-lane read of a value that differs lane by lane (v_readfirstlane_b32 s28, v3): that
    // one needs the other host wave, through a group barrier, under a branch the two console
    // waves may take differently.
    let other = two_wave_program(0x7E38_0503);
    let q = ir::build(&other.code, other.wave64, other.vgprs).expect("ir::build");
    let q = ir::without_skips(&q, true).expect("the early-out and the skip left out");
    let refused = lowered(&other, &q, &HashSet::new()).expect_err("a barrier under a branch of one console wave");
    assert!(
        matches!(refused, ShaderError::Unsupported { what, .. } if what == "a first-lane read where not every lane runs"),
        "{refused:?}"
    );
    let p = ir::without_skips(&p, true).expect("the early-out and the skip left out");
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program (waves of 32?): {e}; not run");
            return;
        }
    };
    const RECORD: u32 = 200;
    // (s0, s1, s2): both console waves pass, one fails, one has no lane left, the update
    // by one console wave, by none
    for (s0, s1, s2) in [(128, 200, 5), (128, 100, 5), (50, 10, 77), (100, 99, 99), (70, 200, 3), (0, 0, 0)] {
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            if let (RootType::Constants, 0, d @ 0..=3) = (u.root, u.slot, u.dword) {
                ud[u.index as usize] = [s0, s1, s2, RECORD][d as usize];
            }
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(256)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        let want: Vec<u32> = (0..256u32)
            .map(|t| {
                if t == RECORD {
                    if s2 < s0.min(128) { 1 << 14 } else { 0 }
                } else if t < s0.min(128) {
                    let wave = t >> 6;
                    let fails = s1 < s0 && s1 >> 6 == wave;
                    if fails { 2 } else { 1 }
                } else {
                    0
                }
            })
            .collect();
        let wrong: Vec<(usize, u32, u32)> = out[0]
            .iter()
            .zip(&want)
            .enumerate()
            .filter(|(_, (got, want))| got != want)
            .map(|(t, (&got, &want))| (t, got, want))
            .take(8)
            .collect();
        assert!(wrong.is_empty(), "s0 {s0}, s1 {s1}, s2 {s2}: threads (index, got, want): {wrong:?}");
    }
    eprintln!("ran on the hardware device: every thread's console wave decided as the console does");
}

/// One console wave of 64 threads, thread t holding t + 1. A branch on s0 (the wave's) picks
/// `WaveActiveMax` (s0 not 0) or `WaveActiveMin`, and the two reductions share their end:
/// each arm saves EXEC and reduces up to the read of lane 32, the join reads lane 0 and
/// restores EXEC (`deferred_tiled:CS_WaterTransmittanceOpticalDepthProcessing`, 0xb3c-0xbe0).
/// Neither block holds a whole section; `v_permlanex16_b32` lowered on its own reads the lanes
/// the host has, so the translation is the one with the join's code in each arm
/// ([`ir::whole_sections`]), where each section is whole.
#[test]
fn two_reductions_that_share_their_end_translate_as_two_sections() {
    // an arm: save EXEC, the identity `init`, then `op` over the wave up to lane 32's read
    let arm = |init: u32, op: u32| {
        let mut w = vec![
            0xBE8C_25C1, // s_or_saveexec_b64 s[12:13], -1
            0x7E04_02FF,
            init,        // v_mov_b32 v2, init
            0xD501_0002,
            0x0032_0902, // v_cndmask_b32 v2, v2, v4, s[12:13]
        ];
        for dpp in [0xDE54_C102, 0x9360_1202, 0x0009_2402] {
            w.extend([op << 25 | 0x0004_04E9, dpp]); // v_<op>_f32 v2, v2, v2 dpp8:[...]
        }
        w.extend([
            op << 25 | 0x0004_04FA,
            0xFF09_4002, // v_<op>_f32 v2, v2, v2 row_mirror bound_ctrl:1
            0xD778_0003,
            0x0201_0102,               // v_permlanex16_b32 v3, v2, 0, 0
            op << 25 | 0x0004_0503,    // v_<op>_f32 v2, v3, v2
            0xD760_000E,
            0x0001_4102,               // v_readlane_b32 s14, v2, 32
            op << 25 | 0x0004_040E,    // v_<op>_f32 v2, s14, v2
        ]);
        w
    };
    let mut words = vec![
        0x7E08_0D00, // v_cvt_f32_u32 v4, v0
        0x0608_08F2, // v_add_f32 v4, 1.0, v4
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0xBF85_0014, // s_cbranch_scc1 to the minimum
    ];
    words.extend(arm(0xFF80_0000, 0x10)); // the maximum from -inf
    words.push(0xBF82_0013); // s_branch to the join
    words.extend(arm(0x7F80_0000, 0x0F)); // the minimum from +inf
    words.extend([
        0xD760_000E,
        0x0001_0102, // v_readlane_b32 s14, v2, 0
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        0x7E00_020E, // v_mov_b32 v0, s14
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ]);
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    // the section starts in the arms and ends in the join; afterwards each block holds whole ones
    let sections = |p: &ir::Program| (0..p.blocks.len()).map(|b| super::wave::sections(p, b).len()).sum::<usize>();
    assert_eq!(sections(&p), 0);
    let q = ir::whole_sections(&p).expect("sections to join");
    assert_eq!(sections(&q), 2);
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let bytes = crate::dxbc::build(&[]);
    let c = super::Container::parse(&bytes).expect("container");
    let st_q = structure::structure(&q).expect("structure");
    let io_q = io::raise(&q, &native).expect("io::raise");
    let res_q = resource::track(&q, &native.layout, native.stage, Some(&st_q)).expect("track");
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let joined = super::emit_any(&c, &native, &q, &st_q, &io_q, &res_q);
    let shader = super::emit(&c, &native, &p, &st, &io, &res);
    lower::TEST_HOST_WAVE.set(was);
    let shader = shader.expect("emit");
    let joined = joined.expect("the joined sections");
    assert!(shader.dxil == joined.dxil, "the split sections' translation was taken");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (s0, expected) in [(1u32, 64.0f32), (0, 1.0)] {
        let Some(got) = on_the_card(&shader, [s0, 0]) else { return };
        assert_eq!(f32::from_bits(got), expected, "s0 = {s0}");
    }
    eprintln!("ran on the hardware device: the maximum or the minimum over the console's 64 lanes");
}

/// One console wave of 64 threads summing doubles over the lanes below s0 as the title's
/// water volumetrics count their lanes (`water_shading:CS_TransmissionVolumetrics`,
/// 0x5c4-0x68c): `v_add_f64` of each dword pair moved by DPP, `v_permlanex16` and lane 32,
/// the sum moved to s[20:21] and taken back as a float (`v_cvt_f32_f64`). The terms here are
/// the thread index as a double (`v_cvt_f64_f32`), so the sum is s0 (s0 - 1) / 2.
#[test]
fn a_wave_sum_of_doubles_is_the_consoles() {
    let mut words = vec![
        0x7E3E_0D00, // v_cvt_f32_u32 v31, v0
        0x7E3A_211F, // v_cvt_f64_f32 v[29:30], v31
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_046A, // s_mov_b64 s[12:13], vcc
        0xBE90_25C1, // s_or_saveexec_b64 s[16:17], -1
        0x7E04_0280, // v_mov_b32 v2, 0
        0x7E06_0280, // v_mov_b32 v3, 0
        0x8792_100C, // s_and_b64 s[18:19], s[12:13], s[16:17]
        0xD501_0002,
        0x004A_3B02, // v_cndmask_b32 v2, v2, v29, s[18:19]
        0xD501_0003,
        0x004A_3D03, // v_cndmask_b32 v3, v3, v30, s[18:19]
    ];
    // v_mov_b32 v4, v2 / v5, v3 with each DPP pattern, then v_add_f64 v[2:3], v[4:5], v[2:3]
    for (code, dpp) in [(0xE9, 0xDE54_C100), (0xE9, 0x9360_1200), (0xE9, 0x0009_2400), (0xFA, 0xFF09_4000)] {
        words.extend([0x7E08_0200 | code, dpp | 2, 0x7E0A_0200 | code, dpp | 3, 0xD564_0002, 0x0002_0504]);
    }
    words.extend([
        0xD778_0004,
        0x0201_0102, // v_permlanex16_b32 v4, v2, 0, 0
        0xD778_0005,
        0x0201_0103, // v_permlanex16_b32 v5, v3, 0, 0
        0xD564_0002,
        0x0002_0504, // v_add_f64 v[2:3], v[4:5], v[2:3]
        0xD760_0012,
        0x0001_4102, // v_readlane_b32 s18, v2, 32
        0xD760_0013,
        0x0001_4103, // v_readlane_b32 s19, v3, 32
        0xD564_0002,
        0x0002_0412, // v_add_f64 v[2:3], s[18:19], v[2:3]
        0xD760_0012,
        0x0001_0102, // v_readlane_b32 s18, v2, 0
        0xD760_0013,
        0x0001_0103, // v_readlane_b32 s19, v3, 0
        0xBE94_0312, // s_mov_b32 s20, s18
        0xBE95_0313, // s_mov_b32 s21, s19
        0xBEFE_0410, // s_mov_b64 exec, s[16:17]
        0x7E00_1E14, // v_cvt_f32_f64 v0, s[20:21]
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ]);
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    // the section leaves the sum's two dwords in s20 and s21
    let block = (0..p.blocks.len()).find(|&b| !super::wave::sections(&p, b).is_empty()).expect("a section");
    let sums = super::wave::sections(&p, block)
        .into_values()
        .flat_map(|s| s.shapes.into_values())
        .filter(|s| matches!(s, super::wave::Shape::Sum64 { .. }))
        .count();
    assert!(sums >= 2, "{sums} dwords of 64-bit sums");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let bytes = crate::dxbc::build(&[]);
    let c = super::Container::parse(&bytes).expect("container");
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let shader = super::emit(&c, &native, &p, &st, &io, &res);
    lower::TEST_HOST_WAVE.set(was);
    let shader = shader.expect("emit");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for s0 in [64u32, 40, 33, 1, 0] {
        let Some(got) = on_the_card(&shader, [s0, 0]) else { return };
        let want = (s0 * s0.saturating_sub(1) / 2) as f32;
        assert_eq!(f32::from_bits(got), want, "the lanes below {s0}");
    }
    eprintln!("ran on the hardware device: the sums of doubles over the console's 64 lanes");
}

// ---- the paired form: a host wave of 32 for each console wave, two lanes a thread ----

// Operands as the hardware numbers them: SGPR n is n, VCC 106 (its high dword 107), EXEC 126,
// an inline integer n 128 + n (-1: 193), a literal (the next dword) 255, VGPR n 256 + n.
const VCC: u32 = 106;
const EXEC: u32 = 126;
const LIT: u32 = 255;

const fn v(n: u32) -> u32 {
    256 + n
}

const fn k(n: u32) -> u32 {
    128 + n
}

const fn sop1(op: u32, sdst: u32, src0: u32) -> u32 {
    0xBE80_0000 | sdst << 16 | op << 8 | src0
}

const fn sop2(op: u32, sdst: u32, src0: u32, src1: u32) -> u32 {
    0x8000_0000 | op << 23 | sdst << 16 | src1 << 8 | src0
}

/// A branch by `imm` dwords from the next instruction (or `s_barrier`, `imm` 0).
const fn sopp(op: u32, imm: i16) -> u32 {
    0xBF80_0000 | op << 16 | imm as u16 as u32
}

const fn vop1(op: u32, vdst: u32, src0: u32) -> u32 {
    0x7E00_0000 | vdst << 17 | op << 9 | src0
}

const fn vop2(op: u32, vdst: u32, src0: u32, vsrc1: u32) -> u32 {
    op << 25 | vdst << 17 | vsrc1 << 9 | src0
}

const fn vopc(op: u32, src0: u32, vsrc1: u32) -> u32 {
    0x7C00_0000 | op << 17 | vsrc1 << 9 | src0
}

const fn vop3(op: u32, dst: u32, src0: u32, src1: u32) -> [u32; 2] {
    [0xD400_0000 | op << 16 | dst, src0 | src1 << 9]
}

const fn ds_at(op: u32, vdst: u32, addr: u32, data: u32, offset: u32) -> [u32; 2] {
    [0xD800_0000 | op << 18 | offset, vdst << 24 | data << 8 | addr]
}

/// `buffer_store_dword v<data>, v<addr>, s[8:11], 0 offen offset:<offset>`
const fn store(data: u32, addr: u32, offset: u32) -> [u32; 2] {
    [0xE070_1000 | offset, 0x8002_0000 | data << 8 | addr]
}

/// `v<regs>` stored as each thread's dwords of the UAV, the thread's index (v0) times 16 on.
fn stores(regs: [u32; 4]) -> Vec<u32> {
    let mut w = vec![vop2(0x1A, 15, k(4), 0)]; // v_lshlrev_b32 v15, 4, v0
    for (n, r) in regs.into_iter().enumerate() {
        w.extend(store(r, 15, 4 * n as u32));
    }
    w
}

/// `native` in the paired form on a host of 32-lane waves, validated by `dxil.dll` when
/// present. Where no other form gives the host the program (each still asks for 64-lane
/// waves, or fails), it is also what [`super::emit`] gives it.
fn paired(native: &NativeProgram) -> HostShader {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, native).expect("io::raise");
    let bytes = crate::dxbc::build(&[]);
    let c = super::Container::parse(&bytes).expect("container");
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let others = super::emit_forms(&c, native, &p, &st, &io, &res);
    let shader = super::paired(&c, native, &p, &io, &res);
    let chosen = super::emit(&c, native, &p, &st, &io, &res);
    lower::TEST_HOST_WAVE.set(was);
    let (shader, wave) = shader.expect("the paired form");
    assert_eq!(wave, Some(32));
    if !matches!(others, Ok((_, wave)) if wave != Some(64)) {
        assert_eq!(chosen.expect("emit").dxil, shader.dxil, "emit's choice");
    }
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    shader
}

/// `shader` on the hardware device: one group, root constants `s0..s3`, a raw UAV of `dwords`
/// zeroed dwords, which come back; None (and a line) without a device that takes it.
fn run_paired(shader: &HostShader, constants: [u32; 4], dwords: u32) -> Option<Vec<u32>> {
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return None;
    };
    let (root, pso) = match host.compute_pipeline(shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program: {e}; not run");
            return None;
        }
    };
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if let (RootType::Constants, 0, d @ 0..=3) = (u.root, u.slot, u.dword) {
            ud[u.index as usize] = constants[d as usize];
        }
    }
    let mut out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(dwords)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(out.swap_remove(0))
}

/// Panics at the first dword of `got` that is not the console's (`want`), naming the thread
/// and which of its four dwords.
fn same_as_console(got: &[u32], want: &[u32]) {
    assert_eq!(got.len(), want.len());
    if let Some(n) = (0..want.len()).find(|&n| got[n] != want[n]) {
        panic!("thread {}, dword {}: {:#x}, the console's {:#x}", n / 4, n % 4, got[n], want[n]);
    }
}

/// One console wave of 64 threads with 256 bytes of LDS. The even lanes write their dword in
/// one arm of a per-lane branch; the odd lanes theirs in the other arm, then, in that arm and
/// with no barrier, read their mirror lane's (63 - n, an even lane's); the even lanes read
/// theirs (an odd lane's) after the join. Then lanes 5 and 40 are read (`v_readlane_b32`), the
/// lanes holding more than 250 are counted (`s_bcnt1_i32_b64` of the compare's mask) and
/// counted below each lane (`v_mbcnt_lo/hi`): each of these takes lanes of both halves.
#[test]
fn paired_lanes_share_lds_and_lanes_across_halves_without_barriers() {
    let mut w = vec![vop2(0x1A, 1, k(2), 0)]; // v_lshlrev_b32 v1, 2, v0: this lane's dword
    w.extend(vop3(0x169, 2, v(0), k(7))); // v_mul_lo_u32 v2, v0, 7
    w.extend([
        vop2(0x26, 5, k(63), 0), // v_sub_nc_u32 v5, 63, v0: the mirror lane
        vop2(0x1A, 5, k(2), 5),  // v_lshlrev_b32 v5, 2, v5: its dword
        vop2(0x1B, 3, k(1), 0),  // v_and_b32 v3, 1, v0
        vopc(0xC2, k(0), 3),     // v_cmp_eq_u32 vcc, 0, v3: the even lanes
        sop1(0x24, 12, VCC),     // s_and_saveexec_b64 s[12:13], vcc
        sopp(0x08, 4),           // s_cbranch_execz past the even lanes' arm
        vop2(0x25, 4, LIT, 2),   // v_add_nc_u32 v4, 100, v2
        100,
    ]);
    w.extend(ds_at(0x0D, 0, 1, 4, 0)); // ds_write_b32 v1, v4
    w.extend([
        sop2(0x15, EXEC, 12, EXEC), // s_andn2_b64 exec, s[12:13], exec: the odd lanes
        sopp(0x08, 6),              // s_cbranch_execz past their arm
        vop2(0x25, 4, LIT, 2),      // v_add_nc_u32 v4, 200, v2
        200,
    ]);
    w.extend(ds_at(0x0D, 0, 1, 4, 0)); // ds_write_b32 v1, v4
    w.extend(ds_at(0x36, 6, 5, 0, 0)); // ds_read_b32 v6, v5: an even lane's, from the other arm
    w.extend([
        sop1(0x04, EXEC, 12), // s_mov_b64 exec, s[12:13]
        sop1(0x24, 12, VCC),  // s_and_saveexec_b64 s[12:13], vcc: the even lanes again
    ]);
    w.extend(ds_at(0x36, 6, 5, 0, 0)); // ds_read_b32 v6, v5: an odd lane's
    w.push(sop1(0x04, EXEC, 12)); // s_mov_b64 exec, s[12:13]
    w.extend(vop3(0x360, 14, v(6), k(5))); // v_readlane_b32 s14, v6, 5
    w.extend(vop3(0x360, 15, v(6), k(40))); // v_readlane_b32 s15, v6, 40
    w.extend([
        sop2(0x00, 14, 14, 15), // s_add_u32 s14, s14, s15
        vopc(0xC1, LIT, 6),     // v_cmp_lt_u32 vcc, 250, v6
        250,
        sop1(0x10, 16, VCC), // s_bcnt1_i32_b64 s16, vcc
    ]);
    w.extend(vop3(0x365, 7, VCC, k(0))); // v_mbcnt_lo_u32_b32 v7, vcc_lo, 0
    w.extend(vop3(0x366, 7, VCC + 1, v(7))); // v_mbcnt_hi_u32_b32 v7, vcc_hi, v7
    w.extend([
        vop1(0x01, 9, 14),  // v_mov_b32 v9, s14
        vop1(0x01, 10, 16), // v_mov_b32 v10, s16
    ]);
    w.extend(stores([6, 9, 10, 7]));
    let shader = paired(&native_with(&w, true, 64, 256));
    // the console: lane n's dword is 7n + 100 (even n) or 7n + 200, and lane n reads 63 - n's
    let dword = |n: u32| 7 * n + if n.is_multiple_of(2) { 100 } else { 200 };
    let read: Vec<u32> = (0..64).map(|n| dword(63 - n)).collect();
    let big = (0..64).filter(|&n| read[n] > 250).fold(0u64, |m, n| m | 1 << n);
    assert!(big as u32 != 0 && big >> 32 != 0, "lanes of both halves counted");
    let want: Vec<u32> = (0..64)
        .flat_map(|n| [read[n], read[5] + read[40], big.count_ones(), (big & ((1 << n) - 1)).count_ones()])
        .collect();
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Some(got) = run_paired(&shader, [0; 4], 256) else { return };
    same_as_console(&got, &want);
    eprintln!("ran on the hardware device: LDS, lane reads and masks across the halves as on the console");
}

/// Two console waves of 64 in a group of 128 (two host waves of 32, 512 bytes of LDS). In
/// each, lane 0 zeroes the wave's dword of LDS; every lane loops (lane & 3) + 1 + wave times,
/// adding its lane number to its sum and 1 to the wave's dword each round (`ds_add_u32`; the
/// loop's branch is the wave's: it goes round while some lane is left); after the loop each
/// lane reads the wave's total, with no barrier; lanes 40-63 read their first lane's sum
/// (`v_readfirstlane_b32`: lane 40, an upper lane), then lanes 20-63 (lane 20, a lower one,
/// though upper lanes have one too); past the barrier, each lane reads the other wave's total.
#[test]
fn paired_lanes_loop_apart_and_meet_the_other_wave_at_the_barrier() {
    let mut w = vec![
        vop2(0x16, 1, k(6), 0),  // v_lshrrev_b32 v1, 6, v0: the console wave
        vop2(0x1B, 2, k(63), 0), // v_and_b32 v2, 63, v0: the lane in it
        vop2(0x1A, 11, k(2), 1), // v_lshlrev_b32 v11, 2, v1: the wave's dword
        vop1(0x01, 12, k(0)),    // v_mov_b32 v12, 0
        sop1(0x04, 22, EXEC),    // s_mov_b64 s[22:23], exec
        vopc(0xD2, k(0), 2),     // v_cmpx_eq_u32 exec, 0, v2: lane 0
    ];
    w.extend(ds_at(0x0D, 0, 11, 12, 0)); // ds_write_b32 v11, v12
    w.extend([
        sop1(0x04, EXEC, 22),   // s_mov_b64 exec, s[22:23]
        vop2(0x1B, 3, k(3), 2), // v_and_b32 v3, 3, v2
        vop2(0x25, 3, k(1), 3), // v_add_nc_u32 v3, 1, v3
        vop2(0x25, 3, v(1), 3), // v_add_nc_u32 v3, v1, v3: the rounds
        vop1(0x01, 4, k(0)),    // v_mov_b32 v4, 0
        vop1(0x01, 13, k(1)),   // v_mov_b32 v13, 1
        sop1(0x04, 20, EXEC),   // s_mov_b64 s[20:21], exec
        vop2(0x25, 4, v(2), 4), // loop: v_add_nc_u32 v4, v2, v4
    ]);
    w.extend(ds_at(0x00, 0, 11, 13, 0)); // ds_add_u32 v11, v13
    w.extend([
        vop2(0x25, 3, 193, 3),       // v_add_nc_u32 v3, -1, v3
        vopc(0xC5, k(0), 3),         // v_cmp_ne_u32 vcc, 0, v3
        sop2(0x0F, EXEC, EXEC, VCC), // s_and_b64 exec, exec, vcc
        sopp(0x09, -7),              // s_cbranch_execnz loop
        sop1(0x04, EXEC, 20),        // s_mov_b64 exec, s[20:21]
    ]);
    w.extend(ds_at(0x36, 5, 11, 0, 0)); // ds_read_b32 v5, v11: the wave's total
    w.extend([
        sop1(0x04, 24, EXEC), // s_mov_b64 s[24:25], exec
        vopc(0xD1, k(39), 2), // v_cmpx_lt_u32 exec, 39, v2: lanes 40-63
        vop1(0x02, 6, v(4)),  // v_readfirstlane_b32 s6, v4
        sop1(0x04, EXEC, 24), // s_mov_b64 exec, s[24:25]
        vopc(0xD1, k(19), 2), // v_cmpx_lt_u32 exec, 19, v2: lanes 20-63
        vop1(0x02, 7, v(4)),  // v_readfirstlane_b32 s7, v4
        sop1(0x04, EXEC, 24), // s_mov_b64 exec, s[24:25]
        vop1(0x01, 6, 6),     // v_mov_b32 v6, s6
        vop1(0x01, 9, 7),     // v_mov_b32 v9, s7
        0xD400_0000 | 0x346 << 16 | 6, // v_lshl_add_u32 v6, v9, 16, v6
        v(9) | k(16) << 9 | v(6) << 18,
        sopp(0x0A, 0),            // s_barrier
        vop2(0x1D, 14, k(4), 11), // v_xor_b32 v14, 4, v11: the other wave's dword
    ]);
    w.extend(ds_at(0x36, 7, 14, 0, 0)); // ds_read_b32 v7, v14
    w.extend(stores([4, 5, 6, 7]));
    let shader = paired(&native_with(&w, true, 128, 512));
    let rounds = |wave: u32, lane: u32| (lane & 3) + 1 + wave;
    let total = |wave: u32| (0..64).map(|lane| rounds(wave, lane)).sum::<u32>();
    let first = |wave: u32, lane: u32| lane * rounds(wave, lane);
    let want: Vec<u32> = (0..128)
        .flat_map(|t| {
            let (wave, lane) = (t / 64, t % 64);
            [first(wave, lane), total(wave), first(wave, 20) << 16 | first(wave, 40), total(1 - wave)]
        })
        .collect();
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Some(got) = run_paired(&shader, [0; 4], 512) else { return };
    same_as_console(&got, &want);
    eprintln!("ran on the hardware device: two console waves, their loops and LDS as on the console");
}

/// One console wave: `ds_bpermute_b32` from the mirror lane; a DPP row shift whose row mask
/// leaves rows 0 and 3 alone (rows 2 and 3 are lanes 32-63: rows are counted in console lanes); a
/// `ds_swizzle_b32` swapping the two 16-lane rows of each 32 lanes; `v_writelane_b32` of s0
/// into lane 45.
#[test]
fn paired_lanes_permute_shift_and_write_lanes_of_both_halves() {
    let mut w = vop3(0x169, 1, v(0), k(3)).to_vec(); // v_mul_lo_u32 v1, v0, 3
    w.extend([
        vop2(0x25, 1, k(1), 1),  // v_add_nc_u32 v1, 1, v1
        vop2(0x26, 2, k(63), 0), // v_sub_nc_u32 v2, 63, v0
        vop2(0x1A, 2, k(2), 2),  // v_lshlrev_b32 v2, 2, v2: the mirror lane's address
    ]);
    w.extend(ds_at(0xB3, 3, 2, 1, 0)); // ds_bpermute_b32 v3, v2, v1
    w.extend([
        vop1(0x01, 4, LIT), // v_mov_b32 v4, 1000
        1000,
        vop1(0x01, 4, 0xFA), // v_mov_b32 v4, v1 row_shr:1 row_mask:0x6 bank_mask:0xf
        0x6 << 28 | 0xF << 24 | 0x111 << 8 | 1,
    ]);
    w.extend(ds_at(0x35, 5, 1, 0, 0x401F)); // ds_swizzle_b32 v5, v1 offset:swizzle(BITMASK_PERM, xor 16)
    w.push(vop1(0x01, 6, v(0))); // v_mov_b32 v6, v0
    w.extend(vop3(0x361, 6, 0, k(45))); // v_writelane_b32 v6, s0, 45
    w.extend(stores([3, 4, 5, 6]));
    let shader = paired(&native_with(&w, true, 64, 0));
    let s0 = 0xABCD_1234;
    let x = |n: u32| 3 * n + 1;
    let want: Vec<u32> = (0..64)
        .flat_map(|n| {
            let shifted = if matches!(n >> 4, 1 | 2) && n & 15 != 0 { x(n - 1) } else { 1000 };
            let swizzled = x(n & 32 | (n & 31) ^ 16);
            [x(63 - n), shifted, swizzled, if n == 45 { s0 } else { n }]
        })
        .collect();
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Some(got) = run_paired(&shader, [s0, 0, 0, 0], 256) else { return };
    same_as_console(&got, &want);
    eprintln!("ran on the hardware device: permutes, row masks, swizzles and lane writes as on the console");
}

/// [`program`]'s wave OR inside a per-lane branch, in the paired form: the OR of both halves'
/// lanes, whichever take the branch; nothing stored when none does.
#[test]
fn paired_lanes_or_a_wave_inside_a_lane_branch() {
    for (skip, name) in [(0xBF88_0000, "s_cbranch_execz"), (0xBF84_0000, "s_cbranch_scc0")] {
        let shader = paired(&program(skip));
        let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let cases = [(64u32, u32::MAX), (40, 0x000F_FFFF), (33, 0x0001_FFFF), (10, 0x1F), (1, 1), (0, 0)];
        for (below, expected) in cases {
            let Some(got) = on_the_card(&shader, [below, 0]) else { return };
            assert_eq!(got, expected, "{name}: {below} threads in the branch");
        }
    }
    eprintln!("ran on the hardware device: the OR of both halves in every case");
}

/// One console wave of 64 threads, thread t holding the bit `1 << (t >> 1)`: the threads below
/// s0 OR theirs over the wave inside `if (t < s0) { ... }` ([`super::wave::WAVE_OR_V7_TO_S7`]
/// into s7, which holds 0x77 before), and every thread stores s7 after the branch: the wave's
/// OR, or 0x77 when no thread took the branch and the wave skipped it. A wave-wide value used
/// past a per-lane branch, as the title's tile classification and light clustering programs
/// use theirs: no form in two host waves takes it (they would meet inside the branch, and the
/// skip cannot be left out: s7 would no longer keep 0x77), and the approximate form, each host
/// wave a wave of its own, gives each half its own OR, 0x77 to the upper half while s0 <= 32.
/// The title's 39 such programs ran approximate; [`super::emit`] now gives the paired form.
#[test]
fn a_wave_value_used_past_a_lane_branch_takes_the_paired_form_not_an_approximate_one() {
    let mut w = vec![
        sop1(0x03, 7, LIT), // s_mov_b32 s7, 0x77
        0x77,
        vop1(0x01, 5, k(1)),    // v_mov_b32 v5, 1
        vop2(0x16, 6, k(1), 0), // v_lshrrev_b32 v6, 1, v0
        vop2(0x1A, 7, v(6), 5), // v_lshlrev_b32 v7, v6, v5
        vopc(0xC4, 0, 0),       // v_cmp_gt_u32 vcc, s0, v0
        sop1(0x24, 12, VCC),    // s_and_saveexec_b64 s[12:13], vcc
        sopp(0x08, 20),         // s_cbranch_execz past the OR
    ];
    w.extend_from_slice(&super::wave::WAVE_OR_V7_TO_S7);
    w.extend([
        sop1(0x04, EXEC, 12), // s_mov_b64 exec, s[12:13]
        vop1(0x01, 8, 7),     // v_mov_b32 v8, s7
    ]);
    w.extend(stores([8, 7, 6, 0]));
    let native = native_with(&w, true, 64, 0);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let bytes = crate::dxbc::build(&[]);
    let c = super::Container::parse(&bytes).expect("container");
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let pairs = super::paired(&c, &native, &p, &io, &res);
    let chosen = super::emit(&c, &native, &p, &st, &io, &res);
    // (as before: the paired form only where no other form takes the program)
    let off = super::TEST_PAIRS.replace(Some(false));
    let approximate = super::emit(&c, &native, &p, &st, &io, &res);
    super::TEST_PAIRS.set(off);
    lower::TEST_HOST_WAVE.set(was);
    let (pairs, wave) = pairs.expect("the paired form");
    assert_eq!(wave, Some(32));
    let (chosen, approximate) = (chosen.expect("emit"), approximate.expect("an approximate form"));
    assert!(chosen.dxil == pairs.dxil, "emit's choice is not the paired form");
    assert!(approximate.dxil != pairs.dxil, "without the paired form, emit gives it anyway");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = chosen.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    // the console: every thread stores the OR of the threads below s0, or 0x77 with none
    let want = |s0: u32| -> Vec<u32> {
        let or = (0..64u32).filter(|&t| t < s0).fold(0u32, |m, t| m | 1 << (t >> 1));
        let s7 = if s0 == 0 { 0x77 } else { or };
        (0..64u32).flat_map(|t| [s7, 1 << (t >> 1), t >> 1, t]).collect()
    };
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for s0 in [64u32, 40, 33, 32, 10, 1, 0] {
        let Some(got) = run_paired(&chosen, [s0, 0, 0, 0], 256) else { return };
        same_as_console(&got, &want(s0));
    }
    // each half on its own: with s0 = 10 the upper half took no branch and kept 0x77
    let Some(halves) = run_paired(&approximate, [10, 0, 0, 0], 256) else { return };
    assert_eq!((want(10)[0], want(10)[4 * 40]), (0x1F, 0x1F));
    assert_eq!((halves[0], halves[4 * 40]), (0x1F, 0x77), "the approximate form's lanes 0 and 40");
    eprintln!("ran on the hardware device: the wave's OR past the branch in every lane, not each half's");
}

/// [`half_sum_program`] in the paired form: each half's tree over its 32 lanes, then the two
/// halves' sums, the console's last addition.
#[test]
fn paired_lanes_sum_halves_as_the_consoles_tree() {
    let (native, tree, _) = half_sum_program();
    let shader = paired(&native);
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Some(got) = on_the_card(&shader, [0, 0]) else { return };
    assert_eq!(got, u32::from(tree));
    eprintln!("ran on the hardware device: the sum {tree:#x} of the console's tree");
}

/// The lens-flare scans' allocation (`lens_flares:CS_ScanMip1`, 0x5c0-0x61c): in a branch
/// of the lanes below s0, the first lane adds the wave's count times a size (s1) to a
/// counter, and each lane takes base + below × size with `v_mad_u32_u24`. That is the
/// idiom a host wave may run alone ([`super::idiom::wave_local`]): each half reserves its own
/// count's slots. The same with the count where the base goes is not.
#[test]
fn an_allocation_in_units_of_a_size_is_the_idiom() {
    let program = |base: u32| -> Vec<u8> {
        let words = [
            0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
            0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
            0xBF88_0016, // s_cbranch_execz past the allocation
            0xBE82_047E, // s_mov_b64 s[2:3], exec
            0xBE84_147E, // s_ff1_i32_b64 s4, exec
            0xD765_0004,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v4, -1, 0
            0xD766_0004,
            0x0002_08C1, // v_mbcnt_hi_u32_b32 v4, -1, v4: the lane
            0xBE86_047E, // s_mov_b64 s[6:7], exec
            0x7DA4_0804, // v_cmpx_eq_u32 exec, s4, v4: the first lane
            0xBE82_1002, // s_bcnt1_i32_b64 s2, s[2:3]: the count
            0x7E02_0201, // v_mov_b32 v1, s1: the size
            0x160A_0202, // v_mul_u32_u24 v5, s2, v1
            0x7E16_0280, // v_mov_b32 v11, 0
            0xE0C8_6000,
            0x8002_050B, // buffer_atomic_add v5, v11, s[8:11], 0 idxen glc
            0xBEFE_0406, // s_mov_b64 exec, s[6:7]
            0xD765_0004,
            0x0001_007E, // v_mbcnt_lo_u32_b32 v4, exec_lo, 0
            0xD766_0004,
            0x0002_087F, // v_mbcnt_hi_u32_b32 v4, exec_hi, v4: the lanes below
            0x7E0A_0505, // v_readfirstlane_b32 s5, v5: the base
            0xD543_0006,
            0x0002_0304 | base << 18, // v_mad_u32_u24 v6, v4, v1, s<base>
            0xBEFE_040C, // s_mov_b64 exec, s[12:13]
            S_ENDPGM,
        ];
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    };
    let idiom = |code: &[u8]| {
        let p = ir::build(code, true, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        super::idiom::wave_local(&p, &st)
    };
    assert_eq!(idiom(&program(5)), Ok(()));
    assert!(idiom(&program(2)).is_err(), "the count as the base");
}
