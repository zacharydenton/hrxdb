
hrx-cache/kernels/566b75e6356c3b833861c6cfe3393e6fb1801e0947deb56f8bee9f1d0b2df72e/kernel.hsaco:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001000 <select>:
	s_load_b64 s[4:5], s[0:1], null                            // 000000001000: F4040100 F8000000
	s_load_b64 s[6:7], s[0:1], 0x8                             // 000000001008: F4040180 F8000008
	v_lshl_add_u32 v1, s2, 10, v0                              // 000000001010: D6460001 04011402
	s_mov_b32 s3, 0x100                                        // 000000001018: BE8300FF 00000100
	v_mov_b32_e32 v2, 0x100                                    // 000000001020: 7E0402FF 00000100
	s_mov_b32 s8, 3                                            // 000000001028: BE880083
	s_delay_alu instid0(VALU_DEP_1) | instid1(VALU_DEP_2)      // 00000000102C: BF870101
	v_add_nc_u32_e32 v2, v1, v2                                // 000000001030: 4A040501
	v_lshl_add_u32 v3, s3, 1, v1                               // 000000001034: D6460003 04050203
	v_lshl_add_u32 v4, s8, 8, v1                               // 00000000103C: D6460004 04051008
	s_waitcnt lgkmcnt(0)                                       // 000000001044: BF89FC07
	v_mov_b32_e32 v5, s4                                       // 000000001048: 7E0A0204
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000104C: BF870001
	v_cmp_lt_u32_e64 s8, v1, v5                                // 000000001050: D4490008 02020B01
	v_cmp_lt_u32_e64 s10, v2, v5                               // 000000001058: D449000A 02020B02
	v_cmp_lt_u32_e64 s12, v3, v5                               // 000000001060: D449000C 02020B03
	v_cmp_lt_u32_e64 s14, v4, v5                               // 000000001068: D449000E 02020B04
	s_delay_alu instid0(VALU_DEP_4)                            // 000000001070: BF870004
	v_cndmask_b32_e64 v1, 0, v1, s8                            // 000000001074: D5010001 00220280
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000107C: BF870004
	v_cndmask_b32_e64 v2, 0, v2, s10                           // 000000001080: D5010002 002A0480
	s_delay_alu instid0(VALU_DEP_4)                            // 000000001088: BF870004
	v_cndmask_b32_e64 v3, 0, v3, s12                           // 00000000108C: D5010003 00320680
	s_delay_alu instid0(VALU_DEP_4)                            // 000000001094: BF870004
	v_cndmask_b32_e64 v4, 0, v4, s14                           // 000000001098: D5010004 003A0880
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000010A0: BF870004
	v_lshlrev_b32_e32 v5, 2, v1                                // 0000000010A4: 300A0282
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000010A8: BF870004
	v_lshlrev_b32_e32 v6, 2, v2                                // 0000000010AC: 300C0482
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000010B0: BF870004
	v_lshlrev_b32_e32 v7, 2, v3                                // 0000000010B4: 300E0682
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000010B8: BF870004
	v_lshlrev_b32_e32 v8, 2, v4                                // 0000000010BC: 30100882
	global_load_b32 v5, v5, s[6:7]                             // 0000000010C0: DC520000 05060005
	global_load_b32 v6, v6, s[6:7]                             // 0000000010C8: DC520000 06060006
	global_load_b32 v7, v7, s[6:7]                             // 0000000010D0: DC520000 07060007
	global_load_b32 v8, v8, s[6:7]                             // 0000000010D8: DC520000 08060008
	s_load_b128 s[16:19], s[0:1], 0x18                         // 0000000010E0: F4080400 F8000018
	s_mul_i32 s0, s2, s5                                       // 0000000010E8: 96000502
	v_cndmask_b32_e64 v1, 0x7fffffff, v1, s8                   // 0000000010EC: D5010001 002202FF 7FFFFFFF
	v_cndmask_b32_e64 v2, 0x7fffffff, v2, s10                  // 0000000010F8: D5010002 002A04FF 7FFFFFFF
	v_cndmask_b32_e64 v3, 0x7fffffff, v3, s12                  // 000000001104: D5010003 003206FF 7FFFFFFF
	v_cndmask_b32_e64 v4, 0x7fffffff, v4, s14                  // 000000001110: D5010004 003A08FF 7FFFFFFF
	s_mov_b32 s1, 0                                            // 00000000111C: BE810080
	s_waitcnt vmcnt(3)                                         // 000000001120: BF890FF7
	v_cndmask_b32_e64 v5, 0xff800000, v5, s8                   // 000000001124: D5010005 00220AFF FF800000
	s_waitcnt vmcnt(2)                                         // 000000001130: BF890BF7
	v_cndmask_b32_e64 v6, 0xff800000, v6, s10                  // 000000001134: D5010006 002A0CFF FF800000
	s_waitcnt vmcnt(1)                                         // 000000001140: BF8907F7
	v_cndmask_b32_e64 v7, 0xff800000, v7, s12                  // 000000001144: D5010007 00320EFF FF800000
	s_waitcnt vmcnt(0)                                         // 000000001150: BF8903F7
	v_cndmask_b32_e64 v8, 0xff800000, v8, s14                  // 000000001154: D5010008 003A10FF FF800000
	s_cmp_lt_i32 s1, s5                                        // 000000001160: BF040501
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001164: BF870009
	s_cbranch_scc0 245                                         // 000000001168: BFA100F5 <select+0x540>
	s_waitcnt lgkmcnt(0)                                       // 00000000116C: BF89FC07
	v_max_f32_e32 v9, 0xff800000, v5                           // 000000001170: 20120AFF FF800000
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001178: BF870001
	v_max_f32_e32 v9, v9, v6                                   // 00000000117C: 20120D09
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001180: BF870001
	v_max_f32_e32 v9, v9, v7                                   // 000000001184: 20120F09
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001188: BF870001
	v_max_f32_e32 v9, v9, v8                                   // 00000000118C: 20121109
	v_and_b32_e32 v10, 31, v0                                  // 000000001190: 3614009F
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 000000001194: BF890007
	s_waitcnt_vscnt null, 0x0                                  // 000000001198: BC7C0000
	v_mov_b32_e32 v11, 0                                       // 00000000119C: 7E160280
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000011A0: BF870003
	v_max_f32_dpp v9, v9, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000011A4: 201212FA FF08B109
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000011AC: BF870001
	v_max_f32_dpp v9, v9, v9 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000011B0: 201212FA FF084E09
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000011B8: BF870001
	v_max_f32_dpp v9, v9, v9 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000011BC: 201212FA FF094109
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000011C4: BF870001
	v_max_f32_dpp v9, v9, v9 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000011C8: 201212FA FF094009
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000011D0: BF870001
	v_permlanex16_b32 v12, v9, 0, 0                            // 0000000011D4: D65C000C 02010109
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000011DC: BF870001
	v_max_f32_e32 v9, v9, v12                                  // 0000000011E0: 20121909
	v_lshrrev_b32_e32 v12, 5, v0                               // 0000000011E4: 32180085
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000011E8: BF870001
	v_lshlrev_b32_e32 v12, 2, v12                              // 0000000011EC: 30181882
	v_cmp_lt_u32_e64 s2, v10, 1                                // 0000000011F0: D4490002 0201030A
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000011F8: BF870001
	s_and_saveexec_b64 s[2:3], s[2:3]                          // 0000000011FC: BE822102
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001200: BF870002
	v_add_nc_u32_e32 v12, v11, v12                             // 000000001204: 4A18190B
	ds_store_b32 v12, v9                                       // 000000001208: D8340000 0000090C
	s_mov_b64 exec, s[2:3]                                     // 000000001210: BEFE0102
	s_waitcnt lgkmcnt(0)                                       // 000000001214: BF89FC07
	s_barrier                                                  // 000000001218: BFBD0000
	v_cmp_lt_u32_e64 s2, v10, 8                                // 00000000121C: D4490002 0201110A
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001224: BF870001
	s_and_saveexec_b64 s[2:3], s[2:3]                          // 000000001228: BE822102
	v_lshlrev_b32_e32 v9, 2, v10                               // 00000000122C: 30121482
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001230: BF870001
	v_add_nc_u32_e32 v9, v11, v9                               // 000000001234: 4A12130B
	ds_load_b32 v9, v9                                         // 000000001238: D8D80000 09000009
	v_xor_b32_e32 v11, 4, v10                                  // 000000001240: 3A161484
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001244: BF870001
	v_lshlrev_b32_e32 v11, 2, v11                              // 000000001248: 30161682
	s_waitcnt lgkmcnt(0)                                       // 00000000124C: BF89FC07
	ds_bpermute_b32 v11, v11, v9                               // 000000001250: DACC0000 0B00090B
	v_xor_b32_e32 v12, 2, v10                                  // 000000001258: 3A181482
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000125C: BF870001
	v_lshlrev_b32_e32 v12, 2, v12                              // 000000001260: 30181882
	s_waitcnt lgkmcnt(0)                                       // 000000001264: BF89FC07
	v_max_f32_e32 v9, v9, v11                                  // 000000001268: 20121709
	ds_bpermute_b32 v11, v12, v9                               // 00000000126C: DACC0000 0B00090C
	v_xor_b32_e32 v10, 1, v10                                  // 000000001274: 3A141481
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001278: BF870001
	v_lshlrev_b32_e32 v10, 2, v10                              // 00000000127C: 30141482
	s_waitcnt lgkmcnt(0)                                       // 000000001280: BF89FC07
	s_delay_alu instid0(VALU_DEP_3)                            // 000000001284: BF870003
	v_max_f32_e32 v9, v9, v11                                  // 000000001288: 20121709
	ds_bpermute_b32 v10, v10, v9                               // 00000000128C: DACC0000 0A00090A
	s_waitcnt lgkmcnt(0)                                       // 000000001294: BF89FC07
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001298: BF870001
	v_max_f32_e32 v9, v9, v10                                  // 00000000129C: 20121509
	s_mov_b64 exec, s[2:3]                                     // 0000000012A0: BEFE0102
	v_mov_b32_e32 v10, 0                                       // 0000000012A4: 7E140280
	ds_bpermute_b32 v9, v10, v9                                // 0000000012A8: DACC0000 0900090A
	v_and_b32_e32 v10, 31, v0                                  // 0000000012B0: 3614009F
	s_waitcnt lgkmcnt(0)                                       // 0000000012B4: BF89FC07
	v_cmp_eq_f32_e64 s2, v5, v9                                // 0000000012B8: D4120002 02021305
	v_cmp_eq_f32_e64 s6, v6, v9                                // 0000000012C0: D4120006 02021306
	v_cmp_eq_f32_e64 s8, v7, v9                                // 0000000012C8: D4120008 02021307
	v_cmp_eq_f32_e64 s10, v8, v9                               // 0000000012D0: D412000A 02021308
	v_mov_b32_e32 v11, 32                                      // 0000000012D8: 7E1602A0
	v_cndmask_b32_e64 v12, 0x7fffffff, v1, s2                  // 0000000012DC: D501000C 000A02FF 7FFFFFFF
	v_cndmask_b32_e64 v13, 0x7fffffff, v2, s6                  // 0000000012E8: D501000D 001A04FF 7FFFFFFF
	v_cndmask_b32_e64 v14, 0x7fffffff, v3, s8                  // 0000000012F4: D501000E 002206FF 7FFFFFFF
	v_cndmask_b32_e64 v15, 0x7fffffff, v4, s10                 // 000000001300: D501000F 002A08FF 7FFFFFFF
	s_delay_alu instid0(VALU_DEP_3) | instid1(VALU_DEP_4)      // 00000000130C: BF870203
	v_min_u32_e32 v12, v12, v13                                // 000000001310: 26181B0C
	s_delay_alu instid0(VALU_DEP_1) | instid1(VALU_DEP_3)      // 000000001314: BF870181
	v_min_u32_e32 v12, v12, v14                                // 000000001318: 26181D0C
	s_delay_alu instid0(VALU_DEP_1) | instid1(VALU_DEP_3)      // 00000000131C: BF870181
	v_min_u32_e32 v12, v12, v15                                // 000000001320: 26181F0C
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001324: BF870001
	v_mov_b32_dpp v13, v12 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001328: 7E1A02FA FF08B10C
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001330: BF870001
	v_min_u32_e32 v12, v12, v13                                // 000000001334: 26181B0C
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001338: BF870001
	v_mov_b32_dpp v13, v12 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1// 00000000133C: 7E1A02FA FF084E0C
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001344: BF870001
	v_min_u32_e32 v12, v12, v13                                // 000000001348: 26181B0C
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000134C: BF870001
	v_mov_b32_dpp v13, v12 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001350: 7E1A02FA FF09410C
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001358: BF870001
	v_min_u32_e32 v12, v12, v13                                // 00000000135C: 26181B0C
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001360: BF870001
	v_mov_b32_dpp v13, v12 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001364: 7E1A02FA FF09400C
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000136C: BF870001
	v_min_u32_e32 v12, v12, v13                                // 000000001370: 26181B0C
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001374: BF870001
	v_permlanex16_b32 v13, v12, 0, 0                           // 000000001378: D65C000D 0201010C
	v_lshrrev_b32_e32 v14, 5, v0                               // 000000001380: 321C0085
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001384: BF870001
	v_lshlrev_b32_e32 v14, 2, v14                              // 000000001388: 301C1C82
	v_cmp_lt_u32_e64 s2, v10, 1                                // 00000000138C: D4490002 0201030A
	s_delay_alu instid0(VALU_DEP_4)                            // 000000001394: BF870004
	v_min_u32_e32 v12, v12, v13                                // 000000001398: 26181B0C
	s_delay_alu instid0(VALU_DEP_2)                            // 00000000139C: BF870002
	s_and_saveexec_b64 s[2:3], s[2:3]                          // 0000000013A0: BE822102
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000013A4: BF870003
	v_add_nc_u32_e32 v13, v11, v14                             // 0000000013A8: 4A1A1D0B
	ds_store_b32 v13, v12                                      // 0000000013AC: D8340000 00000C0D
	s_mov_b64 exec, s[2:3]                                     // 0000000013B4: BEFE0102
	s_waitcnt lgkmcnt(0)                                       // 0000000013B8: BF89FC07
	s_barrier                                                  // 0000000013BC: BFBD0000
	v_cmp_lt_u32_e64 s2, v10, 8                                // 0000000013C0: D4490002 0201110A
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000013C8: BF870001
	s_and_saveexec_b64 s[2:3], s[2:3]                          // 0000000013CC: BE822102
	v_lshlrev_b32_e32 v12, 2, v10                              // 0000000013D0: 30181482
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000013D4: BF870001
	v_add_nc_u32_e32 v11, v11, v12                             // 0000000013D8: 4A16190B
	ds_load_b32 v11, v11                                       // 0000000013DC: D8D80000 0B00000B
	v_xor_b32_e32 v12, 4, v10                                  // 0000000013E4: 3A181484
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000013E8: BF870001
	v_lshlrev_b32_e32 v12, 2, v12                              // 0000000013EC: 30181882
	s_waitcnt lgkmcnt(0)                                       // 0000000013F0: BF89FC07
	ds_bpermute_b32 v12, v12, v11                              // 0000000013F4: DACC0000 0C000B0C
	v_xor_b32_e32 v13, 2, v10                                  // 0000000013FC: 3A1A1482
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001400: BF870001
	v_lshlrev_b32_e32 v13, 2, v13                              // 000000001404: 301A1A82
	s_waitcnt lgkmcnt(0)                                       // 000000001408: BF89FC07
	v_min_u32_e32 v11, v11, v12                                // 00000000140C: 2616190B
	ds_bpermute_b32 v12, v13, v11                              // 000000001410: DACC0000 0C000B0D
	v_xor_b32_e32 v10, 1, v10                                  // 000000001418: 3A141481
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000141C: BF870001
	v_lshlrev_b32_e32 v10, 2, v10                              // 000000001420: 30141482
	s_waitcnt lgkmcnt(0)                                       // 000000001424: BF89FC07
	s_delay_alu instid0(VALU_DEP_3)                            // 000000001428: BF870003
	v_min_u32_e32 v11, v11, v12                                // 00000000142C: 2616190B
	ds_bpermute_b32 v10, v10, v11                              // 000000001430: DACC0000 0A000B0A
	s_waitcnt lgkmcnt(0)                                       // 000000001438: BF89FC07
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000143C: BF870001
	v_min_u32_e32 v10, v11, v10                                // 000000001440: 2614150B
	s_mov_b64 exec, s[2:3]                                     // 000000001444: BEFE0102
	v_mov_b32_e32 v11, 0                                       // 000000001448: 7E160280
	ds_bpermute_b32 v10, v11, v10                              // 00000000144C: DACC0000 0A000A0B
	v_cmp_eq_i32_e64 s2, v0, 0                                 // 000000001454: D4420002 02010100
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000145C: BF870001
	s_and_saveexec_b64 s[2:3], s[2:3]                          // 000000001460: BE822102
	s_cbranch_scc0 56                                          // 000000001464: BFA10038 <select+0x548>
	s_add_u32 s4, s0, s1                                       // 000000001468: 80040100
	s_lshl_b32 s6, s4, 2                                       // 00000000146C: 84068204
	s_waitcnt lgkmcnt(0)                                       // 000000001470: BF89FC07
	s_add_u32 s6, s16, s6                                      // 000000001474: 80060610
	s_mov_b32 s8, 0                                            // 000000001478: BE880080
	s_addc_u32 s7, s17, s8                                     // 00000000147C: 82070811
	v_mov_b32_e32 v11, 0                                       // 000000001480: 7E160280
	s_waitcnt lgkmcnt(0)                                       // 000000001484: BF89FC07
	global_store_b32 v11, v9, s[6:7]                           // 000000001488: DC6A0000 0006090B
	s_lshl_b32 s4, s4, 2                                       // 000000001490: 84048204
	s_add_u32 s6, s18, s4                                      // 000000001494: 80060412
	s_addc_u32 s7, s19, s8                                     // 000000001498: 82070813
	global_store_b32 v11, v10, s[6:7]                          // 00000000149C: DC6A0000 00060A0B
	s_branch 40                                                // 0000000014A4: BFA00028 <select+0x548>
	s_waitcnt lgkmcnt(0)                                       // 0000000014A8: BF89FC07
	v_cmp_eq_i32_e64 s2, v1, v10                               // 0000000014AC: D4420002 02021501
	s_waitcnt lgkmcnt(0)                                       // 0000000014B4: BF89FC07
	v_cmp_eq_i32_e64 s6, v2, v10                               // 0000000014B8: D4420006 02021502
	v_cmp_eq_i32_e64 s8, v3, v10                               // 0000000014C0: D4420008 02021503
	v_cmp_eq_i32_e64 s10, v4, v10                              // 0000000014C8: D442000A 02021504
	s_waitcnt vmcnt(0)                                         // 0000000014D0: BF8903F7
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000014D4: BF870004
	v_cndmask_b32_e64 v5, v5, 0xff800000, s2                   // 0000000014D8: D5010005 0009FF05 FF800000
	v_cndmask_b32_e64 v1, v1, 0x7fffffff, s2                   // 0000000014E4: D5010001 0009FF01 7FFFFFFF
	v_cndmask_b32_e64 v6, v6, 0xff800000, s6                   // 0000000014F0: D5010006 0019FF06 FF800000
	v_cndmask_b32_e64 v2, v2, 0x7fffffff, s6                   // 0000000014FC: D5010002 0019FF02 7FFFFFFF
	v_cndmask_b32_e64 v7, v7, 0xff800000, s8                   // 000000001508: D5010007 0021FF07 FF800000
	v_cndmask_b32_e64 v3, v3, 0x7fffffff, s8                   // 000000001514: D5010003 0021FF03 7FFFFFFF
	v_cndmask_b32_e64 v8, v8, 0xff800000, s10                  // 000000001520: D5010008 0029FF08 FF800000
	v_cndmask_b32_e64 v4, v4, 0x7fffffff, s10                  // 00000000152C: D5010004 0029FF04 7FFFFFFF
	s_add_u32 s1, s1, 1                                        // 000000001538: 80018101
	s_branch 65288                                             // 00000000153C: BFA0FF08 <select+0x160>
	s_waitcnt_vscnt null, 0x0                                  // 000000001540: BC7C0000
	s_endpgm                                                   // 000000001544: BFB00000
	s_mov_b64 exec, s[2:3]                                     // 000000001548: BEFE0102
	s_branch 65494                                             // 00000000154C: BFA0FFD6 <select+0x4a8>
