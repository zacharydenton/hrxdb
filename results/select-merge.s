
hrx-cache/kernels/12a27c46ab742d6370f0f1dda0f2218befef5749a87af530500688c11b50486b/kernel.hsaco:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001000 <select>:
	s_load_b64 s[4:5], s[0:1], null                            // 000000001000: F4040100 F8000000
	s_load_b256 s[8:15], s[0:1], 0x8                           // 000000001008: F40C0200 F8000008
	v_lshl_add_u32 v1, s2, 10, v0                              // 000000001010: D6460001 04011402
	s_mov_b32 s0, 0x100                                        // 000000001018: BE8000FF 00000100
	v_mov_b32_e32 v2, 0x100                                    // 000000001020: 7E0402FF 00000100
	s_mov_b32 s1, 3                                            // 000000001028: BE810083
	s_delay_alu instid0(VALU_DEP_1) | instid1(VALU_DEP_2)      // 00000000102C: BF870101
	v_add_nc_u32_e32 v2, v1, v2                                // 000000001030: 4A040501
	v_lshl_add_u32 v3, s0, 1, v1                               // 000000001034: D6460003 04050200
	v_lshl_add_u32 v4, s1, 8, v1                               // 00000000103C: D6460004 04051001
	s_waitcnt lgkmcnt(0)                                       // 000000001044: BF89FC07
	v_mov_b32_e32 v5, s4                                       // 000000001048: 7E0A0204
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000104C: BF870001
	v_cmp_lt_u32_e64 s0, v1, v5                                // 000000001050: D4490000 02020B01
	v_cmp_lt_u32_e64 s6, v2, v5                                // 000000001058: D4490006 02020B02
	v_cmp_lt_u32_e64 s16, v3, v5                               // 000000001060: D4490010 02020B03
	v_cmp_lt_u32_e64 s18, v4, v5                               // 000000001068: D4490012 02020B04
	s_delay_alu instid0(VALU_DEP_4)                            // 000000001070: BF870004
	v_cndmask_b32_e64 v1, 0, v1, s0                            // 000000001074: D5010001 00020280
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000107C: BF870004
	v_cndmask_b32_e64 v2, 0, v2, s6                            // 000000001080: D5010002 001A0480
	s_delay_alu instid0(VALU_DEP_4)                            // 000000001088: BF870004
	v_cndmask_b32_e64 v3, 0, v3, s16                           // 00000000108C: D5010003 00420680
	s_delay_alu instid0(VALU_DEP_4)                            // 000000001094: BF870004
	v_cndmask_b32_e64 v4, 0, v4, s18                           // 000000001098: D5010004 004A0880
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000010A0: BF870004
	v_lshlrev_b32_e32 v1, 2, v1                                // 0000000010A4: 30020282
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000010A8: BF870004
	v_lshlrev_b32_e32 v2, 2, v2                                // 0000000010AC: 30040482
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000010B0: BF870004
	v_lshlrev_b32_e32 v3, 2, v3                                // 0000000010B4: 30060682
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000010B8: BF870004
	v_lshlrev_b32_e32 v4, 2, v4                                // 0000000010BC: 30080882
	global_load_b32 v5, v1, s[8:9]                             // 0000000010C0: DC520000 05080001
	global_load_b32 v1, v1, s[10:11]                           // 0000000010C8: DC520000 010A0001
	global_load_b32 v6, v2, s[8:9]                             // 0000000010D0: DC520000 06080002
	global_load_b32 v2, v2, s[10:11]                           // 0000000010D8: DC520000 020A0002
	global_load_b32 v7, v3, s[8:9]                             // 0000000010E0: DC520000 07080003
	global_load_b32 v8, v4, s[8:9]                             // 0000000010E8: DC520000 08080004
	global_load_b32 v3, v3, s[10:11]                           // 0000000010F0: DC520000 030A0003
	global_load_b32 v4, v4, s[10:11]                           // 0000000010F8: DC520000 040A0004
	s_mul_i32 s2, s2, s5                                       // 000000001100: 96020502
	s_mov_b32 s3, 0                                            // 000000001104: BE830080
	s_waitcnt vmcnt(7)                                         // 000000001108: BF891FF7
	v_cndmask_b32_e64 v5, 0xff800000, v5, s0                   // 00000000110C: D5010005 00020AFF FF800000
	s_waitcnt vmcnt(6)                                         // 000000001118: BF891BF7
	v_cndmask_b32_e64 v1, 0x7fffffff, v1, s0                   // 00000000111C: D5010001 000202FF 7FFFFFFF
	s_waitcnt vmcnt(5)                                         // 000000001128: BF8917F7
	v_cndmask_b32_e64 v6, 0xff800000, v6, s6                   // 00000000112C: D5010006 001A0CFF FF800000
	s_waitcnt vmcnt(4)                                         // 000000001138: BF8913F7
	v_cndmask_b32_e64 v2, 0x7fffffff, v2, s6                   // 00000000113C: D5010002 001A04FF 7FFFFFFF
	s_waitcnt vmcnt(3)                                         // 000000001148: BF890FF7
	v_cndmask_b32_e64 v7, 0xff800000, v7, s16                  // 00000000114C: D5010007 00420EFF FF800000
	s_waitcnt vmcnt(2)                                         // 000000001158: BF890BF7
	v_cndmask_b32_e64 v8, 0xff800000, v8, s18                  // 00000000115C: D5010008 004A10FF FF800000
	s_waitcnt vmcnt(1)                                         // 000000001168: BF8907F7
	v_cndmask_b32_e64 v3, 0x7fffffff, v3, s16                  // 00000000116C: D5010003 004206FF 7FFFFFFF
	s_waitcnt vmcnt(0)                                         // 000000001178: BF8903F7
	v_cndmask_b32_e64 v4, 0x7fffffff, v4, s18                  // 00000000117C: D5010004 004A08FF 7FFFFFFF
	s_cmp_lt_i32 s3, s5                                        // 000000001188: BF040503
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000118C: BF870009
	s_cbranch_scc0 247                                         // 000000001190: BFA100F7 <select+0x570>
	s_waitcnt lgkmcnt(0)                                       // 000000001194: BF89FC07
	v_max_f32_e32 v9, 0xff800000, v5                           // 000000001198: 20120AFF FF800000
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000011A0: BF870001
	v_max_f32_e32 v9, v9, v6                                   // 0000000011A4: 20120D09
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000011A8: BF870001
	v_max_f32_e32 v9, v9, v7                                   // 0000000011AC: 20120F09
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000011B0: BF870001
	v_max_f32_e32 v9, v9, v8                                   // 0000000011B4: 20121109
	v_and_b32_e32 v10, 31, v0                                  // 0000000011B8: 3614009F
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 0000000011BC: BF890007
	s_waitcnt_vscnt null, 0x0                                  // 0000000011C0: BC7C0000
	v_mov_b32_e32 v11, 0                                       // 0000000011C4: 7E160280
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000011C8: BF870003
	v_max_f32_dpp v9, v9, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000011CC: 201212FA FF08B109
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000011D4: BF870001
	v_max_f32_dpp v9, v9, v9 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000011D8: 201212FA FF084E09
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000011E0: BF870001
	v_max_f32_dpp v9, v9, v9 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000011E4: 201212FA FF094109
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000011EC: BF870001
	v_max_f32_dpp v9, v9, v9 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000011F0: 201212FA FF094009
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000011F8: BF870001
	v_permlanex16_b32 v12, v9, 0, 0                            // 0000000011FC: D65C000C 02010109
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001204: BF870001
	v_max_f32_e32 v9, v9, v12                                  // 000000001208: 20121909
	v_lshrrev_b32_e32 v12, 5, v0                               // 00000000120C: 32180085
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001210: BF870001
	v_lshlrev_b32_e32 v12, 2, v12                              // 000000001214: 30181882
	v_cmp_lt_u32_e64 s0, v10, 1                                // 000000001218: D4490000 0201030A
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001220: BF870001
	s_and_saveexec_b64 s[0:1], s[0:1]                          // 000000001224: BE802100
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001228: BF870002
	v_add_nc_u32_e32 v12, v11, v12                             // 00000000122C: 4A18190B
	ds_store_b32 v12, v9                                       // 000000001230: D8340000 0000090C
	s_mov_b64 exec, s[0:1]                                     // 000000001238: BEFE0100
	s_waitcnt lgkmcnt(0)                                       // 00000000123C: BF89FC07
	s_barrier                                                  // 000000001240: BFBD0000
	v_cmp_lt_u32_e64 s0, v10, 8                                // 000000001244: D4490000 0201110A
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000124C: BF870001
	s_and_saveexec_b64 s[0:1], s[0:1]                          // 000000001250: BE802100
	v_lshlrev_b32_e32 v9, 2, v10                               // 000000001254: 30121482
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001258: BF870001
	v_add_nc_u32_e32 v9, v11, v9                               // 00000000125C: 4A12130B
	ds_load_b32 v9, v9                                         // 000000001260: D8D80000 09000009
	v_xor_b32_e32 v11, 4, v10                                  // 000000001268: 3A161484
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000126C: BF870001
	v_lshlrev_b32_e32 v11, 2, v11                              // 000000001270: 30161682
	s_waitcnt lgkmcnt(0)                                       // 000000001274: BF89FC07
	ds_bpermute_b32 v11, v11, v9                               // 000000001278: DACC0000 0B00090B
	v_xor_b32_e32 v12, 2, v10                                  // 000000001280: 3A181482
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001284: BF870001
	v_lshlrev_b32_e32 v12, 2, v12                              // 000000001288: 30181882
	s_waitcnt lgkmcnt(0)                                       // 00000000128C: BF89FC07
	v_max_f32_e32 v9, v9, v11                                  // 000000001290: 20121709
	ds_bpermute_b32 v11, v12, v9                               // 000000001294: DACC0000 0B00090C
	v_xor_b32_e32 v10, 1, v10                                  // 00000000129C: 3A141481
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000012A0: BF870001
	v_lshlrev_b32_e32 v10, 2, v10                              // 0000000012A4: 30141482
	s_waitcnt lgkmcnt(0)                                       // 0000000012A8: BF89FC07
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000012AC: BF870003
	v_max_f32_e32 v9, v9, v11                                  // 0000000012B0: 20121709
	ds_bpermute_b32 v10, v10, v9                               // 0000000012B4: DACC0000 0A00090A
	s_waitcnt lgkmcnt(0)                                       // 0000000012BC: BF89FC07
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000012C0: BF870001
	v_max_f32_e32 v9, v9, v10                                  // 0000000012C4: 20121509
	s_mov_b64 exec, s[0:1]                                     // 0000000012C8: BEFE0100
	v_mov_b32_e32 v10, 0                                       // 0000000012CC: 7E140280
	ds_bpermute_b32 v9, v10, v9                                // 0000000012D0: DACC0000 0900090A
	v_mov_b32_e32 v10, 0x7fffffff                              // 0000000012D8: 7E1402FF 7FFFFFFF
	v_and_b32_e32 v11, 31, v0                                  // 0000000012E0: 3616009F
	s_waitcnt lgkmcnt(0)                                       // 0000000012E4: BF89FC07
	v_cmp_eq_f32_e64 s0, v5, v9                                // 0000000012E8: D4120000 02021305
	v_cmp_eq_f32_e64 s6, v6, v9                                // 0000000012F0: D4120006 02021306
	v_cmp_eq_f32_e64 s16, v7, v9                               // 0000000012F8: D4120010 02021307
	v_cmp_eq_f32_e64 s18, v8, v9                               // 000000001300: D4120012 02021308
	v_mov_b32_e32 v12, 32                                      // 000000001308: 7E1802A0
	v_cndmask_b32_e64 v13, 0x7fffffff, v1, s0                  // 00000000130C: D501000D 000202FF 7FFFFFFF
	v_cndmask_b32_e64 v14, 0x7fffffff, v2, s6                  // 000000001318: D501000E 001A04FF 7FFFFFFF
	v_cndmask_b32_e64 v15, 0x7fffffff, v3, s16                 // 000000001324: D501000F 004206FF 7FFFFFFF
	v_cndmask_b32_e64 v16, 0x7fffffff, v4, s18                 // 000000001330: D5010010 004A08FF 7FFFFFFF
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000133C: BF870004
	v_min_u32_e32 v10, v10, v13                                // 000000001340: 26141B0A
	s_delay_alu instid0(VALU_DEP_1) | instid1(VALU_DEP_4)      // 000000001344: BF870201
	v_min_u32_e32 v10, v10, v14                                // 000000001348: 26141D0A
	s_delay_alu instid0(VALU_DEP_1) | instid1(VALU_DEP_4)      // 00000000134C: BF870201
	v_min_u32_e32 v10, v10, v15                                // 000000001350: 26141F0A
	s_delay_alu instid0(VALU_DEP_1) | instid1(VALU_DEP_4)      // 000000001354: BF870201
	v_min_u32_e32 v10, v10, v16                                // 000000001358: 2614210A
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000135C: BF870001
	v_mov_b32_dpp v13, v10 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001360: 7E1A02FA FF08B10A
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001368: BF870001
	v_min_u32_e32 v10, v10, v13                                // 00000000136C: 26141B0A
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001370: BF870001
	v_mov_b32_dpp v13, v10 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001374: 7E1A02FA FF084E0A
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000137C: BF870001
	v_min_u32_e32 v10, v10, v13                                // 000000001380: 26141B0A
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001384: BF870001
	v_mov_b32_dpp v13, v10 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001388: 7E1A02FA FF09410A
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001390: BF870001
	v_min_u32_e32 v10, v10, v13                                // 000000001394: 26141B0A
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001398: BF870001
	v_mov_b32_dpp v13, v10 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 00000000139C: 7E1A02FA FF09400A
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000013A4: BF870001
	v_min_u32_e32 v10, v10, v13                                // 0000000013A8: 26141B0A
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000013AC: BF870001
	v_permlanex16_b32 v13, v10, 0, 0                           // 0000000013B0: D65C000D 0201010A
	v_lshrrev_b32_e32 v14, 5, v0                               // 0000000013B8: 321C0085
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000013BC: BF870001
	v_lshlrev_b32_e32 v14, 2, v14                              // 0000000013C0: 301C1C82
	v_cmp_lt_u32_e64 s0, v11, 1                                // 0000000013C4: D4490000 0201030B
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000013CC: BF870004
	v_min_u32_e32 v10, v10, v13                                // 0000000013D0: 26141B0A
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000013D4: BF870002
	s_and_saveexec_b64 s[0:1], s[0:1]                          // 0000000013D8: BE802100
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000013DC: BF870003
	v_add_nc_u32_e32 v13, v12, v14                             // 0000000013E0: 4A1A1D0C
	ds_store_b32 v13, v10                                      // 0000000013E4: D8340000 00000A0D
	s_mov_b64 exec, s[0:1]                                     // 0000000013EC: BEFE0100
	s_waitcnt lgkmcnt(0)                                       // 0000000013F0: BF89FC07
	s_barrier                                                  // 0000000013F4: BFBD0000
	v_cmp_lt_u32_e64 s0, v11, 8                                // 0000000013F8: D4490000 0201110B
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001400: BF870001
	s_and_saveexec_b64 s[0:1], s[0:1]                          // 000000001404: BE802100
	v_lshlrev_b32_e32 v10, 2, v11                              // 000000001408: 30141682
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000140C: BF870001
	v_add_nc_u32_e32 v10, v12, v10                             // 000000001410: 4A14150C
	ds_load_b32 v10, v10                                       // 000000001414: D8D80000 0A00000A
	v_xor_b32_e32 v12, 4, v11                                  // 00000000141C: 3A181684
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001420: BF870001
	v_lshlrev_b32_e32 v12, 2, v12                              // 000000001424: 30181882
	s_waitcnt lgkmcnt(0)                                       // 000000001428: BF89FC07
	ds_bpermute_b32 v12, v12, v10                              // 00000000142C: DACC0000 0C000A0C
	v_xor_b32_e32 v13, 2, v11                                  // 000000001434: 3A1A1682
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001438: BF870001
	v_lshlrev_b32_e32 v13, 2, v13                              // 00000000143C: 301A1A82
	s_waitcnt lgkmcnt(0)                                       // 000000001440: BF89FC07
	v_min_u32_e32 v10, v10, v12                                // 000000001444: 2614190A
	ds_bpermute_b32 v12, v13, v10                              // 000000001448: DACC0000 0C000A0D
	v_xor_b32_e32 v11, 1, v11                                  // 000000001450: 3A161681
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001454: BF870001
	v_lshlrev_b32_e32 v11, 2, v11                              // 000000001458: 30161682
	s_waitcnt lgkmcnt(0)                                       // 00000000145C: BF89FC07
	s_delay_alu instid0(VALU_DEP_3)                            // 000000001460: BF870003
	v_min_u32_e32 v10, v10, v12                                // 000000001464: 2614190A
	ds_bpermute_b32 v11, v11, v10                              // 000000001468: DACC0000 0B000A0B
	s_waitcnt lgkmcnt(0)                                       // 000000001470: BF89FC07
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001474: BF870001
	v_min_u32_e32 v10, v10, v11                                // 000000001478: 2614170A
	s_mov_b64 exec, s[0:1]                                     // 00000000147C: BEFE0100
	v_mov_b32_e32 v11, 0                                       // 000000001480: 7E160280
	ds_bpermute_b32 v10, v11, v10                              // 000000001484: DACC0000 0A000A0B
	v_cmp_eq_i32_e64 s0, v0, 0                                 // 00000000148C: D4420000 02010100
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001494: BF870001
	s_and_saveexec_b64 s[0:1], s[0:1]                          // 000000001498: BE802100
	s_cbranch_scc0 54                                          // 00000000149C: BFA10036 <select+0x578>
	s_add_u32 s4, s2, s3                                       // 0000000014A0: 80040302
	s_lshl_b32 s6, s4, 2                                       // 0000000014A4: 84068204
	s_add_u32 s6, s12, s6                                      // 0000000014A8: 8006060C
	s_mov_b32 s16, 0                                           // 0000000014AC: BE900080
	s_addc_u32 s7, s13, s16                                    // 0000000014B0: 8207100D
	v_mov_b32_e32 v11, 0                                       // 0000000014B4: 7E160280
	s_waitcnt lgkmcnt(0)                                       // 0000000014B8: BF89FC07
	global_store_b32 v11, v9, s[6:7]                           // 0000000014BC: DC6A0000 0006090B
	s_lshl_b32 s4, s4, 2                                       // 0000000014C4: 84048204
	s_add_u32 s6, s14, s4                                      // 0000000014C8: 8006040E
	s_addc_u32 s7, s15, s16                                    // 0000000014CC: 8207100F
	global_store_b32 v11, v10, s[6:7]                          // 0000000014D0: DC6A0000 00060A0B
	s_branch 39                                                // 0000000014D8: BFA00027 <select+0x578>
	s_waitcnt lgkmcnt(0)                                       // 0000000014DC: BF89FC07
	v_cmp_eq_i32_e64 s0, v1, v10                               // 0000000014E0: D4420000 02021501
	v_cmp_eq_i32_e64 s6, v2, v10                               // 0000000014E8: D4420006 02021502
	v_cmp_eq_i32_e64 s16, v3, v10                              // 0000000014F0: D4420010 02021503
	v_cmp_eq_i32_e64 s18, v4, v10                              // 0000000014F8: D4420012 02021504
	s_waitcnt vmcnt(0)                                         // 000000001500: BF8903F7
	s_delay_alu instid0(VALU_DEP_4)                            // 000000001504: BF870004
	v_cndmask_b32_e64 v5, v5, 0xff800000, s0                   // 000000001508: D5010005 0001FF05 FF800000
	v_cndmask_b32_e64 v1, v1, 0x7fffffff, s0                   // 000000001514: D5010001 0001FF01 7FFFFFFF
	v_cndmask_b32_e64 v6, v6, 0xff800000, s6                   // 000000001520: D5010006 0019FF06 FF800000
	v_cndmask_b32_e64 v2, v2, 0x7fffffff, s6                   // 00000000152C: D5010002 0019FF02 7FFFFFFF
	v_cndmask_b32_e64 v7, v7, 0xff800000, s16                  // 000000001538: D5010007 0041FF07 FF800000
	v_cndmask_b32_e64 v3, v3, 0x7fffffff, s16                  // 000000001544: D5010003 0041FF03 7FFFFFFF
	v_cndmask_b32_e64 v8, v8, 0xff800000, s18                  // 000000001550: D5010008 0049FF08 FF800000
	v_cndmask_b32_e64 v4, v4, 0x7fffffff, s18                  // 00000000155C: D5010004 0049FF04 7FFFFFFF
	s_add_u32 s3, s3, 1                                        // 000000001568: 80038103
	s_branch 65286                                             // 00000000156C: BFA0FF06 <select+0x188>
	s_waitcnt_vscnt null, 0x0                                  // 000000001570: BC7C0000
	s_endpgm                                                   // 000000001574: BFB00000
	s_mov_b64 exec, s[0:1]                                     // 000000001578: BEFE0100
	s_branch 65495                                             // 00000000157C: BFA0FFD7 <select+0x4dc>
