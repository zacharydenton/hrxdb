
hrx-cache/kernels/b3eb48d810fba9a5cd35a67c3e979536c23524665c64e41dd0f391382de8aaeb/kernel.hsaco:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001000 <scan>:
	s_load_b64 s[4:5], s[0:1], null                            // 000000001000: F4040100 F8000000
	s_mov_b32 s3, 0x1800                                       // 000000001008: BE8300FF 00001800
	s_mov_b32 s6, 0                                            // 000000001010: BE860080
	s_mulk_i32 s6, 0x1800                                      // 000000001014: B8061800
	s_mul_hi_u32 s7, s2, s3                                    // 000000001018: 96870302
	s_mul_i32 s3, s2, s3                                       // 00000000101C: 96030302
	s_add_u32 s8, s7, s6                                       // 000000001020: 80080607
	s_waitcnt lgkmcnt(0)                                       // 000000001024: BF89FC07
	s_add_u32 s10, s4, s3                                      // 000000001028: 800A0304
	s_addc_u32 s11, s5, s8                                     // 00000000102C: 820B0805
	s_add_u32 s8, s7, s6                                       // 000000001030: 80080607
	s_add_u32 s12, s4, s3                                      // 000000001034: 800C0304
	s_addc_u32 s13, s5, s8                                     // 000000001038: 820D0805
	s_add_u32 s8, s7, s6                                       // 00000000103C: 80080607
	v_and_b32_e32 v1, 31, v0                                   // 000000001040: 3602009F
	v_lshrrev_b32_e32 v0, 5, v0                                // 000000001044: 32000085
	s_add_u32 s14, s4, s3                                      // 000000001048: 800E0304
	s_delay_alu instid0(VALU_DEP_1) | instid1(VALU_DEP_2)      // 00000000104C: BF870101
	v_mad_u32_u24 v2, v0, 0xc0, v1                             // 000000001050: D60B0002 0405FF00 000000C0
	s_addc_u32 s15, s5, s8                                     // 00000000105C: 820F0805
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001060: BF870001
	v_lshlrev_b32_e32 v2, 3, v2                                // 000000001064: 30040483
	s_add_u32 s8, s7, s6                                       // 000000001068: 80080607
	global_load_b64 v[4:5], v2, s[10:11]                       // 00000000106C: DC560000 040A0002
	global_load_b64 v[6:7], v2, s[12:13] offset:768            // 000000001074: DC560300 060C0002
	s_add_u32 s10, s4, s3                                      // 00000000107C: 800A0304
	s_addc_u32 s11, s5, s8                                     // 000000001080: 820B0805
	s_add_u32 s8, s7, s6                                       // 000000001084: 80080607
	s_add_u32 s12, s4, s3                                      // 000000001088: 800C0304
	global_load_b64 v[8:9], v2, s[14:15] offset:256            // 00000000108C: DC560100 080E0002
	global_load_b64 v[10:11], v2, s[10:11] offset:1024         // 000000001094: DC560400 0A0A0002
	s_addc_u32 s13, s5, s8                                     // 00000000109C: 820D0805
	s_add_u32 s6, s7, s6                                       // 0000000010A0: 80060607
	s_add_u32 s4, s4, s3                                       // 0000000010A4: 80040304
	s_addc_u32 s5, s5, s6                                      // 0000000010A8: 82050605
	global_load_b64 v[12:13], v2, s[12:13] offset:512          // 0000000010AC: DC560200 0C0C0002
	global_load_b64 v[2:3], v2, s[4:5] offset:1280             // 0000000010B4: DC560500 02040002
	s_waitcnt vmcnt(5)                                         // 0000000010BC: BF8917F7
	v_cvt_f32_f16_e64 v14, v4.l                                // 0000000010C0: D58B000E 02010104
	v_lshrrev_b32_e32 v4, 16, v4                               // 0000000010C8: 32080890
	s_waitcnt vmcnt(4)                                         // 0000000010CC: BF8913F7
	v_cvt_f32_f16_e64 v15, v6.l                                // 0000000010D0: D58B000F 02010106
	v_lshrrev_b32_e32 v6, 16, v6                               // 0000000010D8: 320C0C90
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000010DC: BF870003
	v_cvt_f32_f16_e64 v4, v4.l                                 // 0000000010E0: D58B0004 02010104
	v_add_f32_e32 v14, lit(0x0), v14                           // 0000000010E8: 061C1CFF 00000000
	s_delay_alu instid0(VALU_DEP_1) | instid1(VALU_DEP_2)      // 0000000010F0: BF870101
	v_add_f32_e32 v4, v14, v4                                  // 0000000010F4: 0608090E
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000010F8: BF870004
	v_cvt_f32_f16_e64 v6, v6.l                                 // 0000000010FC: D58B0006 02010106
	v_add_f32_e32 v14, lit(0x0), v15                           // 000000001104: 061C1EFF 00000000
	s_delay_alu instid0(VALU_DEP_1) | instid1(VALU_DEP_2)      // 00000000110C: BF870101
	v_add_f32_e32 v6, v14, v6                                  // 000000001110: 060C0D0E
	v_cvt_f32_f16_e64 v14, v5.l                                // 000000001114: D58B000E 02010105
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000111C: BF870001
	v_add_f32_e32 v4, v4, v14                                  // 000000001120: 06081D04
	v_lshrrev_b32_e32 v5, 16, v5                               // 000000001124: 320A0A90
	v_cvt_f32_f16_e64 v14, v7.l                                // 000000001128: D58B000E 02010107
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001130: BF870001
	v_add_f32_e32 v6, v6, v14                                  // 000000001134: 060C1D06
	v_lshrrev_b32_e32 v7, 16, v7                               // 000000001138: 320E0E90
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000113C: BF870004
	v_cvt_f32_f16_e64 v5, v5.l                                 // 000000001140: D58B0005 02010105
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001148: BF870001
	v_add_f32_e32 v4, v4, v5                                   // 00000000114C: 06080B04
	s_delay_alu instid0(VALU_DEP_3)                            // 000000001150: BF870003
	v_cvt_f32_f16_e64 v5, v7.l                                 // 000000001154: D58B0005 02010107
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000115C: BF870001
	v_add_f32_e32 v5, v6, v5                                   // 000000001160: 060A0B06
	s_waitcnt vmcnt(3)                                         // 000000001164: BF890FF7
	v_cvt_f32_f16_e64 v6, v8.l                                 // 000000001168: D58B0006 02010108
	s_delay_alu instid0(VALU_DEP_1) | instid1(VALU_DEP_4)      // 000000001170: BF870201
	v_add_f32_e32 v4, v4, v6                                   // 000000001174: 06080D04
	v_lshrrev_b32_e32 v6, 16, v8                               // 000000001178: 320C1090
	s_waitcnt vmcnt(2)                                         // 00000000117C: BF890BF7
	v_cvt_f32_f16_e64 v7, v10.l                                // 000000001180: D58B0007 0201010A
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001188: BF870001
	v_add_f32_e32 v5, v5, v7                                   // 00000000118C: 060A0F05
	v_lshrrev_b32_e32 v7, 16, v10                              // 000000001190: 320E1490
	s_delay_alu instid0(VALU_DEP_4)                            // 000000001194: BF870004
	v_cvt_f32_f16_e64 v6, v6.l                                 // 000000001198: D58B0006 02010106
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000011A0: BF870001
	v_add_f32_e32 v4, v4, v6                                   // 0000000011A4: 06080D04
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000011A8: BF870003
	v_cvt_f32_f16_e64 v6, v7.l                                 // 0000000011AC: D58B0006 02010107
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000011B4: BF870001
	v_add_f32_e32 v5, v5, v6                                   // 0000000011B8: 060A0D05
	v_cvt_f32_f16_e64 v6, v9.l                                 // 0000000011BC: D58B0006 02010109
	s_delay_alu instid0(VALU_DEP_1) | instid1(VALU_DEP_4)      // 0000000011C4: BF870201
	v_add_f32_e32 v4, v4, v6                                   // 0000000011C8: 06080D04
	v_lshrrev_b32_e32 v6, 16, v9                               // 0000000011CC: 320C1290
	v_cvt_f32_f16_e64 v7, v11.l                                // 0000000011D0: D58B0007 0201010B
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000011D8: BF870001
	v_add_f32_e32 v5, v5, v7                                   // 0000000011DC: 060A0F05
	v_lshrrev_b32_e32 v7, 16, v11                              // 0000000011E0: 320E1690
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000011E4: BF870004
	v_cvt_f32_f16_e64 v6, v6.l                                 // 0000000011E8: D58B0006 02010106
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000011F0: BF870001
	v_add_f32_e32 v4, v4, v6                                   // 0000000011F4: 06080D04
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000011F8: BF870003
	v_cvt_f32_f16_e64 v6, v7.l                                 // 0000000011FC: D58B0006 02010107
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001204: BF870001
	v_add_f32_e32 v5, v5, v6                                   // 000000001208: 060A0D05
	s_waitcnt vmcnt(1)                                         // 00000000120C: BF8907F7
	v_cvt_f32_f16_e64 v6, v12.l                                // 000000001210: D58B0006 0201010C
	s_delay_alu instid0(VALU_DEP_1) | instid1(VALU_DEP_4)      // 000000001218: BF870201
	v_add_f32_e32 v4, v4, v6                                   // 00000000121C: 06080D04
	v_lshrrev_b32_e32 v6, 16, v12                              // 000000001220: 320C1890
	s_waitcnt vmcnt(0)                                         // 000000001224: BF8903F7
	v_cvt_f32_f16_e64 v7, v2.l                                 // 000000001228: D58B0007 02010102
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001230: BF870001
	v_add_f32_e32 v5, v5, v7                                   // 000000001234: 060A0F05
	v_lshrrev_b32_e32 v2, 16, v2                               // 000000001238: 32040490
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000123C: BF870004
	v_cvt_f32_f16_e64 v6, v6.l                                 // 000000001240: D58B0006 02010106
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001248: BF870001
	v_add_f32_e32 v4, v4, v6                                   // 00000000124C: 06080D04
	s_delay_alu instid0(VALU_DEP_3)                            // 000000001250: BF870003
	v_cvt_f32_f16_e64 v2, v2.l                                 // 000000001254: D58B0002 02010102
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000125C: BF870001
	v_add_f32_e32 v2, v5, v2                                   // 000000001260: 06040505
	v_cvt_f32_f16_e64 v5, v13.l                                // 000000001264: D58B0005 0201010D
	s_delay_alu instid0(VALU_DEP_1) | instid1(VALU_DEP_4)      // 00000000126C: BF870201
	v_add_f32_e32 v4, v4, v5                                   // 000000001270: 06080B04
	v_lshrrev_b32_e32 v5, 16, v13                              // 000000001274: 320A1A90
	v_cvt_f32_f16_e64 v6, v3.l                                 // 000000001278: D58B0006 02010103
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001280: BF870001
	v_add_f32_e32 v2, v2, v6                                   // 000000001284: 06040D02
	v_lshrrev_b32_e32 v3, 16, v3                               // 000000001288: 32060690
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000128C: BF870004
	v_cvt_f32_f16_e64 v5, v5.l                                 // 000000001290: D58B0005 02010105
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001298: BF870001
	v_add_f32_e32 v4, v4, v5                                   // 00000000129C: 06080B04
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000012A0: BF870003
	v_cvt_f32_f16_e64 v3, v3.l                                 // 0000000012A4: D58B0003 02010103
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000012AC: BF870001
	v_add_f32_e32 v2, v2, v3                                   // 0000000012B0: 06040702
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000012B4: BF870003
	v_add_f32_dpp v3, v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000012B8: 060608FA FF08B104
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000012C0: BF870001
	v_add_f32_dpp v3, v3, v3 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000012C4: 060606FA FF084E03
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000012CC: BF870001
	v_add_f32_dpp v3, v3, v3 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000012D0: 060606FA FF094103
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000012D8: BF870001
	v_add_f32_dpp v3, v3, v3 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000012DC: 060606FA FF094003
	v_add_f32_dpp v2, v2, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000012E4: 060404FA FF08B102
	s_load_b64 s[4:5], s[0:1], 0x18                            // 0000000012EC: F4040100 F8000018
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000012F4: BF870001
	v_add_f32_dpp v2, v2, v2 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000012F8: 060404FA FF084E02
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001300: BF870001
	v_add_f32_dpp v2, v2, v2 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001304: 060404FA FF094102
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000130C: BF870001
	v_add_f32_dpp v2, v2, v2 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001310: 060404FA FF094002
	v_permlanex16_b32 v4, v3, 0, 0                             // 000000001318: D65C0004 02010103
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001320: BF870001
	v_add_f32_e32 v3, v3, v4                                   // 000000001324: 06060903
	s_delay_alu instid0(VALU_DEP_3)                            // 000000001328: BF870003
	v_permlanex16_b32 v4, v2, 0, 0                             // 00000000132C: D65C0004 02010102
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001334: BF870001
	v_add_f32_e32 v2, v2, v4                                   // 000000001338: 06040902
	v_cmp_eq_i32_e64 s0, v1, 0                                 // 00000000133C: D4420000 02010101
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001344: BF870001
	s_and_saveexec_b64 s[6:7], s[0:1]                          // 000000001348: BE862100
	s_cbranch_scc0 21                                          // 00000000134C: BFA10015 <scan+0x3a4>
	s_lshl_b32 s3, s2, 5                                       // 000000001350: 84038502
	s_waitcnt lgkmcnt(0)                                       // 000000001354: BF89FC07
	s_add_u32 s8, s4, s3                                       // 000000001358: 80080304
	s_mov_b32 s3, 0                                            // 00000000135C: BE830080
	s_addc_u32 s9, s5, s3                                      // 000000001360: 82090305
	v_lshlrev_b32_e32 v1, 3, v0                                // 000000001364: 30020083
	global_store_b32 v1, v3, s[8:9]                            // 000000001368: DC6A0000 00080301
	s_branch 12                                                // 000000001370: BFA0000C <scan+0x3a4>
	s_and_saveexec_b64 s[0:1], s[0:1]                          // 000000001374: BE802100
	s_cbranch_scc0 12                                          // 000000001378: BFA1000C <scan+0x3ac>
	s_lshl_b32 s2, s2, 5                                       // 00000000137C: 84028502
	s_waitcnt lgkmcnt(0)                                       // 000000001380: BF89FC07
	s_add_u32 s8, s4, s2                                       // 000000001384: 80080204
	s_mov_b32 s2, 0                                            // 000000001388: BE820080
	s_addc_u32 s9, s5, s2                                      // 00000000138C: 82090205
	v_lshlrev_b32_e32 v0, 3, v0                                // 000000001390: 30000083
	s_waitcnt vmcnt(0)                                         // 000000001394: BF8903F7
	global_store_b32 v0, v2, s[8:9] offset:4                   // 000000001398: DC6A0004 00080200
	s_branch 2                                                 // 0000000013A0: BFA00002 <scan+0x3ac>
	s_mov_b64 exec, s[6:7]                                     // 0000000013A4: BEFE0106
	s_branch 65522                                             // 0000000013A8: BFA0FFF2 <scan+0x374>
	s_mov_b64 exec, s[0:1]                                     // 0000000013AC: BEFE0100
	s_waitcnt_vscnt null, 0x0                                  // 0000000013B0: BC7C0000
	s_endpgm                                                   // 0000000013B4: BFB00000
