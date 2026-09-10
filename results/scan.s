
hrx-cache/kernels/eec59d8b64f44c11b1a65bbb6991ef73038a9e5ec64ca5c0d9916633bcec4497/kernel.hsaco:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001000 <scan>:
	s_load_b256 s[8:15], s[0:1], null                          // 000000001000: F40C0200 F8000000
	s_mov_b32 s0, 0x1800                                       // 000000001008: BE8000FF 00001800
	s_mov_b32 s1, 0                                            // 000000001010: BE810080
	s_mulk_i32 s1, 0x1800                                      // 000000001014: B8011800
	s_mul_hi_u32 s3, s2, s0                                    // 000000001018: 96830002
	s_mul_i32 s0, s2, s0                                       // 00000000101C: 96000002
	s_add_u32 s4, s3, s1                                       // 000000001020: 80040103
	s_waitcnt lgkmcnt(0)                                       // 000000001024: BF89FC07
	s_add_u32 s6, s8, s0                                       // 000000001028: 80060008
	s_addc_u32 s7, s9, s4                                      // 00000000102C: 82070409
	s_add_u32 s4, s3, s1                                       // 000000001030: 80040103
	s_add_u32 s16, s8, s0                                      // 000000001034: 80100008
	s_addc_u32 s17, s9, s4                                     // 000000001038: 82110409
	s_add_u32 s4, s3, s1                                       // 00000000103C: 80040103
	v_and_b32_e32 v1, 31, v0                                   // 000000001040: 3602009F
	v_lshrrev_b32_e32 v0, 5, v0                                // 000000001044: 32000085
	s_add_u32 s18, s8, s0                                      // 000000001048: 80120008
	s_delay_alu instid0(VALU_DEP_1) | instid1(VALU_DEP_2)      // 00000000104C: BF870101
	v_mad_u32_u24 v2, v0, 0xc0, v1                             // 000000001050: D60B0002 0405FF00 000000C0
	s_addc_u32 s19, s9, s4                                     // 00000000105C: 82130409
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001060: BF870001
	v_lshlrev_b32_e32 v2, 3, v2                                // 000000001064: 30040483
	s_add_u32 s4, s3, s1                                       // 000000001068: 80040103
	global_load_b64 v[4:5], v2, s[6:7]                         // 00000000106C: DC560000 04060002
	global_load_b64 v[6:7], v2, s[16:17] offset:768            // 000000001074: DC560300 06100002
	v_lshlrev_b32_e32 v3, 4, v1                                // 00000000107C: 30060284
	s_add_u32 s6, s8, s0                                       // 000000001080: 80060008
	s_addc_u32 s7, s9, s4                                      // 000000001084: 82070409
	global_load_b128 v[8:11], v3, s[10:11]                     // 000000001088: DC5E0000 080A0003
	s_add_u32 s4, s3, s1                                       // 000000001090: 80040103
	s_add_u32 s16, s8, s0                                      // 000000001094: 80100008
	global_load_b64 v[12:13], v2, s[18:19] offset:256          // 000000001098: DC560100 0C120002
	global_load_b64 v[14:15], v2, s[6:7] offset:1024           // 0000000010A0: DC560400 0E060002
	global_load_b128 v[16:19], v3, s[10:11] offset:512         // 0000000010A8: DC5E0200 100A0003
	global_load_b128 v[20:23], v3, s[10:11] offset:1024        // 0000000010B0: DC5E0400 140A0003
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v24, 0              // 0000000010B8: CA100080 03180080
	s_waitcnt vmcnt(6)                                         // 0000000010C0: BF891BF7
	v_cvt_f32_f16_e64 v25, v4.l                                // 0000000010C4: D58B0019 02010104
	v_lshrrev_b32_e32 v4, 16, v4                               // 0000000010CC: 32080890
	s_waitcnt vmcnt(5)                                         // 0000000010D0: BF8917F7
	v_cvt_f32_f16_e64 v26, v6.l                                // 0000000010D4: D58B001A 02010106
	v_lshrrev_b32_e32 v6, 16, v6                               // 0000000010DC: 320C0C90
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000010E0: BF870003
	v_cvt_f32_f16_e64 v4, v4.l                                 // 0000000010E4: D58B0004 02010104
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000010EC: BF870002
	v_cvt_f32_f16_e64 v6, v6.l                                 // 0000000010F0: D58B0006 02010106
	s_waitcnt vmcnt(4)                                         // 0000000010F8: BF8913F7
	s_delay_alu instid0(VALU_DEP_4)                            // 0000000010FC: BF870004
	v_dual_fmac_f32 v3, v8, v25 :: v_dual_fmac_f32 v24, v26, v8// 000000001100: C8003308 0318111A
	s_addc_u32 s17, s9, s4                                     // 000000001108: 82110409
	s_add_u32 s1, s3, s1                                       // 00000000110C: 80010103
	s_add_u32 s0, s8, s0                                       // 000000001110: 80000008
	s_addc_u32 s1, s9, s1                                      // 000000001114: 82010109
	global_load_b64 v[26:27], v2, s[16:17] offset:512          // 000000001118: DC560200 1A100002
	v_dual_fmac_f32 v3, v9, v4 :: v_dual_fmac_f32 v24, v6, v9  // 000000001120: C8000909 03181306
	v_cvt_f32_f16_e64 v4, v5.l                                 // 000000001128: D58B0004 02010105
	v_lshrrev_b32_e32 v5, 16, v5                               // 000000001130: 320A0A90
	v_cvt_f32_f16_e64 v6, v7.l                                 // 000000001134: D58B0006 02010107
	v_lshrrev_b32_e32 v7, 16, v7                               // 00000000113C: 320E0E90
	s_delay_alu instid0(VALU_DEP_3)                            // 000000001140: BF870003
	v_cvt_f32_f16_e64 v5, v5.l                                 // 000000001144: D58B0005 02010105
	s_delay_alu instid0(VALU_DEP_2)                            // 00000000114C: BF870002
	v_cvt_f32_f16_e64 v7, v7.l                                 // 000000001150: D58B0007 02010107
	v_fmac_f32_e32 v3, v4, v10                                 // 000000001158: 56061504
	s_delay_alu instid0(VALU_DEP_1) | instid1(VALU_DEP_3)      // 00000000115C: BF870181
	v_dual_fmac_f32 v24, v6, v10 :: v_dual_fmac_f32 v3, v5, v11// 000000001160: C8001506 18021705
	s_delay_alu instid0(VALU_DEP_1) | instid1(VALU_DEP_3)      // 000000001168: BF870181
	v_fmac_f32_e32 v24, v7, v11                                // 00000000116C: 56301707
	global_load_b64 v[4:5], v2, s[0:1] offset:1280             // 000000001170: DC560500 04000002
	s_waitcnt vmcnt(5)                                         // 000000001178: BF8917F7
	v_cvt_f32_f16_e64 v2, v12.l                                // 00000000117C: D58B0002 0201010C
	v_lshrrev_b32_e32 v6, 16, v12                              // 000000001184: 320C1890
	s_waitcnt vmcnt(4)                                         // 000000001188: BF8913F7
	v_cvt_f32_f16_e64 v7, v14.l                                // 00000000118C: D58B0007 0201010E
	v_lshrrev_b32_e32 v8, 16, v14                              // 000000001194: 32101C90
	s_delay_alu instid0(VALU_DEP_3)                            // 000000001198: BF870003
	v_cvt_f32_f16_e64 v6, v6.l                                 // 00000000119C: D58B0006 02010106
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000011A4: BF870002
	v_cvt_f32_f16_e64 v8, v8.l                                 // 0000000011A8: D58B0008 02010108
	s_waitcnt vmcnt(3)                                         // 0000000011B0: BF890FF7
	v_fma_f32 v2, v2, v16, v3                                  // 0000000011B4: D6130002 040E2102
	v_fma_f32 v3, v7, v16, v24                                 // 0000000011BC: D6130003 04622107
	s_nop 2                                                    // 0000000011C4: BF800002
	v_dual_fmac_f32 v2, v17, v6 :: v_dual_fmac_f32 v3, v8, v17 // 0000000011C8: C8000D11 02022308
	v_cvt_f32_f16_e64 v6, v13.l                                // 0000000011D0: D58B0006 0201010D
	v_lshrrev_b32_e32 v7, 16, v13                              // 0000000011D8: 320E1A90
	v_cvt_f32_f16_e64 v8, v15.l                                // 0000000011DC: D58B0008 0201010F
	v_lshrrev_b32_e32 v9, 16, v15                              // 0000000011E4: 32121E90
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000011E8: BF870003
	v_cvt_f32_f16_e64 v7, v7.l                                 // 0000000011EC: D58B0007 02010107
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000011F4: BF870002
	v_cvt_f32_f16_e64 v9, v9.l                                 // 0000000011F8: D58B0009 02010109
	v_fmac_f32_e32 v2, v6, v18                                 // 000000001200: 56042506
	s_delay_alu instid0(VALU_DEP_1) | instid1(VALU_DEP_3)      // 000000001204: BF870181
	v_dual_fmac_f32 v3, v8, v18 :: v_dual_fmac_f32 v2, v7, v19 // 000000001208: C8002508 03022707
	s_delay_alu instid0(VALU_DEP_1) | instid1(VALU_DEP_3)      // 000000001210: BF870181
	v_fmac_f32_e32 v3, v9, v19                                 // 000000001214: 56062709
	s_waitcnt vmcnt(1)                                         // 000000001218: BF8907F7
	v_cvt_f32_f16_e64 v6, v26.l                                // 00000000121C: D58B0006 0201011A
	v_lshrrev_b32_e32 v7, 16, v26                              // 000000001224: 320E3490
	s_waitcnt vmcnt(0)                                         // 000000001228: BF8903F7
	v_cvt_f32_f16_e64 v8, v4.l                                 // 00000000122C: D58B0008 02010104
	v_lshrrev_b32_e32 v4, 16, v4                               // 000000001234: 32080890
	s_delay_alu instid0(VALU_DEP_3)                            // 000000001238: BF870003
	v_cvt_f32_f16_e64 v7, v7.l                                 // 00000000123C: D58B0007 02010107
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001244: BF870002
	v_cvt_f32_f16_e64 v4, v4.l                                 // 000000001248: D58B0004 02010104
	v_fma_f32 v2, v6, v20, v2                                  // 000000001250: D6130002 040A2906
	v_fma_f32 v3, v8, v20, v3                                  // 000000001258: D6130003 040E2908
	s_nop 2                                                    // 000000001260: BF800002
	v_dual_fmac_f32 v2, v21, v7 :: v_dual_fmac_f32 v3, v4, v21 // 000000001264: C8000F15 02022B04
	v_cvt_f32_f16_e64 v4, v27.l                                // 00000000126C: D58B0004 0201011B
	v_lshrrev_b32_e32 v6, 16, v27                              // 000000001274: 320C3690
	v_cvt_f32_f16_e64 v7, v5.l                                 // 000000001278: D58B0007 02010105
	v_lshrrev_b32_e32 v5, 16, v5                               // 000000001280: 320A0A90
	s_delay_alu instid0(VALU_DEP_3)                            // 000000001284: BF870003
	v_cvt_f32_f16_e64 v6, v6.l                                 // 000000001288: D58B0006 02010106
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001290: BF870002
	v_cvt_f32_f16_e64 v5, v5.l                                 // 000000001294: D58B0005 02010105
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000129C: BF870004
	v_dual_fmac_f32 v2, v22, v4 :: v_dual_fmac_f32 v3, v7, v22 // 0000000012A0: C8000916 02022D07
	s_nop 2                                                    // 0000000012A8: BF800002
	v_dual_fmac_f32 v2, v23, v6 :: v_dual_fmac_f32 v3, v5, v23 // 0000000012AC: C8000D17 02022F05
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000012B4: BF870001
	v_add_f32_dpp v2, v2, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000012B8: 060404FA FF08B102
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000012C0: BF870001
	v_add_f32_dpp v2, v2, v2 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000012C4: 060404FA FF084E02
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000012CC: BF870001
	v_add_f32_dpp v2, v2, v2 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000012D0: 060404FA FF094102
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000012D8: BF870001
	v_add_f32_dpp v2, v2, v2 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000012DC: 060404FA FF094002
	v_add_f32_dpp v3, v3, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000012E4: 060606FA FF08B103
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000012EC: BF870001
	v_add_f32_dpp v3, v3, v3 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000012F0: 060606FA FF084E03
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000012F8: BF870001
	v_add_f32_dpp v3, v3, v3 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000012FC: 060606FA FF094103
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001304: BF870001
	v_add_f32_dpp v3, v3, v3 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001308: 060606FA FF094003
	v_permlanex16_b32 v4, v2, 0, 0                             // 000000001310: D65C0004 02010102
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001318: BF870001
	v_add_f32_e32 v2, v2, v4                                   // 00000000131C: 06040902
	s_delay_alu instid0(VALU_DEP_3)                            // 000000001320: BF870003
	v_permlanex16_b32 v4, v3, 0, 0                             // 000000001324: D65C0004 02010103
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000132C: BF870001
	v_add_f32_e32 v3, v3, v4                                   // 000000001330: 06060903
	v_cmp_eq_i32_e64 s0, v1, 0                                 // 000000001334: D4420000 02010101
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000133C: BF870001
	s_and_saveexec_b64 s[4:5], s[0:1]                          // 000000001340: BE842100
	s_cbranch_scc0 34                                          // 000000001344: BFA10022 <scan+0x3d0>
	s_lshl_b32 s3, s2, 5                                       // 000000001348: 84038502
	s_add_u32 s6, s12, s3                                      // 00000000134C: 8006030C
	s_mov_b32 s3, 0                                            // 000000001350: BE830080
	v_lshlrev_b32_e32 v1, 3, v0                                // 000000001354: 30020083
	s_addc_u32 s7, s13, s3                                     // 000000001358: 8207030D
	global_load_b32 v4, v1, s[6:7]                             // 00000000135C: DC520000 04060001
	s_lshl_b32 s6, s2, 5                                       // 000000001364: 84068502
	s_add_u32 s6, s14, s6                                      // 000000001368: 8006060E
	s_addc_u32 s7, s15, s3                                     // 00000000136C: 8207030F
	s_waitcnt vmcnt(0)                                         // 000000001370: BF8903F7
	v_mul_f32_e32 v2, v2, v4                                   // 000000001374: 10040902
	global_store_b32 v1, v2, s[6:7]                            // 000000001378: DC6A0000 00060201
	s_branch 19                                                // 000000001380: BFA00013 <scan+0x3d0>
	s_and_saveexec_b64 s[0:1], s[0:1]                          // 000000001384: BE802100
	s_cbranch_scc0 19                                          // 000000001388: BFA10013 <scan+0x3d8>
	s_lshl_b32 s3, s2, 5                                       // 00000000138C: 84038502
	s_add_u32 s6, s12, s3                                      // 000000001390: 8006030C
	s_mov_b32 s3, 0                                            // 000000001394: BE830080
	v_lshlrev_b32_e32 v0, 3, v0                                // 000000001398: 30000083
	s_addc_u32 s7, s13, s3                                     // 00000000139C: 8207030D
	s_waitcnt_vscnt null, 0x0                                  // 0000000013A0: BC7C0000
	global_load_b32 v1, v0, s[6:7] offset:4                    // 0000000013A4: DC520004 01060000
	s_lshl_b32 s2, s2, 5                                       // 0000000013AC: 84028502
	s_add_u32 s6, s14, s2                                      // 0000000013B0: 8006020E
	s_addc_u32 s7, s15, s3                                     // 0000000013B4: 8207030F
	s_waitcnt vmcnt(0)                                         // 0000000013B8: BF8903F7
	v_mul_f32_e32 v1, v3, v1                                   // 0000000013BC: 10020303
	s_waitcnt lgkmcnt(0)                                       // 0000000013C0: BF89FC07
	global_store_b32 v0, v1, s[6:7] offset:4                   // 0000000013C4: DC6A0004 00060100
	s_branch 2                                                 // 0000000013CC: BFA00002 <scan+0x3d8>
	s_mov_b64 exec, s[4:5]                                     // 0000000013D0: BEFE0104
	s_branch 65515                                             // 0000000013D4: BFA0FFEB <scan+0x384>
	s_mov_b64 exec, s[0:1]                                     // 0000000013D8: BEFE0100
	s_waitcnt_vscnt null, 0x0                                  // 0000000013DC: BC7C0000
	s_endpgm                                                   // 0000000013E0: BFB00000
