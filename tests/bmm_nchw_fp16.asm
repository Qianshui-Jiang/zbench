//.kernel bmm_nchw_fp16
//.platform DG2
//.thread_config numGRF=256, numAcc=8, numSWSB=16
//.options_string "-enableHalfLSC -dumpcommonisa -output -binary -printregusage -hasNoInt64Add -TotalGRFNum 256 -fusedCallWA 1 -abiver 2 -LSCFenceWA "
//.full_options "-abiver 2 -printregusage -TotalGRFNum 256 -output -binary -dumpcommonisa -enableHalfLSC -hasNoInt64Add -fusedCallWA 1 -LSCFenceWA "
//.instCount 95
//.RA type	TRIVIAL_RA
//.git-hash 61adfa08f1610456ac8dd5539a6657645ff14b77
//.GRF count 41

//.declare BuiltInR0 (0)  rf=r size=32 type=ud align=16 words (r0.0) IsBuiltin
//.declare  (1)  rf=r size=32 type=ud alias=BuiltInR0+0 align=16 words (r0.0) IsBuiltin
//.declare BuiltinA0 (2)  rf=a size=4 type=ud align=1 words (a0.0) IsBuiltin
//.declare BuiltinA0Dot2 (3)  rf=a size=4 type=ud align=1 words (a0.2) IsBuiltin
//.declare BuiltinSR0Dot1 (5)  rf=r size=4 type=ud align=2 words IsBuiltin
//.declare %null (10)  rf=r size=4 type=ud align=2 words
//.declare %local_id_x (13)  rf=r size=4 type=ud align=2 words (r3.7)
//.declare %local_id_y (14)  rf=r size=4 type=ud align=2 words (r4.0)
//.declare %local_size_x (15)  rf=r size=4 type=ud align=2 words (r3.3)
//.declare %local_size_y (16)  rf=r size=4 type=ud align=2 words (r3.4)
//.declare %group_id_x (17)  rf=r size=4 type=ud align=2 words (r0.1)
//.declare %group_id_y (18)  rf=r size=4 type=ud align=2 words (r0.6)
//.declare %group_id_z (19)  rf=r size=4 type=ud align=2 words (r0.7)
//.declare %group_count_x (20)  rf=r size=4 type=ud align=2 words (r3.5)
//.declare %group_count_y (21)  rf=r size=4 type=ud align=2 words (r3.6)
//.declare %tsc (22)  rf=r size=20 type=ud align=2 words
//.declare %arg (23)  rf=r size=0 type=ud align=16 words (r26.0)
//.declare %retval (24)  rf=r size=0 type=ud align=16 words (r26.0) Output
//.declare %sp (25)  rf=r size=8 type=uq align=4 words (r255.3)
//.declare %fp (26)  rf=r size=8 type=uq align=4 words (r255.2)
//.declare %sr0 (27)  rf=r size=16 type=ud align=2 words
//.declare %cr0 (28)  rf=r size=12 type=ud align=2 words
//.declare %ce0 (29)  rf=r size=4 type=ud align=2 words
//.declare %dbg0 (30)  rf=r size=8 type=ud align=2 words
//.declare implBufPtr (32)  rf=r size=8 type=uq align=4 words (r254.0)
//.declare localIdBufPtr (33)  rf=r size=8 type=uq align=4 words (r254.3)
//.declare %msg0 (34)  rf=r size=12 type=ud align=2 words
//.declare T6 (39)  rf=r size=4 type=ud align=2 words (r2.6)
//.declare T7 (40)  rf=r size=4 type=ud align=2 words (r3.0)
//.declare T8 (41)  rf=r size=4 type=ud align=2 words (r3.2)
//.declare V32 (42)  rf=r size=6 type=w align=1 words (r1.0)
//.declare V33 (43)  rf=r size=12 type=d align=2 words (r2.0)
//.declare V34 (44)  rf=r size=8 type=q align=4 words (r2.2)
//.declare V35 (45)  rf=r size=12 type=d align=2 words (r1.2)
//.declare V36 (46)  rf=r size=4 type=d align=16 words (r3.0)
//.declare V37 (47)  rf=r size=4 type=d align=16 words (r4.0)
//.declare V38 (48)  rf=r size=4 type=d align=2 words (r1.5)
//.declare P1 (49)  rf=f1  size=2 type=uw align=1 words (f1.1)
//.declare V40 (51)  rf=r size=32 type=hf align=16 words (r5.0)
//.declare V41 (52)  rf=r size=4 type=d align=16 words (r6.0)
//.declare V42 (53)  rf=r size=4 type=d align=2 words (r1.6)
//.declare V43 (54)  rf=r size=4 type=d align=2 words (r1.7)
//.declare V44 (55)  rf=r size=4 type=d align=16 words (r7.0)
//.declare V45 (56)  rf=r size=32 type=hf align=16 words (r8.0)
//.declare V46 (57)  rf=r size=4 type=d align=16 words (r9.0)
//.declare V47 (58)  rf=r size=32 type=hf align=16 words (r10.0)
//.declare V48 (59)  rf=r size=32 type=hf align=16 words (r11.0)
//.declare V49 (60)  rf=r size=4 type=d align=16 words (r12.0)
//.declare V50 (61)  rf=r size=4 type=d align=16 words (r13.0)
//.declare V51 (62)  rf=r size=32 type=hf align=16 words (r14.0)
//.declare V52 (63)  rf=r size=32 type=hf align=16 words (r15.0)
//.declare V53 (64)  rf=r size=4 type=d align=16 words (r16.0)
//.declare V54 (65)  rf=r size=4 type=d align=16 words (r17.0)
//.declare V55 (66)  rf=r size=32 type=hf align=16 words (r18.0)
//.declare V56 (67)  rf=r size=32 type=hf align=16 words (r19.0)
//.declare V57 (68)  rf=r size=4 type=d align=16 words (r20.0)
//.declare V58 (69)  rf=r size=4 type=d align=16 words (r21.0)
//.declare V59 (70)  rf=r size=32 type=hf align=16 words (r22.0)
//.declare V60 (71)  rf=r size=32 type=hf align=16 words (r23.0)
//.declare V61 (72)  rf=r size=4 type=d align=16 words (r24.0)
//.declare V62 (73)  rf=r size=4 type=d align=16 words (r25.0)
//.declare V63 (74)  rf=r size=32 type=hf align=16 words (r26.0)
//.declare V64 (75)  rf=r size=32 type=hf align=16 words (r27.0)
//.declare V65 (76)  rf=r size=4 type=d align=16 words (r28.0)
//.declare V66 (77)  rf=r size=4 type=d align=16 words (r29.0)
//.declare V67 (78)  rf=r size=32 type=hf align=16 words (r30.0)
//.declare V68 (79)  rf=r size=32 type=hf align=16 words (r31.0)
//.declare V69 (80)  rf=r size=4 type=d align=16 words (r32.0)
//.declare V70 (81)  rf=r size=4 type=d align=16 words (r33.0)
//.declare V71 (82)  rf=r size=32 type=hf align=16 words (r34.0)
//.declare V72 (83)  rf=r size=32 type=hf align=16 words (r35.0)
//.declare V73 (84)  rf=r size=4 type=d align=16 words (r36.0)
//.declare V74 (85)  rf=r size=4 type=d align=16 words (r37.0)
//.declare V75 (86)  rf=r size=32 type=hf align=16 words (r38.0)
//.declare P2 (87)  rf=f1  size=2 type=uw align=1 words (f1.0)
//.declare V76 (88)  rf=r size=4 type=d align=16 words (r39.0)
//.declare V77 (89)  rf=r size=4 type=d alias=V36+0 align=2 words (r3.0)
//.declare V78 (90)  rf=r size=12 type=d alias=V33+0 align=2 words (r2.0)
//.declare V79 (91)  rf=r size=6 type=uw alias=V32+0 align=1 words (r1.0)
//.declare V80 (92)  rf=r size=12 type=d alias=V35+0 align=2 words (r1.2)
//.declare V81 (93)  rf=r size=4 type=d alias=V37+0 align=2 words (r4.0)
//.declare V82 (94)  rf=r size=4 type=d alias=V38+0 align=2 words (r1.5)
//.declare V83 (95)  rf=r size=4 type=ud alias=V37+0 align=2 words (r4.0)
//.declare V84 (96)  rf=r size=4 type=ud alias=V41+0 align=2 words (r6.0)
//.declare V85 (97)  rf=r size=4 type=d alias=V43+0 align=2 words (r1.7)
//.declare V86 (98)  rf=r size=4 type=d alias=V42+0 align=2 words (r1.6)
//.declare V87 (99)  rf=r size=4 type=d alias=V44+0 align=2 words (r7.0)
//.declare V88 (100)  rf=r size=32 type=d alias=V45+0 align=16 words (r8.0)
//.declare V89 (101)  rf=r size=4 type=ud alias=V44+0 align=2 words (r7.0)
//.declare V90 (102)  rf=r size=32 type=d alias=V47+0 align=16 words (r10.0)
//.declare V91 (103)  rf=r size=4 type=ud alias=V46+0 align=2 words (r9.0)
//.declare V92 (104)  rf=r size=32 type=d alias=V48+0 align=16 words (r11.0)
//.declare V93 (105)  rf=r size=4 type=ud alias=V49+0 align=2 words (r12.0)
//.declare V94 (106)  rf=r size=32 type=d alias=V51+0 align=16 words (r14.0)
//.declare V95 (107)  rf=r size=4 type=ud alias=V50+0 align=2 words (r13.0)
//.declare V96 (108)  rf=r size=32 type=d alias=V52+0 align=16 words (r15.0)
//.declare V97 (109)  rf=r size=4 type=ud alias=V53+0 align=2 words (r16.0)
//.declare V98 (110)  rf=r size=32 type=d alias=V55+0 align=16 words (r18.0)
//.declare V99 (111)  rf=r size=4 type=ud alias=V54+0 align=2 words (r17.0)
//.declare V100 (112)  rf=r size=32 type=d alias=V56+0 align=16 words (r19.0)
//.declare V101 (113)  rf=r size=4 type=ud alias=V57+0 align=2 words (r20.0)
//.declare V102 (114)  rf=r size=32 type=d alias=V59+0 align=16 words (r22.0)
//.declare V103 (115)  rf=r size=4 type=ud alias=V58+0 align=2 words (r21.0)
//.declare V104 (116)  rf=r size=32 type=d alias=V60+0 align=16 words (r23.0)
//.declare V105 (117)  rf=r size=4 type=ud alias=V61+0 align=2 words (r24.0)
//.declare V106 (118)  rf=r size=32 type=d alias=V63+0 align=16 words (r26.0)
//.declare V107 (119)  rf=r size=4 type=ud alias=V62+0 align=2 words (r25.0)
//.declare V108 (120)  rf=r size=32 type=d alias=V64+0 align=16 words (r27.0)
//.declare V109 (121)  rf=r size=4 type=ud alias=V65+0 align=2 words (r28.0)
//.declare V110 (122)  rf=r size=32 type=d alias=V67+0 align=16 words (r30.0)
//.declare V111 (123)  rf=r size=4 type=ud alias=V66+0 align=2 words (r29.0)
//.declare V112 (124)  rf=r size=32 type=d alias=V68+0 align=16 words (r31.0)
//.declare V113 (125)  rf=r size=4 type=ud alias=V69+0 align=2 words (r32.0)
//.declare V114 (126)  rf=r size=32 type=d alias=V71+0 align=16 words (r34.0)
//.declare V115 (127)  rf=r size=4 type=ud alias=V70+0 align=2 words (r33.0)
//.declare V116 (128)  rf=r size=32 type=d alias=V72+0 align=16 words (r35.0)
//.declare V117 (129)  rf=r size=4 type=ud alias=V73+0 align=2 words (r36.0)
//.declare V118 (130)  rf=r size=32 type=d alias=V75+0 align=16 words (r38.0)
//.declare V119 (131)  rf=r size=4 type=ud alias=V74+0 align=2 words (r37.0)
//.declare V120 (132)  rf=r size=32 type=d alias=V40+0 align=16 words (r5.0)
//.declare V121 (133)  rf=r size=4 type=d alias=V76+0 align=2 words (r39.0)
//.declare V122 (134)  rf=r size=4 type=ud alias=V76+0 align=2 words (r39.0)
//.declare  (135)  rf=f16  size=2 type=uw align=1 words (f0.1)
//.declare  (136)  rf=f16  size=2 type=uw align=1 words (f0.0)
//.declare  (137)  rf=r size=32 type=ud align=16 words (r255.0)
//.declare  (138)  rf=f16  size=2 type=uw align=1 words (f0.0)
//.declare  (139)  rf=r size=8 type=w align=1 words (r1.0)
//.declare  (140)  rf=r size=16 type=f align=2 words (r1.4)
//.declare r0 (141)  rf=r size=32 type=ud align=16 words (r0.0)
//.declare rtmp (142)  rf=r size=32 type=ud align=16 words (r255.0)
//.declare  (143)  rf=r size=32 type=ud align=16 words (r1.0)
//.declare  (144)  rf=r size=64 type=ud align=16 words (r2.0)

// .inputs
// +----------+----------+--------+----------+------------------+
// | id       | type     |  bytes | at       | from             |
// +----------+----------+--------+----------+------------------+
// | V32      | :w x 3   |    0x6 | r1       | pti[tid]+0x0     |
// | V33      | :d x 3   |    0xC | r2       | cti+0x0          |
// | V34      | :q       |    0x8 | r2+0x10  | cti+0x10         |
// | T6       | :ud      |    0x4 | r2+0x18  | cti+0x18         |
// | T7       | :ud      |    0x4 | r3       | cti+0x20         |
// | T8       | :ud      |    0x4 | r3+0x8   | cti+0x28         |
// +----------+----------+--------+----------+------------------+


// B000: Preds:{},  Succs:{B001}
per_thread_prolog:
(W)     mov (8|M0)               r255.0<1>:ud  0x0:ud                              {A@1}             //  ALU pipe: int; 
(W)     and (1|M0)               r255.2<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud                       //  ALU pipe: int; 
(W)     and (1|M0)               r255.0<1>:uw  r0.4<0;1,0>:uw    0xFF:uw                             //  ALU pipe: int; 
(W)     add (1|M0)               r255.2<1>:ud  r255.2<0;1,0>:ud  0x40:ud              {I@2}          //  ALU pipe: int; 
(W)     mad (1|M0)               r255.2<1>:ud  r255.2<0;0>:ud    r255.0<0;0>:uw    0x20:uw              {I@1} //  ALU pipe: int; 
(W)     send.dc0 (8|M0)          r1       r255    null:0  0x0            0x021842FD           {A@1,$0} // wr:1h+0, rd:1; oword aligned block read x2 // 
        nop                                                                                          // 
        nop                                                                                          // 
// B001: Preds:{B000},  Succs:{B002}
// cross_thread_prolog:
(W)     mov (8|M0)               r255.0<1>:ud  0x0:ud                              {$0.src}          //  ALU pipe: int; 
(W)     and (1|M0)               r255.2<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud                       //  ALU pipe: int; 
(W)     send.dc0 (8|M0)          r2       r255    null:0  0x0            0x022843FD           {A@1,$1} // wr:1h+0, rd:2; oword aligned block read x4 // 
// B002: Preds:{B001},  Succs:{B003, B006}
// bmm_nchw_fp16_BB_0:
        mov (1|M0)               r3.0<1>:d     r0.1<0;1,0>:ud                   {$1.dst}             //  ALU pipe: int; $1
        mov (1|M0)               r4.0<1>:d     r0.6<0;1,0>:ud                                        //  ALU pipe: int; $2
        mov (2|M0)               r1.2<1>:d     r1.0<1;1,0>:uw                   {$0.dst}             //  ALU pipe: int; $4
(W)     mul (1|M0)               acc0.0<1>:d   r3.0<0;1,0>:d     r2.0<0;1,0>:uw   {Compacted,I@3}    //  ALU pipe: int; $3 R{} IR{}{O:0,O:0,},  {BC=1}
        mach (1|M0)              r3.0<1>:d     r3.0<0;1,0>:d     r2.0<0;1,0>:d    {Compacted}        //  ALU pipe: int; $4
(W)     mul (1|M0)               acc0.0<1>:d   r4.0<0;1,0>:d     r2.2<0;1,0>:uw   {Compacted,I@4}    //  ALU pipe: int; $6
        mach (1|M0)              r4.0<1>:d     r4.0<0;1,0>:d     r2.1<0;1,0>:d    {Compacted}        //  ALU pipe: int; $7
        add (1|M0)               r3.0<1>:d     r3.0<0;1,0>:d     r1.2<0;1,0>:d    {Compacted,I@3}    //  ALU pipe: int; $5
        add (1|M0)               r4.0<1>:d     r4.0<0;1,0>:d     r1.3<0;1,0>:d    {Compacted,I@2}    //  ALU pipe: int; $7
        mul (1|M0)               r1.5<1>:d     r3.0<0;1,0>:d     448:w               {Compacted,I@2} //  ALU pipe: int; $8
        cmp (1|M0)    (gt)f1.1   null<1>:ud    r4.0<0;1,0>:ud    0x4:uw              {I@2}           //  ALU pipe: int; $9
(W)     and (1|M0)               f0.1<1>:uw    f1.1<0;1,0>:uw    0x1:uw                              //  ALU pipe: int; $10
(f0.1.any16h) goto (16|M0)                   BB_1              BB_1                                  //  ALU pipe: int; $10
// B003: Preds:{B002},  Succs:{B004}
_bmm_nchw_fp16_k0_0_:
        shl (1|M0)               r6.0<1>:ud    r4.0<0;1,0>:ud    0x5:uw                              //  ALU pipe: int; $11
        mov (16|M0)              r5.0<1>:hf    0.0:hf                                                //  ALU pipe: float; $13
        mov (1|M0)               r1.6<1>:d     0:w                                                   //  ALU pipe: int; $14
// B004: Preds:{B004, B003},  Succs:{B005, B004}
BB_2:
        shl (1|M0)               r1.7<1>:d     r1.6<0;1,0>:d     5:w               {I@1}             //  ALU pipe: int; $16
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r10:1 bti[1][r6:1]       {$2} // ex_desc:0x1000000; desc:0x6218C500 // $21
        add (1|M0)               r9.0<1>:ud    r6.0<0;1,0>:ud    0x80:uw                             //  ALU pipe: int; $22
        add (1|M0)               r12.0<1>:ud   r6.0<0;1,0>:ud    0x100:uw                            //  ALU pipe: int; $27
        add (1|M0)               r13.0<1>:ud   r6.0<0;1,0>:ud    0x180:uw                            //  ALU pipe: int; $30
        add (1|M0)               r7.0<1>:d     r1.7<0;1,0>:d     r1.5<0;1,0>:d    {I@4}              //  ALU pipe: int; $17
        add (1|M0)               r16.0<1>:ud   r6.0<0;1,0>:ud    0x200:uw                            //  ALU pipe: int; $35
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r11:1 bti[1][r9:1]       {A@5,$3} // ex_desc:0x1000000; desc:0x6218C500 // $24
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r14:1 bti[1][r12:1]      {A@4,$4} // ex_desc:0x1000000; desc:0x6218C500 // $29
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r15:1 bti[1][r13:1]      {A@3,$5} // ex_desc:0x1000000; desc:0x6218C500 // $32
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r8:1  bti[0][r7:1]       {A@2,$6} // ex_desc:0x0; desc:0x6218C500 // $19
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r18:1 bti[1][r16:1]      {A@1,$7} // ex_desc:0x1000000; desc:0x6218C500 // $37
        add (1|M0)               r17.0<1>:ud   r6.0<0;1,0>:ud    0x280:uw                            //  ALU pipe: int; $38
        add (1|M0)               r20.0<1>:ud   r6.0<0;1,0>:ud    0x300:uw                            //  ALU pipe: int; $43
        add (1|M0)               r21.0<1>:ud   r6.0<0;1,0>:ud    0x380:uw                            //  ALU pipe: int; $46
        add (1|M0)               r24.0<1>:ud   r6.0<0;1,0>:ud    0x400:uw                            //  ALU pipe: int; $51
        add (1|M0)               r25.0<1>:ud   r6.0<0;1,0>:ud    0x480:uw                            //  ALU pipe: int; $54
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r19:1 bti[1][r17:1]      {A@5,$8} // ex_desc:0x1000000; desc:0x6218C500 // $40
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r22:1 bti[1][r20:1]      {A@4,$9} // ex_desc:0x1000000; desc:0x6218C500 // $45
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r23:1 bti[1][r21:1]      {A@3,$10} // ex_desc:0x1000000; desc:0x6218C500 // $48
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r26:1 bti[1][r24:1]      {A@2,$11} // ex_desc:0x1000000; desc:0x6218C500 // $53
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r27:1 bti[1][r25:1]      {A@1,$12} // ex_desc:0x1000000; desc:0x6218C500 // $56
        add (1|M0)               r28.0<1>:ud   r6.0<0;1,0>:ud    0x500:uw                            //  ALU pipe: int; $59
        add (1|M0)               r29.0<1>:ud   r6.0<0;1,0>:ud    0x580:uw                            //  ALU pipe: int; $62
        add (1|M0)               r32.0<1>:ud   r6.0<0;1,0>:ud    0x600:uw                            //  ALU pipe: int; $67
        add (1|M0)               r33.0<1>:ud   r6.0<0;1,0>:ud    0x680:uw                            //  ALU pipe: int; $70
        add (1|M0)               r36.0<1>:ud   r6.0<0;1,0>:ud    0x700:uw                            //  ALU pipe: int; $75
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r30:1 bti[1][r28:1]      {A@5,$13} // ex_desc:0x1000000; desc:0x6218C500 // $61
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r31:1 bti[1][r29:1]      {A@4,$14} // ex_desc:0x1000000; desc:0x6218C500 // $64
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r34:1 bti[1][r32:1]      {A@3,$15} // ex_desc:0x1000000; desc:0x6218C500 // $69
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r35:1 bti[1][r33:1]      {A@2,$0} // ex_desc:0x1000000; desc:0x6218C500 // $72
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r38:1 bti[1][r36:1]      {A@1,$1} // ex_desc:0x1000000; desc:0x6218C500 // $77
        sync.allwr                           ($2,$6)                                                 // $25
        mad (16|M0)              acc0.0<1>:hf  r5.0<1;0>:hf      r10.0<1;0>:hf     r8.0<0>:hf       {F@1} //  ALU pipe: float; $25
        add (1|M0)               r37.0<1>:ud   r6.0<0;1,0>:ud    0x780:uw                            //  ALU pipe: int; $78
        add (1|M0)               r1.6<1>:d     r1.6<0;1,0>:d     1:w                                 //  ALU pipe: int; $84
        add (1|M0)               r6.0<1>:ud    r6.0<0;1,0>:ud    0x800:uw                            //  ALU pipe: int; $83
        mad (16|M0)              acc0.0<1>:hf  acc0.0<1;0>:hf    r11.0<1;0>:hf     r8.1<0>:hf       {$3.dst} //  ALU pipe: float; $26
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r5:1  bti[1][r37:1]      {A@2,$2} // ex_desc:0x1000000; desc:0x6218C500 // $80
        cmp (1|M0)    (eq)f1.0   null<1>:d     r1.6<0;1,0>:d     14:w               {I@2}            //  ALU pipe: int; $85
        mad (16|M0)              acc0.0<1>:hf  acc0.0<1;0>:hf    r14.0<1;0>:hf     r8.2<0>:hf       {$4.dst} //  ALU pipe: float; $33
        mad (16|M0)              acc0.0<1>:hf  acc0.0<1;0>:hf    r15.0<1;0>:hf     r8.3<0>:hf       {$5.dst} //  ALU pipe: float; $34
(W)     and (1|M0)               f0.0<1>:uw    f1.0<0;1,0>:uw    0x1:uw                              //  ALU pipe: int; $86
        mad (16|M0)              acc0.0<1>:hf  acc0.0<1;0>:hf    r18.0<1;0>:hf     r8.4<0>:hf       {$7.dst} //  ALU pipe: float; $41
        mad (16|M0)              acc0.0<1>:hf  acc0.0<1;0>:hf    r19.0<1;0>:hf     r8.5<0>:hf       {$8.dst} //  ALU pipe: float; $42
        mad (16|M0)              acc0.0<1>:hf  acc0.0<1;0>:hf    r22.0<1;0>:hf     r8.6<0>:hf       {$9.dst} //  ALU pipe: float; $49
        mad (16|M0)              acc0.0<1>:hf  acc0.0<1;0>:hf    r23.0<1;0>:hf     r8.7<0>:hf       {$10.dst} //  ALU pipe: float; $50
        mad (16|M0)              acc0.0<1>:hf  acc0.0<1;0>:hf    r26.0<1;0>:hf     r8.8<0>:hf       {$11.dst} //  ALU pipe: float; $57
        mad (16|M0)              acc0.0<1>:hf  acc0.0<1;0>:hf    r27.0<1;0>:hf     r8.9<0>:hf       {$12.dst} //  ALU pipe: float; $58
        mad (16|M0)              acc0.0<1>:hf  acc0.0<1;0>:hf    r30.0<1;0>:hf     r8.10<0>:hf      {$13.dst} //  ALU pipe: float; $65
        mad (16|M0)              acc0.0<1>:hf  acc0.0<1;0>:hf    r31.0<1;0>:hf     r8.11<0>:hf      {$14.dst} //  ALU pipe: float; $66
        mad (16|M0)              acc0.0<1>:hf  acc0.0<1;0>:hf    r34.0<1;0>:hf     r8.12<0>:hf      {$15.dst} //  ALU pipe: float; $73
        mad (16|M0)              acc0.0<1>:hf  acc0.0<1;0>:hf    r35.0<1;0>:hf     r8.13<0>:hf      {$0.dst} //  ALU pipe: float; $74
        mad (16|M0)              acc0.0<1>:hf  acc0.0<1;0>:hf    r38.0<1;0>:hf     r8.14<0>:hf      {$1.dst} //  ALU pipe: float; $81
        mad (16|M0)              r5.0<1>:hf    acc0.0<1;0>:hf    r5.0<1;0>:hf      r8.15<0>:hf      {$2.dst} //  ALU pipe: float; $82
(~f0.0.any16h) goto.b (16|M0)                _bmm_nchw_fp16_k0_1_  BB_2                              //  ALU pipe: int; $86
// B005: Preds:{B004},  Succs:{B006}
_bmm_nchw_fp16_k0_1_:
        join (16|M0)                         BB_1                                                    // 
L1288:
        shl (1|M0)               r4.0<1>:ud    r4.0<0;1,0>:ud    0x4:uw                              //  ALU pipe: int; $87
        shl (1|M0)               r3.0<1>:d     r3.0<0;1,0>:d     6:w               {Compacted}       //  ALU pipe: int; $88
        add (1|M0)               r4.0<1>:d     r4.0<0;1,0>:d     r3.0<0;1,0>:d    {Compacted,I@1}    //  ALU pipe: int; $89
        shl (1|M0)               r39.0<1>:d    r4.0<0;1,0>:d     1:w               {Compacted,I@1}   //  ALU pipe: int; $90
        store.ugm.d32x8t.a32.wb.wb (1|M0)  bti[2][r39:1] r5:1      {A@1,$3} // ex_desc:0x2000000; desc:0x620EC504 // $91
// B006: Preds:{B005, B002},  Succs:{}
BB_1:
        join (16|M0)                         L1360                                                   // 
L1360:
(W)     mov (8|M0)               r255.0<1>:f   r0.0<1;1,0>:f                    {Compacted}          //  ALU pipe: float; $93
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<1;0>:w       r1.0<1;0>:w       r1.0<1>:w         //  ALU pipe: int; $93
(W)     csel (4|M0)   (eq)f0.0   r1.4<1>:f     r1.4<1;0>:f       r1.4<1;0>:f       r1.4<1>:f        {I@1} //  ALU pipe: float; $93
(W)     send.gtwy (1|M0)         null     r255    null:0  0x0            0x02000010           {EOT,A@1} // wr:1+0, rd:0; end of thread // $93
L1416:
        nop                                                                                          // $93


//.BankConflicts: 1
//.ByteRMWs: 0
//


//.numALUInst: 68
//.accSubDef: 15
//.accSubUse: 15
//.accSubCandidateDef: 15
//.accSubCandidateUse: 15
//
//
//.singlePipeAtOneDistNum: 9
//.allAtOneDistNum: 5
//.syncInstCount: 0
//.tokenReuseCount: 0
//.AfterWriteTokenDepCount: 19
//.AfterReadTokenDepCount: 1
