//.kernel bmm_nchw_fp16
//.platform DG2
//.thread_config numGRF=256, numAcc=8, numSWSB=16
//.options_string "-enableHalfLSC -dumpcommonisa -output -binary -printregusage -hasNoInt64Add -TotalGRFNum 256 -fusedCallWA 1 -abiver 2 -LSCFenceWA "
//.full_options "-abiver 2 -printregusage -TotalGRFNum 256 -output -binary -dumpcommonisa -enableHalfLSC -hasNoInt64Add -fusedCallWA 1 -LSCFenceWA "
//.instCount 172
//.RA type	TRIVIAL_RA
//.git-hash ee9d4992e9bf803c53080043aa3f5c1df6a6234f
//.GRF count 118

//.declare BuiltInR0 (0)  rf=r size=32 type=ud align=16 words (r0.0) IsBuiltin
//.declare  (1)  rf=r size=32 type=ud alias=BuiltInR0+0 align=16 words (r0.0) IsBuiltin
//.declare BuiltinA0 (2)  rf=a size=4 type=ud align=1 words (a0.0) IsBuiltin
//.declare BuiltinA0Dot2 (3)  rf=a size=4 type=ud align=1 words (a0.2) IsBuiltin
//.declare BuiltinSR0Dot1 (5)  rf=r size=4 type=ud align=2 words IsBuiltin
//.declare %null (10)  rf=r size=4 type=ud align=2 words
//.declare %local_id_x (13)  rf=r size=4 type=ud align=2 words (r4.1)
//.declare %local_id_y (14)  rf=r size=4 type=ud align=2 words (r4.2)
//.declare %local_size_x (15)  rf=r size=4 type=ud align=2 words (r3.5)
//.declare %local_size_y (16)  rf=r size=4 type=ud align=2 words (r3.6)
//.declare %group_id_x (17)  rf=r size=4 type=ud align=2 words (r0.1)
//.declare %group_id_y (18)  rf=r size=4 type=ud align=2 words (r0.6)
//.declare %group_id_z (19)  rf=r size=4 type=ud align=2 words (r0.7)
//.declare %group_count_x (20)  rf=r size=4 type=ud align=2 words (r3.7)
//.declare %group_count_y (21)  rf=r size=4 type=ud align=2 words (r4.0)
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
//.declare T6 (40)  rf=r size=4 type=ud align=2 words (r3.0)
//.declare T7 (41)  rf=r size=4 type=ud align=2 words (r3.2)
//.declare T8 (42)  rf=r size=4 type=ud align=2 words (r3.4)
//.declare V33 (43)  rf=r size=6 type=w align=1 words (r1.0)
//.declare V34 (44)  rf=r size=12 type=d align=2 words (r2.0)
//.declare V35 (45)  rf=r size=8 type=q align=4 words (r2.2)
//.declare V36 (46)  rf=r size=8 type=q align=4 words (r2.3)
//.declare V37 (47)  rf=r size=8 type=d align=16 words (r4.0)
//.declare V38 (48)  rf=r size=12 type=d align=2 words (r1.2)
//.declare V39 (49)  rf=r size=4 type=d align=16 words (r3.0)
//.declare V40 (50)  rf=r size=4 type=d align=16 words (r5.0)
//.declare V41 (51)  rf=r size=4 type=d align=16 words (r6.0)
//.declare V42 (52)  rf=r size=4 type=d align=16 words (r7.0)
//.declare V43 (53)  rf=r size=8 type=d align=2 words (r1.5)
//.declare V45 (55)  rf=r size=4 type=d align=2 words (r1.7)
//.declare V46 (56)  rf=r size=8 type=d align=16 words (r8.0)
//.declare V47 (57)  rf=r size=8 type=d align=2 words (r2.6)
//.declare V48 (58)  rf=r size=4 type=d align=16 words (r9.0)
//.declare V49 (59)  rf=r size=4 type=d align=16 words (r10.0)
//.declare V50 (60)  rf=r size=8 type=q align=4 words (r3.1)
//.declare V51 (61)  rf=r size=16 type=d align=16 words (r12.0)
//.declare V52 (62)  rf=r size=8 type=d align=2 words (r3.4)
//.declare V53 (63)  rf=r size=4 type=d align=16 words (r11.0)
//.declare V54 (64)  rf=r size=4 type=d align=16 words (r13.0)
//.declare V55 (65)  rf=r size=8 type=q align=16 words (r14.0)
//.declare V56 (66)  rf=r size=4 type=d align=16 words (r16.0)
//.declare V57 (67)  rf=r size=4 type=d align=2 words (r2.3)
//.declare P1 (68)  rf=f1  size=2 type=uw align=1 words (f1.1)
//.declare V59 (70)  rf=r size=64 type=f align=16 words (r17.0)
//.declare V60 (71)  rf=r size=4 type=d align=16 words (r15.0)
//.declare V61 (72)  rf=r size=4 type=d align=2 words (r3.1)
//.declare V62 (73)  rf=r size=4 type=d align=2 words (r3.6)
//.declare V63 (74)  rf=r size=4 type=d align=16 words (r19.0)
//.declare V64 (75)  rf=r size=32 type=hf align=16 words (r20.0)
//.declare V65 (76)  rf=r size=32 type=hf align=16 words (r21.0)
//.declare V66 (77)  rf=r size=64 type=f align=16 words (r22.0)
//.declare V67 (78)  rf=r size=4 type=d align=16 words (r24.0)
//.declare V68 (79)  rf=r size=32 type=hf align=16 words (r25.0)
//.declare V69 (80)  rf=r size=64 type=f align=16 words (r26.0)
//.declare V70 (81)  rf=r size=64 type=f align=16 words (r28.0)
//.declare V71 (82)  rf=r size=64 type=f align=16 words (r30.0)
//.declare V72 (83)  rf=r size=4 type=d align=16 words (r32.0)
//.declare V73 (84)  rf=r size=32 type=hf align=16 words (r33.0)
//.declare V74 (85)  rf=r size=64 type=f align=16 words (r34.0)
//.declare V75 (86)  rf=r size=4 type=d align=16 words (r36.0)
//.declare V76 (87)  rf=r size=32 type=hf align=16 words (r37.0)
//.declare V77 (88)  rf=r size=64 type=f align=16 words (r38.0)
//.declare V78 (89)  rf=r size=64 type=f align=16 words (r40.0)
//.declare V79 (90)  rf=r size=64 type=f align=16 words (r42.0)
//.declare V80 (91)  rf=r size=4 type=d align=16 words (r44.0)
//.declare V81 (92)  rf=r size=32 type=hf align=16 words (r45.0)
//.declare V82 (93)  rf=r size=64 type=f align=16 words (r46.0)
//.declare V83 (94)  rf=r size=4 type=d align=16 words (r48.0)
//.declare V84 (95)  rf=r size=32 type=hf align=16 words (r49.0)
//.declare V85 (96)  rf=r size=64 type=f align=16 words (r50.0)
//.declare V86 (97)  rf=r size=64 type=f align=16 words (r52.0)
//.declare V87 (98)  rf=r size=64 type=f align=16 words (r54.0)
//.declare V88 (99)  rf=r size=4 type=d align=16 words (r56.0)
//.declare V89 (100)  rf=r size=32 type=hf align=16 words (r57.0)
//.declare V90 (101)  rf=r size=64 type=f align=16 words (r58.0)
//.declare V91 (102)  rf=r size=4 type=d align=16 words (r60.0)
//.declare V92 (103)  rf=r size=32 type=hf align=16 words (r61.0)
//.declare V93 (104)  rf=r size=64 type=f align=16 words (r62.0)
//.declare V94 (105)  rf=r size=64 type=f align=16 words (r64.0)
//.declare V95 (106)  rf=r size=64 type=f align=16 words (r66.0)
//.declare V96 (107)  rf=r size=4 type=d align=16 words (r68.0)
//.declare V97 (108)  rf=r size=32 type=hf align=16 words (r69.0)
//.declare V98 (109)  rf=r size=64 type=f align=16 words (r70.0)
//.declare V99 (110)  rf=r size=4 type=d align=16 words (r72.0)
//.declare V100 (111)  rf=r size=32 type=hf align=16 words (r73.0)
//.declare V101 (112)  rf=r size=64 type=f align=16 words (r74.0)
//.declare V102 (113)  rf=r size=64 type=f align=16 words (r76.0)
//.declare V103 (114)  rf=r size=64 type=f align=16 words (r78.0)
//.declare V104 (115)  rf=r size=4 type=d align=16 words (r80.0)
//.declare V105 (116)  rf=r size=32 type=hf align=16 words (r81.0)
//.declare V106 (117)  rf=r size=64 type=f align=16 words (r82.0)
//.declare V107 (118)  rf=r size=4 type=d align=16 words (r84.0)
//.declare V108 (119)  rf=r size=32 type=hf align=16 words (r85.0)
//.declare V109 (120)  rf=r size=64 type=f align=16 words (r86.0)
//.declare V110 (121)  rf=r size=64 type=f align=16 words (r88.0)
//.declare V111 (122)  rf=r size=64 type=f align=16 words (r90.0)
//.declare V112 (123)  rf=r size=4 type=d align=16 words (r92.0)
//.declare V113 (124)  rf=r size=32 type=hf align=16 words (r93.0)
//.declare V114 (125)  rf=r size=64 type=f align=16 words (r94.0)
//.declare V115 (126)  rf=r size=4 type=d align=16 words (r96.0)
//.declare V116 (127)  rf=r size=32 type=hf align=16 words (r97.0)
//.declare V117 (128)  rf=r size=64 type=f align=16 words (r98.0)
//.declare V118 (129)  rf=r size=64 type=f align=16 words (r100.0)
//.declare V119 (130)  rf=r size=64 type=f align=16 words (r102.0)
//.declare V120 (131)  rf=r size=4 type=d align=16 words (r104.0)
//.declare V121 (132)  rf=r size=32 type=hf align=16 words (r105.0)
//.declare V122 (133)  rf=r size=64 type=f align=16 words (r106.0)
//.declare V123 (134)  rf=r size=4 type=d align=16 words (r108.0)
//.declare V124 (135)  rf=r size=32 type=hf align=16 words (r109.0)
//.declare V125 (136)  rf=r size=64 type=f align=16 words (r110.0)
//.declare V126 (137)  rf=r size=64 type=f align=16 words (r112.0)
//.declare P2 (138)  rf=f1  size=2 type=uw align=1 words (f1.0)
//.declare V127 (139)  rf=r size=4 type=d align=16 words (r114.0)
//.declare V128 (140)  rf=r size=32 type=d align=16 words (r115.0)
//.declare V129 (141)  rf=r size=8 type=q alias=V46+0 align=4 words (r8.0)
//.declare V130 (142)  rf=r size=8 type=uq alias=V37+0 align=4 words (r4.0)
//.declare V131 (143)  rf=r size=4 type=d alias=V39+0 align=2 words (r3.0)
//.declare V132 (144)  rf=r size=12 type=d alias=V34+0 align=2 words (r2.0)
//.declare V133 (145)  rf=r size=6 type=uw alias=V33+0 align=1 words (r1.0)
//.declare V134 (146)  rf=r size=12 type=d alias=V38+0 align=2 words (r1.2)
//.declare V135 (147)  rf=r size=4 type=d alias=V40+0 align=2 words (r5.0)
//.declare V136 (148)  rf=r size=8 type=uq alias=V46+0 align=4 words (r8.0)
//.declare V137 (149)  rf=r size=8 type=d alias=V43+0 align=2 words (r1.5)
//.declare V138 (150)  rf=r size=4 type=d alias=V42+0 align=2 words (r7.0)
//.declare V140 (152)  rf=r size=4 type=ud alias=V45+0 align=2 words (r1.7)
//.declare V141 (153)  rf=r size=8 type=ud alias=V43+0 align=2 words (r1.5)
//.declare V142 (154)  rf=r size=8 type=d alias=V47+0 align=2 words (r2.6)
//.declare V143 (155)  rf=r size=4 type=d alias=V45+0 align=2 words (r1.7)
//.declare V144 (156)  rf=r size=4 type=ud alias=V48+0 align=2 words (r9.0)
//.declare V145 (157)  rf=r size=4 type=ud alias=V49+0 align=2 words (r10.0)
//.declare V146 (158)  rf=r size=8 type=ud alias=V46+0 align=2 words (r8.0)
//.declare V147 (159)  rf=r size=8 type=ud alias=V47+0 align=2 words (r2.6)
//.declare V148 (160)  rf=r size=8 type=d alias=V50+0 align=2 words (r3.2)
//.declare V149 (161)  rf=r size=8 type=d alias=V50+0 align=2 words (r3.2)
//.declare V150 (162)  rf=r size=4 type=d alias=V49+0 align=2 words (r10.0)
//.declare V151 (163)  rf=r size=8 type=d alias=V46+0 align=2 words (r8.0)
//.declare V152 (164)  rf=r size=16 type=q alias=V51+0 align=4 words (r12.0)
//.declare V153 (165)  rf=r size=16 type=uq alias=V51+0 align=4 words (r12.0)
//.declare V154 (166)  rf=r size=4 type=ud alias=V53+0 align=2 words (r11.0)
//.declare V155 (167)  rf=r size=4 type=ud alias=V54+0 align=2 words (r13.0)
//.declare V156 (168)  rf=r size=8 type=ud alias=V52+0 align=2 words (r3.4)
//.declare V157 (169)  rf=r size=8 type=d alias=V55+0 align=2 words (r14.0)
//.declare V158 (170)  rf=r size=8 type=d alias=V55+0 align=2 words (r14.0)
//.declare V159 (171)  rf=r size=4 type=d alias=V54+0 align=2 words (r13.0)
//.declare V160 (172)  rf=r size=8 type=d alias=V52+0 align=2 words (r3.4)
//.declare V161 (173)  rf=r size=8 type=uq alias=V55+0 align=4 words (r14.0)
//.declare V162 (174)  rf=r size=4 type=d alias=V57+0 align=2 words (r2.3)
//.declare V163 (175)  rf=r size=4 type=ud alias=V40+0 align=2 words (r5.0)
//.declare V164 (176)  rf=r size=4 type=ud alias=V60+0 align=2 words (r15.0)
//.declare V165 (177)  rf=r size=4 type=d alias=V62+0 align=2 words (r3.6)
//.declare V166 (178)  rf=r size=4 type=d alias=V61+0 align=2 words (r3.1)
//.declare V167 (179)  rf=r size=4 type=d alias=V63+0 align=2 words (r19.0)
//.declare V168 (180)  rf=r size=32 type=d alias=V64+0 align=16 words (r20.0)
//.declare V169 (181)  rf=r size=4 type=ud alias=V63+0 align=2 words (r19.0)
//.declare V170 (182)  rf=r size=32 type=d alias=V65+0 align=16 words (r21.0)
//.declare V171 (183)  rf=r size=4 type=ud alias=V67+0 align=2 words (r24.0)
//.declare V172 (184)  rf=r size=32 type=d alias=V68+0 align=16 words (r25.0)
//.declare V173 (185)  rf=r size=4 type=ud alias=V72+0 align=2 words (r32.0)
//.declare V174 (186)  rf=r size=32 type=d alias=V73+0 align=16 words (r33.0)
//.declare V175 (187)  rf=r size=4 type=ud alias=V75+0 align=2 words (r36.0)
//.declare V176 (188)  rf=r size=32 type=d alias=V76+0 align=16 words (r37.0)
//.declare V177 (189)  rf=r size=4 type=ud alias=V80+0 align=2 words (r44.0)
//.declare V178 (190)  rf=r size=32 type=d alias=V81+0 align=16 words (r45.0)
//.declare V179 (191)  rf=r size=4 type=ud alias=V83+0 align=2 words (r48.0)
//.declare V180 (192)  rf=r size=32 type=d alias=V84+0 align=16 words (r49.0)
//.declare V181 (193)  rf=r size=4 type=ud alias=V88+0 align=2 words (r56.0)
//.declare V182 (194)  rf=r size=32 type=d alias=V89+0 align=16 words (r57.0)
//.declare V183 (195)  rf=r size=4 type=ud alias=V91+0 align=2 words (r60.0)
//.declare V184 (196)  rf=r size=32 type=d alias=V92+0 align=16 words (r61.0)
//.declare V185 (197)  rf=r size=4 type=ud alias=V96+0 align=2 words (r68.0)
//.declare V186 (198)  rf=r size=32 type=d alias=V97+0 align=16 words (r69.0)
//.declare V187 (199)  rf=r size=4 type=ud alias=V99+0 align=2 words (r72.0)
//.declare V188 (200)  rf=r size=32 type=d alias=V100+0 align=16 words (r73.0)
//.declare V189 (201)  rf=r size=4 type=ud alias=V104+0 align=2 words (r80.0)
//.declare V190 (202)  rf=r size=32 type=d alias=V105+0 align=16 words (r81.0)
//.declare V191 (203)  rf=r size=4 type=ud alias=V107+0 align=2 words (r84.0)
//.declare V192 (204)  rf=r size=32 type=d alias=V108+0 align=16 words (r85.0)
//.declare V193 (205)  rf=r size=4 type=ud alias=V112+0 align=2 words (r92.0)
//.declare V194 (206)  rf=r size=32 type=d alias=V113+0 align=16 words (r93.0)
//.declare V195 (207)  rf=r size=4 type=ud alias=V115+0 align=2 words (r96.0)
//.declare V196 (208)  rf=r size=32 type=d alias=V116+0 align=16 words (r97.0)
//.declare V197 (209)  rf=r size=4 type=ud alias=V120+0 align=2 words (r104.0)
//.declare V198 (210)  rf=r size=32 type=d alias=V121+0 align=16 words (r105.0)
//.declare V199 (211)  rf=r size=4 type=ud alias=V123+0 align=2 words (r108.0)
//.declare V200 (212)  rf=r size=32 type=d alias=V124+0 align=16 words (r109.0)
//.declare V201 (213)  rf=r size=4 type=d alias=V127+0 align=2 words (r114.0)
//.declare V202 (214)  rf=r size=32 type=hf alias=V128+0 align=16 words (r115.0)
//.declare V203 (215)  rf=r size=4 type=ud alias=V127+0 align=2 words (r114.0)
//.declare  (216)  rf=f16  size=2 type=uw align=1 words (f0.1)
//.declare  (217)  rf=f16  size=2 type=uw align=1 words (f0.0)
//.declare  (218)  rf=r size=32 type=ud align=16 words (r255.0)
//.declare  (219)  rf=r size=32 type=ud align=16 words (r116.0)
//.declare  (220)  rf=f16  size=2 type=uw align=1 words (f0.0)
//.declare  (221)  rf=r size=8 type=w align=1 words (r1.0)
//.declare  (222)  rf=r size=16 type=f align=2 words (r1.4)
//.declare r0 (223)  rf=r size=32 type=ud align=16 words (r0.0)
//.declare rtmp (224)  rf=r size=32 type=ud align=16 words (r255.0)
//.declare  (225)  rf=r size=32 type=ud align=16 words (r1.0)
//.declare  (226)  rf=r size=64 type=ud align=16 words (r2.0)

// .inputs
// +----------+----------+--------+----------+------------------+
// | id       | type     |  bytes | at       | from             |
// +----------+----------+--------+----------+------------------+
// | V33      | :w x 3   |    0x6 | r1       | pti[tid]+0x0     |
// | V34      | :d x 3   |    0xC | r2       | cti+0x0          |
// | V35      | :q       |    0x8 | r2+0x10  | cti+0x10         |
// | V36      | :q       |    0x8 | r2+0x18  | cti+0x18         |
// | T6       | :ud      |    0x4 | r3       | cti+0x20         |
// | T7       | :ud      |    0x4 | r3+0x8   | cti+0x28         |
// | T8       | :ud      |    0x4 | r3+0x10  | cti+0x30         |
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
(W)     mov (2|M0)               r8.0<1>:ud    r2.4<1;1,0>:ud                   {$1.dst}             //  ALU pipe: int; $1
        mov (1|M0)               r6.0<1>:d     8:w                                                   //  ALU pipe: int; $10
        mov (1|M0)               r3.0<1>:d     r0.1<0;1,0>:ud                                        //  ALU pipe: int; $3
        mov (1|M0)               r5.0<1>:d     r0.6<0;1,0>:ud                                        //  ALU pipe: int; $4
        mov (2|M0)               r1.2<1>:d     r1.0<1;1,0>:uw                   {$0.dst}             //  ALU pipe: int; $6
(W)     mov (1|M0)               r4.0<1>:ud    0x6E10CA2E:ud                                         //  R_SYM_ADDR_32: .str; ALU pipe: int; $2
        atomic_iadd.ugm.d32.a64.uc.uc (1|M0)  r7:1 [r8:2]         r6:1               {A@5,$2} // ex_desc:0x0; desc:0x412058C // $11
(W)     mul (1|M0)               acc0.0<1>:d   r3.0<0;1,0>:d     r2.0<0;1,0>:uw   {Compacted,I@4}    //  ALU pipe: int; $5 R{} IR{}{O:0,O:0,},  {BC=1}
(W)     mov (1|M0)               r16.0<1>:ud   0x6E10CA2E:ud                                         //  R_SYM_ADDR_32_HI: .str; ALU pipe: int; $2
        mach (1|M0)              r3.0<1>:d     r3.0<0;1,0>:d     r2.0<0;1,0>:d    {Compacted}        //  ALU pipe: int; $6 R{} IR{}{O:0,O:0,},  {BC=1}
(W)     mul (1|M0)               acc0.0<1>:d   r5.0<0;1,0>:d     r2.2<0;1,0>:uw   {Compacted,I@6}    //  ALU pipe: int; $8
        mach (1|M0)              r5.0<1>:d     r5.0<0;1,0>:d     r2.1<0;1,0>:d    {Compacted}        //  ALU pipe: int; $9
        add (1|M0)               r3.0<1>:d     r3.0<0;1,0>:d     r1.2<0;1,0>:d    {Compacted,I@3}    //  ALU pipe: int; $7
        add (1|M0)               r5.0<1>:d     r5.0<0;1,0>:d     r1.3<0;1,0>:d    {Compacted,I@2}    //  ALU pipe: int; $9
        mul (1|M0)               r2.3<1>:d     r3.0<0;1,0>:d     448:w               {Compacted,I@2} //  ALU pipe: int; $29
        cmp (1|M0)    (gt)f1.1   null<1>:ud    r5.0<0;1,0>:ud    0x4:uw              {I@2}           //  ALU pipe: int; $30
(W)     and (1|M0)               f0.1<1>:uw    f1.1<0;1,0>:uw    0x1:uw                              //  ALU pipe: int; $31
        asr (1|M0)               r1.5<1>:d     r7.0<0;1,0>:d     2:w               {Compacted,$2.dst} //  ALU pipe: int; $12
        shl (1|M0)               r2.6<1>:d     r1.5<0;1,0>:d     2:w               {I@1}             //  ALU pipe: int; $16
        shr (1|M0)               r1.7<1>:ud    r1.5<0;1,0>:ud    0x1E:uw                             //  ALU pipe: int; $15
        addc (1|M0)              r9.0<1>:ud    r8.0<0;1,0>:ud    r2.6<0;1,0>:ud   {AccWrEn,I@2}      //  ALU pipe: int; $18
        or (1|M0)                r2.7<1>:d     r1.7<0;1,0>:d     0:w               {I@2}             //  ALU pipe: int; $17
        mov (1|M0)               r10.0<1>:ud   acc0.0<0;1,0>:ud                 {Compacted}          //  ALU pipe: int; $18
        mov (1|M0)               r3.2<1>:d     r9.0<0;1,0>:d                    {I@3}                //  ALU pipe: int; $19
        add3 (1|M0)              r3.3<1>:d     r10.0<0;0>:d      r8.1<0;0>:d       r2.7<0>:d        {I@2} //  ALU pipe: int; $20
(W)     mov (2|M0)               r12.0<1>:ud   r3.2<1;1,0>:ud                   {I@1}                //  ALU pipe: int; $21
        mov (2|M0)               r3.4<1>:d     r12.0<1;1,0>:d                   {I@1}                //  ALU pipe: int; $23
        store.ugm.d32.a64 (1|M0)  [r12:2]       r4:1               {$3} // ex_desc:0x0; desc:0x4000584 // $22
        addc (1|M0)              r11.0<1>:ud   r3.4<0;1,0>:ud    0x4:ud              {AccWrEn,I@1}   //  ALU pipe: int; $24
        mov (1|M0)               r13.0<1>:ud   acc0.0<0;1,0>:ud                 {Compacted,$3.src}   //  ALU pipe: int; $24
        mov (1|M0)               r14.0<1>:f    r11.0<0;1,0>:f                   {Compacted,I@2}      //  ALU pipe: float; $25
        add3 (1|M0)              r14.1<1>:d    r13.0<0;0>:d      r3.5<0;0>:d       0:w               {I@1} //  ALU pipe: int; $26
        store.ugm.d32.a64 (1|M0)  [r14:2]       r16:1              {A@1,$4} // ex_desc:0x0; desc:0x4000584 // $28
(f0.1.any16h) goto (16|M0)                   BB_1              BB_1                                  //  ALU pipe: int; $31
// B003: Preds:{B002},  Succs:{B004}
_bmm_nchw_fp16_k0_0_:
        mov (16|M0)              r17.0<1>:f    0.0:f                               {Compacted}       //  ALU pipe: float; $34
        shl (1|M0)               r15.0<1>:ud   r5.0<0;1,0>:ud    0x5:uw              {$4.src}        //  ALU pipe: int; $32
        mov (1|M0)               r3.1<1>:d     0:w                                                   //  ALU pipe: int; $35
// B004: Preds:{B004, B003},  Succs:{B005, B004}
BB_2:
        shl (1|M0)               r3.6<1>:d     r3.1<0;1,0>:d     5:w               {I@1}             //  ALU pipe: int; $37
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r21:1 bti[1][r15:1]      {$5} // ex_desc:0x1000000; desc:0x6218C500 // $42
        add (1|M0)               r24.0<1>:ud   r15.0<0;1,0>:ud   0x80:uw                             //  ALU pipe: int; $45
        add (1|M0)               r32.0<1>:ud   r15.0<0;1,0>:ud   0x100:uw                            //  ALU pipe: int; $52
        add (1|M0)               r36.0<1>:ud   r15.0<0;1,0>:ud   0x180:uw                            //  ALU pipe: int; $57
        add (1|M0)               r19.0<1>:d    r3.6<0;1,0>:d     r2.3<0;1,0>:d    {I@4}              //  ALU pipe: int; $38 R{} IR{}{O:0,O:0,},  {BC=1}
        add (1|M0)               r44.0<1>:ud   r15.0<0;1,0>:ud   0x200:uw                            //  ALU pipe: int; $64
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r25:1 bti[1][r24:1]      {A@5,$6} // ex_desc:0x1000000; desc:0x6218C500 // $47
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r33:1 bti[1][r32:1]      {A@4,$7} // ex_desc:0x1000000; desc:0x6218C500 // $54
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r37:1 bti[1][r36:1]      {A@3,$8} // ex_desc:0x1000000; desc:0x6218C500 // $59
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r20:1 bti[0][r19:1]      {A@2,$9} // ex_desc:0x0; desc:0x6218C500 // $40
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r45:1 bti[1][r44:1]      {A@1,$10} // ex_desc:0x1000000; desc:0x6218C500 // $66
        add (1|M0)               r48.0<1>:ud   r15.0<0;1,0>:ud   0x280:uw                            //  ALU pipe: int; $69
        add (1|M0)               r56.0<1>:ud   r15.0<0;1,0>:ud   0x300:uw                            //  ALU pipe: int; $76
        add (1|M0)               r60.0<1>:ud   r15.0<0;1,0>:ud   0x380:uw                            //  ALU pipe: int; $81
        add (1|M0)               r68.0<1>:ud   r15.0<0;1,0>:ud   0x400:uw                            //  ALU pipe: int; $88
        add (1|M0)               r72.0<1>:ud   r15.0<0;1,0>:ud   0x480:uw                            //  ALU pipe: int; $93
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r49:1 bti[1][r48:1]      {A@5,$11} // ex_desc:0x1000000; desc:0x6218C500 // $71
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r57:1 bti[1][r56:1]      {A@4,$12} // ex_desc:0x1000000; desc:0x6218C500 // $78
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r61:1 bti[1][r60:1]      {A@3,$13} // ex_desc:0x1000000; desc:0x6218C500 // $83
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r69:1 bti[1][r68:1]      {A@2,$14} // ex_desc:0x1000000; desc:0x6218C500 // $90
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r73:1 bti[1][r72:1]      {A@1,$15} // ex_desc:0x1000000; desc:0x6218C500 // $95
        add (1|M0)               r80.0<1>:ud   r15.0<0;1,0>:ud   0x500:uw                            //  ALU pipe: int; $100
        mov (8|M0)               r28.0<1>:f    r21.0<1;1,0>:hf                  {$5.dst}             //  ALU pipe: float; $43
        mov (8|M0)               r29.0<1>:f    r21.8<1;1,0>:hf                                       //  ALU pipe: float; $43
        add (1|M0)               r84.0<1>:ud   r15.0<0;1,0>:ud   0x580:uw                            //  ALU pipe: int; $105
        add (1|M0)               r92.0<1>:ud   r15.0<0;1,0>:ud   0x600:uw                            //  ALU pipe: int; $112
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r81:1 bti[1][r80:1]      {A@3,$0} // ex_desc:0x1000000; desc:0x6218C500 // $102
        add (1|M0)               r96.0<1>:ud   r15.0<0;1,0>:ud   0x680:uw                            //  ALU pipe: int; $117
        mov (8|M0)               r30.0<1>:f    r25.0<1;1,0>:hf                  {$6.dst}             //  ALU pipe: float; $48
        mov (8|M0)               r31.0<1>:f    r25.8<1;1,0>:hf                                       //  ALU pipe: float; $48
        mov (8|M0)               r40.0<1>:f    r33.0<1;1,0>:hf                  {$7.dst}             //  ALU pipe: float; $55
        mov (16|M0)              acc0.0<1>:f   r20.0<0;1,0>:hf                  {$9.dst}             //  ALU pipe: float; $44
        mov (16|M0)              acc2.0<1>:f   r20.1<0;1,0>:hf                                       //  ALU pipe: float; $49
        mov (8|M0)               r41.0<1>:f    r33.8<1;1,0>:hf                                       //  ALU pipe: float; $55
        mad (16|M0)              acc0.0<1>:f   r17.0<1;0>:f      acc0.0<1;0>:f     r28.0<1>:f       {Compacted,F@7} //  ALU pipe: float; $50
        mov (16|M0)              acc4.0<1>:f   r20.2<0;1,0>:hf                                       //  ALU pipe: float; $56
        mov (8|M0)               r42.0<1>:f    r37.0<1;1,0>:hf                  {$8.dst}             //  ALU pipe: float; $60
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc2.0<1;0>:f     r30.0<1>:f       {F@7} //  ALU pipe: float; $51
        mov (8|M0)               r43.0<1>:f    r37.8<1;1,0>:hf                                       //  ALU pipe: float; $60
        mov (16|M0)              acc2.0<1>:f   r20.3<0;1,0>:hf                                       //  ALU pipe: float; $61
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc4.0<1;0>:f     r40.0<1>:f       {F@7} //  ALU pipe: float; $62
        mov (16|M0)              acc4.0<1>:f   r20.4<0;1,0>:hf                                       //  ALU pipe: float; $68
        mov (8|M0)               r52.0<1>:f    r45.0<1;1,0>:hf                  {$10.dst}            //  ALU pipe: float; $67
        mov (8|M0)               r53.0<1>:f    r45.8<1;1,0>:hf                                       //  ALU pipe: float; $67
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc2.0<1;0>:f     r42.0<1>:f       {F@6} //  ALU pipe: float; $63
        mov (16|M0)              acc2.0<1>:f   r20.5<0;1,0>:hf                                       //  ALU pipe: float; $73
        mov (8|M0)               r54.0<1>:f    r49.0<1;1,0>:hf                  {$11.dst}            //  ALU pipe: float; $72
        mov (8|M0)               r55.0<1>:f    r49.8<1;1,0>:hf                                       //  ALU pipe: float; $72
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc4.0<1;0>:f     r52.0<1>:f       {F@5} //  ALU pipe: float; $74
        mov (16|M0)              acc4.0<1>:f   r20.6<0;1,0>:hf                                       //  ALU pipe: float; $80
        mov (8|M0)               r64.0<1>:f    r57.0<1;1,0>:hf                  {$12.dst}            //  ALU pipe: float; $79
        mov (8|M0)               r65.0<1>:f    r57.8<1;1,0>:hf                                       //  ALU pipe: float; $79
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc2.0<1;0>:f     r54.0<1>:f       {F@5} //  ALU pipe: float; $75
        sync.nop                             null                             {Compacted,I@3}        // $107
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r85:1 bti[1][r84:1]      {$1} // ex_desc:0x1000000; desc:0x6218C500 // $107
        mov (16|M0)              acc2.0<1>:f   r20.7<0;1,0>:hf                                       //  ALU pipe: float; $85
        mov (8|M0)               r66.0<1>:f    r61.0<1;1,0>:hf                  {$13.dst}            //  ALU pipe: float; $84
        mov (8|M0)               r67.0<1>:f    r61.8<1;1,0>:hf                                       //  ALU pipe: float; $84
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc4.0<1;0>:f     r64.0<1>:f       {F@5} //  ALU pipe: float; $86
        sync.nop                             null                             {Compacted,I@2}        // $114
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r93:1 bti[1][r92:1]      {$2} // ex_desc:0x1000000; desc:0x6218C500 // $114
        mov (16|M0)              acc4.0<1>:f   r20.8<0;1,0>:hf                                       //  ALU pipe: float; $92
        mov (8|M0)               r76.0<1>:f    r69.0<1;1,0>:hf                  {$14.dst}            //  ALU pipe: float; $91
        mov (8|M0)               r77.0<1>:f    r69.8<1;1,0>:hf                                       //  ALU pipe: float; $91
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc2.0<1;0>:f     r66.0<1>:f       {F@5} //  ALU pipe: float; $87
        sync.nop                             null                             {Compacted,I@1}        // $119
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r97:1 bti[1][r96:1]      {$3} // ex_desc:0x1000000; desc:0x6218C500 // $119
        add (1|M0)               r104.0<1>:ud  r15.0<0;1,0>:ud   0x700:uw                            //  ALU pipe: int; $124
        mov (16|M0)              acc2.0<1>:f   r20.9<0;1,0>:hf                                       //  ALU pipe: float; $97
        mov (8|M0)               r78.0<1>:f    r73.0<1;1,0>:hf                  {$15.dst}            //  ALU pipe: float; $96
        mov (8|M0)               r79.0<1>:f    r73.8<1;1,0>:hf                                       //  ALU pipe: float; $96
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc4.0<1;0>:f     r76.0<1>:f       {F@5} //  ALU pipe: float; $98
        sync.nop                             null                             {Compacted,I@1}        // $126
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r105:1 bti[1][r104:1]    {$4} // ex_desc:0x1000000; desc:0x6218C500 // $126
        add (1|M0)               r108.0<1>:ud  r15.0<0;1,0>:ud   0x780:uw                            //  ALU pipe: int; $129
        mov (16|M0)              acc4.0<1>:f   r20.10<0;1,0>:hf                                      //  ALU pipe: float; $104
        mov (8|M0)               r88.0<1>:f    r81.0<1;1,0>:hf                  {$0.dst}             //  ALU pipe: float; $103
        mov (8|M0)               r89.0<1>:f    r81.8<1;1,0>:hf                                       //  ALU pipe: float; $103
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc2.0<1;0>:f     r78.0<1>:f       {F@5} //  ALU pipe: float; $99
        sync.nop                             null                             {Compacted,I@1}        // $131
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r109:1 bti[1][r108:1]    {$5} // ex_desc:0x1000000; desc:0x6218C500 // $131
        mov (16|M0)              acc2.0<1>:f   r20.11<0;1,0>:hf                                      //  ALU pipe: float; $109
        mov (8|M0)               r90.0<1>:f    r85.0<1;1,0>:hf                  {$1.dst}             //  ALU pipe: float; $108
        mov (8|M0)               r91.0<1>:f    r85.8<1;1,0>:hf                                       //  ALU pipe: float; $108
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc4.0<1;0>:f     r88.0<1>:f       {F@5} //  ALU pipe: float; $110
        add (1|M0)               r3.1<1>:d     r3.1<0;1,0>:d     1:w               {Compacted}       //  ALU pipe: int; $137
        mov (16|M0)              acc4.0<1>:f   r20.12<0;1,0>:hf                                      //  ALU pipe: float; $116
        mov (8|M0)               r100.0<1>:f   r93.0<1;1,0>:hf                  {$2.dst}             //  ALU pipe: float; $115
        mov (8|M0)               r101.0<1>:f   r93.8<1;1,0>:hf                                       //  ALU pipe: float; $115
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc2.0<1;0>:f     r90.0<1>:f       {F@5} //  ALU pipe: float; $111
        mov (16|M0)              acc2.0<1>:f   r20.13<0;1,0>:hf                                      //  ALU pipe: float; $121
        mov (8|M0)               r102.0<1>:f   r97.0<1;1,0>:hf                  {$3.dst}             //  ALU pipe: float; $120
        mov (8|M0)               r103.0<1>:f   r97.8<1;1,0>:hf                                       //  ALU pipe: float; $120
        cmp (1|M0)    (eq)f1.0   null<1>:d     r3.1<0;1,0>:d     14:w               {I@1}            //  ALU pipe: int; $138
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc4.0<1;0>:f     r100.0<1>:f      {F@5} //  ALU pipe: float; $122
        mov (16|M0)              acc4.0<1>:f   r20.14<0;1,0>:hf                                      //  ALU pipe: float; $128
        mov (8|M0)               r112.0<1>:f   r105.0<1;1,0>:hf                 {$4.dst}             //  ALU pipe: float; $127
        mov (8|M0)               r113.0<1>:f   r105.8<1;1,0>:hf                                      //  ALU pipe: float; $127
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc2.0<1;0>:f     r102.0<1>:f      {F@5} //  ALU pipe: float; $123
(W)     and (1|M0)               f0.0<1>:uw    f1.0<0;1,0>:uw    0x1:uw                              //  ALU pipe: int; $139
        mov (16|M0)              acc2.0<1>:f   r20.15<0;1,0>:hf                                      //  ALU pipe: float; $133
        mov (8|M0)               r17.0<1>:f    r109.0<1;1,0>:hf                 {$5.dst}             //  ALU pipe: float; $132
        mov (8|M0)               r18.0<1>:f    r109.8<1;1,0>:hf                                      //  ALU pipe: float; $132
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc4.0<1;0>:f     r112.0<1>:f      {F@5} //  ALU pipe: float; $134
        add (1|M0)               r15.0<1>:ud   r15.0<0;1,0>:ud   0x800:uw                            //  ALU pipe: int; $136
        mad (16|M0)              r17.0<1>:f    acc0.0<1;0>:f     acc2.0<1;0>:f     r17.0<1>:f       {F@2} //  ALU pipe: float; $135
(~f0.0.any16h) goto.b (16|M0)                _bmm_nchw_fp16_k0_1_  BB_2                              //  ALU pipe: int; $139
// B005: Preds:{B004},  Succs:{B006}
_bmm_nchw_fp16_k0_1_:
        join (16|M0)                         BB_1                                                    // 
L2360:
        shl (1|M0)               r5.0<1>:ud    r5.0<0;1,0>:ud    0x4:uw                              //  ALU pipe: int; $140
        shl (1|M0)               r3.0<1>:d     r3.0<0;1,0>:d     6:w               {Compacted}       //  ALU pipe: int; $141
        mov (8|M0)               r115.0<1>:hf  r17.0<1;1,0>:f                   {F@1}                //  ALU pipe: float; $144
        mov (8|M0)               r115.8<1>:hf  r18.0<1;1,0>:f                                        //  ALU pipe: float; $144
        add (1|M0)               r5.0<1>:d     r5.0<0;1,0>:d     r3.0<0;1,0>:d    {Compacted,I@1}    //  ALU pipe: int; $142
        shl (1|M0)               r114.0<1>:d   r5.0<0;1,0>:d     1:w               {Compacted,I@1}   //  ALU pipe: int; $143
        store.ugm.d32x8t.a32.wb.wb (1|M0)  bti[2][r114:1] r115:1   {A@1,$6} // ex_desc:0x2000000; desc:0x620EC504 // $145
// B006: Preds:{B005, B002},  Succs:{}
BB_1:
        join (16|M0)                         L2464                                                   // 
L2464:
(W)     mov (8|M0)               r255.0<1>:f   r0.0<1;1,0>:f                    {Compacted}          //  ALU pipe: float; $147
(W)     send.ugm (1|M0)          r116     r0      null:0  0x0            0x0210641F           {$7} // wr:1+0, rd:1; fence invalid flush type scoped to tile // $147
(W)     mov (8|M0)               null<1>:ud    r116.0<1;1,0>:ud                 {$7.dst}             //  memory fence commit; ALU pipe: int; $147
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<1;0>:w       r1.0<1;0>:w       r1.0<1>:w         //  ALU pipe: int; $147
(W)     csel (4|M0)   (eq)f0.0   r1.4<1>:f     r1.4<1;0>:f       r1.4<1;0>:f       r1.4<1>:f        {I@1} //  ALU pipe: float; $147
(W)     send.gtwy (1|M0)         null     r255    null:0  0x0            0x02000010           {EOT,A@1} // wr:1+0, rd:0; end of thread // $147
L2552:
        nop                                                                                          // $147


//.BankConflicts: 3
//.ByteRMWs: 0
//


//.numALUInst: 141
//.accSubDef: 31
//.accSubUse: 31
//.accSubCandidateDef: 31
//.accSubCandidateUse: 31
//
//
//.singlePipeAtOneDistNum: 17
//.allAtOneDistNum: 6
//.syncInstCount: 5
//.tokenReuseCount: 0
//.AfterWriteTokenDepCount: 21
//.AfterReadTokenDepCount: 3
