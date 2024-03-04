//.kernel gemm_nchw_fp16
//.platform DG2
//.thread_config numGRF=256, numAcc=8, numSWSB=16
//.options_string "-enableHalfLSC -dumpcommonisa -output -binary -printregusage -hasNoInt64Add -TotalGRFNum 256 -fusedCallWA 1 -abiver 1 -LSCFenceWA "
//.full_options "-printregusage -TotalGRFNum 256 -output -binary -dumpcommonisa -enableHalfLSC -hasNoInt64Add -fusedCallWA 1 -LSCFenceWA "
//.instCount 258
//.RA type	TRIVIAL_RA
//.git-hash 0725d6fd9e247c17cac748fc32fa6555470a5c0d
//.GRF count 203

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
//.declare V36 (46)  rf=r size=4 type=d align=2 words (r1.5)
//.declare V38 (48)  rf=r size=64 type=f align=16 words (r3.0)
//.declare V39 (49)  rf=r size=4 type=d align=16 words (r5.0)
//.declare V40 (50)  rf=r size=4 type=d align=2 words (r1.6)
//.declare V41 (51)  rf=r size=4 type=d align=2 words (r1.7)
//.declare V42 (52)  rf=r size=4 type=d align=16 words (r6.0)
//.declare V43 (53)  rf=r size=64 type=hf align=16 words (r7.0)
//.declare V44 (54)  rf=r size=32 type=hf align=16 words (r9.0)
//.declare V45 (55)  rf=r size=64 type=f align=16 words (r10.0)
//.declare V46 (56)  rf=r size=4 type=d align=16 words (r12.0)
//.declare V47 (57)  rf=r size=32 type=hf align=16 words (r13.0)
//.declare V48 (58)  rf=r size=64 type=f align=16 words (r14.0)
//.declare V49 (59)  rf=r size=64 type=f align=16 words (r16.0)
//.declare V50 (60)  rf=r size=64 type=f align=16 words (r18.0)
//.declare V51 (61)  rf=r size=4 type=d align=16 words (r20.0)
//.declare V52 (62)  rf=r size=32 type=hf align=16 words (r21.0)
//.declare V53 (63)  rf=r size=64 type=f align=16 words (r22.0)
//.declare V54 (64)  rf=r size=4 type=d align=16 words (r24.0)
//.declare V55 (65)  rf=r size=32 type=hf align=16 words (r25.0)
//.declare V56 (66)  rf=r size=64 type=f align=16 words (r26.0)
//.declare V57 (67)  rf=r size=64 type=f align=16 words (r28.0)
//.declare V58 (68)  rf=r size=64 type=f align=16 words (r30.0)
//.declare V59 (69)  rf=r size=4 type=d align=16 words (r32.0)
//.declare V60 (70)  rf=r size=32 type=hf align=16 words (r33.0)
//.declare V61 (71)  rf=r size=64 type=f align=16 words (r34.0)
//.declare V62 (72)  rf=r size=4 type=d align=16 words (r36.0)
//.declare V63 (73)  rf=r size=32 type=hf align=16 words (r37.0)
//.declare V64 (74)  rf=r size=64 type=f align=16 words (r38.0)
//.declare V65 (75)  rf=r size=64 type=f align=16 words (r40.0)
//.declare V66 (76)  rf=r size=64 type=f align=16 words (r42.0)
//.declare V67 (77)  rf=r size=4 type=d align=16 words (r44.0)
//.declare V68 (78)  rf=r size=32 type=hf align=16 words (r45.0)
//.declare V69 (79)  rf=r size=64 type=f align=16 words (r46.0)
//.declare V70 (80)  rf=r size=4 type=d align=16 words (r48.0)
//.declare V71 (81)  rf=r size=32 type=hf align=16 words (r49.0)
//.declare V72 (82)  rf=r size=64 type=f align=16 words (r50.0)
//.declare V73 (83)  rf=r size=64 type=f align=16 words (r52.0)
//.declare V74 (84)  rf=r size=64 type=f align=16 words (r54.0)
//.declare V75 (85)  rf=r size=4 type=d align=16 words (r56.0)
//.declare V76 (86)  rf=r size=32 type=hf align=16 words (r57.0)
//.declare V77 (87)  rf=r size=64 type=f align=16 words (r58.0)
//.declare V78 (88)  rf=r size=4 type=d align=16 words (r60.0)
//.declare V79 (89)  rf=r size=32 type=hf align=16 words (r61.0)
//.declare V80 (90)  rf=r size=64 type=f align=16 words (r62.0)
//.declare V81 (91)  rf=r size=64 type=f align=16 words (r64.0)
//.declare V82 (92)  rf=r size=64 type=f align=16 words (r66.0)
//.declare V83 (93)  rf=r size=4 type=d align=16 words (r68.0)
//.declare V84 (94)  rf=r size=32 type=hf align=16 words (r69.0)
//.declare V85 (95)  rf=r size=64 type=f align=16 words (r70.0)
//.declare V86 (96)  rf=r size=4 type=d align=16 words (r72.0)
//.declare V87 (97)  rf=r size=32 type=hf align=16 words (r73.0)
//.declare V88 (98)  rf=r size=64 type=f align=16 words (r74.0)
//.declare V89 (99)  rf=r size=64 type=f align=16 words (r76.0)
//.declare V90 (100)  rf=r size=64 type=f align=16 words (r78.0)
//.declare V91 (101)  rf=r size=4 type=d align=16 words (r80.0)
//.declare V92 (102)  rf=r size=32 type=hf align=16 words (r81.0)
//.declare V93 (103)  rf=r size=64 type=f align=16 words (r82.0)
//.declare V94 (104)  rf=r size=4 type=d align=16 words (r84.0)
//.declare V95 (105)  rf=r size=32 type=hf align=16 words (r85.0)
//.declare V96 (106)  rf=r size=64 type=f align=16 words (r86.0)
//.declare V97 (107)  rf=r size=64 type=f align=16 words (r88.0)
//.declare V98 (108)  rf=r size=64 type=f align=16 words (r90.0)
//.declare V99 (109)  rf=r size=4 type=d align=16 words (r92.0)
//.declare V100 (110)  rf=r size=32 type=hf align=16 words (r93.0)
//.declare V101 (111)  rf=r size=64 type=f align=16 words (r94.0)
//.declare V102 (112)  rf=r size=4 type=d align=16 words (r96.0)
//.declare V103 (113)  rf=r size=32 type=hf align=16 words (r97.0)
//.declare V104 (114)  rf=r size=64 type=f align=16 words (r98.0)
//.declare V105 (115)  rf=r size=64 type=f align=16 words (r100.0)
//.declare V106 (116)  rf=r size=64 type=f align=16 words (r102.0)
//.declare V107 (117)  rf=r size=4 type=d align=16 words (r104.0)
//.declare V108 (118)  rf=r size=32 type=hf align=16 words (r105.0)
//.declare V109 (119)  rf=r size=64 type=f align=16 words (r106.0)
//.declare V110 (120)  rf=r size=4 type=d align=16 words (r108.0)
//.declare V111 (121)  rf=r size=32 type=hf align=16 words (r109.0)
//.declare V112 (122)  rf=r size=64 type=f align=16 words (r110.0)
//.declare V113 (123)  rf=r size=64 type=f align=16 words (r112.0)
//.declare V114 (124)  rf=r size=64 type=f align=16 words (r114.0)
//.declare V115 (125)  rf=r size=4 type=d align=16 words (r116.0)
//.declare V116 (126)  rf=r size=32 type=hf align=16 words (r117.0)
//.declare V117 (127)  rf=r size=64 type=f align=16 words (r118.0)
//.declare V118 (128)  rf=r size=4 type=d align=16 words (r120.0)
//.declare V119 (129)  rf=r size=32 type=hf align=16 words (r121.0)
//.declare V120 (130)  rf=r size=64 type=f align=16 words (r122.0)
//.declare V121 (131)  rf=r size=64 type=f align=16 words (r124.0)
//.declare V122 (132)  rf=r size=64 type=f align=16 words (r126.0)
//.declare V123 (133)  rf=r size=4 type=d align=16 words (r128.0)
//.declare V124 (134)  rf=r size=32 type=hf align=16 words (r129.0)
//.declare V125 (135)  rf=r size=64 type=f align=16 words (r130.0)
//.declare V126 (136)  rf=r size=4 type=d align=16 words (r132.0)
//.declare V127 (137)  rf=r size=32 type=hf align=16 words (r133.0)
//.declare V128 (138)  rf=r size=64 type=f align=16 words (r134.0)
//.declare V129 (139)  rf=r size=64 type=f align=16 words (r136.0)
//.declare V130 (140)  rf=r size=64 type=f align=16 words (r138.0)
//.declare V131 (141)  rf=r size=4 type=d align=16 words (r140.0)
//.declare V132 (142)  rf=r size=32 type=hf align=16 words (r141.0)
//.declare V133 (143)  rf=r size=64 type=f align=16 words (r142.0)
//.declare V134 (144)  rf=r size=4 type=d align=16 words (r144.0)
//.declare V135 (145)  rf=r size=32 type=hf align=16 words (r145.0)
//.declare V136 (146)  rf=r size=64 type=f align=16 words (r146.0)
//.declare V137 (147)  rf=r size=64 type=f align=16 words (r148.0)
//.declare V138 (148)  rf=r size=64 type=f align=16 words (r150.0)
//.declare V139 (149)  rf=r size=4 type=d align=16 words (r152.0)
//.declare V140 (150)  rf=r size=32 type=hf align=16 words (r153.0)
//.declare V141 (151)  rf=r size=64 type=f align=16 words (r154.0)
//.declare V142 (152)  rf=r size=4 type=d align=16 words (r156.0)
//.declare V143 (153)  rf=r size=32 type=hf align=16 words (r157.0)
//.declare V144 (154)  rf=r size=64 type=f align=16 words (r158.0)
//.declare V145 (155)  rf=r size=64 type=f align=16 words (r160.0)
//.declare V146 (156)  rf=r size=64 type=f align=16 words (r162.0)
//.declare V147 (157)  rf=r size=4 type=d align=16 words (r164.0)
//.declare V148 (158)  rf=r size=32 type=hf align=16 words (r165.0)
//.declare V149 (159)  rf=r size=64 type=f align=16 words (r166.0)
//.declare V150 (160)  rf=r size=4 type=d align=16 words (r168.0)
//.declare V151 (161)  rf=r size=32 type=hf align=16 words (r169.0)
//.declare V152 (162)  rf=r size=64 type=f align=16 words (r170.0)
//.declare V153 (163)  rf=r size=64 type=f align=16 words (r172.0)
//.declare V154 (164)  rf=r size=64 type=f align=16 words (r174.0)
//.declare V155 (165)  rf=r size=4 type=d align=16 words (r176.0)
//.declare V156 (166)  rf=r size=32 type=hf align=16 words (r177.0)
//.declare V157 (167)  rf=r size=64 type=f align=16 words (r178.0)
//.declare V158 (168)  rf=r size=4 type=d align=16 words (r180.0)
//.declare V159 (169)  rf=r size=32 type=hf align=16 words (r181.0)
//.declare V160 (170)  rf=r size=64 type=f align=16 words (r182.0)
//.declare V161 (171)  rf=r size=64 type=f align=16 words (r184.0)
//.declare V162 (172)  rf=r size=64 type=f align=16 words (r186.0)
//.declare V163 (173)  rf=r size=4 type=d align=16 words (r188.0)
//.declare V164 (174)  rf=r size=32 type=hf align=16 words (r189.0)
//.declare V165 (175)  rf=r size=64 type=f align=16 words (r190.0)
//.declare V166 (176)  rf=r size=4 type=d align=16 words (r192.0)
//.declare V167 (177)  rf=r size=32 type=hf align=16 words (r193.0)
//.declare V168 (178)  rf=r size=64 type=f align=16 words (r194.0)
//.declare V169 (179)  rf=r size=64 type=f align=16 words (r196.0)
//.declare P1 (180)  rf=f1  size=2 type=uw align=1 words (f0.1)
//.declare V170 (181)  rf=r size=4 type=d align=16 words (r198.0)
//.declare V171 (182)  rf=r size=4 type=d align=16 words (r199.0)
//.declare V172 (183)  rf=r size=4 type=d align=16 words (r200.0)
//.declare V173 (184)  rf=r size=32 type=d align=16 words (r201.0)
//.declare V174 (185)  rf=r size=4 type=d alias=V170+0 align=2 words (r198.0)
//.declare V175 (186)  rf=r size=12 type=d alias=V33+0 align=2 words (r2.0)
//.declare V176 (187)  rf=r size=6 type=uw alias=V32+0 align=1 words (r1.0)
//.declare V177 (188)  rf=r size=12 type=d alias=V35+0 align=2 words (r1.2)
//.declare V178 (189)  rf=r size=4 type=d alias=V171+0 align=2 words (r199.0)
//.declare V179 (190)  rf=r size=4 type=d alias=V36+0 align=2 words (r1.5)
//.declare V180 (191)  rf=r size=4 type=d alias=V39+0 align=2 words (r5.0)
//.declare V181 (192)  rf=r size=4 type=d alias=V41+0 align=2 words (r1.7)
//.declare V182 (193)  rf=r size=4 type=d alias=V40+0 align=2 words (r1.6)
//.declare V183 (194)  rf=r size=4 type=d alias=V42+0 align=2 words (r6.0)
//.declare V184 (195)  rf=r size=64 type=d alias=V43+0 align=16 words (r7.0)
//.declare V185 (196)  rf=r size=4 type=ud alias=V42+0 align=2 words (r6.0)
//.declare V186 (197)  rf=r size=32 type=d alias=V44+0 align=16 words (r9.0)
//.declare V187 (198)  rf=r size=4 type=ud alias=V39+0 align=2 words (r5.0)
//.declare V188 (199)  rf=r size=4 type=ud alias=V46+0 align=2 words (r12.0)
//.declare V189 (200)  rf=r size=32 type=d alias=V47+0 align=16 words (r13.0)
//.declare V190 (201)  rf=r size=4 type=ud alias=V51+0 align=2 words (r20.0)
//.declare V191 (202)  rf=r size=32 type=d alias=V52+0 align=16 words (r21.0)
//.declare V192 (203)  rf=r size=4 type=ud alias=V54+0 align=2 words (r24.0)
//.declare V193 (204)  rf=r size=32 type=d alias=V55+0 align=16 words (r25.0)
//.declare V194 (205)  rf=r size=4 type=ud alias=V59+0 align=2 words (r32.0)
//.declare V195 (206)  rf=r size=32 type=d alias=V60+0 align=16 words (r33.0)
//.declare V196 (207)  rf=r size=4 type=ud alias=V62+0 align=2 words (r36.0)
//.declare V197 (208)  rf=r size=32 type=d alias=V63+0 align=16 words (r37.0)
//.declare V198 (209)  rf=r size=4 type=ud alias=V67+0 align=2 words (r44.0)
//.declare V199 (210)  rf=r size=32 type=d alias=V68+0 align=16 words (r45.0)
//.declare V200 (211)  rf=r size=4 type=ud alias=V70+0 align=2 words (r48.0)
//.declare V201 (212)  rf=r size=32 type=d alias=V71+0 align=16 words (r49.0)
//.declare V202 (213)  rf=r size=4 type=ud alias=V75+0 align=2 words (r56.0)
//.declare V203 (214)  rf=r size=32 type=d alias=V76+0 align=16 words (r57.0)
//.declare V204 (215)  rf=r size=4 type=ud alias=V78+0 align=2 words (r60.0)
//.declare V205 (216)  rf=r size=32 type=d alias=V79+0 align=16 words (r61.0)
//.declare V206 (217)  rf=r size=4 type=ud alias=V83+0 align=2 words (r68.0)
//.declare V207 (218)  rf=r size=32 type=d alias=V84+0 align=16 words (r69.0)
//.declare V208 (219)  rf=r size=4 type=ud alias=V86+0 align=2 words (r72.0)
//.declare V209 (220)  rf=r size=32 type=d alias=V87+0 align=16 words (r73.0)
//.declare V210 (221)  rf=r size=4 type=ud alias=V91+0 align=2 words (r80.0)
//.declare V211 (222)  rf=r size=32 type=d alias=V92+0 align=16 words (r81.0)
//.declare V212 (223)  rf=r size=4 type=ud alias=V94+0 align=2 words (r84.0)
//.declare V213 (224)  rf=r size=32 type=d alias=V95+0 align=16 words (r85.0)
//.declare V214 (225)  rf=r size=4 type=ud alias=V99+0 align=2 words (r92.0)
//.declare V215 (226)  rf=r size=32 type=d alias=V100+0 align=16 words (r93.0)
//.declare V216 (227)  rf=r size=4 type=ud alias=V102+0 align=2 words (r96.0)
//.declare V217 (228)  rf=r size=32 type=d alias=V103+0 align=16 words (r97.0)
//.declare V218 (229)  rf=r size=4 type=ud alias=V107+0 align=2 words (r104.0)
//.declare V219 (230)  rf=r size=32 type=d alias=V108+0 align=16 words (r105.0)
//.declare V220 (231)  rf=r size=4 type=ud alias=V110+0 align=2 words (r108.0)
//.declare V221 (232)  rf=r size=32 type=d alias=V111+0 align=16 words (r109.0)
//.declare V222 (233)  rf=r size=4 type=ud alias=V115+0 align=2 words (r116.0)
//.declare V223 (234)  rf=r size=32 type=d alias=V116+0 align=16 words (r117.0)
//.declare V224 (235)  rf=r size=4 type=ud alias=V118+0 align=2 words (r120.0)
//.declare V225 (236)  rf=r size=32 type=d alias=V119+0 align=16 words (r121.0)
//.declare V226 (237)  rf=r size=4 type=ud alias=V123+0 align=2 words (r128.0)
//.declare V227 (238)  rf=r size=32 type=d alias=V124+0 align=16 words (r129.0)
//.declare V228 (239)  rf=r size=4 type=ud alias=V126+0 align=2 words (r132.0)
//.declare V229 (240)  rf=r size=32 type=d alias=V127+0 align=16 words (r133.0)
//.declare V230 (241)  rf=r size=4 type=ud alias=V131+0 align=2 words (r140.0)
//.declare V231 (242)  rf=r size=32 type=d alias=V132+0 align=16 words (r141.0)
//.declare V232 (243)  rf=r size=4 type=ud alias=V134+0 align=2 words (r144.0)
//.declare V233 (244)  rf=r size=32 type=d alias=V135+0 align=16 words (r145.0)
//.declare V234 (245)  rf=r size=4 type=ud alias=V139+0 align=2 words (r152.0)
//.declare V235 (246)  rf=r size=32 type=d alias=V140+0 align=16 words (r153.0)
//.declare V236 (247)  rf=r size=4 type=ud alias=V142+0 align=2 words (r156.0)
//.declare V237 (248)  rf=r size=32 type=d alias=V143+0 align=16 words (r157.0)
//.declare V238 (249)  rf=r size=4 type=ud alias=V147+0 align=2 words (r164.0)
//.declare V239 (250)  rf=r size=32 type=d alias=V148+0 align=16 words (r165.0)
//.declare V240 (251)  rf=r size=4 type=ud alias=V150+0 align=2 words (r168.0)
//.declare V241 (252)  rf=r size=32 type=d alias=V151+0 align=16 words (r169.0)
//.declare V242 (253)  rf=r size=4 type=ud alias=V155+0 align=2 words (r176.0)
//.declare V243 (254)  rf=r size=32 type=d alias=V156+0 align=16 words (r177.0)
//.declare V244 (255)  rf=r size=4 type=ud alias=V158+0 align=2 words (r180.0)
//.declare V245 (256)  rf=r size=32 type=d alias=V159+0 align=16 words (r181.0)
//.declare V246 (257)  rf=r size=4 type=ud alias=V163+0 align=2 words (r188.0)
//.declare V247 (258)  rf=r size=32 type=d alias=V164+0 align=16 words (r189.0)
//.declare V248 (259)  rf=r size=4 type=ud alias=V166+0 align=2 words (r192.0)
//.declare V249 (260)  rf=r size=32 type=d alias=V167+0 align=16 words (r193.0)
//.declare V250 (261)  rf=r size=4 type=d alias=V172+0 align=2 words (r200.0)
//.declare V251 (262)  rf=r size=32 type=hf alias=V173+0 align=16 words (r201.0)
//.declare V252 (263)  rf=r size=32 type=hf alias=V173+0 align=16 words (r201.0)
//.declare V253 (264)  rf=r size=4 type=ud alias=V172+0 align=2 words (r200.0)
//.declare  (265)  rf=f16  size=2 type=uw align=1 words (f0.0)
//.declare  (266)  rf=r size=32 type=ud align=16 words (r255.0)
//.declare  (267)  rf=f16  size=2 type=uw align=1 words (f0.0)
//.declare  (268)  rf=r size=8 type=w align=1 words (r1.0)
//.declare  (269)  rf=r size=16 type=f align=2 words (r1.4)
//.declare r0 (270)  rf=r size=32 type=ud align=16 words (r0.0)
//.declare rtmp (271)  rf=r size=32 type=ud align=16 words (r255.0)
//.declare  (272)  rf=r size=32 type=ud align=16 words (r1.0)
//.declare  (273)  rf=r size=64 type=ud align=16 words (r2.0)

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
// B002: Preds:{B001},  Succs:{B003}
// gemm_nchw_fp16_BB_0:
        mov (1|M0)               r198.0<1>:d   r0.1<0;1,0>:ud                                        //  ALU pipe: int; $1
        mov (1|M0)               r199.0<1>:d   r0.6<0;1,0>:ud                                        //  ALU pipe: int; $2
        mov (2|M0)               r1.2<1>:d     r1.0<1;1,0>:uw                   {$0.dst}             //  ALU pipe: int; $4
        mov (16|M0)              r3.0<1>:f     0.0:f                               {Compacted,$1.dst} //  ALU pipe: float; $11
(W)     mul (1|M0)               acc0.0<1>:d   r198.0<0;1,0>:d   r2.0<0;1,0>:uw   {Compacted,I@3}    //  ALU pipe: int; $3
        mov (1|M0)               r1.6<1>:d     0:w                                                   //  ALU pipe: int; $12
        mach (1|M0)              r198.0<1>:d   r198.0<0;1,0>:d   r2.0<0;1,0>:d    {Compacted}        //  ALU pipe: int; $4
(W)     mul (1|M0)               acc0.0<1>:d   r199.0<0;1,0>:d   r2.2<0;1,0>:uw   {Compacted,I@5}    //  ALU pipe: int; $6
        mach (1|M0)              r199.0<1>:d   r199.0<0;1,0>:d   r2.1<0;1,0>:d    {Compacted}        //  ALU pipe: int; $7
        add (1|M0)               r198.0<1>:d   r198.0<0;1,0>:d   r1.2<0;1,0>:d    {Compacted,I@3}    //  ALU pipe: int; $5
        add (1|M0)               r199.0<1>:d   r199.0<0;1,0>:d   r1.3<0;1,0>:d    {Compacted,I@2}    //  ALU pipe: int; $7
        shl (1|M0)               r1.5<1>:d     r198.0<0;1,0>:d   13:w               {Compacted,I@2}  //  ALU pipe: int; $8
        shl (1|M0)               r5.0<1>:d     r199.0<0;1,0>:d   5:w               {Compacted,I@2}   //  ALU pipe: int; $9
// B003: Preds:{B003, B002},  Succs:{B004, B003}
BB_1:
        shl (1|M0)               r1.7<1>:d     r1.6<0;1,0>:d     6:w                                 //  ALU pipe: int; $14
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r9:1  bti[1][r5:1]       {A@2,$2} // ex_desc:0x1000000; desc:0x6218C500 // $19
        add (1|M0)               r12.0<1>:ud   r5.0<0;1,0>:ud    0x2000:uw                           //  ALU pipe: int; $22
        add (1|M0)               r20.0<1>:ud   r5.0<0;1,0>:ud    0x4000:uw                           //  ALU pipe: int; $29
        add (1|M0)               r24.0<1>:ud   r5.0<0;1,0>:ud    0x6000:uw                           //  ALU pipe: int; $34
        add (1|M0)               r6.0<1>:d     r1.7<0;1,0>:d     r1.5<0;1,0>:d    {I@4}              //  ALU pipe: int; $15
        add (1|M0)               r32.0<1>:ud   r5.0<0;1,0>:ud    0x8000:uw                           //  ALU pipe: int; $41
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r13:1 bti[1][r12:1]      {A@5,$3} // ex_desc:0x1000000; desc:0x6218C500 // $24
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r21:1 bti[1][r20:1]      {A@4,$4} // ex_desc:0x1000000; desc:0x6218C500 // $31
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r25:1 bti[1][r24:1]      {A@3,$5} // ex_desc:0x1000000; desc:0x6218C500 // $36
        load.ugm.d32x16t.a32.ca.ca (1|M0)  r7:2 bti[0][r6:1]       {A@2,$6} // ex_desc:0x0; desc:0x6228D500 // $17
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r33:1 bti[1][r32:1]      {A@1,$7} // ex_desc:0x1000000; desc:0x6218C500 // $43
        add (1|M0)               r36.0<1>:ud   r5.0<0;1,0>:ud    0xA000:uw                           //  ALU pipe: int; $46
        add (1|M0)               r44.0<1>:ud   r5.0<0;1,0>:ud    0xC000:uw                           //  ALU pipe: int; $53
        add (1|M0)               r48.0<1>:ud   r5.0<0;1,0>:ud    0xE000:uw                           //  ALU pipe: int; $58
        add (1|M0)               r56.0<1>:ud   r5.0<0;1,0>:ud    0x10000:ud                          //  ALU pipe: int; $65
        add (1|M0)               r60.0<1>:ud   r5.0<0;1,0>:ud    0x12000:ud                          //  ALU pipe: int; $70
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r37:1 bti[1][r36:1]      {A@5,$8} // ex_desc:0x1000000; desc:0x6218C500 // $48
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r45:1 bti[1][r44:1]      {A@4,$9} // ex_desc:0x1000000; desc:0x6218C500 // $55
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r49:1 bti[1][r48:1]      {A@3,$10} // ex_desc:0x1000000; desc:0x6218C500 // $60
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r57:1 bti[1][r56:1]      {A@2,$11} // ex_desc:0x1000000; desc:0x6218C500 // $67
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r61:1 bti[1][r60:1]      {A@1,$12} // ex_desc:0x1000000; desc:0x6218C500 // $72
        add (1|M0)               r68.0<1>:ud   r5.0<0;1,0>:ud    0x14000:ud                          //  ALU pipe: int; $77
        mov (8|M0)               r16.0<1>:f    r9.0<1;1,0>:hf                   {$2.dst}             //  ALU pipe: float; $20
        mov (8|M0)               r17.0<1>:f    r9.8<1;1,0>:hf                                        //  ALU pipe: float; $20
        add (1|M0)               r72.0<1>:ud   r5.0<0;1,0>:ud    0x16000:ud                          //  ALU pipe: int; $82
        add (1|M0)               r80.0<1>:ud   r5.0<0;1,0>:ud    0x18000:ud                          //  ALU pipe: int; $89
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r69:1 bti[1][r68:1]      {A@3,$13} // ex_desc:0x1000000; desc:0x6218C500 // $79
        add (1|M0)               r84.0<1>:ud   r5.0<0;1,0>:ud    0x1A000:ud                          //  ALU pipe: int; $94
        mov (8|M0)               r18.0<1>:f    r13.0<1;1,0>:hf                  {$3.dst}             //  ALU pipe: float; $25
        mov (8|M0)               r19.0<1>:f    r13.8<1;1,0>:hf                                       //  ALU pipe: float; $25
        mov (8|M0)               r28.0<1>:f    r21.0<1;1,0>:hf                  {$4.dst}             //  ALU pipe: float; $32
        mov (16|M0)              acc0.0<1>:f   r7.0<0;1,0>:hf                   {$6.dst}             //  ALU pipe: float; $21
        mov (16|M0)              acc2.0<1>:f   r7.1<0;1,0>:hf                                        //  ALU pipe: float; $26
        mov (8|M0)               r29.0<1>:f    r21.8<1;1,0>:hf                                       //  ALU pipe: float; $32
        mad (16|M0)              acc0.0<1>:f   r3.0<1;0>:f       acc0.0<1;0>:f     r16.0<1>:f       {Compacted,F@7} //  ALU pipe: float; $27
        mov (16|M0)              acc4.0<1>:f   r7.2<0;1,0>:hf                                        //  ALU pipe: float; $33
        mov (8|M0)               r30.0<1>:f    r25.0<1;1,0>:hf                  {$5.dst}             //  ALU pipe: float; $37
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc2.0<1;0>:f     r18.0<1>:f       {F@7} //  ALU pipe: float; $28
        mov (8|M0)               r31.0<1>:f    r25.8<1;1,0>:hf                                       //  ALU pipe: float; $37
        mov (16|M0)              acc2.0<1>:f   r7.3<0;1,0>:hf                                        //  ALU pipe: float; $38
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc4.0<1;0>:f     r28.0<1>:f       {F@7} //  ALU pipe: float; $39
        mov (16|M0)              acc4.0<1>:f   r7.4<0;1,0>:hf                                        //  ALU pipe: float; $45
        mov (8|M0)               r40.0<1>:f    r33.0<1;1,0>:hf                  {$7.dst}             //  ALU pipe: float; $44
        mov (8|M0)               r41.0<1>:f    r33.8<1;1,0>:hf                                       //  ALU pipe: float; $44
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc2.0<1;0>:f     r30.0<1>:f       {F@6} //  ALU pipe: float; $40
        mov (16|M0)              acc2.0<1>:f   r7.5<0;1,0>:hf                                        //  ALU pipe: float; $50
        mov (8|M0)               r42.0<1>:f    r37.0<1;1,0>:hf                  {$8.dst}             //  ALU pipe: float; $49
        mov (8|M0)               r43.0<1>:f    r37.8<1;1,0>:hf                                       //  ALU pipe: float; $49
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc4.0<1;0>:f     r40.0<1>:f       {F@5} //  ALU pipe: float; $51
        mov (16|M0)              acc4.0<1>:f   r7.6<0;1,0>:hf                                        //  ALU pipe: float; $57
        mov (8|M0)               r52.0<1>:f    r45.0<1;1,0>:hf                  {$9.dst}             //  ALU pipe: float; $56
        mov (8|M0)               r53.0<1>:f    r45.8<1;1,0>:hf                                       //  ALU pipe: float; $56
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc2.0<1;0>:f     r42.0<1>:f       {F@5} //  ALU pipe: float; $52
        sync.nop                             null                             {Compacted,I@3}        // $84
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r73:1 bti[1][r72:1]      {$14} // ex_desc:0x1000000; desc:0x6218C500 // $84
        mov (16|M0)              acc2.0<1>:f   r7.7<0;1,0>:hf                                        //  ALU pipe: float; $62
        mov (8|M0)               r54.0<1>:f    r49.0<1;1,0>:hf                  {$10.dst}            //  ALU pipe: float; $61
        mov (8|M0)               r55.0<1>:f    r49.8<1;1,0>:hf                                       //  ALU pipe: float; $61
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc4.0<1;0>:f     r52.0<1>:f       {F@5} //  ALU pipe: float; $63
        sync.nop                             null                             {Compacted,I@2}        // $91
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r81:1 bti[1][r80:1]      {$15} // ex_desc:0x1000000; desc:0x6218C500 // $91
        mov (16|M0)              acc4.0<1>:f   r7.8<0;1,0>:hf                                        //  ALU pipe: float; $69
        mov (8|M0)               r64.0<1>:f    r57.0<1;1,0>:hf                  {$11.dst}            //  ALU pipe: float; $68
        mov (8|M0)               r65.0<1>:f    r57.8<1;1,0>:hf                                       //  ALU pipe: float; $68
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc2.0<1;0>:f     r54.0<1>:f       {F@5} //  ALU pipe: float; $64
        sync.nop                             null                             {Compacted,I@1}        // $96
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r85:1 bti[1][r84:1]      {$0} // ex_desc:0x1000000; desc:0x6218C500 // $96
        add (1|M0)               r92.0<1>:ud   r5.0<0;1,0>:ud    0x1C000:ud                          //  ALU pipe: int; $101
        mov (16|M0)              acc2.0<1>:f   r7.9<0;1,0>:hf                                        //  ALU pipe: float; $74
        mov (8|M0)               r66.0<1>:f    r61.0<1;1,0>:hf                  {$12.dst}            //  ALU pipe: float; $73
        mov (8|M0)               r67.0<1>:f    r61.8<1;1,0>:hf                                       //  ALU pipe: float; $73
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc4.0<1;0>:f     r64.0<1>:f       {F@5} //  ALU pipe: float; $75
        sync.nop                             null                             {Compacted,I@1}        // $103
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r93:1 bti[1][r92:1]      {$1} // ex_desc:0x1000000; desc:0x6218C500 // $103
        add (1|M0)               r96.0<1>:ud   r5.0<0;1,0>:ud    0x1E000:ud                          //  ALU pipe: int; $106
        mov (16|M0)              acc4.0<1>:f   r7.10<0;1,0>:hf                                       //  ALU pipe: float; $81
        mov (8|M0)               r76.0<1>:f    r69.0<1;1,0>:hf                  {$13.dst}            //  ALU pipe: float; $80
        mov (8|M0)               r77.0<1>:f    r69.8<1;1,0>:hf                                       //  ALU pipe: float; $80
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc2.0<1;0>:f     r66.0<1>:f       {F@5} //  ALU pipe: float; $76
        sync.nop                             null                             {Compacted,I@1}        // $108
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r97:1 bti[1][r96:1]      {$2} // ex_desc:0x1000000; desc:0x6218C500 // $108
        add (1|M0)               r104.0<1>:ud  r5.0<0;1,0>:ud    0x20000:ud                          //  ALU pipe: int; $113
        mov (16|M0)              acc2.0<1>:f   r7.11<0;1,0>:hf                                       //  ALU pipe: float; $86
        mov (8|M0)               r78.0<1>:f    r73.0<1;1,0>:hf                  {$14.dst}            //  ALU pipe: float; $85
        mov (8|M0)               r79.0<1>:f    r73.8<1;1,0>:hf                                       //  ALU pipe: float; $85
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc4.0<1;0>:f     r76.0<1>:f       {F@5} //  ALU pipe: float; $87
        sync.nop                             null                             {Compacted,I@1}        // $115
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r105:1 bti[1][r104:1]    {$3} // ex_desc:0x1000000; desc:0x6218C500 // $115
        add (1|M0)               r108.0<1>:ud  r5.0<0;1,0>:ud    0x22000:ud                          //  ALU pipe: int; $118
        mov (16|M0)              acc4.0<1>:f   r7.12<0;1,0>:hf                                       //  ALU pipe: float; $93
        mov (8|M0)               r88.0<1>:f    r81.0<1;1,0>:hf                  {$15.dst}            //  ALU pipe: float; $92
        mov (8|M0)               r89.0<1>:f    r81.8<1;1,0>:hf                                       //  ALU pipe: float; $92
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc2.0<1;0>:f     r78.0<1>:f       {F@5} //  ALU pipe: float; $88
        sync.nop                             null                             {Compacted,I@1}        // $120
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r109:1 bti[1][r108:1]    {$4} // ex_desc:0x1000000; desc:0x6218C500 // $120
        add (1|M0)               r116.0<1>:ud  r5.0<0;1,0>:ud    0x24000:ud                          //  ALU pipe: int; $125
        mov (16|M0)              acc2.0<1>:f   r7.13<0;1,0>:hf                                       //  ALU pipe: float; $98
        mov (8|M0)               r90.0<1>:f    r85.0<1;1,0>:hf                  {$0.dst}             //  ALU pipe: float; $97
        mov (8|M0)               r91.0<1>:f    r85.8<1;1,0>:hf                                       //  ALU pipe: float; $97
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc4.0<1;0>:f     r88.0<1>:f       {F@5} //  ALU pipe: float; $99
        sync.nop                             null                             {Compacted,I@1}        // $127
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r117:1 bti[1][r116:1]    {$5} // ex_desc:0x1000000; desc:0x6218C500 // $127
        add (1|M0)               r120.0<1>:ud  r5.0<0;1,0>:ud    0x26000:ud                          //  ALU pipe: int; $130
        mov (16|M0)              acc4.0<1>:f   r7.14<0;1,0>:hf                                       //  ALU pipe: float; $105
        mov (8|M0)               r100.0<1>:f   r93.0<1;1,0>:hf                  {$1.dst}             //  ALU pipe: float; $104
        mov (8|M0)               r101.0<1>:f   r93.8<1;1,0>:hf                                       //  ALU pipe: float; $104
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc2.0<1;0>:f     r90.0<1>:f       {F@5} //  ALU pipe: float; $100
        sync.nop                             null                             {Compacted,I@1}        // $132
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r121:1 bti[1][r120:1]    {$6} // ex_desc:0x1000000; desc:0x6218C500 // $132
        add (1|M0)               r128.0<1>:ud  r5.0<0;1,0>:ud    0x28000:ud                          //  ALU pipe: int; $137
        mov (16|M0)              acc2.0<1>:f   r7.15<0;1,0>:hf                                       //  ALU pipe: float; $110
        mov (8|M0)               r102.0<1>:f   r97.0<1;1,0>:hf                  {$2.dst}             //  ALU pipe: float; $109
        mov (8|M0)               r103.0<1>:f   r97.8<1;1,0>:hf                                       //  ALU pipe: float; $109
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc4.0<1;0>:f     r100.0<1>:f      {F@5} //  ALU pipe: float; $111
        sync.nop                             null                             {Compacted,I@1}        // $139
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r129:1 bti[1][r128:1]    {$7} // ex_desc:0x1000000; desc:0x6218C500 // $139
        add (1|M0)               r132.0<1>:ud  r5.0<0;1,0>:ud    0x2A000:ud                          //  ALU pipe: int; $142
        mov (16|M0)              acc4.0<1>:f   r8.0<0;1,0>:hf                                        //  ALU pipe: float; $117
        mov (8|M0)               r112.0<1>:f   r105.0<1;1,0>:hf                 {$3.dst}             //  ALU pipe: float; $116
        mov (8|M0)               r113.0<1>:f   r105.8<1;1,0>:hf                                      //  ALU pipe: float; $116
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc2.0<1;0>:f     r102.0<1>:f      {F@5} //  ALU pipe: float; $112
        sync.nop                             null                             {Compacted,I@1}        // $144
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r133:1 bti[1][r132:1]    {$8} // ex_desc:0x1000000; desc:0x6218C500 // $144
        add (1|M0)               r140.0<1>:ud  r5.0<0;1,0>:ud    0x2C000:ud                          //  ALU pipe: int; $149
        mov (16|M0)              acc2.0<1>:f   r8.1<0;1,0>:hf                                        //  ALU pipe: float; $122
        mov (8|M0)               r114.0<1>:f   r109.0<1;1,0>:hf                 {$4.dst}             //  ALU pipe: float; $121
        mov (8|M0)               r115.0<1>:f   r109.8<1;1,0>:hf                                      //  ALU pipe: float; $121
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc4.0<1;0>:f     r112.0<1>:f      {F@5} //  ALU pipe: float; $123
        sync.nop                             null                             {Compacted,I@1}        // $151
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r141:1 bti[1][r140:1]    {$9} // ex_desc:0x1000000; desc:0x6218C500 // $151
        add (1|M0)               r144.0<1>:ud  r5.0<0;1,0>:ud    0x2E000:ud                          //  ALU pipe: int; $154
        mov (16|M0)              acc4.0<1>:f   r8.2<0;1,0>:hf                                        //  ALU pipe: float; $129
        mov (8|M0)               r124.0<1>:f   r117.0<1;1,0>:hf                 {$5.dst}             //  ALU pipe: float; $128
        mov (8|M0)               r125.0<1>:f   r117.8<1;1,0>:hf                                      //  ALU pipe: float; $128
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc2.0<1;0>:f     r114.0<1>:f      {F@5} //  ALU pipe: float; $124
        sync.nop                             null                             {Compacted,I@1}        // $156
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r145:1 bti[1][r144:1]    {$10} // ex_desc:0x1000000; desc:0x6218C500 // $156
        add (1|M0)               r152.0<1>:ud  r5.0<0;1,0>:ud    0x30000:ud                          //  ALU pipe: int; $161
        mov (16|M0)              acc2.0<1>:f   r8.3<0;1,0>:hf                                        //  ALU pipe: float; $134
        mov (8|M0)               r126.0<1>:f   r121.0<1;1,0>:hf                 {$6.dst}             //  ALU pipe: float; $133
        mov (8|M0)               r127.0<1>:f   r121.8<1;1,0>:hf                                      //  ALU pipe: float; $133
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc4.0<1;0>:f     r124.0<1>:f      {F@5} //  ALU pipe: float; $135
        sync.nop                             null                             {Compacted,I@1}        // $163
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r153:1 bti[1][r152:1]    {$11} // ex_desc:0x1000000; desc:0x6218C500 // $163
        add (1|M0)               r156.0<1>:ud  r5.0<0;1,0>:ud    0x32000:ud                          //  ALU pipe: int; $166
        mov (16|M0)              acc4.0<1>:f   r8.4<0;1,0>:hf                                        //  ALU pipe: float; $141
        mov (8|M0)               r136.0<1>:f   r129.0<1;1,0>:hf                 {$7.dst}             //  ALU pipe: float; $140
        mov (8|M0)               r137.0<1>:f   r129.8<1;1,0>:hf                                      //  ALU pipe: float; $140
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc2.0<1;0>:f     r126.0<1>:f      {F@5} //  ALU pipe: float; $136
        sync.nop                             null                             {Compacted,I@1}        // $168
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r157:1 bti[1][r156:1]    {$12} // ex_desc:0x1000000; desc:0x6218C500 // $168
        add (1|M0)               r164.0<1>:ud  r5.0<0;1,0>:ud    0x34000:ud                          //  ALU pipe: int; $173
        mov (16|M0)              acc2.0<1>:f   r8.5<0;1,0>:hf                                        //  ALU pipe: float; $146
        mov (8|M0)               r138.0<1>:f   r133.0<1;1,0>:hf                 {$8.dst}             //  ALU pipe: float; $145
        mov (8|M0)               r139.0<1>:f   r133.8<1;1,0>:hf                                      //  ALU pipe: float; $145
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc4.0<1;0>:f     r136.0<1>:f      {F@5} //  ALU pipe: float; $147
        sync.nop                             null                             {Compacted,I@1}        // $175
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r165:1 bti[1][r164:1]    {$13} // ex_desc:0x1000000; desc:0x6218C500 // $175
        add (1|M0)               r168.0<1>:ud  r5.0<0;1,0>:ud    0x36000:ud                          //  ALU pipe: int; $178
        mov (16|M0)              acc4.0<1>:f   r8.6<0;1,0>:hf                                        //  ALU pipe: float; $153
        mov (8|M0)               r148.0<1>:f   r141.0<1;1,0>:hf                 {$9.dst}             //  ALU pipe: float; $152
        mov (8|M0)               r149.0<1>:f   r141.8<1;1,0>:hf                                      //  ALU pipe: float; $152
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc2.0<1;0>:f     r138.0<1>:f      {F@5} //  ALU pipe: float; $148
        sync.nop                             null                             {Compacted,I@1}        // $180
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r169:1 bti[1][r168:1]    {$14} // ex_desc:0x1000000; desc:0x6218C500 // $180
        add (1|M0)               r176.0<1>:ud  r5.0<0;1,0>:ud    0x38000:ud                          //  ALU pipe: int; $185
        mov (16|M0)              acc2.0<1>:f   r8.7<0;1,0>:hf                                        //  ALU pipe: float; $158
        mov (8|M0)               r150.0<1>:f   r145.0<1;1,0>:hf                 {$10.dst}            //  ALU pipe: float; $157
        mov (8|M0)               r151.0<1>:f   r145.8<1;1,0>:hf                                      //  ALU pipe: float; $157
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc4.0<1;0>:f     r148.0<1>:f      {F@5} //  ALU pipe: float; $159
        sync.nop                             null                             {Compacted,I@1}        // $187
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r177:1 bti[1][r176:1]    {$15} // ex_desc:0x1000000; desc:0x6218C500 // $187
        add (1|M0)               r180.0<1>:ud  r5.0<0;1,0>:ud    0x3A000:ud                          //  ALU pipe: int; $190
        mov (16|M0)              acc4.0<1>:f   r8.8<0;1,0>:hf                                        //  ALU pipe: float; $165
        mov (8|M0)               r160.0<1>:f   r153.0<1;1,0>:hf                 {$11.dst}            //  ALU pipe: float; $164
        mov (8|M0)               r161.0<1>:f   r153.8<1;1,0>:hf                                      //  ALU pipe: float; $164
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc2.0<1;0>:f     r150.0<1>:f      {F@5} //  ALU pipe: float; $160
        sync.nop                             null                             {Compacted,I@1}        // $192
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r181:1 bti[1][r180:1]    {$0} // ex_desc:0x1000000; desc:0x6218C500 // $192
        add (1|M0)               r188.0<1>:ud  r5.0<0;1,0>:ud    0x3C000:ud                          //  ALU pipe: int; $197
        mov (16|M0)              acc2.0<1>:f   r8.9<0;1,0>:hf                                        //  ALU pipe: float; $170
        mov (8|M0)               r162.0<1>:f   r157.0<1;1,0>:hf                 {$12.dst}            //  ALU pipe: float; $169
        mov (8|M0)               r163.0<1>:f   r157.8<1;1,0>:hf                                      //  ALU pipe: float; $169
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc4.0<1;0>:f     r160.0<1>:f      {F@5} //  ALU pipe: float; $171
        sync.nop                             null                             {Compacted,I@1}        // $199
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r189:1 bti[1][r188:1]    {$1} // ex_desc:0x1000000; desc:0x6218C500 // $199
        add (1|M0)               r192.0<1>:ud  r5.0<0;1,0>:ud    0x3E000:ud                          //  ALU pipe: int; $202
        mov (16|M0)              acc4.0<1>:f   r8.10<0;1,0>:hf                                       //  ALU pipe: float; $177
        mov (8|M0)               r172.0<1>:f   r165.0<1;1,0>:hf                 {$13.dst}            //  ALU pipe: float; $176
        mov (8|M0)               r173.0<1>:f   r165.8<1;1,0>:hf                                      //  ALU pipe: float; $176
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc2.0<1;0>:f     r162.0<1>:f      {F@5} //  ALU pipe: float; $172
        sync.nop                             null                             {Compacted,I@1}        // $204
        load.ugm.d32x8t.a32.ca.ca (1|M0)  r193:1 bti[1][r192:1]    {$2} // ex_desc:0x1000000; desc:0x6218C500 // $204
        mov (16|M0)              acc2.0<1>:f   r8.11<0;1,0>:hf                                       //  ALU pipe: float; $182
        mov (8|M0)               r174.0<1>:f   r169.0<1;1,0>:hf                 {$14.dst}            //  ALU pipe: float; $181
        mov (8|M0)               r175.0<1>:f   r169.8<1;1,0>:hf                                      //  ALU pipe: float; $181
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc4.0<1;0>:f     r172.0<1>:f      {F@5} //  ALU pipe: float; $183
        add (1|M0)               r1.6<1>:d     r1.6<0;1,0>:d     1:w                                 //  ALU pipe: int; $210
        mov (16|M0)              acc4.0<1>:f   r8.12<0;1,0>:hf                                       //  ALU pipe: float; $189
        mov (8|M0)               r184.0<1>:f   r177.0<1;1,0>:hf                 {$15.dst}            //  ALU pipe: float; $188
        mov (8|M0)               r185.0<1>:f   r177.8<1;1,0>:hf                                      //  ALU pipe: float; $188
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc2.0<1;0>:f     r174.0<1>:f      {F@5} //  ALU pipe: float; $184
        mov (16|M0)              acc2.0<1>:f   r8.13<0;1,0>:hf                                       //  ALU pipe: float; $194
        mov (8|M0)               r186.0<1>:f   r181.0<1;1,0>:hf                 {$0.dst}             //  ALU pipe: float; $193
        mov (8|M0)               r187.0<1>:f   r181.8<1;1,0>:hf                                      //  ALU pipe: float; $193
        cmp (1|M0)    (eq)f0.1   null<1>:d     r1.6<0;1,0>:d     128:w               {I@1}           //  ALU pipe: int; $211
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc4.0<1;0>:f     r184.0<1>:f      {F@5} //  ALU pipe: float; $195
        mov (16|M0)              acc4.0<1>:f   r8.14<0;1,0>:hf                                       //  ALU pipe: float; $201
        mov (8|M0)               r196.0<1>:f   r189.0<1;1,0>:hf                 {$1.dst}             //  ALU pipe: float; $200
        mov (8|M0)               r197.0<1>:f   r189.8<1;1,0>:hf                                      //  ALU pipe: float; $200
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc2.0<1;0>:f     r186.0<1>:f      {F@5} //  ALU pipe: float; $196
(W)     and (1|M0)               f0.0<1>:uw    f0.1<0;1,0>:uw    0x1:uw                              //  ALU pipe: int; $212
        mov (16|M0)              acc2.0<1>:f   r8.15<0;1,0>:hf                                       //  ALU pipe: float; $206
        mov (8|M0)               r3.0<1>:f     r193.0<1;1,0>:hf                 {$2.dst}             //  ALU pipe: float; $205
        mov (8|M0)               r4.0<1>:f     r193.8<1;1,0>:hf                                      //  ALU pipe: float; $205
        mad (16|M0)              acc0.0<1>:f   acc0.0<1;0>:f     acc4.0<1;0>:f     r196.0<1>:f      {F@5} //  ALU pipe: float; $207
        add (1|M0)               r5.0<1>:ud    r5.0<0;1,0>:ud    0x40000:ud                          //  ALU pipe: int; $209
        mad (16|M0)              r3.0<1>:f     acc0.0<1;0>:f     acc2.0<1;0>:f     r3.0<1>:f        {F@2} //  ALU pipe: float; $208
(~f0.0.any16h) goto.b (16|M0)                _gemm_nchw_fp16_k0_0_  BB_1                             //  ALU pipe: int; $212
// B004: Preds:{B003},  Succs:{}
_gemm_nchw_fp16_k0_0_:
        join (16|M0)                         L3672                                                   // 
L3672:
        shl (1|M0)               r198.0<1>:d   r198.0<0;1,0>:d   12:w               {Compacted}      //  ALU pipe: int; $213
        shl (1|M0)               r199.0<1>:d   r199.0<0;1,0>:d   4:w               {Compacted}       //  ALU pipe: int; $214
        mov (8|M0)               r201.0<1>:hf  r3.0<1;1,0>:f                    {F@1}                //  ALU pipe: float; $217
        mov (8|M0)               r201.8<1>:hf  r4.0<1;1,0>:f                                         //  ALU pipe: float; $217
(W)     mov (8|M0)               r255.0<1>:f   r0.0<1;1,0>:f                    {Compacted}          //  ALU pipe: float; $220
        add (1|M0)               r199.0<1>:d   r199.0<0;1,0>:d   r198.0<0;1,0>:d  {Compacted,I@1}    //  ALU pipe: int; $215 R{} IR{}{O:1,O:1,},  {BC=1}
        add (16|M0)              r201.0<1>:hf  r201.0<1;1,0>:hf  0.0:hf              {F@2}           //  ALU pipe: float; $218
        shl (1|M0)               r200.0<1>:d   r199.0<0;1,0>:d   1:w               {Compacted,I@1}   //  ALU pipe: int; $216
        store.ugm.d32x8t.a32.wb.wb (1|M0)  bti[2][r200:1] r201:1   {A@1,$3} // ex_desc:0x2000000; desc:0x620EC504 // $219
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<1;0>:w       r1.0<1;0>:w       r1.0<1>:w         //  ALU pipe: int; $220
(W)     csel (4|M0)   (eq)f0.0   r1.4<1>:f     r1.4<1;0>:f       r1.4<1;0>:f       r1.4<1>:f        {I@1} //  ALU pipe: float; $220
(W)     send.gtwy (1|M0)         null     r255    null:0  0x0            0x02000010           {EOT,A@1} // wr:1+0, rd:0; end of thread // $220
L3824:
        nop                                                                                          // $220


//.BankConflicts: 1
//.ByteRMWs: 0
//


//.numALUInst: 217
//.accSubDef: 63
//.accSubUse: 63
//.accSubCandidateDef: 63
//.accSubCandidateUse: 63
//
//
//.singlePipeAtOneDistNum: 27
//.allAtOneDistNum: 5
//.syncInstCount: 21
//.tokenReuseCount: 0
//.AfterWriteTokenDepCount: 35
//.AfterReadTokenDepCount: 1
