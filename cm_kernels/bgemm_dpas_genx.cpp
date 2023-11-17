#define GEN_KERNEL

#undef CM_DEBUG
#include <cm/cm.h>
#include "share.h"

#define LOGe_BASE2 1.4426950408889394
#define HALF half
#define FLOAT float
#define GROUP_SIZE_IN_THREADS 16
#define DIM_X 0
#define DIM_Y 1

#if defined(RELU_PACKMASK) || defined(BWD_RELU_UNPACKMASK)
#define TGRP_H      (WG_TILE_M/SG_TILE_M)   // Number of threads in a group height
#define TGRP_W      (WG_TILE_N/SG_TILE_N)   // Number of threads in a group width
#define MASK_TW     8                       // DWords in width per 32x32
#define MASK_TH     4                       // Height of mask per 32x32
#define MASK_GW     (MASK_TW * TGRP_W)      // DWords in width per 128x128 group
#define MASK_GH     (MASK_TH * TGRP_H)      // Height of mask per 128x128 group
#define MASK_T_SZ   (MASK_TW * MASK_TH)     // Mask Dwords per 32x32
#define MASK_G_SZ   (MASK_GW * MASK_GH)     // Mask DWords per 128x128
#endif

#define SPLIT_BARRIER_SIGNAL 1
#define SPLIT_BARRIER_WAIT 0

const   short  init_offsetDW[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
const   int    init_offsetReadOffset[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
const   int    init_offsetReadOffset_2[4] = { 0, 1, 2, 3 };
const int init_prefetchOffset[8] = { 0,0,0,0,0,0,0,0 };

_GENX_ inline void myDPAS(matrix_ref<HALF, 4, 16> matA,
                          matrix_ref<HALF, 8, 16> matB,
                          matrix_ref<FLOAT, 8, 8> result) {
  result = cm_dpasw<CM_PRECISION_BF, CM_PRECISION_BF, 8, 8>(
      result.format<FLOAT>(), matB.format<U32>(), matA.format<U32>());
}

// A matrix format - [K/16][M][16K]
// B matrix format - [K/16][N/8][8K][8N][2K]
// C matrix format - [N/16][M][16N]
extern "C" _GENX_MAIN_ void
PVCBGEMM(SurfaceIndex INMTXa[[type("buffer_t bfloat")]], //
         SurfaceIndex INMTXb[[type("buffer_t bfloat")]], //
         SurfaceIndex OUTMTX[[type("buffer_t bfloat")]], //
         int M,                                          //
         int K,                                          //
         int N,                                          //
         int repeat_count,                               //
         int MatAReadIncSizeByte,                        //
         int MatBReadIncSizeByte,                        //
         int StepSizeForSecondReadByte,                  //
         int groupHeight) {
    matrix<FLOAT, 8, 8> result11; //first number indicates row of 8x8 inside 32x32 output tile. Second number is column of 8x8 inside 32x32.
    matrix<FLOAT, 8, 8> result12; //since there are 16 8x8 tiles inside 32x32, 16 sets of these accum are required.
    matrix<FLOAT, 8, 8> result13;
    matrix<FLOAT, 8, 8> result14;
    matrix<FLOAT, 8, 8> result21;
    matrix<FLOAT, 8, 8> result22;
    matrix<FLOAT, 8, 8> result23;
    matrix<FLOAT, 8, 8> result24;
    matrix<FLOAT, 8, 8> result31;
    matrix<FLOAT, 8, 8> result32;
    matrix<FLOAT, 8, 8> result33;
    matrix<FLOAT, 8, 8> result34;
    matrix<FLOAT, 8, 8> result41;
    matrix<FLOAT, 8, 8> result42;
    matrix<FLOAT, 8, 8> result43;
    matrix<FLOAT, 8, 8> result44;

    vector<uint, 16> offset(init_offsetDW);

    matrix_ref<FLOAT, 8, 8> result11ref = result11;
    matrix_ref<FLOAT, 8, 8> result12ref = result12;
    matrix_ref<FLOAT, 8, 8> result13ref = result13;
    matrix_ref<FLOAT, 8, 8> result14ref = result14;
    matrix_ref<FLOAT, 8, 8> result21ref = result21;
    matrix_ref<FLOAT, 8, 8> result22ref = result22;
    matrix_ref<FLOAT, 8, 8> result23ref = result23;
    matrix_ref<FLOAT, 8, 8> result24ref = result24;
    matrix_ref<FLOAT, 8, 8> result31ref = result31;
    matrix_ref<FLOAT, 8, 8> result32ref = result32;
    matrix_ref<FLOAT, 8, 8> result33ref = result33;
    matrix_ref<FLOAT, 8, 8> result34ref = result34;
    matrix_ref<FLOAT, 8, 8> result41ref = result41;
    matrix_ref<FLOAT, 8, 8> result42ref = result42;
    matrix_ref<FLOAT, 8, 8> result43ref = result43;
    matrix_ref<FLOAT, 8, 8> result44ref = result44;

    // B tile registers - 32Nx16K tile - [K/16][N/8][8K][8N][2K]
    vector<HALF, 128> readB1;
    vector<HALF, 128> readB2;
    vector<HALF, 128> readB3;
    vector<HALF, 128> readB4;

    matrix_ref<HALF, 8, 16> readB1_m = readB1.format<HALF, 8, 16>(); //N=0..7,K=0..15
    matrix_ref<HALF, 8, 16> readB2_m = readB2.format<HALF, 8, 16>(); //N=8..15,K=0..15
    matrix_ref<HALF, 8, 16> readB3_m = readB3.format<HALF, 8, 16>();
    matrix_ref<HALF, 8, 16> readB4_m = readB4.format<HALF, 8, 16>(); //N=24..31,K=0..15

    //A tile registers - [K/16][M][16K]
    vector<HALF, 64> readA1;
    vector<HALF, 64> readA2;
    vector<HALF, 64> readA3;
    vector<HALF, 64> readA4;

    matrix_ref<HALF, 4, 16> readA1_m = readA1.format<HALF, 4, 16>();//M=0..3,K=0..15
    matrix_ref<HALF, 4, 16> readA2_m = readA2.format<HALF, 4, 16>();//M=8..11,K=0..15
    matrix_ref<HALF, 4, 16> readA3_m = readA3.format<HALF, 4, 16>();//M=16..19,K=0..15
    matrix_ref<HALF, 4, 16> readA4_m = readA4.format<HALF, 4, 16>();//M=24..27,K=0..15


    uint gidY = cm_group_id(DIM_Y);
    uint gidX = cm_group_id(DIM_X);
    uint tidX = cm_local_id(DIM_X);
    uint tidY = cm_local_id(DIM_Y);
    uint linear_thread_id = (tidY * 4) + tidX;          // 0,1,2,3....14,15
    uint fused_thread_id = tidX % 2;                    // 0,1,0,1....0,1
    uint fused_thread_group_id = linear_thread_id >> 1; // 0,0,1,1,...7,7

    vector<int, 4> MatAReadIndexOffsetByte(init_offsetReadOffset_2);

    MatAReadIndexOffsetByte = MatAReadIndexOffsetByte * StepSizeForSecondReadByte * 2 +
        StepSizeForSecondReadByte * fused_thread_id +
        gidY * WG_TILE_M * CONTIGUOUS_K_SIZE * SIZE_OF_BF16_BYTE +
        tidY * SG_TILE_M * CONTIGUOUS_K_SIZE * SIZE_OF_BF16_BYTE;

    vector<int, 8> MatBReadIndexOffsetByte(init_offsetReadOffset);
    MatBReadIndexOffsetByte = MatBReadIndexOffsetByte * StepSizeForSecondReadByte +
        gidX * WG_TILE_N * CONTIGUOUS_K_SIZE * SIZE_OF_BF16_BYTE +
        tidX * SG_TILE_N * CONTIGUOUS_K_SIZE * SIZE_OF_BF16_BYTE;

    //Write surface address
    vector<uint, 8> WriteIndex;
    uint MatCWriteStepVert = 4 * 16 * sizeof(ushort);
    uint MatCWriteStepHorz = 16 * M * SIZE_OF_BF16_BYTE;
    uint MatCGroupWriteIndexByte = (gidX * WG_TILE_N + tidX * SG_TILE_N) * M * SIZE_OF_BF16_BYTE
        + (gidY * WG_TILE_M + tidY * SG_TILE_M) * CONTIGUOUS_K_SIZE * SIZE_OF_BF16_BYTE;
    uint WriteIndexIncrement = (groupHeight * WG_TILE_M) * CONTIGUOUS_K_SIZE * SIZE_OF_BF16_BYTE - MatCWriteStepHorz;

    WriteIndex = offset.select<8, 1>(0) * MatCWriteStepVert + MatCGroupWriteIndexByte;

    for (int i = 0; i < repeat_count;
         i++) // this is done so that there is only single kernel wave.
              // Repeat across the group_y direction
    {
      // init the accumulators
      result11 = 0.0;
      result12 = 0.0;
      result13 = 0.0;
      result14 = 0.0;
      result21 = 0.0;
      result22 = 0.0;
      result23 = 0.0;
      result24 = 0.0;
      result31 = 0.0;
      result32 = 0.0;
      result33 = 0.0;
      result34 = 0.0;
      result41 = 0.0;
      result42 = 0.0;
      result43 = 0.0;
      result44 = 0.0;

      // iterates to process the entire K for A and B.
      for (int j = 0; j < (K >> 4); j += 1) {
        // Read 32x16 B tile
        read(DWALIGNED(INMTXb), MatBReadIndexOffsetByte(0),
             readB1.select<64, 1>(0));
        read(DWALIGNED(INMTXb), MatBReadIndexOffsetByte(1),
             readB1.select<64, 1>(64));
        read(DWALIGNED(INMTXb), MatBReadIndexOffsetByte(2),
             readB2.select<64, 1>(0));
        read(DWALIGNED(INMTXb), MatBReadIndexOffsetByte(3),
             readB2.select<64, 1>(64));
        read(DWALIGNED(INMTXb), MatBReadIndexOffsetByte(4),
             readB3.select<64, 1>(0));
        read(DWALIGNED(INMTXb), MatBReadIndexOffsetByte(5),
             readB3.select<64, 1>(64));
        read(DWALIGNED(INMTXb), MatBReadIndexOffsetByte(6),
             readB4.select<64, 1>(0));
        read(DWALIGNED(INMTXb), MatBReadIndexOffsetByte(7),
             readB4.select<64, 1>(64));
        // Read A tile
        read(DWALIGNED(INMTXa), MatAReadIndexOffsetByte(0),
             readA1.select<64, 1>(0)); // 4Mx16K , M=0..3
        read(DWALIGNED(INMTXa), MatAReadIndexOffsetByte(1),
             readA2.select<64, 1>(0)); // M=8..11
        read(DWALIGNED(INMTXa), MatAReadIndexOffsetByte(2),
             readA3.select<64, 1>(0)); // M=16..19
        read(DWALIGNED(INMTXa), MatAReadIndexOffsetByte(3),
             readA4.select<64, 1>(0)); // M=24..27

        myDPAS(readA1_m, readB1_m, result11ref);
        myDPAS(readA2_m, readB1_m, result21ref);
        myDPAS(readA3_m, readB1_m, result31ref);
        myDPAS(readA4_m, readB1_m, result41ref);
        myDPAS(readA1_m, readB2_m, result12ref);
        myDPAS(readA2_m, readB2_m, result22ref);
        myDPAS(readA3_m, readB2_m, result32ref);
        myDPAS(readA4_m, readB2_m, result42ref);
        myDPAS(readA1_m, readB3_m, result13ref);
        myDPAS(readA2_m, readB3_m, result23ref);
        myDPAS(readA3_m, readB3_m, result33ref);
        myDPAS(readA4_m, readB3_m, result43ref);
        myDPAS(readA1_m, readB4_m, result14ref);
        myDPAS(readA2_m, readB4_m, result24ref);
        myDPAS(readA3_m, readB4_m, result34ref);
        myDPAS(readA4_m, readB4_m, result44ref);

        MatAReadIndexOffsetByte += MatAReadIncSizeByte;
        MatBReadIndexOffsetByte += MatBReadIncSizeByte;
      }

      // convert the fp32 to bf16 before writing out the results.
      {

        matrix_ref<ushort, 8, 16> result11ref_ushort =
            result11.format<ushort, 8, 16>();
        matrix_ref<ushort, 8, 16> result12ref_ushort =
            result12.format<ushort, 8, 16>();
        matrix_ref<ushort, 8, 16> result13ref_ushort =
            result13.format<ushort, 8, 16>();
        matrix_ref<ushort, 8, 16> result14ref_ushort =
            result14.format<ushort, 8, 16>();
        matrix_ref<ushort, 8, 16> result21ref_ushort =
            result21.format<ushort, 8, 16>();
        matrix_ref<ushort, 8, 16> result22ref_ushort =
            result22.format<ushort, 8, 16>();
        matrix_ref<ushort, 8, 16> result23ref_ushort =
            result23.format<ushort, 8, 16>();
        matrix_ref<ushort, 8, 16> result24ref_ushort =
            result24.format<ushort, 8, 16>();
        matrix_ref<ushort, 8, 16> result31ref_ushort =
            result31.format<ushort, 8, 16>();
        matrix_ref<ushort, 8, 16> result32ref_ushort =
            result32.format<ushort, 8, 16>();
        matrix_ref<ushort, 8, 16> result33ref_ushort =
            result33.format<ushort, 8, 16>();
        matrix_ref<ushort, 8, 16> result34ref_ushort =
            result34.format<ushort, 8, 16>();
        matrix_ref<ushort, 8, 16> result41ref_ushort =
            result41.format<ushort, 8, 16>();
        matrix_ref<ushort, 8, 16> result42ref_ushort =
            result42.format<ushort, 8, 16>();
        matrix_ref<ushort, 8, 16> result43ref_ushort =
            result43.format<ushort, 8, 16>();
        matrix_ref<ushort, 8, 16> result44ref_ushort =
            result44.format<ushort, 8, 16>();

        matrix<short, 8, 16> result_bf16_out1;
        matrix<short, 8, 16> result_bf16_out2;
        matrix<short, 8, 16> result_bf16_out3;
        matrix<short, 8, 16> result_bf16_out4;
        matrix<short, 8, 16> result_bf16_out5;
        matrix<short, 8, 16> result_bf16_out6;
        matrix<short, 8, 16> result_bf16_out7;
        matrix<short, 8, 16> result_bf16_out8;

        result_bf16_out1.select<8, 1, 8, 1>(0, 0) =
            result11ref_ushort.select<8, 1, 8, 2>(0, 1);
        result_bf16_out1.select<8, 1, 8, 1>(0, 8) =
            result12ref_ushort.select<8, 1, 8, 2>(0, 1);
        result_bf16_out2.select<8, 1, 8, 1>(0, 0) =
            result21ref_ushort.select<8, 1, 8, 2>(0, 1);
        result_bf16_out2.select<8, 1, 8, 1>(0, 8) =
            result22ref_ushort.select<8, 1, 8, 2>(0, 1);
        result_bf16_out3.select<8, 1, 8, 1>(0, 0) =
            result31ref_ushort.select<8, 1, 8, 2>(0, 1);
        result_bf16_out3.select<8, 1, 8, 1>(0, 8) =
            result32ref_ushort.select<8, 1, 8, 2>(0, 1);
        result_bf16_out4.select<8, 1, 8, 1>(0, 0) =
            result41ref_ushort.select<8, 1, 8, 2>(0, 1);
        result_bf16_out4.select<8, 1, 8, 1>(0, 8) =
            result42ref_ushort.select<8, 1, 8, 2>(0, 1);
        result_bf16_out5.select<8, 1, 8, 1>(0, 0) =
            result13ref_ushort.select<8, 1, 8, 2>(0, 1);
        result_bf16_out5.select<8, 1, 8, 1>(0, 8) =
            result14ref_ushort.select<8, 1, 8, 2>(0, 1);
        result_bf16_out6.select<8, 1, 8, 1>(0, 0) =
            result23ref_ushort.select<8, 1, 8, 2>(0, 1);
        result_bf16_out6.select<8, 1, 8, 1>(0, 8) =
            result24ref_ushort.select<8, 1, 8, 2>(0, 1);
        result_bf16_out7.select<8, 1, 8, 1>(0, 0) =
            result33ref_ushort.select<8, 1, 8, 2>(0, 1);
        result_bf16_out7.select<8, 1, 8, 1>(0, 8) =
            result34ref_ushort.select<8, 1, 8, 2>(0, 1);
        result_bf16_out8.select<8, 1, 8, 1>(0, 0) =
            result43ref_ushort.select<8, 1, 8, 2>(0, 1);
        result_bf16_out8.select<8, 1, 8, 1>(0, 8) =
            result44ref_ushort.select<8, 1, 8, 2>(0, 1);

        // write the results
        // output C format: [N/16][M][16N]. Following is the order of data
        // (row=0,col=0..15)(row=2,col=0..15)(row=3,col=0..15). row and
        // colume maps to output C[M][N]
        write(OUTMTX, WriteIndex(0),
              result_bf16_out1.select<4, 1, 16, 1>(0, 0).format<ushort>());
        write(OUTMTX, WriteIndex(1),
              result_bf16_out1.select<4, 1, 16, 1>(4, 0).format<ushort>());
        write(OUTMTX, WriteIndex(2),
              result_bf16_out2.select<4, 1, 16, 1>(0, 0).format<ushort>());
        write(OUTMTX, WriteIndex(3),
              result_bf16_out2.select<4, 1, 16, 1>(4, 0).format<ushort>());
        write(OUTMTX, WriteIndex(4),
              result_bf16_out3.select<4, 1, 16, 1>(0, 0).format<ushort>());
        write(OUTMTX, WriteIndex(5),
              result_bf16_out3.select<4, 1, 16, 1>(4, 0).format<ushort>());
        write(OUTMTX, WriteIndex(6),
              result_bf16_out4.select<4, 1, 16, 1>(0, 0).format<ushort>());
        write(OUTMTX, WriteIndex(7),
              result_bf16_out4.select<4, 1, 16, 1>(4, 0).format<ushort>());
        WriteIndex += MatCWriteStepHorz;
        write(OUTMTX, WriteIndex(0),
              result_bf16_out5.select<4, 1, 16, 1>(0, 0).format<ushort>());
        write(OUTMTX, WriteIndex(1),
              result_bf16_out5.select<4, 1, 16, 1>(4, 0).format<ushort>());
        write(OUTMTX, WriteIndex(2),
              result_bf16_out6.select<4, 1, 16, 1>(0, 0).format<ushort>());
        write(OUTMTX, WriteIndex(3),
              result_bf16_out6.select<4, 1, 16, 1>(4, 0).format<ushort>());
        write(OUTMTX, WriteIndex(4),
              result_bf16_out7.select<4, 1, 16, 1>(0, 0).format<ushort>());
        write(OUTMTX, WriteIndex(5),
              result_bf16_out7.select<4, 1, 16, 1>(4, 0).format<ushort>());
        write(OUTMTX, WriteIndex(6),
              result_bf16_out8.select<4, 1, 16, 1>(0, 0).format<ushort>());
        write(OUTMTX, WriteIndex(7),
              result_bf16_out8.select<4, 1, 16, 1>(4, 0).format<ushort>());
      } // Write logic

      // increment the groupY and continue recycling the thread until all
      // the data are processed.
      MatAReadIndexOffsetByte +=
          -(MatAReadIncSizeByte * (K >> 4)) +
          groupHeight * WG_TILE_M * CONTIGUOUS_K_SIZE * SIZE_OF_BF16_BYTE;
      MatBReadIndexOffsetByte += -(MatBReadIncSizeByte * (K >> 4));
      gidY += groupHeight;
      WriteIndex += WriteIndexIncrement;

    } // Repeat loop
}

