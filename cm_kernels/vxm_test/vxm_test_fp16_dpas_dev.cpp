#include <cm/cm.h>
#include <cm/cmtl.h>

#define HALF half
#define FLOAT float

#define SIZE_OF_FP16 2
#define SIZE_PER_DPAS_HF16 128  // DPAS works for half float matrix [8x16] [16x8]

// -----------------------------
#define ALPHA 1.000000 
#define BETA 0.000000 
// -----------------------------
#if 0
_GENX_ inline void myDPAS8(matrix_ref<HALF, 8, 16> matA,
                            matrix_ref<HALF, 8, 16> matB,
                            matrix_ref<FLOAT, 8, 8> result)
{
	result = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(result.format<FLOAT>(), matB.format<U32>(), matA.format<U32>());
}

extern "C" _GENX_MAIN_ void vxm_test_fp16(
	SurfaceIndex surface_input_a [[type("buffer_t")]],
	SurfaceIndex surface_input_b [[type("buffer_t")]],
	SurfaceIndex surface_output [[type("buffer_t")]]
)
{
	// use byte to calcute offset
	const uint32_t global_tid_x = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
	const uint32_t global_tid_y = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
	const uint32_t thread_id_2 = cm_group_id(2) * cm_local_size(2) + cm_local_id(2);
	const uint32_t base_offset_a =  global_tid_x * TILE_M * SIZE_K * SIZE_OF_FP16;
	const uint32_t base_offset_b =  global_tid_y * TILE_N * SIZE_OF_FP16;
	const uint32_t base_offset_output =  (global_tid_x * TILE_M * SIZE_N + global_tid_y * TILE_N) * SIZE_OF_FP16;
	//printf("%d,%d,%d\n", global_tid_x, global_tid_y, thread_id_2);

	// init TILE_A
	vector<HALF, SIZE_PER_DPAS_HF16 * (TILE_M / 8) * (TILE_K / 16) > readA = 0.0; 	// M=0..7,  K=0..15		// A tile: (8*TILE_M/8)M x 16K
	matrix_ref<HALF, TILE_M, TILE_K> readA_m = readA.format<HALF, TILE_M, TILE_K>();

	// init TILE_B
	vector<HALF, SIZE_PER_DPAS_HF16> readB = 0.0; 	// N=0..7,  K=0..15		//B tile: 16Kx8N
	matrix_ref<HALF, 8, 16> readB_m = readB.format<HALF, 8, 16>();

	//init the accumulators
	matrix<FLOAT, TILE_M, TILE_N> result1 = 0.0;  
	matrix_ref<FLOAT, TILE_M, TILE_N> result1ref = result1;
	
	for( int step = 0; step < SIZE_K; step += TILE_K)
	{
		const uint32_t step_base_offset_a = base_offset_a + step * SIZE_OF_FP16;
		const uint32_t step_base_offset_b = base_offset_b + (step / TILE_K) * SIZE_N * TILE_K * SIZE_OF_FP16;
		// TILE_B: read two lines, and ordered into DPAS required format
		matrix<HALF, TILE_K/2, TILE_N> rowX2_0 = 0.0;  
		matrix<HALF, TILE_K/2, TILE_N> rowX2_1 = 0.0;

		//cache elements in matrix B 
		#pragma unroll
		for(int row = 0; row < TILE_K/2; row++)
		{
			const uint32_t rowX2 = row * 2;
			const uint32_t read_offset_b = step_base_offset_b + (rowX2 * SIZE_N)* SIZE_OF_FP16;
			rowX2_0.select<1,1,TILE_N,1>(row,0).format<U32>() = cm_load<U32, TILE_N/2, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, read_offset_b);  
			rowX2_1.select<1,1,TILE_N,1>(row,0).format<U32>() = cm_load<U32, TILE_N/2, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, read_offset_b + SIZE_N* SIZE_OF_FP16);  
		}		


		#pragma unroll
		for(int m=0; m < TILE_M/8; m++)
		{
			//cache elements in matrix A
			#pragma unroll
			for(int row = 0; row < 8; row++)
			{
				const unsigned read_offset_a = step_base_offset_a + (row * SIZE_K) * SIZE_OF_FP16;
				// Read from inputs surfaces row M x 16K
				readA_m.select<1,1,TILE_K,1>(row + m * 8, 0).format<U32>() = cm_load<U32, TILE_K/2 , DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_a, read_offset_a + SIZE_K * (8 * m) * SIZE_OF_FP16);
			}	
			
			//calcute DPAS, small 8x16x8
			#pragma unroll	
			for(int n = 0; n < TILE_N/8; n++)  
			{
				#pragma unroll	
				for(int k = 0; k < TILE_K/16; k++)
				{
					#pragma unroll
					for (int row = 0; row < 8; row++)
					{		
						readB_m.select<1,1,8,2>(row, 0)= rowX2_0.select<1,1,8,1>(row + k*8, 8*n);
						readB_m.select<1,1,8,2>(row, 1)= rowX2_1.select<1,1,8,1>(row + k*8, 8*n);	
					}	
					// if (global_tid_x == 0 && global_tid_y == 0)
					// {
					// 	printf("rowX2_0:\n");
					// 	for (int i = 0; i < TILE_K/2; i++)
					// 	{
					// 		  printf(" row%d", i);
					// 		 for (int j = 0; j < TILE_N;j++)
					// 		{ 
					// 		    printf(" %f", rowX2_0(i , j));
					// 		}
					// 		printf("\n");\
					// 	}
					// 	printf("readB_m:\n");
					// 	for (int i = 0; i < 8; i++)
					// 	{
					// 		  printf(" row%d", i);
					// 		 for (int j = 0; j < 16;j++)
					// 		{ 
					// 		    printf(" %f", readB_m(i , j));
					// 		}
					// 		printf("\n");\
					// 	}
					// }
					// return;
					myDPAS8(readA_m.select<8,1,16,1>(m * 8, k*16),  readB_m, result1ref.select<8,1,8,1>(m * 8,n * 8));  
				}
			}

		}

		// if (global_tid_x == 11 && global_tid_y == 0)
		// {
		// 	for (int i = 0; i < TILE_M; i++)
    	// 	{
        // 		  printf(" row%d", i);
       	// 		 for (int j = 0; j < TILE_N;j++)
        // 		{ 
        // 		    printf(" %f", result1ref(i , j));
        // 		}
        // 		printf("\n");\
    	// 	}
		// }
		
		// if (global_tid_x == 11 && global_tid_y == 0)
		// {
		// 	for (int i = 0; i < 16; i++)
    	// 	{
        // 		  printf(" row%d", i);
       	// 		 for (int j = 0; j < TILE_K;j++)
        // 		{ 
        // 		    printf(" %f",readA(i *TILE_K+j));
		// 			// printf(" %f",readB(i *16+j));
        // 		}
        // 		printf("\n");\
    	// 	}
		// }

	}

	vector<HALF, TILE_N> result_hf16_CL1 = 0.0;
	result1 *= HALF(ALPHA);
	

	#pragma unroll
	for(int i = 0; i < TILE_M; i++)
	{
		const unsigned write_index = base_offset_output + i * SIZE_N * SIZE_OF_FP16;	
		result_hf16_CL1.select<TILE_N, 1>(0)  = result1ref.select<1, 1, TILE_N, 1>(i, 0);
		cm_store<U32, TILE_N/2, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, write_index, result_hf16_CL1.format<U32>());
	}

}

#else

#if 1
_GENX_ inline void myDPAS8(matrix_ref<HALF, 8, 16> matA,
                            matrix_ref<HALF, 8, 16> matB,
                            matrix_ref<FLOAT, 8, 8> result)
{
	result = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(result.format<FLOAT>(), matB.format<U32>(), matA.format<U32>());
}

extern "C" _GENX_MAIN_ void vxm_test_fp16(
	SurfaceIndex surface_input_a [[type("buffer_t")]],
	SurfaceIndex surface_input_b [[type("buffer_t")]],
	SurfaceIndex surface_output [[type("buffer_t")]]
)
{
	// use byte to calcute offset
	const uint32_t global_tid_x = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
	const uint32_t global_tid_y = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
	const uint32_t base_offset_a =  0;
	const uint32_t base_offset_b =  global_tid_y * TILE_N * SIZE_OF_FP16;
	const uint32_t base_offset_output =  global_tid_y * TILE_N * SIZE_OF_FP16;
	//printf("%d,%d,%d\n", global_tid_x, global_tid_y, thread_id_2);
    
	// init TILE_A
	matrix<HALF, 1, TILE_K> row_a = 0.0;  

	vector<HALF, SIZE_PER_DPAS_HF16 * (TILE_N / 8) * (TILE_K / 16) > readA = 0.0; 	// M=0..7,  K=0..15		// A tile: (8*TILE_M/8)M x 16K
	matrix_ref<HALF, TILE_N, TILE_K> readA_m = readA.format<HALF, TILE_N, TILE_K>();

	// init TILE_B
	vector<HALF, SIZE_PER_DPAS_HF16> readB = 0.0; 	// N=0..7,  K=0..15		//B tile: 16Kx8N
	vector_ref<uint32_t, SIZE_PER_DPAS_HF16/2> input_b_packed = readB.format<uint32_t>();  // reinterprete to uint32
	matrix_ref<HALF, 8, 16> readB_m = readB.format<HALF, 8, 16>();

	//init the accumulators
	matrix<FLOAT, TILE_N, TILE_M> result1 = 0.0;  
	matrix_ref<FLOAT, TILE_N, TILE_M> result1ref = result1;
	
	for( int step = 0; step < SIZE_K/TILE_K; step ++ )
	{
		const uint32_t step_base_offset_a = base_offset_a + step * TILE_K * SIZE_OF_FP16;
		const uint32_t step_base_offset_b = base_offset_b + step * TILE_K * SIZE_N * SIZE_OF_FP16;

		//Load from input buffer A, used for DPAS A, 8X16, actuall only 8x1
		const uint32_t read_offset_a = step_base_offset_a;
		row_a.select<1,1,TILE_K,1>(0,step*TILE_K).format<U32>() = cm_load<U32, TILE_K/2, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_a, step_base_offset_a);  
		
		
		// if (step == 1 && global_tid_x == 0 && global_tid_y == 0)
		// {
		// 	int i = 0;
		// 	printf(" row%d", i);
		// 	for (int j = 0; j < TILE_K;j++)
		// 	{ 
		// 		printf(" %f", row_a(i , j));
		// 	}
		// 	printf("\n");\
		// return;
		// }

	
		// Load from input buffer B, used for DPAS A, 16x8
		// TODO: optimized for 2D transposed loading to save cycles
		for(int row = 0; row < TILE_K; row++)
		{
			const unsigned read_offset_b = step_base_offset_b + (row * SIZE_N) * SIZE_OF_FP16;
			readA_m.column(row).format<U32>() = cm_load<U32, TILE_N/2 , DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, read_offset_b);
		}	

		// if (step == 1 && global_tid_x == 0 && global_tid_y == 0)
		// {
		// 	printf("readA_m:\n");
		// 	for (int i = 0; i < TILE_N; i++)
		// 	{
		// 			printf(" row%d", i);
		// 			for (int j = 0; j < TILE_K;j++)
		// 		{ 
		// 			printf(" %d", (int)readA_m(i , j));
		// 		}
		// 		printf("\n");\
		// 	}
		// 	return;
		// }

		//calcute DPAS, small 8x16x8
		#pragma unroll	
		for(int n = 0; n < TILE_N/8; n++)  
		{
			#pragma unroll	
			for(int k = 0; k < TILE_K/16; k++)
			{
				readB_m.select<8,1,2,1>(0, 0)= row_a.select<1,1,16,1>(0, k*16);
			
				// if (global_tid_x == 0 && global_tid_y == 0)
				// {
				// 	printf("rowX2_0:\n");
				// 	for (int i = 0; i < TILE_K/2; i++)
				// 	{
				// 		  printf(" row%d", i);
				// 		 for (int j = 0; j < TILE_N;j++)
				// 		{ 
				// 		    printf(" %f", rowX2_0(i , j));
				// 		}
				// 		printf("\n");\
				// 	}
				// 	printf("readB_m:\n");
				// 	for (int i = 0; i < 8; i++)
				// 	{
				// 		  printf(" row%d", i);
				// 		 for (int j = 0; j < 16;j++)
				// 		{ 
				// 		    printf(" %f", readB_m(i , j));
				// 		}
				// 		printf("\n");\
				// 	}
				// }
				// return;

				// TODO: optimized repeat count for dpas 8x1 to save cycles
				myDPAS8(readA_m.select<8,1,16,1>(n * 8, k*16),  readB_m, result1ref.select<8,1,8,1>(n * 8, 0));  
			}
		}
	}

	// if (global_tid_x == 0 && global_tid_y == 1)
	// {
	// 	printf("result1ref:\n");
	// 	for (int i = 0; i < TILE_N; i++)
	// 	{
	// 			printf(" row%d", i);
	// 			for (int j = 0; j < TILE_M;j++)
	// 		{ 
	// 			printf(" %f", result1ref(i , j));
	// 		}
	// 		printf("\n");\
	// 	}
	// }

	vector<HALF, TILE_N> result_hf16_CL1 = 0.0;
	result1 *= HALF(ALPHA);
	


	result_hf16_CL1.select<TILE_N, 1>(0)  = result1ref.select<TILE_N, 1, 1, 1>(0, 0);
	cm_store<U32, TILE_N/2, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, base_offset_output, result_hf16_CL1.format<U32>());
	

}

#else  // for success backup
_GENX_ inline void myDPAS8(matrix_ref<HALF, 8, 16> matA,
                            matrix_ref<HALF, 8, 16> matB,
                            matrix_ref<FLOAT, 8, 8> result)
{
	result = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(result.format<FLOAT>(), matB.format<U32>(), matA.format<U32>());
}

extern "C" _GENX_MAIN_ void gemm_nchw_dpas(
	SurfaceIndex surface_input_a [[type("buffer_t")]],
	SurfaceIndex surface_input_b [[type("buffer_t")]],
	SurfaceIndex surface_output [[type("buffer_t")]]
)
{
	// use byte to calcute offset
	const uint32_t global_tid_x = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
	const uint32_t global_tid_y = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
	const uint32_t base_offset_a =  0;
	const uint32_t base_offset_b =  global_tid_y * TILE_N * SIZE_OF_FP16;
	const uint32_t base_offset_output =  global_tid_y * TILE_N * SIZE_OF_FP16;
	//printf("%d,%d,%d\n", global_tid_x, global_tid_y, thread_id_2);
    
	// init TILE_A
	matrix<HALF, 1, TILE_K> row_a = 0.0;  

	vector<HALF, SIZE_PER_DPAS_HF16 * (TILE_N / 8) * (TILE_K / 16) > readA = 0.0; 	// M=0..7,  K=0..15		// A tile: (8*TILE_M/8)M x 16K
	matrix_ref<HALF, TILE_N, TILE_K> readA_m = readA.format<HALF, TILE_N, TILE_K>();

	// init TILE_B
	vector<HALF, SIZE_PER_DPAS_HF16> readB = 0.0; 	// N=0..7,  K=0..15		//B tile: 16Kx8N
	vector_ref<uint32_t, SIZE_PER_DPAS_HF16/2> input_b_packed = readB.format<uint32_t>();  // reinterprete to uint32
	matrix_ref<HALF, 8, 16> readB_m = readB.format<HALF, 8, 16>();

	//init the accumulators
	matrix<FLOAT, TILE_N, TILE_M> result1 = 0.0;  
	matrix_ref<FLOAT, TILE_N, TILE_M> result1ref = result1;
	
	for( int step = 0; step < SIZE_K/TILE_K; step ++ )
	{
		const uint32_t step_base_offset_a = base_offset_a + step * TILE_K * SIZE_OF_FP16;
		const uint32_t step_base_offset_b = base_offset_b + step * TILE_K * SIZE_N * SIZE_OF_FP16;

		//Load from input buffer A, used for DPAS A, 8X16, actuall only 8x1
		const uint32_t read_offset_a = step_base_offset_a;
		row_a.select<1,1,TILE_K,1>(0,step*TILE_K).format<U32>() = cm_load<U32, TILE_K/2, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_a, step_base_offset_a);  
		
		
		// if (step == 1 && global_tid_x == 0 && global_tid_y == 0)
		// {
		// 	int i = 0;
		// 	printf(" row%d", i);
		// 	for (int j = 0; j < TILE_K;j++)
		// 	{ 
		// 		printf(" %f", row_a(i , j));
		// 	}
		// 	printf("\n");\
		// return;
		// }

	
		// Load from input buffer B, used for DPAS A, 16x8
		// TODO: optimized for 2D transposed loading to save cycles
		
		for(int row = 0; row < TILE_K; row++)
		{
			const unsigned read_offset_b = step_base_offset_b + (row * SIZE_N) * SIZE_OF_FP16;
			readA_m.select<TILE_N,1,1,1>(0, row).format<U32>() = cm_load<U32, TILE_N/2 , DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, read_offset_b);
		}	

		// if (step == 1 && global_tid_x == 0 && global_tid_y == 0)
		// {
		// 	printf("readA_m:\n");
		// 	for (int i = 0; i < TILE_N; i++)
		// 	{
		// 			printf(" row%d", i);
		// 			for (int j = 0; j < TILE_K;j++)
		// 		{ 
		// 			printf(" %d", (int)readA_m(i , j));
		// 		}
		// 		printf("\n");\
		// 	}
		// 	return;
		// }

		//calcute DPAS, small 8x16x8
		#pragma unroll	
		for(int n = 0; n < TILE_N/8; n++)  
		{
			#pragma unroll	
			for(int k = 0; k < TILE_K/16; k++)
			{
				readB_m.select<8,1,2,1>(0, 0)= row_a.select<1,1,16,1>(0, k*16);

				// if (global_tid_x == 0 && global_tid_y == 0)
				// {
				// 	printf("rowX2_0:\n");
				// 	for (int i = 0; i < TILE_K/2; i++)
				// 	{
				// 		  printf(" row%d", i);
				// 		 for (int j = 0; j < TILE_N;j++)
				// 		{ 
				// 		    printf(" %f", rowX2_0(i , j));
				// 		}
				// 		printf("\n");\
				// 	}
				// 	printf("readB_m:\n");
				// 	for (int i = 0; i < 8; i++)
				// 	{
				// 		  printf(" row%d", i);
				// 		 for (int j = 0; j < 16;j++)
				// 		{ 
				// 		    printf(" %f", readB_m(i , j));
				// 		}
				// 		printf("\n");\
				// 	}
				// }
				// return;

				// TODO: optimized repeat count for dpas 8x1 to save cycles
				myDPAS8(readA_m.select<8,1,16,1>(n * 8, k*16),  readB_m, result1ref.select<8,1,8,1>(n * 8, 0));  
			}
		}
	}

	// if (global_tid_x == 0 && global_tid_y == 1)
	// {
	// 	printf("result1ref:\n");
	// 	for (int i = 0; i < TILE_N; i++)
	// 	{
	// 			printf(" row%d", i);
	// 			for (int j = 0; j < TILE_M;j++)
	// 		{ 
	// 			printf(" %f", result1ref(i , j));
	// 		}
	// 		printf("\n");\
	// 	}
	// }

	vector<HALF, TILE_N> result_hf16_CL1 = 0.0;
	result1 *= HALF(ALPHA);
	


	result_hf16_CL1.select<TILE_N, 1>(0)  = result1ref.select<TILE_N, 1, 1, 1>(0, 0);
	cm_store<U32, TILE_N/2, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, base_offset_output, result_hf16_CL1.format<U32>());
	

}
#endif

#endif