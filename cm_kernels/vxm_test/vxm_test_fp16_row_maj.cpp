#include <cm/cm.h>
#include <cm/cmtl.h>

// #define SIZE_M 2048
// #define SIZE_K 1280 
// #define SIZE_N 1280 
// #define ALPHA 1.000000 
// #define BETA 0.000000 
// #define TILE_M 16 
// #define TILE_K 16 
// #define TILE_N 64 
// #define LWS_SIZE_X 1 
// #define LWS_SIZE_Y 1 
// #define LWS_SIZE_Z 1 
// #define GWS_SIZE_X 128 
// #define GWS_SIZE_Y 20 
// #define GWS_SIZE_Z 1 
// #define ACCU_IS_FP32 0 
// #define CM_BINDLESS 1 


#define ACCU_IS_FP32 1
#define DT half 

#if ACCU_IS_FP32
#define DT_ACCU float
#else
#define DT_ACCU DT
#endif

#define ALPHA 1.000000 
#define BETA 0.000000 

#if 1

extern "C" _GENX_MAIN_ void vxm_test_fp16_row_maj(
	SurfaceIndex surface_input_a [[type("buffer_t" )]],
	SurfaceIndex surface_input_b [[type("buffer_t" )]],
	SurfaceIndex surface_output  [[type("buffer_t" )]]
)
{
    const uint32_t thread_id_0 = cm_group_id(0) * cm_local_size(0) + cm_local_id(0); // 0
    const uint32_t thread_id_1 = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
	// printf("thread_id_0: %d, thread_id_1: %d \n", thread_id_0, thread_id_1);
    const uint32_t input_a_load_size = (TILE_K * sizeof(DT)) / sizeof(uint32_t);
    const uint32_t input_b_load_size = (TILE_N * sizeof(DT)) / sizeof(uint32_t);
        
    const uint32_t input_a_base_offset =0;
    
	vector<DT_ACCU, TILE_N> accu(0.0f);
    uint32_t input_b_localbase = thread_id_1 * TILE_N * sizeof(DT);
	
    vector<DT, TILE_K> input_a(0.0f);   
    vector_ref<uint32_t, TILE_K/2> input_a_packed = input_a.format<uint32_t>();  // reinterprete to uint32
    
	    
	matrix<DT, TILE_K, TILE_N> input_b(0.0f);   
    matrix_ref<uint32_t, TILE_K, TILE_N/2> input_b_packed = input_b.format<uint32_t, TILE_K, TILE_N/2>();  // reinterprete to uint32
    

	// #pragma unroll
	for(uint32_t i = 0; i < SIZE_K / TILE_K; i++) // outter reduced Tile in K axis 
	// for(uint32_t i = 0; i < 65; i++) // outter reduced Tile in K axis 
	{

		// Load A
		const uint32_t input_a_offset = input_a_base_offset + (i * TILE_K) * sizeof(DT);
		input_a_packed = cm_load<uint32_t, input_a_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_a, input_a_offset);

		
		for(int k = 0; k < TILE_K; k++)
		{
			uint32_t input_b_offset = input_b_localbase + (i*TILE_K+k) * SIZE_N * sizeof(DT);
			cm_prefetch<input_b_load_size, DataSize::U64, CacheHint::Cached, CacheHint::Cached>(surface_input_b, input_b_offset);
			input_b_packed.row(k) = cm_load<uint32_t, input_b_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, input_b_offset);  
		}
		
		// if (i == 0 && thread_id_0 == 0 && thread_id_1 == 0)
		// {
		// 	for (int i = 0; i < TILE_K; i++)
    	// 	{
        // 		  printf(" row%d", i);
       	// 		 for (int j = 0; j < TILE_N;j++)
        // 		{ 
        // 		    printf(" %f", input_b(i , j));
        // 		}
        // 		printf("\n");\
    	// 	}
		// 	return;
		// }

		// #pragma unroll
		for(uint32_t k = 0; k < TILE_K; k++) // reduce for inner product
		// for(uint32_t k = 0; k < 32; k++) // reduce for inner product
		{
			// // Acc
#if ACCU_IS_FP32
				vector<DT_ACCU, TILE_N> input_b_fp32 = vector<DT_ACCU, TILE_N>(input_b.row(k));
				vector<DT_ACCU, TILE_N> input_a_fp32 = vector<DT_ACCU, TILE_N>(input_a.select<1, 1>(k).replicate<TILE_N>());
				accu.select<TILE_N, 1>(0) += input_b_fp32 * input_a_fp32;
#else	
			accu.select<TILE_N, 1>(0) += input_b * input_a.select<1, 1>(k).replicate<TILE_N>();
#endif
		}

    }

    const uint32_t output_store_size = (TILE_N * sizeof(DT)) / sizeof(uint32_t);
    uint32_t output_offset = (thread_id_0  * SIZE_N + thread_id_1 * TILE_N) * sizeof(DT);
		
	vector<DT, TILE_N> accu_out = accu;  // if DT_ACCU == DT then compiler removes this line
	accu_out *= DT(ALPHA);
	accu_out += DT(BETA);
	
	// Store C
	vector_ref<uint32_t, output_store_size> accu_0_packed = accu_out.select<TILE_N, 1>(0).format<uint32_t>();
	cm_store<uint32_t, output_store_size, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, output_offset, accu_0_packed);

}

#else
extern "C" _GENX_MAIN_ void vxm_test_fp16_row_maj(
	SurfaceIndex surface_input_a [[type("buffer_t" )]],
	SurfaceIndex surface_input_b [[type("buffer_t" )]],
	SurfaceIndex surface_output  [[type("buffer_t" )]]
)
{
    const uint32_t thread_id_0 = cm_group_id(0) * cm_local_size(0) + cm_local_id(0); // 0
    const uint32_t thread_id_1 = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
	// printf("thread_id_0: %d, thread_id_1: %d \n", thread_id_0, thread_id_1);
    const uint32_t input_a_load_size = (TILE_K * sizeof(DT)) / sizeof(uint32_t);
    const uint32_t input_b_load_size = (TILE_N * sizeof(DT)) / sizeof(uint32_t);
        
    const uint32_t input_a_base_offset =0;
    
	vector<DT_ACCU, TILE_N> accu(0.0f);
    uint32_t input_b_localbase = thread_id_1 * TILE_N * sizeof(DT);
	
    vector<DT, TILE_K> input_a(0.0f);   
    vector_ref<uint32_t, TILE_K/2> input_a_packed = input_a.format<uint32_t>();  // reinterprete to uint32
    
	    
	matrix<DT, TILE_N, TILE_K> input_b(0.0f);   
    // matrix_ref<uint32_t, TILE_K, TILE_N/2> input_b_packed = input_b.format<uint32_t, TILE_K, TILE_N/2>();  // reinterprete to uint32
    matrix_ref<DT, TILE_N, TILE_K> input_b_packed_trans = input_b;  // reinterprete to uint32
    

	// #pragma unroll
	for(uint32_t i = 0; i < SIZE_K / TILE_K; i++) // outter reduced Tile in K axis 
	// for(uint32_t i = 0; i < 65; i++) // outter reduced Tile in K axis 
	{

		// Load A
		const uint32_t input_a_offset = input_a_base_offset + (i * TILE_K) * sizeof(DT);
		input_a_packed = cm_load<uint32_t, input_a_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_a, input_a_offset);

		
		for(int k = 0; k < TILE_K; k++)
		{
			uint32_t input_b_offset = input_b_localbase + (i*TILE_K+k) * SIZE_N * sizeof(DT);
			cm_prefetch<input_b_load_size, DataSize::U64, CacheHint::Cached, CacheHint::Cached>(surface_input_b, input_b_offset);
			// input_b_packed.row(k) = cm_load<uint32_t, input_b_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, input_b_offset);  
			input_b_packed_trans.column(k).format<U32>() = cm_load<uint32_t, input_b_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, input_b_offset);  
		}
		
		// if (i == 0 && thread_id_0 == 0 && thread_id_1 == 0)
		// {
		// 	printf(" input_b_packed_trans: \n");
		// 	for (int i = 0; i < TILE_N; i++)
    	// 	{
        // 		  printf(" row%d", i);
       	// 		 for (int j = 0; j < TILE_K; j++)
        // 		{ 
        // 		    printf(" %f", input_b(i,  j));
        // 		}
        // 		printf("\n");\
    	// 	}
		// }

		for(int ks = 0; ks < TILE_K/4; ks++){

			vector<DT_ACCU, TILE_N*4> input_b_fp32 = vector<DT_ACCU, TILE_N*4>(input_b.select<TILE_N, 1, 4, 1>(0, ks*4).format<DT>());
			// if (ks == 0 && thread_id_0 == 0 && thread_id_1 == 0)
			// {
			// 	printf(" input_b_fp32: ");
			// 	for (int i = 0; i < TILE_N*4; i++)
			// 	{
			// 		printf(" %f", input_b_fp32(i));
			// 	}
			// 	return;
			// }
			vector<DT_ACCU, TILE_N*4> input_a_fp32 = vector<DT_ACCU, TILE_N*4>(input_a.select<4, 1>(ks*4).replicate<TILE_N>());
			accu.select<TILE_N, 1>(0) += cm_dp4<float>(input_b_fp32, input_a_fp32).select<TILE_N, 4>(0);

		}

		// printf("thread_id_0: %d, thread_id_1: %d i= %d \n", TILE_K, TILE_N, i);
		// if (i == 0 && thread_id_0 == 0 && thread_id_1 == 0)
		// {
		// 	printf(" accu: \n");
		// 	for (int i = 0; i < TILE_N; i++)
    	// 	{
		// 	    printf(" %f", accu(i));
		// 	}
		// 	printf("\n");\
		// 	return;
		// }

    }

    const uint32_t output_store_size = (TILE_N * sizeof(DT)) / sizeof(uint32_t);
    uint32_t output_offset = (thread_id_0  * SIZE_N + thread_id_1 * TILE_N) * sizeof(DT);
		
	vector<DT, TILE_N> accu_out = accu;  // if DT_ACCU == DT then compiler removes this line
	accu_out *= DT(ALPHA);
	accu_out += DT(BETA);
	
	// Store C
	vector_ref<uint32_t, output_store_size> accu_0_packed = accu_out.select<TILE_N, 1>(0).format<uint32_t>();
	cm_store<uint32_t, output_store_size, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, output_offset, accu_0_packed);

}


#endif