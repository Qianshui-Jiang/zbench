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


extern "C" _GENX_MAIN_ void vxm_test_fp16(
	SurfaceIndex surface_input_a [[type("buffer_t" )]],
	SurfaceIndex surface_input_b [[type("buffer_t" )]],
	SurfaceIndex surface_output  [[type("buffer_t" )]]
)
{
    const uint32_t thread_id_0 = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
    const uint32_t thread_id_1 = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
	// printf("thread_id_0: %d, thread_id_1: %d \n", thread_id_0, thread_id_1);
    const uint32_t input_a_load_size = (TILE_K * sizeof(DT)) / sizeof(uint32_t);
    const uint32_t input_b_load_size = (TILE_N * sizeof(DT)) / sizeof(uint32_t);
        
    const uint32_t input_a_base_offset =0;
    
	vector<DT_ACCU, TILE_N> accu(0.0f);
    uint32_t input_b_offset = thread_id_1 * TILE_N * sizeof(DT);
	
    vector<DT, TILE_K> input_a;   
    vector_ref<uint32_t, TILE_K/2> input_a_packed = input_a.format<uint32_t>();  // reinterprete to uint32
    
	// #pragma unroll
	for(uint32_t i = 0; i < SIZE_K / TILE_K; i++) // outter reduced Tile in K axis 
	{
		// Load A
		const uint32_t input_a_offset = input_a_base_offset + (i * TILE_K) * sizeof(DT);
		input_a_packed = cm_load<uint32_t, input_a_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_a, input_a_offset);
	   
		// #pragma unroll
		for(uint32_t k = 0; k < TILE_K; k++) // reduce for inner product
		{
			// Load B
			vector<uint32_t, input_b_load_size> input_b_packed = cm_load<uint32_t, input_b_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, input_b_offset);  
			vector_ref<DT, TILE_N> input_b = input_b_packed.format<DT>();        
			input_b_offset += SIZE_N * sizeof(DT);
		
			// // Acc
#if ACCU_IS_FP32
				vector<DT_ACCU, TILE_N> input_b_fp32 = vector<DT_ACCU, TILE_N>(input_b);
				vector<DT_ACCU, TILE_N> input_a_fp32 = vector<DT_ACCU, TILE_N>(input_a.select< 1, 1>(k).replicate<TILE_N>());
				accu.select<TILE_N, 1>(0) += input_b_fp32 * input_a_fp32;
#else
			accu.select<TILE_N, 1>(0) += input_b * input_a.select<1, 1>(k).replicate<TILE_N>();
#endif

		}
		// printf("thread_id_0: %d, thread_id_1: %d i= %d ,k= %d\n", thread_id_0, thread_id_1, i, k);
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

