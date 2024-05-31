#include <cm/cm.h>
#include <cm/cmtl.h>

#define DT half
#define DT_ACCU float 

// #define Q_SEQ_LEN 1
// #define KV_SEQ_LEN 2048
// #define HEAD_DIM 128
// #define TILE_Q 1
// #define TILE_KV 32
// #define HEAD_SCALE 
// #define SPLIT_KV 32

#define MATH_E 2.718281828459045235360287471352f
#define FLOAT_MAX 3.402823466e+38f
// #define KV_PER_THREAD (KV_SEQ_LEN / SPLIT_KV)
// #define KV_PER_THREAD 1

// In this kernel, Q_SEQ_LEN == TILE_Q == 1
// -> x axis for Head_count & Batch parallel
// -> y axis for Q_SEQ_LEN parallel, for TILE_Q items per thread
// -> z axis for KV_SEQ_LEN parallel and reduce, speciall for flash decoding (flat shape MHA)

#define LD_ST_SIZE ((HEAD_DIM * sizeof(DT))/sizeof(uint32_t))


// simplified matmul by cm_mul
extern "C" _GENX_MAIN_ void mha_q_k_v_flash_decoding(
		SurfaceIndex surface_input_q        [[type("buffer_t half")]],
		SurfaceIndex surface_input_k        [[type("buffer_t half")]],
		SurfaceIndex surface_input_v        [[type("buffer_t half")]],
		SurfaceIndex surface_past_seq_len   [[type("buffer_t half")]],
		SurfaceIndex surface_output_present_k  [[type("buffer_t half")]],
		SurfaceIndex surface_output_present_v  [[type("buffer_t half")]],
		SurfaceIndex surface_output         [[type("buffer_t half")]]
)
{
    const uint32_t global_x = cm_group_id(0) * LWS_SIZE_X + cm_local_id(0);
    const uint32_t global_y = cm_group_id(1) * LWS_SIZE_Y + cm_local_id(1);
    const uint32_t global_z = cm_group_id(2) * LWS_SIZE_Z + cm_local_id(2);
	
    vector<DT, HEAD_DIM> input_q;
    vector_ref<uint32_t, HEAD_DIM/2> input_q_packed = input_q.format<uint32_t>();
    
	vector<DT, HEAD_DIM> input_k;
    vector_ref<uint32_t, HEAD_DIM/2> input_k_packed = input_k.format<uint32_t>();
	
	vector<DT, HEAD_DIM> input_v;
    vector_ref<uint32_t, HEAD_DIM/2> input_v_packed = input_v.format<uint32_t>();

	DT_ACCU qk;
	DT_ACCU p;
	DT_ACCU m_prev= 0 - FLOAT_MAX;  // m --> max
	DT_ACCU m_cur;      // m --> max
	DT_ACCU f = 0;      // f --> exp(m_prev - m_cur); 
	DT_ACCU l_prev = 0;	// l --> sum of exp(Xi-m)
	DT_ACCU l_cur;	    // l --> sum of exp(Xi-m)
	DT_ACCU l_rcp;	    // l --> sum of exp(Xi-m)
	vector<DT_ACCU, HEAD_DIM> acc(0);

	const uint32_t threads_offset = global_x * HEAD_DIM * sizeof(DT) ;
	uint32_t output_offset = threads_offset ;
	uint32_t kv_offset = 0;
	const uint32_t output_store_size = (HEAD_DIM * sizeof(DT)) / sizeof(uint32_t);

	input_q_packed = cm_load<uint32_t, LD_ST_SIZE, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_q, threads_offset);

	vector<uint32_t,1> past_seq_len = cm_load<uint32_t, 1, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_past_seq_len, 0);

	// load KV, store to Present K/V
	kv_offset = (global_x) * HEAD_DIM * sizeof(DT) ;
	output_offset = (global_x * KV_SEQ_LEN + past_seq_len(0)) * HEAD_DIM * sizeof(DT);

	input_k_packed  = cm_load<uint32_t, LD_ST_SIZE, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_k, kv_offset);
	cm_store<uint32_t, output_store_size, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output_present_k, output_offset, input_k_packed);

	input_v_packed  = cm_load<uint32_t, LD_ST_SIZE, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_v, kv_offset);
	cm_store<uint32_t, output_store_size, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output_present_v, output_offset, input_v_packed);

	// Main loop of K/v seq_len
	#pragma unroll
	for(int j=0; j<past_seq_len(0)+1; j++){  // Loop on tiled K/V --> Bc in paper
		kv_offset = (global_x * KV_SEQ_LEN +  j ) * HEAD_DIM * sizeof(DT) ;
		input_k_packed  = cm_load<uint32_t, LD_ST_SIZE, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_output_present_k, kv_offset);
		input_v_packed  = cm_load<uint32_t, LD_ST_SIZE, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_output_present_v, kv_offset);


		// Q*K^T sub GEMM
		qk =  (DT_ACCU)HEAD_SCALE * cm_sum<DT_ACCU>(input_q * input_k);


		m_cur = m_prev>qk ? m_prev : qk;

		f = cm_pow((DT_ACCU)MATH_E, (m_prev - m_cur));
		l_prev *= f;

		p =  cm_pow((DT_ACCU)MATH_E, (qk - m_cur));

		l_cur =  l_prev + p;  // p idx was wight?

		// 1. For flash attention
		// l_rcp = 1/l_cur;
		// p *=  l_rcp;  // s
		// acc *=  (l_prev * l_rcp);

		// 2. For flash attention 2
		acc *= f ;

		// S*V sub GEMM
		acc += p * input_v;

		m_prev = m_cur;
		l_prev = l_cur;

	}

	// if (global_x == 0 && global_y == 0 && global_z==0)
	// {	
	// 	printf("acc :\n");
	// 	for (int y = 0; y < 8; y++)
	// 	{
	// 		printf("%f, ", acc(y));
	// 	}	
	// 	printf("\n");
	// }


	// 1. For flash attention
	// vector<DT, HEAD_DIM> acc_out = acc;
	
	// 2. For flash attention
	vector<DT, HEAD_DIM> acc_out = acc/l_cur;	
	cm_store<uint32_t, LD_ST_SIZE, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, threads_offset, acc_out.format<uint32_t>());
}