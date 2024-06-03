#include <cm/cm.h>
#include <cm/cmtl.h>

// In this kernel, Q_SEQ_LEN == TILE_Q == 1
// 	-> x axis for Head_count & Batch parallel
// 	-> y axis for Q_SEQ_LEN parallel, for TILE_Q items per thread, in LLMs Q_SEQ_LEN=1
// 	-> z axis for KV_SEQ_LEN parallel and reduction, speciall for flash decoding (flat shape MHA)

// TODOs:
// 	-> KV_SEQ_LEN parallel[Flash decoding], need dynamic dispatch depending on 'pase_seq_len' number.
// 	-> KV_SEQ_LEN tiling in single thread[Main loop + Tail loop] 
// 	-> SLM for Q tensor parallel loading and shared by KV_SEQ_LEN parallel threads.
// 	-> optimize the QK^T gemm of flash decoding shader.

// #define KV_PER_THREAD (KV_SEQ_LEN / SPLIT_KV)
// #define KV_PER_THREAD 1

#define MATH_E 2.718281828459045235360287471352f
#define FLOAT_MAX 3.402823466e+38f

#define DT half
#define DT_ACCU float

#define LD_ST_SIZE ((HEAD_DIM * sizeof(DT))/sizeof(uint32_t))

extern "C" _GENX_MAIN_ void mha_q_k_v_flash_1st_token(
		SurfaceIndex surface_input_q        	[[type("buffer_t")]],
		SurfaceIndex surface_input_k        	[[type("buffer_t")]],
		SurfaceIndex surface_input_v        	[[type("buffer_t")]],
		SurfaceIndex surface_past_seq_len   	[[type("buffer_t")]],
		SurfaceIndex surface_output_present_k   [[type("buffer_t")]],
		SurfaceIndex surface_output_present_v   [[type("buffer_t")]],
		SurfaceIndex surface_output             [[type("buffer_t")]]
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

	// vector<DT_ACCU, SPLIT_KV> calibre_sum_exp;
	DT_ACCU m_prev= 0 - FLOAT_MAX;      // m --> max
	DT_ACCU m_cur;      			    // m --> max
	DT_ACCU f = 0;      			    // f --> exp(m_prev - m_cur); 
	DT_ACCU l_prev = 0;				    // l --> sum of exp(Xi-m)
	DT_ACCU l_cur;	    			    // l --> sum of exp(Xi-m)
	DT_ACCU l_rcp;	    			    // l --> sum of exp(Xi-m)
	vector<DT_ACCU, HEAD_DIM> acc(0);

									// Q_SEQ_LEN pralell						 // Head count pralell
	const uint32_t threads_offset = (global_y * TILE_Q * HEAD_COUNT * HEAD_DIM + global_x * HEAD_DIM) * sizeof(DT) ;
	uint32_t output_offset =0 ;
	uint32_t kv_offset = 0;
	const uint32_t output_store_size = (HEAD_DIM * sizeof(DT)) / sizeof(uint32_t);

	input_q_packed = cm_load<uint32_t, LD_ST_SIZE, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_q, threads_offset);

	// if (global_x == 0 && global_y == 5 && global_z==0)
	// if (global_x == 0 && global_y &&  global_z==0)
	// {	
	// 	printf("global_y : %d , threads_offset : %d\n", global_y, threads_offset/ sizeof(DT));
	// 	printf("HEAD_COUNT: %d\n", HEAD_COUNT);
	// 	printf("HEAD_DIM: %d\n", HEAD_DIM);
	// 	printf("latest Q: ");
	// 	for (int y = 0; y < 8; y++)
	// 	{
	// 		printf(" %f, ", input_q(y));
	// 	}	
	// 	printf("\n");
	// }
	
	// Main loop of K/v seq_len
	// for(int j=0; j<Q_SEQ_LEN; j++){  // Loop on tiled K/V --> Bc in paper
	// #pragma unroll
	// jth col in Q  ---> KV cols < j
	for(int j=0; j<global_y+1; j++){  // Loop on tiled K/V --> Bc in paper
		kv_offset  = j * TILE_Q * HEAD_COUNT * HEAD_DIM * sizeof(DT) + global_x  * HEAD_DIM * sizeof(DT) ;
		input_k_packed  = cm_load<uint32_t, LD_ST_SIZE, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_k, kv_offset);
		input_v_packed  = cm_load<uint32_t, LD_ST_SIZE, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_v, kv_offset);

		// // store to Output Present K/V
		if(global_y==Q_SEQ_LEN-1){
			output_offset = (global_x * KV_SEQ_LEN +  j ) * HEAD_DIM * sizeof(DT) ;
			cm_store<uint32_t, output_store_size, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output_present_k, output_offset, input_k_packed.format<uint32_t>());
			cm_store<uint32_t, output_store_size, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output_present_v, output_offset, input_v_packed.format<uint32_t>());
		}
		

		// Q*K
		qk =  (DT_ACCU)HEAD_SCALE * cm_sum<DT_ACCU>(input_q * input_k);

		m_cur = qk;

		if(m_prev > m_cur){
			m_cur = m_prev;
		}	

		f = cm_pow((DT_ACCU)MATH_E, m_prev - m_cur);
		// f = cm_exp(m_prev - m_cur);
		l_prev *= f ;

		p =  cm_pow((DT_ACCU)MATH_E, (qk - m_cur));
		// p =  cm_exp((qk - m_cur));

		l_cur =  l_prev + p;  // p idx was wight?

		// 1. For flash attention
		// l_rcp = 1/l_cur;
		// p *=  l_rcp;  // s
		// acc *=  (l_prev * l_rcp);

		// 2. For flash attention 2
		acc *= f ;

		acc += p * input_v;

		m_prev = m_cur;
		l_prev = l_cur;
	}


	// 1. For flash attention
	// vector<DT, HEAD_DIM> acc_out = acc;
	
	// 2. For flash attention
	vector<DT, HEAD_DIM> acc_out = acc/l_prev;
	
	output_offset = threads_offset;


	vector_ref<uint32_t, output_store_size> accu_0_packed = acc_out.format<uint32_t>();
	cm_store<uint32_t, output_store_size, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, output_offset, accu_0_packed);
}
