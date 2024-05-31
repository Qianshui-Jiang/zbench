#include <cm/cm.h>
#include <cm/cmtl.h>

// In this kernel, Q_SEQ_LEN == TILE_Q == 1
// -> x axis for Head_count & Batch parallel
// -> y axis for Q_SEQ_LEN parallel, for TILE_Q items per thread, in LLMs Q_SEQ_LEN=1
// -> z axis for KV_SEQ_LEN parallel and reduction, speciall for flash decoding (flat shape MHA)

// TODOs:
//  -> KV_SEQ_LEN parallel[Flash decoding], need dynamic dispatch depending on 'pase_seq_len' number.
//  -> KV_SEQ_LEN tiling in single thread[Main loop + Tail loop] 
//  -> SLM for Q tensor parallel loading and shared by KV_SEQ_LEN parallel threads.
//  -> optimize the QK^T gemm of flash decoding shader.

// #define KV_PER_THREAD (KV_SEQ_LEN / SPLIT_KV)
// #define KV_PER_THREAD 1

#define MATH_E 2.718281828459045235360287471352f
#define FLOAT_MAX 3.402823466e+38f

#define DT half
#define DT_ACCU float

#define LD_ST_SIZE ((HEAD_DIM * sizeof(DT))/sizeof(uint32_t))
// #define KV_HEAD_PER_GROUP (Q_HEAD_COUNT/KV_HEAD_COUNT)

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
	
 	matrix<DT, TILE_Q, HEAD_DIM> input_q;
    matrix_ref<uint32_t, TILE_Q, HEAD_DIM/2> input_q_packed = input_q.format<uint32_t, TILE_Q, LD_ST_SIZE>();
    
	vector<DT, HEAD_DIM> input_k;
    vector_ref<uint32_t, HEAD_DIM/2> input_k_packed = input_k.format<uint32_t>();
	
	vector<DT, HEAD_DIM> input_v;
    vector_ref<uint32_t, HEAD_DIM/2> input_v_packed = input_v.format<uint32_t>();


	vector<DT_ACCU, TILE_Q> m_prev(0 - FLOAT_MAX);  // m --> max
	vector<DT_ACCU, TILE_Q> m_cur;    		// m --> max
	vector<DT_ACCU, TILE_Q> f(0.0f);  		// f --> exp(m_prev - m_cur); 
	vector<DT_ACCU, TILE_Q> l_prev(0.0f);	// l --> sum of exp(Xi-m)
	vector<DT_ACCU, TILE_Q> l_cur;	  		// l --> sum of exp(Xi-m)
	vector<DT_ACCU, TILE_Q> l_rcp;	  		// l --> sum of exp(Xi-m)

	vector<DT_ACCU, TILE_Q> qk(0); 
	vector<DT_ACCU, TILE_Q> p;
	matrix<DT_ACCU, TILE_Q, HEAD_DIM> acc(0.0f);

									// Q_SEQ_LEN pralell						// Head count pralell
	const uint32_t threads_offset = (global_y * TILE_Q * HEAD_COUNT * HEAD_DIM + global_x * HEAD_DIM) * sizeof(DT) ;
	uint32_t output_offset = threads_offset ;
	uint32_t q_offset = 0;
	uint32_t kv_offset = 0;

	#pragma unroll
	for(int t_q=0; t_q < TILE_Q; t_q++){  // Load Tile Q of Q_SEQ_LEN
		q_offset = threads_offset + t_q * HEAD_COUNT * HEAD_DIM * sizeof(DT) ;
		input_q_packed.row(t_q) = cm_load<uint32_t, LD_ST_SIZE, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_q, q_offset);
	}

	// Tail loop of K/v seq_len
	// for(int j=0; j<Q_SEQ_LEN; j++){  // Loop on tiled K/V --> Bc in paper
	for(int j=0; j<global_y+1; j++){  // Loop on tiled K/V --> Bc in paper
		kv_offset  = j * HEAD_COUNT * HEAD_DIM * sizeof(DT) + global_x  * HEAD_DIM * sizeof(DT) ;
		input_k_packed = cm_load<uint32_t, LD_ST_SIZE, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_k, kv_offset);
		input_v_packed = cm_load<uint32_t, LD_ST_SIZE, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_v, kv_offset);
		
		// store to Present K/V
		// if(global_y==0){
		if(global_y==Q_SEQ_LEN-1){
			output_offset = (global_x * KV_SEQ_LEN +  j ) * HEAD_DIM * sizeof(DT) ;
			cm_store<uint32_t, LD_ST_SIZE, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output_present_k, output_offset, input_k_packed);
			cm_store<uint32_t, LD_ST_SIZE, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output_present_v, output_offset, input_v_packed);
		}

		// Q*K
		for(int q_idx=0; q_idx<TILE_Q; q_idx ++){
			qk(q_idx) = cm_sum<DT_ACCU>(input_q.row(q_idx) * input_k);
		}
		qk *=  (DT_ACCU)HEAD_SCALE;
		
		m_cur = qk; // lack of max of
		
		for(int m_cur_idx=0; m_cur_idx<TILE_Q; m_cur_idx ++){
			m_cur(m_cur_idx) = m_prev(m_cur_idx) > qk(m_cur_idx) ? m_prev(m_cur_idx) : qk(m_cur_idx); // lack of max of
		}

		
		f = cm_pow(MATH_E, m_prev - m_cur);
		l_prev *= f ;

		p =  cm_pow(MATH_E, (qk - m_cur));

		l_cur =  l_prev + p;  // p idx was wight?

		// 1. For flash attention
		// l_rcp = 1/l_cur;
		// p *=  l_rcp;  // s
		// acc *=  (l_prev * l_rcp);

		// 2. For flash attention 2
		for(int s_idx=0; s_idx<TILE_Q; s_idx++){
			acc.row(s_idx) *= f(s_idx);
			acc.row(s_idx) += p(s_idx) * input_v;
		}

		m_prev = m_cur;
		l_prev = l_cur;
	}


	// 1. For flash attention
	// matrix<DT, TILE_Q, HEAD_DIM> acc_out = acc;
	
	// 2. For flash attention
	#pragma unroll
	for(int acc_idx=0; acc_idx<TILE_Q; acc_idx ++){
		acc.row(acc_idx)*= cm_inv(l_prev(acc_idx));
	}
	
	
	output_offset = threads_offset;
	matrix<DT, TILE_Q, HEAD_DIM> acc_out = acc;
	
	#pragma unroll
	for(int t_q=0; t_q < TILE_Q; t_q++){  // Load Tile Q
		// printf("q_offset : %d \n", (i * TILE_Q  + t_q));
		output_offset = threads_offset + t_q * HEAD_COUNT * HEAD_DIM * sizeof(DT) ;
		cm_store<uint32_t, LD_ST_SIZE, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, output_offset, acc_out.row(t_q).format<uint32_t>());
	}
}
