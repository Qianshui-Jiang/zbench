#include <cm/cm.h>
#include <cm/cmtl.h>

#define DT half
#define DT_ACCU float 

#define MATH_E 2.718281828459045235360287471352f
#define FLOAT_MAX 3.402823466e+38f

// In this kernel, 
// Q_SEQ_LEN == 100
// TILE_Q == 1
// Parallel as below:
// - x thread group axis for  Batch & Head_count parallel
// - y thread group axis for Q_SEQ_LEN parallel, for TILE_Q items per thread
// - z thread group axis for KV_SEQ_LEN parallel and reduce, specially for flash decoding (flat shape MHA)

// #define Q_SEQ_LEN
// #define KV_SEQ_LEN
// #define HEAD_COUNT
// #define HEAD_DIM
// #define TILE_Q
// #define TILE_HEAD
// #define HEAD_SCALE 

extern "C" _GENX_MAIN_ void flash_decoding(
		SurfaceIndex surface_input_q [[type("buffer_t half")]],
		SurfaceIndex surface_input_k [[type("buffer_t half")]],
		SurfaceIndex surface_input_v [[type("buffer_t half")]],
		SurfaceIndex surface_output [[type("buffer_t half")]]
)
{
    const uint32_t global_x = cm_group_id(0) * LWS_SIZE_X + cm_local_id(0);
    const uint32_t global_y = cm_group_id(1) * LWS_SIZE_Y + cm_local_id(1);
    const uint32_t global_z = cm_group_id(2) * LWS_SIZE_Z + cm_local_id(2);
	// if (global_x == 0 && global_y == 0 && global_z==0)
	// {	
	
	// 	printf("global_x : %d \n", global_x);
	// 	printf("global_y : %d \n", global_y);
	// 	printf("global_z : %d \n", global_z);

	// 	// printf("Q_SEQ_LEN : %d \n", Q_SEQ_LEN);
	// 	// printf("KV_SEQ_LEN : %d \n", KV_SEQ_LEN);
	// 	// printf("HEAD_DIM : %d \n", HEAD_DIM);
	// 	// printf("HEAD_COUNT : %d \n", HEAD_COUNT);
	// 	// printf("TILE_Q : %d \n", TILE_Q);
	// 	// printf("TILE_HEAD : %d \n", TILE_HEAD);
	// 	// printf("HEAD_SCALE : %f \n", HEAD_SCALE);

	// }
 	matrix<DT, TILE_Q, HEAD_DIM> input_q;
    matrix_ref<uint32_t, TILE_Q, HEAD_DIM/2> input_q_packed = input_q.format<uint32_t, TILE_Q, HEAD_DIM/2>();
    
	vector<DT, HEAD_DIM> input_k;
    vector_ref<uint32_t, HEAD_DIM/2> input_k_packed = input_k.format<uint32_t>();
	
	vector<DT, HEAD_DIM> input_v;
    vector_ref<uint32_t, HEAD_DIM/2> input_v_packed = input_v.format<uint32_t>();



	vector<DT_ACCU, TILE_Q> m_prev(0 - FLOAT_MAX);  // m --> max
	vector<DT_ACCU, TILE_Q> m_cur;  // m --> max
	vector<DT_ACCU, TILE_Q> f(0.0f);  // f --> exp(m_prev - m_cur); 
	vector<DT_ACCU, TILE_Q> l_prev(0.0f);	// l --> sum of exp(Xi-m)
	vector<DT_ACCU, TILE_Q> l_cur;	// l --> sum of exp(Xi-m)
	vector<DT_ACCU, TILE_Q> l_rcp;	// l --> sum of exp(Xi-m)

	vector<DT_ACCU, TILE_Q> qk(0); 
	vector<DT_ACCU, TILE_Q> p;
	matrix<DT_ACCU, TILE_Q, HEAD_DIM> acc(0.0f);

									// Q_SEQ_LEN pralell						// Head count pralell
	const uint32_t threads_offset = (global_y * TILE_Q * HEAD_COUNT * HEAD_DIM + global_x * HEAD_DIM) * sizeof(DT) ;
	uint32_t q_offset = 0;
	uint32_t kv_offset = 0;
	uint32_t output_offset = 0;


	#pragma unroll
	for(int t_q=0; t_q < TILE_Q; t_q++){  // Load Tile Q of Q_SEQ_LEN
		q_offset = threads_offset + t_q * HEAD_COUNT * HEAD_DIM * sizeof(DT) ;
		input_q_packed.row(t_q)  = cm_load<uint32_t, (HEAD_DIM * sizeof(DT))/sizeof(uint32_t), DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_q, q_offset);
	}


	m_prev = 0 - FLOAT_MAX;
	l_prev = 0;
	acc = 0;

	if (global_x == 0 && global_y == 0 && global_z==0)
	{	
		printf("-------PRINT TEST--------- :\n");
		printf("-------PRINT TEST--------- :\n");
		printf("-------PRINT TEST--------- :\n");
		printf("-------PRINT TEST--------- :\n");
		// for (int y = 0; y < HEAD_DIM; y++)
		// {
		// 	printf("%f, ", input_q(y));
		// }	
		// printf("\n");
	}

	// #pragma unroll
	for(int j=0; j<KV_SEQ_LEN; j++){  // Loop on tiled K/V --> Bc in paper
		// printf("Loop  Q : %d, Loop K/V: %d\n", i, j);
		kv_offset = j * HEAD_COUNT * HEAD_DIM * sizeof(DT) + global_x  * HEAD_DIM * sizeof(DT);
		input_k_packed  = cm_load<uint32_t, (HEAD_DIM * sizeof(DT))/sizeof(uint32_t), DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_k, kv_offset);
		input_v_packed  = cm_load<uint32_t, (HEAD_DIM * sizeof(DT))/sizeof(uint32_t), DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_v, kv_offset);


		// Q*K
		for(int q_idx=0; q_idx<TILE_Q; q_idx ++){
			qk(q_idx) = cm_sum<DT_ACCU>(input_q.row(q_idx) * input_k);
		}

		qk *=  (DT_ACCU)HEAD_SCALE;

		// m_cur = qk; // lack of max 
		// #pragma unroll
		// for(int m_cur_idx=0; m_cur_idx<TILE_Q; m_cur_idx ++){
		// 	if(m_prev(m_cur_idx) > m_cur(m_cur_idx)){
		// 		m_cur(m_cur_idx) = m_prev(m_cur_idx);
		// 	}	
		// }

		for(int m_cur_idx=0; m_cur_idx<TILE_Q; m_cur_idx ++){
			m_cur(m_cur_idx) = m_prev(m_cur_idx) > qk(m_cur_idx) ? m_prev(m_cur_idx) : qk(m_cur_idx); // lack of max of
		}

		f = cm_pow(MATH_E, m_prev - m_cur);
		l_prev *= f;

		p =  cm_pow(MATH_E, (qk - m_cur));
		l_cur =  l_prev + p;  // p idx was wight?

		
		// // 1. For flash attention
		// l_rcp = 1/l_cur;
		// #pragma unroll
		// for(int p_idx=0; p_idx<TILE_KV; p_idx ++){
		// 	p.column(p_idx) *=  l_rcp;  // s
		// }
		
		// #pragma unroll
		// for(int acc_idx=0; acc_idx<TILE_HEAD; acc_idx ++){
		// 	acc.column(acc_idx) *=  (l_rcp*l_prev);
		// }
		
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
	
	matrix<DT, TILE_Q, HEAD_DIM> acc_out = acc;
	const uint32_t output_store_size = (HEAD_DIM * sizeof(DT)) / sizeof(uint32_t);

	#pragma unroll
	for(int t_q=0; t_q < TILE_Q; t_q++){  // Load Tile Q
		// printf("q_offset : %d \n", (i * TILE_Q  + t_q));
		output_offset = threads_offset + t_q * HEAD_COUNT * HEAD_DIM * sizeof(DT) ;
		vector_ref<uint32_t, output_store_size> accu_0_packed = acc_out.row(t_q).format<uint32_t>();
		cm_store<uint32_t, output_store_size, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, output_offset, accu_0_packed);
	}

}
