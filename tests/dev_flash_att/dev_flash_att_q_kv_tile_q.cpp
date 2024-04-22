/*========================== begin_copyright_notice ============================

INTEL CONFIDENTIAL

Copyright (C) 2023 Intel Corporation

This software and the related documents are Intel copyrighted materials,
and your use of them is governed by the express license under which they were
provided to you ("License"). Unless the License provides otherwise,
you may not use, modify, copy, publish, distribute, disclose or transmit this
software or the related documents without Intel's prior written permission.

This software and the related documents are provided as is, with no express or
implied warranties, other than those that are expressly stated in the License.

============================= end_copyright_notice ===========================*/
#include <cm/cm.h>
#include <cm/cmtl.h>

#define DT half
#define DT_ACCU float 

#define MATH_E 2.718281828459045235360287471352f
#define FLOAT_MAX 3.402823466e+38f

// Parallel as below:
// - x thread group axis for Batch size parallel
// - y thread group axis for Q_SEQ_LEN parallel, for TILE_Q items per thread
// - z thread group axis for HEAD_COUNT parallel


// #define Q_SEQ_LEN
// #define KV_SEQ_LEN
// #define HEAD_DIM
// #define TILE_Q
// #define TILE_KV
// #define TILE_HEAD

#define KV_ITER_NUM (KV_SEQ_LEN/TILE_KV)
#define KV_TAIL_NUM (KV_SEQ_LEN%TILE_KV)



extern "C" _GENX_MAIN_ void flash_att(
		SurfaceIndex surface_input_q [[type("buffer_t half")]],
		SurfaceIndex surface_input_kv [[type("buffer_t half")]],
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

	// 	printf("BATCH_SIZE : %d \n", BATCH_SIZE);
	// 	printf("Q_SEQ_LEN : %d \n", Q_SEQ_LEN);
	// 	printf("KV_SEQ_LEN : %d \n", KV_SEQ_LEN);
	// 	printf("HEAD_COUNT : %d \n", HEAD_COUNT);
	// 	printf("HEAD_DIM : %d \n", HEAD_DIM);

	// 	printf("TILE_Q : %d \n", TILE_Q);
	// 	printf("TILE_KV : %d \n", TILE_KV);
	// 	printf("TILE_HEAD : %d \n", TILE_HEAD);
	// }

    matrix<DT, TILE_Q, HEAD_DIM> input_q;
    matrix_ref<uint32_t, TILE_Q, HEAD_DIM/2> input_q_packed = input_q.format<uint32_t, TILE_Q, HEAD_DIM/2>();
    
	matrix<DT, TILE_KV, HEAD_DIM> input_k;
    matrix_ref<uint32_t, TILE_KV, HEAD_DIM/2> input_k_packed = input_k.format<uint32_t, TILE_KV, HEAD_DIM/2>();
	
	matrix<DT, TILE_KV, HEAD_DIM> input_v;
    matrix_ref<uint32_t, TILE_KV, HEAD_DIM/2> input_v_packed = input_v.format<uint32_t, TILE_KV, HEAD_DIM/2>();


	vector<DT_ACCU, TILE_Q> m_prev(0 - FLOAT_MAX);  // m --> max
	vector<DT_ACCU, TILE_Q> m_cur;  // m --> max
	vector<DT_ACCU, TILE_Q> f(0.0f);  // f --> exp(m_prev - m_cur); 
	vector<DT_ACCU, TILE_Q> l_prev(0.0f);	// l --> sum of exp(Xi-m)
	vector<DT_ACCU, TILE_Q> l_cur;	// l --> sum of exp(Xi-m)
	vector<DT_ACCU, TILE_Q> l_rcp;	// l --> sum of exp(Xi-m)

	matrix<DT_ACCU, TILE_Q, TILE_KV> qk; // 4x2
	matrix<DT_ACCU, TILE_Q, TILE_KV> p;
	matrix<DT_ACCU, TILE_Q, HEAD_DIM> acc(0.0f);

	const uint32_t q_batch_offset = global_x * Q_SEQ_LEN * HEAD_COUNT * HEAD_DIM * sizeof(DT) ;
	const uint32_t q_seq_len_tile_offset = q_batch_offset + global_y * TILE_Q * HEAD_COUNT * HEAD_DIM * sizeof(DT) ;
	const uint32_t q_head_count_offset = q_seq_len_tile_offset + global_z  * HEAD_DIM * sizeof(DT) ;

	const uint32_t kv_batch_offset = global_x * KV_SEQ_LEN * HEAD_COUNT * 2 * HEAD_DIM * sizeof(DT) ;
	const uint32_t kv_head_count_offset = kv_batch_offset + global_z  * 2 *  HEAD_DIM * sizeof(DT) ;

	uint32_t q_offset = 0;
	uint32_t k_offset = 0;
	uint32_t v_offset = 0;
	uint32_t output_offset = 0;



	#pragma unroll
	for(int t_q=0; t_q< TILE_Q; t_q++){  // Load Tile Q of Q_SEQ_LEN
		q_offset = q_head_count_offset + t_q * HEAD_COUNT * HEAD_DIM * sizeof(DT) ;
#if HEAD_DIM == 160
		input_q_packed.row(t_q).select<64, 1>()  = cm_load<uint32_t, 64, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_q, q_offset);
		input_q_packed.row(t_q).select<16, 1>(64)  = cm_load<uint32_t, 16, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_q, q_offset+64 * sizeof(uint32_t));
#elif HEAD_DIM == 80	
		input_q_packed.row(t_q).select<32, 1>()  = cm_load<uint32_t, 32, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_q, q_offset);
		input_q_packed.row(t_q).select<8, 1>(32)  = cm_load<uint32_t, 8, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_q, q_offset+32 * sizeof(uint32_t));
#elif HEAD_DIM == 40	
		input_q_packed.row(t_q).select<16, 1>()  = cm_load<uint32_t, 16, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_q, q_offset);
		input_q_packed.row(t_q).select<4, 1>(16)  = cm_load<uint32_t, 4, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_q, q_offset+16 * sizeof(uint32_t));
#else
		input_q_packed.row(t_q)  = cm_load<uint32_t, (HEAD_DIM * sizeof(DT))/sizeof(uint32_t), DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_q, q_offset);
#endif
	}


	// if (global_x == 0 && global_y == 2 && global_z==0)
	// {	
	// 	printf("input_q :\n");
	// 	for (int x = 0; x < TILE_Q; x++){
	// 		for (int y = 0; y < HEAD_DIM; y++)
	// 		{
	// 			printf("%f, ", input_q(x, y));
	// 		}	
	// 		printf("\n");
	// 	}
	// }

	#pragma unroll
	for(int j=0; j<KV_ITER_NUM; j++){  // Loop K/V --> Bc in paper

		#pragma unroll
		for(int t_kv=0; t_kv< TILE_KV; t_kv++){   // Load Tile K/V of KV_SEQ
		// printf("Loop  Q : %d, Loop K/V: %d\n", i, j);
			k_offset = kv_head_count_offset + ((j * TILE_KV  + t_kv) * HEAD_COUNT * 2)* HEAD_DIM * sizeof(DT) ;
			v_offset = kv_head_count_offset + ((j * TILE_KV  + t_kv) * HEAD_COUNT * 2 + 1) * HEAD_DIM * sizeof(DT) ;
#if HEAD_DIM == 160
			input_k_packed.row(t_kv).select<64, 1>()   = cm_load<uint32_t, 64, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, k_offset);
			input_k_packed.row(t_kv).select<16, 1>(64) = cm_load<uint32_t, 16, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, k_offset+ 64 * sizeof(uint32_t));
			
			input_v_packed.row(t_kv).select<64, 1>()  = cm_load<uint32_t, 64, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, v_offset);
			input_v_packed.row(t_kv).select<16, 1>(64)  = cm_load<uint32_t, 16, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, v_offset+64 * sizeof(uint32_t));
#elif HEAD_DIM == 80	
			input_k_packed.row(t_kv).select<32, 1>()   = cm_load<uint32_t, 32, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, k_offset);
			input_k_packed.row(t_kv).select<8, 1>(32) = cm_load<uint32_t, 8, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, k_offset+ 32 * sizeof(uint32_t));
			
			input_v_packed.row(t_kv).select<32, 1>()  = cm_load<uint32_t, 32, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, v_offset);
			input_v_packed.row(t_kv).select<8, 1>(32)  = cm_load<uint32_t, 8, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, v_offset+32 * sizeof(uint32_t));
#elif HEAD_DIM == 40
			input_k_packed.row(t_kv).select<16, 1>()   = cm_load<uint32_t, 16, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, k_offset);
			input_k_packed.row(t_kv).select<4, 1>(16) = cm_load<uint32_t, 4, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, k_offset+ 16 * sizeof(uint32_t));
			
			input_v_packed.row(t_kv).select<16, 1>()  = cm_load<uint32_t, 16, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, v_offset);
			input_v_packed.row(t_kv).select<4, 1>(16)  = cm_load<uint32_t, 4, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, v_offset+16 * sizeof(uint32_t));
#else
			input_k_packed.row(t_kv)  = cm_load<uint32_t, (HEAD_DIM * sizeof(DT))/sizeof(uint32_t), DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, k_offset);
			input_v_packed.row(t_kv)  = cm_load<uint32_t, (HEAD_DIM * sizeof(DT))/sizeof(uint32_t), DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, v_offset);
#endif

		}

		// if (global_x == 1 && global_y == 0 && global_z==1)
		// {	
		// 	// printf("input_k :\n");
		// 	// for (int x = 0; x < TILE_KV; x++){
		// 	// 	for (int y = 0; y < HEAD_DIM; y++)
		// 	// 	{
		// 	// 		printf("%f, ", input_k(x, y));
		// 	// 	}	
		// 	// 	printf("\n");
		// 	// }

		// 	printf("input_v :\n");
		// 	for (int x = 0; x < TILE_KV; x++){
		// 		for (int y = 0; y < HEAD_DIM; y++)
		// 		{
		// 			printf("%f, ", input_v(x, y));
		// 		}	
		// 		printf("\n");
		// 	}
		// }

		// Q*K
		// #pragma unroll
		// for(int hd_idx=0; hd_idx<HEAD_DIM; hd_idx ++){
		// 	vector<DT_ACCU, TILE_KV> k_fp32 = vector<DT_ACCU, TILE_KV>(input_k.column(hd_idx));
		// 	for(int q_idx=0; q_idx<TILE_Q; q_idx ++){
		// 		vector<DT_ACCU, TILE_KV> q_fp32 = vector<DT_ACCU, TILE_KV>(input_q.select<1, 1, 1, 1>(q_idx, hd_idx).replicate<TILE_KV>());
		// 		qk.select<1, 1, TILE_KV, 1>(q_idx, 0) += q_fp32 * k_fp32;
		// 	}
		// }


		#pragma unroll
		for(int k_idx=0; k_idx<TILE_KV; k_idx ++){
			vector<DT_ACCU, HEAD_DIM> k_fp32 = vector<DT_ACCU, HEAD_DIM>(input_k.row(k_idx));
			for(int q_idx=0; q_idx<TILE_Q; q_idx ++){
				vector<DT_ACCU, HEAD_DIM> q_fp32 = vector<DT_ACCU, HEAD_DIM>(input_q.row(q_idx));
				qk.select<1, 1, 1, 1>(q_idx, k_idx) = cm_sum<DT_ACCU>(q_fp32 * k_fp32);
			}
		}


		qk *=  (DT_ACCU)HEAD_SCALE;

		#pragma unroll
		for(int m_cur_idx=0; m_cur_idx<TILE_Q; m_cur_idx ++){
			m_cur(m_cur_idx) = cm_reduced_max<DT_ACCU>(qk.row(m_cur_idx)); // lack of max of
			if(m_prev(m_cur_idx) > m_cur(m_cur_idx)){
				m_cur(m_cur_idx) = m_prev(m_cur_idx);
			}	
		}


		f = cm_pow(MATH_E, m_prev - m_cur);
		l_prev *= f;

		#pragma unroll
		for(int qk_idx=0; qk_idx<TILE_KV; qk_idx ++){
			p.column(qk_idx) =  cm_pow(MATH_E, (qk.column(qk_idx) - m_cur));
		}

		#pragma unroll
		for(int l_sum_idx=0; l_sum_idx<TILE_Q; l_sum_idx ++){
			l_cur(l_sum_idx) =  l_prev(l_sum_idx) + cm_sum<DT_ACCU>(p.row(l_sum_idx));  // p idx was wight?
		}
		
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
		// #pragma unroll
		// for(int acc_idx=0; acc_idx<HEAD_DIM; acc_idx ++){
		// 	acc.column(acc_idx)*= f;
		// }
		// #pragma unroll
		// for(int acc_idx=0; acc_idx<TILE_Q; acc_idx ++){
		// }


		#pragma unroll
		for(int s_idx=0; s_idx<TILE_Q; s_idx++){
			acc.row(s_idx)*= f.select<1, 1>(s_idx).replicate<HEAD_DIM>();
			for(int v_idx=0; v_idx<TILE_KV; v_idx++){
				vector<DT_ACCU, HEAD_DIM> v_fp32 = vector<DT_ACCU, HEAD_DIM>(input_v.row(v_idx));
				vector<DT_ACCU, HEAD_DIM> s_fp32 = vector<DT_ACCU, HEAD_DIM>(p.select<1, 1, 1, 1>(s_idx, v_idx).replicate<HEAD_DIM>());
				acc.select<1, 1, HEAD_DIM, 1>(s_idx, 0) += s_fp32 * v_fp32;
				// acc.row(s_idx) += s_fp32 * v_fp32;
			}
		}

		// #pragma unroll
		// for(int s_idx=0; s_idx<TILE_Q; s_idx ++){
		// 	for(int v_idx=0; v_idx<HEAD_DIM; v_idx ++){
		// 		vector<DT_ACCU, TILE_KV> s_fp32 = vector<DT_ACCU, TILE_KV>(p.row(s_idx));
		// 		vector<DT_ACCU, TILE_KV> v_fp32 = vector<DT_ACCU, TILE_KV>(input_v.column(v_idx));
		// 		acc.select<1, 1, 1, 1>(s_idx, v_idx) += cm_sum<DT_ACCU>(s_fp32 * v_fp32);
		// 	}
		// }

		m_prev = m_cur;
		l_prev = l_cur;
	}
	


// ------------------Process tail out of KV TILEs-------------------------
#if KV_TAIL_NUM

	#pragma unroll
	for(int t_kv=0; t_kv< KV_TAIL_NUM; t_kv++){   // Load Tile K/V of KV_SEQ
	// printf("Loop  Q : %d, Loop K/V: %d\n", i, j);
		k_offset = kv_head_count_offset + ((KV_ITER_NUM * TILE_KV  + t_kv) * HEAD_COUNT * 2)* HEAD_DIM * sizeof(DT) ;
		v_offset = kv_head_count_offset + ((KV_ITER_NUM * TILE_KV  + t_kv) * HEAD_COUNT * 2 + 1) * HEAD_DIM * sizeof(DT) ;
		// input_k_packed.row(t_kv)  = cm_load<uint32_t, (HEAD_DIM * sizeof(DT))/sizeof(uint32_t), DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, k_offset);
		// input_v_packed.row(t_kv)  = cm_load<uint32_t, (HEAD_DIM * sizeof(DT))/sizeof(uint32_t), DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, v_offset);
#if HEAD_DIM == 160
			input_k_packed.row(t_kv).select<64, 1>()   = cm_load<uint32_t, 64, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, k_offset);
			input_k_packed.row(t_kv).select<16, 1>(64) = cm_load<uint32_t, 16, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, k_offset+ 64 * sizeof(uint32_t));
			
			input_v_packed.row(t_kv).select<64, 1>()  = cm_load<uint32_t, 64, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, v_offset);
			input_v_packed.row(t_kv).select<16, 1>(64)  = cm_load<uint32_t, 16, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, v_offset+64 * sizeof(uint32_t));
#elif HEAD_DIM == 80	
			input_k_packed.row(t_kv).select<32, 1>()   = cm_load<uint32_t, 32, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, k_offset);
			input_k_packed.row(t_kv).select<8, 1>(32) = cm_load<uint32_t, 8, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, k_offset+ 32 * sizeof(uint32_t));
			
			input_v_packed.row(t_kv).select<32, 1>()  = cm_load<uint32_t, 32, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, v_offset);
			input_v_packed.row(t_kv).select<8, 1>(32)  = cm_load<uint32_t, 8, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, v_offset+32 * sizeof(uint32_t));
#elif HEAD_DIM == 40
			input_k_packed.row(t_kv).select<16, 1>()   = cm_load<uint32_t, 16, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, k_offset);
			input_k_packed.row(t_kv).select<4, 1>(16) = cm_load<uint32_t, 4, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, k_offset+ 16 * sizeof(uint32_t));
			
			input_v_packed.row(t_kv).select<16, 1>()  = cm_load<uint32_t, 16, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, v_offset);
			input_v_packed.row(t_kv).select<4, 1>(16)  = cm_load<uint32_t, 4, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, v_offset+16 * sizeof(uint32_t));
#else
			input_k_packed.row(t_kv)  = cm_load<uint32_t, (HEAD_DIM * sizeof(DT))/sizeof(uint32_t), DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, k_offset);
			input_v_packed.row(t_kv)  = cm_load<uint32_t, (HEAD_DIM * sizeof(DT))/sizeof(uint32_t), DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, v_offset);
#endif
	}

	// if (global_x == 0 && global_y == 0 && global_z==0)
	// {	
	// 	printf("KV_TAIL_NUM : %d\n", KV_TAIL_NUM);
	// 	// printf("input_k :\n");
	// 	// for (int x = 0; x < TILE_KV; x++){
	// 	// 	for (int y = 0; y < HEAD_DIM; y++)
	// 	// 	{
	// 	// 		printf("%f, ", input_k(x, y));
	// 	// 	}	
	// 	// 	printf("\n");
	// 	// }

	// 	printf("input_v :\n");
	// 	for (int x = 0; x < TILE_KV; x++){
	// 		for (int y = 0; y < HEAD_DIM; y++)
	// 		{
	// 			printf("%f, ", input_v(x, y));
	// 		}	
	// 		printf("\n");
	// 	}
	// }

	// Q*K
	qk = 0 - FLOAT_MAX;
	#pragma unroll
	for(int k_idx=0; k_idx<KV_TAIL_NUM; k_idx ++){
		vector<DT_ACCU, HEAD_DIM> k_fp32 = vector<DT_ACCU, HEAD_DIM>(input_k.row(k_idx));
		for(int q_idx=0; q_idx<TILE_Q; q_idx ++){
			vector<DT_ACCU, HEAD_DIM> q_fp32 = vector<DT_ACCU, HEAD_DIM>(input_q.row(q_idx));
			qk.select<1, 1, 1, 1>(q_idx, k_idx) = cm_sum<DT_ACCU>(q_fp32 * k_fp32);
		}
	}


	qk *=  (DT_ACCU)HEAD_SCALE;

	#pragma unroll
	for(int m_cur_idx=0; m_cur_idx<TILE_Q; m_cur_idx ++){
		m_cur(m_cur_idx) = cm_reduced_max<DT_ACCU>(qk.row(m_cur_idx)); // lack of max of
		if(m_prev(m_cur_idx) > m_cur(m_cur_idx)){
			m_cur(m_cur_idx) = m_prev(m_cur_idx);
		}	
	}



	f = cm_pow(MATH_E, m_prev - m_cur);
	l_prev *= f;

	p = 0;
	#pragma unroll
	for(int qk_idx=0; qk_idx<KV_TAIL_NUM; qk_idx ++){
		p.column(qk_idx) =  cm_pow(MATH_E, (qk.column(qk_idx) - m_cur));
	}

	#pragma unroll
	for(int l_sum_idx=0; l_sum_idx<TILE_Q; l_sum_idx ++){
		l_cur(l_sum_idx) =  l_prev(l_sum_idx) + cm_sum<DT_ACCU>(p.row(l_sum_idx));  // p idx was wight?
	}

	// 1. For flash attention
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
	#pragma unroll
	for(int acc_idx=0; acc_idx<HEAD_DIM; acc_idx ++){
		acc.column(acc_idx)*= f;
	}
	// #pragma unroll
	// for(int acc_idx=0; acc_idx<TILE_Q; acc_idx ++){
	// 	acc.row(acc_idx)*= f.replicate<HEAD_DIM, 0, 1 , 1>(acc_idx);
	// }


	#pragma unroll
	for(int s_idx=0; s_idx<TILE_Q; s_idx++){
		for(int v_idx=0; v_idx<KV_TAIL_NUM; v_idx++){
			vector<DT_ACCU, HEAD_DIM> v_fp32 = vector<DT_ACCU, HEAD_DIM>(input_v.row(v_idx));
			vector<DT_ACCU, HEAD_DIM> s_fp32 = vector<DT_ACCU, HEAD_DIM>(p.select<1, 1, 1, 1>(s_idx, v_idx).replicate<HEAD_DIM>());
			acc.select<1, 1, HEAD_DIM, 1>(s_idx, 0) += s_fp32 * v_fp32;
		}
	}

	// #pragma unroll
	// for(int s_idx=0; s_idx<TILE_Q; s_idx ++){
	// 	for(int v_idx=0; v_idx<HEAD_DIM; v_idx ++){
	// 		vector<DT_ACCU, TILE_KV> s_fp32 = vector<DT_ACCU, TILE_KV>(p.row(s_idx));
	// 		vector<DT_ACCU, TILE_KV> v_fp32 = vector<DT_ACCU, TILE_KV>(input_v.column(v_idx));
	// 		acc.select<1, 1, 1, 1>(s_idx, v_idx) += cm_sum<DT_ACCU>(s_fp32 * v_fp32);
	// 	}
	// }

	m_prev = m_cur;
	l_prev = l_cur;

#endif
// ------------------------------------------------------

	// if (global_x == 0 && global_y == 0 && global_z==0)
	// {	
	// 	printf("acc :\n");
	// 	for (int x = 0; x < TILE_Q; x++){
	// 		for (int y = 0; y < HEAD_DIM; y++)
	// 		{
	// 			printf("%f, ", acc(x, y));
	// 		}	
	// 		printf("\n");
	// 	}
	// }

	// #pragma unroll
	// for(int acc_idx=0; acc_idx<HEAD_DIM; acc_idx ++){
	// 	acc.column(acc_idx) *=  cm_inv(l_prev);
	// }

	#pragma unroll
	for(int acc_idx=0; acc_idx<TILE_Q; acc_idx ++){
		acc.row(acc_idx)*= cm_inv(l_prev.select<1, 1>(acc_idx).replicate<HEAD_DIM>());

	}


	matrix<DT, TILE_Q, HEAD_DIM> acc_out = acc;
	const uint32_t output_store_size = (HEAD_DIM * sizeof(DT)) / sizeof(uint32_t);

	#pragma unroll
	for(int t_q=0; t_q < TILE_Q; t_q++){  // Load Tile Q
		// printf("q_offset : %d \n", (i * TILE_Q  + t_q));
		output_offset = q_head_count_offset + t_q * HEAD_COUNT * HEAD_DIM * sizeof(DT) ;
		vector_ref<uint32_t, output_store_size> accu_0_packed = acc_out.row(t_q).format<uint32_t>();
#if HEAD_DIM == 160
		cm_store<uint32_t, 64, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, output_offset, accu_0_packed.select<64, 1>());
		cm_store<uint32_t, 16, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, output_offset + 64 * sizeof(uint32_t), accu_0_packed.select<16, 1>(64));
#elif HEAD_DIM == 80
		cm_store<uint32_t, 32, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, output_offset, accu_0_packed.select<32, 1>());
		cm_store<uint32_t, 8, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, output_offset + 32 * sizeof(uint32_t), accu_0_packed.select<8, 1>(32));
#elif HEAD_DIM == 40
		cm_store<uint32_t, 16, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, output_offset, accu_0_packed.select<16, 1>());
		cm_store<uint32_t, 4, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, output_offset + 16 * sizeof(uint32_t), accu_0_packed.select<4, 1>(16));
#else
		cm_store<uint32_t, output_store_size, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, output_offset, accu_0_packed);
#endif
	}
}
