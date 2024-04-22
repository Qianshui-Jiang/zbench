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

// Single Thread flash attention 2 for demostration.

// #define Q_SEQ_LEN
// #define KV_SEQ_LEN
// #define HEAD_DIM
// #define TILE_Q
// #define TILE_KV
// #define TILE_HEAD


// 使用FP16来进行acc的中间计算，
// SUM如何避免一些iteration，如何更好地使用vactorization来进行相关
extern "C" _GENX_MAIN_ void flash_att(
		SurfaceIndex surface_input_q [[type("buffer_t half")]],
		SurfaceIndex surface_input_k [[type("buffer_t half")]],
		SurfaceIndex surface_input_v [[type("buffer_t half")]],
		SurfaceIndex surface_output [[type("buffer_t half")]]
)
{

    const uint32_t global_x = cm_group_id(0) * LWS_SIZE_X + cm_local_id(0);
    const uint32_t global_y = cm_group_id(1) * LWS_SIZE_Y + cm_local_id(1);
    const uint32_t global_z = cm_group_id(2) * LWS_SIZE_Z + cm_local_id(2);
	
	printf("global_x : %d \n", global_x);
	printf("global_y : %d \n", global_y);
	printf("global_z : %d \n", global_z);

	printf("Q_SEQ_LEN : %d \n", Q_SEQ_LEN);
	printf("KV_SEQ_LEN : %d \n", KV_SEQ_LEN);
	printf("HEAD_DIM : %d \n", HEAD_DIM);
	printf("TILE_Q : %d \n", TILE_Q);
	printf("TILE_KV : %d \n", TILE_KV);
	printf("TILE_HEAD : %d \n", TILE_HEAD);


    matrix<DT, TILE_Q, HEAD_DIM> input_q;
    matrix_ref<uint32_t, TILE_Q, HEAD_DIM/2> input_q_packed = input_q.format<uint32_t, TILE_Q, HEAD_DIM/2>();
    
	matrix<DT, TILE_KV, HEAD_DIM> input_k;
    matrix_ref<uint32_t, TILE_KV, HEAD_DIM/2> input_k_packed = input_k.format<uint32_t, TILE_KV, HEAD_DIM/2>();
	
	matrix<DT, TILE_KV, HEAD_DIM> input_v;
    matrix_ref<uint32_t, TILE_KV, HEAD_DIM/2> input_v_packed = input_v.format<uint32_t, TILE_KV, HEAD_DIM/2>();


	matrix<DT_ACCU, TILE_Q, TILE_KV> qk; // 4x2

	matrix<DT_ACCU, TILE_Q, TILE_KV> p;
	vector<DT_ACCU, TILE_Q> m_prev;  // m --> max
	vector<DT_ACCU, TILE_Q> m_cur;  // m --> max
	vector<DT_ACCU, TILE_Q> l_prev;	// l --> sum of exp(Xi-m)
	vector<DT_ACCU, TILE_Q> l_cur;	// l --> sum of exp(Xi-m)
	vector<DT_ACCU, TILE_Q> l_rcp;	// l --> sum of exp(Xi-m)
	matrix<DT_ACCU, TILE_Q, HEAD_DIM> acc;

	const uint32_t threads_offset = 0;
	// const uint32_t threads_offset = (global_x * ITEMNUM_PER_HW + global_y * INOUT_WIDTH + global_z * (INOUT_WIDTH * INOUT_HEIGHT)) * sizeof(DT);
	uint32_t q_offset = threads_offset + (BASE_INPUT_OFFSET * sizeof(DT));
	uint32_t k_offset = threads_offset + (BASE_INPUT_OFFSET * sizeof(DT));
	uint32_t v_offset = threads_offset + (BASE_INPUT_OFFSET * sizeof(DT));
	uint32_t output_offset = threads_offset + (BASE_OUTPUT_OFFSET * sizeof(DT));

	for(int i=0; i < Q_SEQ_LEN/TILE_Q; i++){ // Loop on tiled Q --> Br in paper
	// for(int i=0; i < 1; i++){ // Loop on tiled Q --> Br in paper

		for(int t_q=0; t_q< TILE_Q; t_q++){  // Load Tile Q
			// printf("q_offset : %d \n", (i * TILE_Q  + t_q));
			q_offset = (i * TILE_Q  + t_q) * HEAD_DIM * sizeof(DT) ;
			input_q_packed.row(t_q)  = cm_load<uint32_t, (HEAD_DIM * sizeof(DT))/sizeof(uint32_t), DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_q, q_offset);
		}

		m_prev = 0 - FLOAT_MAX;
		l_prev = 0;
		acc = 0;

		// if (global_x == 0 && global_y == 0 && global_z==0)
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

		for(int j=0; j<KV_SEQ_LEN/TILE_KV; j++){  // Loop on tiled K/V --> Bc in paper
		// for(int j=0; j<1; j++){  // Loop on tiled K/V --> Bc in paper
			for(int t_kv=0; t_kv< TILE_KV; t_kv++){   // Load Tile K/V
			// printf("Loop  Q : %d, Loop K/V: %d\n", i, j);
				k_offset = (j * TILE_KV  + t_kv) * HEAD_DIM * sizeof(DT) ;
				v_offset = (j * TILE_KV  + t_kv) * HEAD_DIM * sizeof(DT) ;
				input_k_packed.row(t_kv)  = cm_load<uint32_t, (HEAD_DIM * sizeof(DT))/sizeof(uint32_t), DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_k, k_offset);
				input_v_packed.row(t_kv)  = cm_load<uint32_t, (HEAD_DIM * sizeof(DT))/sizeof(uint32_t), DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_v, v_offset);
			}

			// if (global_x == 0 && global_y == 0 && global_z==0)
			// {	
			// 	printf("input_k :\n");
			// 	for (int x = 0; x < TILE_KV; x++){
			// 		for (int y = 0; y < HEAD_DIM; y++)
			// 		{
			// 			printf("%f, ", input_k(x, y));
			// 		}	
			// 		printf("\n");
			// 	}

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
			for(int k_idx=0; k_idx<TILE_KV; k_idx ++){
				for(int q_idx=0; q_idx<TILE_Q; q_idx ++){
					vector<DT_ACCU, HEAD_DIM> q_fp32 = vector<DT_ACCU, HEAD_DIM>(input_q.row(q_idx));
					vector<DT_ACCU, HEAD_DIM> k_fp32 = vector<DT_ACCU, HEAD_DIM>(input_k.row(k_idx));
					qk.select<1, 1, 1, 1>(q_idx, k_idx) = cm_sum<DT_ACCU>(q_fp32 * k_fp32);
				}
			}

			for(int m_cur_idx=0; m_cur_idx<TILE_Q; m_cur_idx ++){
				m_cur(m_cur_idx) = cm_reduced_max<DT_ACCU>(qk.row(m_cur_idx)); // lack of max of
			}
			
			for(int m_cur_idx=0; m_cur_idx<TILE_Q; m_cur_idx ++){
				if(m_prev(m_cur_idx) > m_cur(m_cur_idx)){
					m_cur(m_cur_idx) = m_prev(m_cur_idx);
				}	
			}

			l_prev *= cm_pow(MATH_E, m_prev - m_cur);


			for(int qk_idx=0; qk_idx<TILE_KV; qk_idx ++){
				p.column(qk_idx) =  cm_pow(MATH_E, (qk.column(qk_idx) - m_cur));
			}


			for(int l_sum_idx=0; l_sum_idx<TILE_Q; l_sum_idx ++){
				l_cur(l_sum_idx) =  l_prev(l_sum_idx) + cm_sum<DT_ACCU>(p.row(l_sum_idx));  // p idx was wight?
			}

			l_rcp = 1/l_cur;

			for(int p_idx=0; p_idx<TILE_KV; p_idx ++){
				p.column(p_idx) *=  l_rcp;  // s
			}
			
			for(int acc_idx=0; acc_idx<TILE_HEAD; acc_idx ++){
				acc.column(acc_idx) *=  (l_rcp*l_prev);
			}



			for(int s_idx=0; s_idx<TILE_Q; s_idx ++){
				for(int v_idx=0; v_idx<HEAD_DIM; v_idx ++){
					vector<DT_ACCU, TILE_KV> s_fp32 = vector<DT_ACCU, TILE_KV>(p.row(s_idx));
					vector<DT_ACCU, TILE_KV> v_fp32 = vector<DT_ACCU, TILE_KV>(input_v.column(v_idx));
					acc.select<1, 1, 1, 1>(s_idx, v_idx) += cm_sum<DT_ACCU>(s_fp32 * v_fp32);
				}
			}

			m_prev = m_cur;
			l_prev = l_cur;
		}

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


		matrix<DT, TILE_Q, HEAD_DIM> acc_out = acc;
	 	const uint32_t output_store_size = (HEAD_DIM * sizeof(DT)) / sizeof(uint32_t);
		for(int t_q=0; t_q < TILE_Q; t_q++){  // Load Tile Q
			// printf("q_offset : %d \n", (i * TILE_Q  + t_q));
			output_offset = (i * TILE_Q  + t_q) * HEAD_DIM * sizeof(DT) ;
			vector_ref<uint32_t, output_store_size> accu_0_packed = acc_out.row(t_q).format<uint32_t>();
			cm_store<uint32_t, output_store_size, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, output_offset, accu_0_packed);
		}
	}

	// // PASS 1.0, Get partial max. and partial D, of ITEMNUM_PER_HW threads (Tile size)
	// vector<DT_ACCU, ITEMNUM_PER_HW> my_data_f32 = vector<DT_ACCU, ITEMNUM_PER_HW>(my_data);

	// DT_ACCU my_local_max = cm_reduced_max<DT_ACCU>(my_data_f32);
	// surface_output = surface_input_q;
	
}
