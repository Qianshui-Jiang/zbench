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
#define KV_PER_THREAD (KV_SEQ_LEN / SPLIT_KV)

// In this kernel, Q_SEQ_LEN == TILE_Q == 1
// -> x thread group axis for Head_count & Batch parallel
// -> y thread group axis for Q_SEQ_LEN parallel, for TILE_Q items per thread
// -> z thread group axis for KV_SEQ_LEN parallel and reduce, speciall for flash decoding (flat shape MHA)

// #define Q_SEQ_LEN
// #define KV_SEQ_LEN
// #define HEAD_DIM
// #define TILE_Q
// #define TILE_KV
// #define TILE_HEAD
// #define HEAD_SCALE 
// #define SPLIT_KV 

#define LD_ST_SIZE ((HEAD_DIM * sizeof(DT))/sizeof(uint32_t))

// simplified matmul by cm_mul
extern "C" _GENX_MAIN_ void flash_decoding(
		SurfaceIndex surface_input_q [[type("buffer_t half")]],
		SurfaceIndex surface_input_k [[type("buffer_t half")]],
		SurfaceIndex surface_input_v [[type("buffer_t half")]],
		SurfaceIndex surface_output [[type("buffer_t half")]]
)
{
	// use SLM for final output reduce, need 3 spaces for LWS
    //  1st part:  local output --> (o: Q_SEQ_LEN x HEAD_DIM x SPLIT_KV)
    //  2nd part:  local max value of each thread --> (m: 1 scalar x SPLIT_KV)
    //  3rd part:  local sum of the exp(x-m_local) values per thread --> (l: 1 scalar x SPLIT_KV)
	cm_slm_init((HEAD_DIM + 1 + 1) *  SPLIT_KV * sizeof(DT_ACCU));  
    uint slm_buffer = cm_slm_alloc((HEAD_DIM + 1 + 1) *  SPLIT_KV * sizeof(DT_ACCU));
	


    const uint32_t global_x = cm_group_id(0) * LWS_SIZE_X + cm_local_id(0);
    const uint32_t global_y = cm_group_id(1) * LWS_SIZE_Y + cm_local_id(1);
    const uint32_t global_z = cm_group_id(2) * LWS_SIZE_Z + cm_local_id(2);
	
	// printf("global_x : %d \n", global_x);
	// printf("global_y : %d \n", global_y);
	// printf("global_z : %d \n", global_z);

	// printf("Q_SEQ_LEN : %d \n", Q_SEQ_LEN);
	// printf("KV_SEQ_LEN : %d \n", KV_SEQ_LEN);
	// printf("SPLIT_KV : %d \n", SPLIT_KV);
	
	// printf("HEAD_DIM : %d \n", HEAD_DIM);
	// printf("TILE_Q : %d \n", TILE_Q);
	// printf("TILE_KV : %d \n", TILE_KV);
	// printf("TILE_HEAD : %d \n", TILE_HEAD);
	// printf("HEAD_SCALE : %f \n", HEAD_SCALE);


    vector<DT, HEAD_DIM> input_q;
    vector_ref<uint32_t, HEAD_DIM/2> input_q_packed = input_q.format<uint32_t>();
    
	matrix<DT, TILE_KV, HEAD_DIM> input_k;
    matrix_ref<uint32_t, TILE_KV, HEAD_DIM/2> input_k_packed = input_k.format<uint32_t, TILE_KV, HEAD_DIM/2>();
	
	matrix<DT, TILE_KV, HEAD_DIM> input_v;
    matrix_ref<uint32_t, TILE_KV, HEAD_DIM/2> input_v_packed = input_v.format<uint32_t, TILE_KV, HEAD_DIM/2>();

	vector<DT_ACCU, TILE_KV> qk;         // 
	vector<DT_ACCU, TILE_KV> p;

	DT_ACCU m_prev= 0 - FLOAT_MAX;;  // m --> max
	DT_ACCU m_cur;      // m --> max
	DT_ACCU f = 0;      // f --> exp(m_prev - m_cur); 
	DT_ACCU l_prev = 0;	// l --> sum of exp(Xi-m)
	DT_ACCU l_cur;	// l --> sum of exp(Xi-m)
	DT_ACCU l_rcp;	// l --> sum of exp(Xi-m)
	vector<DT_ACCU, HEAD_DIM> acc(0);

	uint32_t q_offset = (global_x * Q_SEQ_LEN * HEAD_DIM + global_y * TILE_Q * HEAD_DIM) * sizeof(DT) ;
	uint32_t output_offset = (global_x * Q_SEQ_LEN * HEAD_DIM + global_y * TILE_Q * HEAD_DIM) * sizeof(DT) ;
	uint32_t kv_offset = 0;


	input_q_packed  = cm_load<uint32_t, LD_ST_SIZE, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_q, q_offset);

	// if (global_x == 0 && global_y == 0 && global_z==1)
	// {	
	// 	printf("input_q_packed :\n");
	// 	for (int y = 0; y < HEAD_DIM; y++)
	// 	{
	// 		printf("%f, ", input_q(y));
	// 	}	
	// 	printf("\n");
	// }

	#pragma unroll
	for(int j=0; j<KV_PER_THREAD/TILE_KV; j++){  // Loop on tiled K/V --> Bc in paper
	// for(int j=0; j<1; j++){  // Loop on tiled K/V --> Bc in paper
		#pragma unroll
		for(int t_kv=0; t_kv < TILE_KV; t_kv++){   // Load Tile K/V
			kv_offset = (global_x * KV_SEQ_LEN + global_z * KV_PER_THREAD + j * TILE_KV  + t_kv ) * HEAD_DIM * sizeof(DT) ;
			input_k_packed.row(t_kv)  = cm_load<uint32_t, LD_ST_SIZE, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_k, kv_offset);
			input_v_packed.row(t_kv)  = cm_load<uint32_t, LD_ST_SIZE, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_v, kv_offset);

		}


		// if (global_x == 0 && global_y == 0 && global_z==1)
		// {	
		// 	printf("input_k :\n");
		// 	for (int x = 0; x < TILE_KV; x++){
		// 		for (int y = 0; y < HEAD_DIM; y++)
		// 		{
		// 			printf("%f, ", input_k(x, y));
		// 		}	
		// 		printf("\n");
		// 	}

		// 	// printf("input_v :\n");
		// 	// for (int x = 0; x < TILE_KV; x++){
		// 	// 	for (int y = 0; y < HEAD_DIM; y++)
		// 	// 	{
		// 	// 		printf("%f, ", input_v(x, y));
		// 	// 	}	
		// 	// 	printf("\n");
		// 	// }
		// }

		// Q*K
		#pragma unroll
		for(int k_idx=0; k_idx<TILE_KV; k_idx ++){
			vector<DT_ACCU, HEAD_DIM> q_fp32 = vector<DT_ACCU, HEAD_DIM>(input_q);
			vector<DT_ACCU, HEAD_DIM> k_fp32 = vector<DT_ACCU, HEAD_DIM>(input_k.row(k_idx));
			qk(k_idx) =   (DT_ACCU)HEAD_SCALE * cm_sum<DT_ACCU>(q_fp32 * k_fp32);
		}

		m_cur = cm_reduced_max<DT_ACCU>(qk); // lack of max 
		
		if(m_prev > m_cur){
			m_cur = m_prev;
		}	
		

		f = cm_pow(MATH_E, m_prev - m_cur);
		l_prev *= f ;

		p =  cm_pow(MATH_E, (qk - m_cur));
		l_cur =  l_prev + cm_sum<DT_ACCU>(p);  // p idx was wight?


		// 1. For flash attention
		// l_rcp = 1/l_cur;
		// p *=  l_rcp;  // s
		// acc *=  (l_prev * l_rcp);

		// 2. For flash attention 2
		acc *= f;
	



		#pragma unroll
		for(int v_idx=0; v_idx<HEAD_DIM; v_idx ++){
			vector<DT_ACCU, TILE_KV> s_fp32 = vector<DT_ACCU, TILE_KV>(p);
			vector<DT_ACCU, TILE_KV> v_fp32 = vector<DT_ACCU, TILE_KV>(input_v.column(v_idx));
			acc(v_idx) += cm_sum<DT_ACCU>(s_fp32 * v_fp32);
		}

		m_prev = m_cur;
		l_prev = l_cur;
	}
	acc = acc/l_prev;

	// // output reduce, from each KV chunck
	cm_store_slm<DT_ACCU, HEAD_DIM>(global_z * HEAD_DIM * sizeof(DT_ACCU), acc);
	cm_store_slm<DT_ACCU, 1>((HEAD_DIM * SPLIT_KV + global_z) * sizeof(DT_ACCU), m_cur);
	cm_store_slm<DT_ACCU, 1>((HEAD_DIM * SPLIT_KV + SPLIT_KV + global_z) * sizeof(DT_ACCU), l_cur);
	cm_slm_fence(CM_GLOBAL_COHERENT_FENCE);
	cm_barrier();
	// // read from slm and further reduce

	 
	matrix<DT_ACCU, SPLIT_KV, HEAD_DIM>  all_accs = cm_load_slm<DT_ACCU, HEAD_DIM * SPLIT_KV>(0);
	vector<DT_ACCU, SPLIT_KV> all_maxs = cm_load_slm<DT_ACCU, SPLIT_KV>(HEAD_DIM * SPLIT_KV * sizeof(DT_ACCU));
	vector<DT_ACCU, SPLIT_KV> all_esums = cm_load_slm<DT_ACCU, SPLIT_KV>((HEAD_DIM * SPLIT_KV + SPLIT_KV )* sizeof(DT_ACCU));

	// if (global_x == 0 && global_y == 0 && global_z==0)
	// {	
	// 	// printf("all_accs :\n");
	// 	// for (int y = 0; y < HEAD_DIM; y++)
	// 	// {
	// 	// 	printf("%f, ", all_accs(0, y));
	// 	// }	
	// 	printf("\n");
	// 	for (int y = 0; y < SPLIT_KV; y++)
	// 	{
	// 		printf("%f, ", all_maxs(y));
	// 	}	
	// 	printf("m_cur: %f, ", m_cur);
	// // 	printf("l_cur: %f, ", l_cur);
	// // 	printf("\nall_maxs :\n");
	// // 	printf("\n");
	// // 	printf("all_esums :\n");
	// // 	for (int y = 0; y < SPLIT_KV; y++)
	// // 	{
	// // 		printf("%f, ", all_esums(y));
	// // 	}	
	// // 	printf("\n");
	// }

	DT_ACCU global_max = cm_reduced_max<DT_ACCU>(all_maxs);
	all_maxs = all_maxs - global_max;
	all_maxs = all_esums * cm_pow(MATH_E, all_maxs);
	DT_ACCU global_sum = cm_sum<DT_ACCU>(all_maxs);

	// PASS 2 final output
	for(int ic=0; ic<SPLIT_KV; ic++){
		all_accs.row(ic) *= all_esums(ic);
		all_accs.row(ic) *= cm_pow(MATH_E, m_cur - global_max); // Calibrate local output
		all_accs.row(ic) *= cm_inv(global_sum); // Calculate division to SUM
	}



	for(int r=1; r<SPLIT_KV; r++){
		all_accs.row(0) += all_accs.row(r); // Calibrate local output
	}

	// if (global_x == 1 && global_y == 0 && global_z==0){
	// 	// local_output_f32 = local_output_f32 * cm_exp(local_max - global_max); // Calibrate local output use cm_exp

	// 	// 1. For flash attention
	// 	// matrix<DT, TILE_Q, HEAD_DIM> acc_out = acc;
		
	// 	// 2. For flash attention2
	// 		printf("acc :\n");
	// 		for (int y = 0; y < HEAD_DIM; y++)
	// 		{
	// 			printf("%f, ", acc(y));
	// 		}	
	// 		printf("\n");
	// }
	vector<DT, HEAD_DIM> acc_out =  all_accs.row(0);
	vector_ref<uint32_t, LD_ST_SIZE> accu_0_packed = acc_out.format<uint32_t>();
	cm_store<uint32_t, LD_ST_SIZE, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, output_offset, accu_0_packed);	
}
