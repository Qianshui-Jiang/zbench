#include <cm/cm.h>
#include <cm/cmtl.h>

#define DT half
#define DT_ACCU float 

#define MATH_E 2.718281828459045235360287471352f
// #define MATH_E 2.71828182845904523536028747135266249775724709369995f

#define FLOAT_MAX 3.402823466e+38f
#define KV_PER_THREAD (KV_SEQ_LEN / SPLIT_KV)
#define LD_ST_SIZE ((HEAD_DIM * sizeof(DT))/sizeof(DT_ACCU))

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

// TODO optimization: 
// --> SLM in FP16 datatype for more KV split

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

// if (global_x == 0 && global_y == 0 && global_z==3)
// {	
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
	// printf("HEAD_SCALE : %f \n", (DT_ACCU)HEAD_SCALE);
	// printf("KV_PER_THREAD : %d \n", KV_PER_THREAD);
// }

    vector<DT, HEAD_DIM> input_q;
    vector_ref<uint32_t, HEAD_DIM/2> input_q_packed = input_q.format<uint32_t>();
    
	vector<DT, HEAD_DIM> input_k;
    vector_ref<uint32_t, HEAD_DIM/2> input_k_packed = input_k.format<uint32_t>();
	
	vector<DT, HEAD_DIM> input_v;
    vector_ref<uint32_t, HEAD_DIM/2> input_v_packed = input_v.format<uint32_t>();

	DT_ACCU qk;         // 
	DT_ACCU p;
	DT_ACCU m_prev= 0 - FLOAT_MAX;  // m --> max
	DT_ACCU m_cur;      // m --> max
	DT_ACCU f = 0;      // f --> exp(m_prev - m_cur); 
	DT_ACCU l_prev = 0;	// l --> sum of exp(Xi-m)
	DT_ACCU l_cur;	    // l --> sum of exp(Xi-m)
	DT_ACCU l_rcp;	    // l --> sum of exp(Xi-m)
	vector<DT_ACCU, HEAD_DIM> acc(0);
	vector<DT_ACCU, SPLIT_KV> calibre_sum_exp;

	uint32_t q_offset = (global_y * TILE_Q * HEAD_COUNT * HEAD_DIM + global_x * HEAD_DIM) * sizeof(DT) ;
	uint32_t output_offset = (global_y * TILE_Q * HEAD_COUNT * HEAD_DIM + global_x * HEAD_DIM) * sizeof(DT) ;
	uint32_t kv_offset = 0;


	input_q_packed = cm_load<uint32_t, LD_ST_SIZE, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_q, q_offset);

	// if (global_x == 1 && global_y == 0 && global_z==0)
	// {	
	// 	printf("input_q_packed :\n");
	// 	for (int y = 0; y < HEAD_DIM; y++)
	// 	{
	// 		printf("%f, ", input_q(y));
	// 	}	
	// 	printf("\n");
	// }

	// #pragma unroll
	for(int j=0; j<KV_PER_THREAD; j++){  // Loop on tiled K/V --> Bc in paper
		// simplified matmul by cm_mul
		// kv_offset = (global_x * KV_SEQ_LEN + global_z * KV_PER_THREAD + j * TILE_KV  + t_kv ) * HEAD_DIM * sizeof(DT) ;
		kv_offset = j * HEAD_COUNT * HEAD_DIM * sizeof(DT) + global_z * KV_PER_THREAD * HEAD_COUNT * HEAD_DIM * sizeof(DT) + global_x  * HEAD_DIM * sizeof(DT);
		input_k_packed = cm_load<uint32_t, LD_ST_SIZE, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_k, kv_offset);
		input_v_packed = cm_load<uint32_t, LD_ST_SIZE, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_v, kv_offset);

		// if (global_x == 0 && global_y == 0 && global_z==3)
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
		// vector<DT_ACCU, HEAD_DIM> q_fp32 = vector<DT_ACCU, HEAD_DIM>(input_q);
		// vector<DT_ACCU, HEAD_DIM> k_fp32 = vector<DT_ACCU, HEAD_DIM>(input_k);
		qk =   (DT_ACCU)HEAD_SCALE * cm_sum<DT_ACCU>(input_q * input_k);

		m_cur = qk; 
		if(m_prev > m_cur){
			m_cur = m_prev;
		}	
		

		f = cm_pow(MATH_E, m_prev - m_cur);
		// f = cm_exp(m_prev - m_cur);
		l_prev *= f ;

		p =  cm_pow(MATH_E, (qk - m_cur));
		// p =  cm_exp((qk - m_cur));
		l_cur =  l_prev + p;  // p idx was wight?


		// 1. For flash attention
		// l_rcp = 1/l_cur;
		// p *=  l_rcp;  // s
		// acc *=  (l_prev * l_rcp);

		// 2. For flash attention 2
		acc *= f;

		// S*V
		// #pragma unroll
		// for(int v_idx=0; v_idx<TILE_KV; v_idx ++){
		// 	vector<DT_ACCU, HEAD_DIM> s_fp32 = vector<DT_ACCU, HEAD_DIM>(p.select<1, 1>(v_idx).replicate<HEAD_DIM>()); 
		// 	vector<DT_ACCU, HEAD_DIM> v_fp32 = vector<DT_ACCU, HEAD_DIM>(input_v.row(v_idx));
		// 	acc += v_fp32 * s_fp32;
		// }

		acc += p * input_v;

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
// if(global_z==0){ 

	matrix<DT_ACCU, SPLIT_KV, HEAD_DIM>  all_accs = cm_load_slm<DT_ACCU, HEAD_DIM * SPLIT_KV>(0);
	vector<DT_ACCU, SPLIT_KV> all_maxs = cm_load_slm<DT_ACCU, SPLIT_KV>(HEAD_DIM * SPLIT_KV * sizeof(DT_ACCU));
	vector<DT_ACCU, SPLIT_KV> all_sum_exp = cm_load_slm<DT_ACCU, SPLIT_KV>((HEAD_DIM * SPLIT_KV + SPLIT_KV )* sizeof(DT_ACCU));

	// if (global_x == 0 && global_y == 0 && global_z==0)
	// {	
	// 	printf("all_accs :\n");
	// 	// for (int y = 0; y < HEAD_DIM; y++)
	// 	// {
	// 	// 	printf("%f, ", all_accs(0, y));
	// 	// }	
	// 	// printf("\n");
	// 	// for (int y = 0; y < SPLIT_KV; y++)
	// 	// {
	// 	// 	printf("%f, ", all_maxs(y));
	// 	// }	
	// 	// printf("m_cur: %f, ", m_cur);
	// 	printf("l_cur: %f, ", l_cur);
	// // 	printf("\nall_maxs :\n");
	// // 	printf("\n");
	// 	printf("all_sum_exp :\n");
	// 	for (int y = 0; y < SPLIT_KV; y++)
	// 	{
	// 		printf("%f, ", all_sum_exp(y));
	// 	}	
	// // 	printf("\n");
	// }

	DT_ACCU global_max = cm_reduced_max<DT_ACCU>(all_maxs);
	calibre_sum_exp = all_sum_exp*cm_pow(MATH_E, all_maxs - global_max);
	DT_ACCU global_sum = cm_sum<DT_ACCU>(calibre_sum_exp);
	// global_sum = cm_pow(MATH_E, m_cur - global_max) * cm_inv(global_sum);

	// PASS 2 final output
	#pragma unroll
	for(int ic=0; ic<SPLIT_KV; ic++){
		all_accs.row(ic) *= all_sum_exp(ic);
		all_accs.row(ic) *= cm_pow(MATH_E, all_maxs(ic) - global_max); // Calibrate local output
		all_accs.row(ic) *= cm_inv(global_sum); // Calculate division to SUM
	}
	// for(int r=0; r<HEAD_DIM; r++){
	// 	acc(r) = cm_sum<DT_ACCU>(all_accs.column(r)); // Calibrate local output
	// }
	#pragma unroll
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
// }
}