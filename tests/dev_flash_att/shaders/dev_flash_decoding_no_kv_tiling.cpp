#include <cm/cm.h>
#include <cm/cmtl.h>

#define DT half
#define DT_ACCU float 

#define MATH_E 2.718281828459045235360287471352f
#define FLOAT_MAX 3.402823466e+38f

// In this kernel, Q_SEQ_LEN == TILE_Q == 1
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
	
	// printf("global_x : %d \n", global_x);
	// printf("global_y : %d \n", global_y);
	// printf("global_z : %d \n", global_z);

	// printf("Q_SEQ_LEN : %d \n", Q_SEQ_LEN);
	// printf("KV_SEQ_LEN : %d \n", KV_SEQ_LEN);
	// printf("HEAD_DIM : %d \n", HEAD_DIM);
	// printf("HEAD_COUNT : %d \n", HEAD_COUNT);
	// printf("TILE_Q : %d \n", TILE_Q);
	// printf("TILE_HEAD : %d \n", TILE_HEAD);
	// printf("HEAD_SCALE : %f \n", HEAD_SCALE);


    vector<DT, HEAD_DIM> input_q;
    vector_ref<uint32_t, HEAD_DIM/2> input_q_packed = input_q.format<uint32_t>();
    
	vector<DT, HEAD_DIM> input_k;
    vector_ref<uint32_t, HEAD_DIM/2> input_k_packed = input_k.format<uint32_t>();
	
	vector<DT, HEAD_DIM> input_v;
    vector_ref<uint32_t, HEAD_DIM/2> input_v_packed = input_v.format<uint32_t>();

	DT_ACCU qk;
	DT_ACCU p;

	DT_ACCU m_prev;  // m --> max
	DT_ACCU m_cur;   // m --> max 
	DT_ACCU f=0;     // f --> exp(m_prev - m_cur); 
	DT_ACCU l_prev;	 // l --> sum of exp(Xi-m)
	DT_ACCU l_cur;	 // l --> sum of exp(Xi-m)
	DT_ACCU l_rcp;	 // l --> sum of exp(Xi-m)

	vector<DT_ACCU, HEAD_DIM> acc;
									// Q_SEQ_LEN pralell						// Head count pralell
	const uint32_t threads_offset = (global_y * TILE_Q * HEAD_COUNT * HEAD_DIM + global_x * HEAD_DIM) * sizeof(DT) ;
	uint32_t q_offset = 0;
	uint32_t kv_offset = 0;
	uint32_t output_offset = 0;

	q_offset = threads_offset;
	input_q_packed  = cm_load<uint32_t, (HEAD_DIM * sizeof(DT))/sizeof(uint32_t), DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_q, q_offset);

	m_prev = 0 - FLOAT_MAX;
	l_prev = 0;
	acc = 0;

	// if (global_x == 0 && global_y == 0 && global_z==0)
	// {	
	// 	printf("input_q_packed :\n");
	// 	for (int y = 0; y < HEAD_DIM; y++)
	// 	{
	// 		printf("%f, ", input_q(y));
	// 	}	
	// 	printf("\n");
	// }

	for(int j=0; j<KV_SEQ_LEN; j++){  // Loop on tiled K/V --> Bc in paper
		// printf("Loop  Q : %d, Loop K/V: %d\n", i, j);
		kv_offset = j * HEAD_COUNT * HEAD_DIM * sizeof(DT) + global_x  * HEAD_DIM * sizeof(DT);
		input_k_packed  = cm_load<uint32_t, (HEAD_DIM * sizeof(DT))/sizeof(uint32_t), DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_k, kv_offset);
		input_v_packed  = cm_load<uint32_t, (HEAD_DIM * sizeof(DT))/sizeof(uint32_t), DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_v, kv_offset);

		// if (global_x == 0 && global_y == 0 && global_z==0)
		// {	
		// 	printf("input_k :\n");
		// 		for (int y = 0; y < HEAD_DIM; y++)
		// 		{
		// 			printf("%f, ", input_k(y));
		// 		}	
		// 		printf("\n");

		// 	printf("input_v :\n");
		// 		for (int y = 0; y < HEAD_DIM; y++)
		// 		{
		// 			printf("%f, ", input_v(y));
		// 		}	
		// 		printf("\n");
		// }

		// Q*K
		qk =  (DT_ACCU)HEAD_SCALE * cm_sum<DT_ACCU>(input_q * input_k);

		
		m_cur = qk; // lack of max of
		
		if(m_prev > m_cur){
			m_cur = m_prev;
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
		acc *= f ;

		acc += p * input_v;
		// for(int v_idx=0; v_idx<HEAD_DIM; v_idx ++){
			// acc(v_idx) += p * (DT_ACCU)input_v(v_idx);
		// }

		m_prev = m_cur;
		l_prev = l_cur;
	}
	// if (global_x == 1 && global_y == 0 && global_z==0)
	// {	
	// 	printf("acc :\n");
	// 	for (int y = 0; y < HEAD_DIM; y++)
	// 	{
	// 		printf("%f, ", acc(y));
	// 	}	
	// 	printf("\n");
	// }


	// 1. For flash attention
	// matrix<DT, TILE_Q, HEAD_DIM> acc_out = acc;
	
	// 2. For flash attention
	matrix<DT, TILE_Q, HEAD_DIM> acc_out = acc/l_prev;
	
	const uint32_t output_store_size = (HEAD_DIM * sizeof(DT)) / sizeof(uint32_t);
	output_offset = threads_offset;
	// printf("output_offset : %d \n", output_offset);
	vector_ref<uint32_t, output_store_size> accu_0_packed = acc_out.format<uint32_t>();
	cm_store<uint32_t, output_store_size, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, output_offset, accu_0_packed);

	
}
