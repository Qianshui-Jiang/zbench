#include "sgemm.h"
#ifndef _WIN32
#define FALSE 0
#define TRUE  1
#endif


int run_gemm(int m, int niterations, int gx, int gy)
{
    storage_type_t st = RowMajor;
    float alpha=+1.0, beta=+1.0;

    // Each thread computes 32x16 block of result matrix
    unsigned nthreadsY    = GEMM_BLOCK/32;
    unsigned nthreadsX    = GEMM_BLOCK/16;

    int n=m, k=m;

    // Initialization
    m = (m / TILE_m) * TILE_m;
    n=k=m;

    int lda = ((k+15)&~15);
    int ldb = ((n+15)&~15);
    int ldc = ldb;
    printf("SGEMM: C(%d, %d) = %.2f * C(%d, %d) + %.2f A(%d, %d) * B(%d, %d)\n", m,n,beta, m,n, alpha, m,k,k,n);
    printf("Row Threads:%d Col Threads:%d\n", nthreadsY, nthreadsX);
    printf("Thread-group setting: %d x %d \n", gx, gy);

    // Find a driver instance with a GPU device
    auto [hDriver, hDevice, hContext] = findDriverAndDevice();
    auto hCommandList = createImmCommandList(hContext, hDevice);

    // Allocate matrices
    Matrix A(m, k, lda, NULL, true, "A", st);
    Matrix B(k, n, ldb, NULL, true, "B", st);
    Matrix C_gold(m, n, ldc, NULL, false, "C_gold",  st);
    Matrix C(C_gold, "C");
    Matrix zero(C_gold, "C");

    if (niterations == 1) {
        printf("** validation run, only one iteration **\n");
        printf("** For performance run, add cmd-args: Sgemm 2048 1000 ** \n");
        // Compute gold result
        printf("Compute gold result\n");

        sgemmNxN(m, n, k, alpha, &A(0,0), A.l_dim(),
                 &B(0,0), B.l_dim(), beta, &C_gold(0,0), C_gold.l_dim());

    }
    else
        printf("CPU result not computed: Make #iterations=1 to compute CPU result\n");

    ze_image_format_t fmt = {ZE_IMAGE_FORMAT_LAYOUT_32, ZE_IMAGE_FORMAT_TYPE_FLOAT};
    auto hAImage = createImage2D(hContext, hDevice, hCommandList, fmt, A.l_dim(), m, &A(0,0));
    auto hBImage = createImage2D(hContext, hDevice, hCommandList, fmt, B.l_dim(), B.n_row(), &B(0,0));
    auto hCImage = createImage2D(hContext, hDevice, hCommandList, fmt, C.l_dim(), m, &C(0,0));

    ze_group_count_t launchArgs = {nthreadsX/gx, nthreadsY/gy, 1};
    const char* kname = "sgemm_kernel";

    char *BINNAME = "sgemm_genx.bin";
    auto hKernel = createKernel(hContext, hDevice, BINNAME, kname);

    L0_SAFE_CALL(zeKernelSetGroupSize(hKernel, gx, gy, 1));

    ze_event_handle_t hEvent = createEvent(hContext, hDevice);
    double thost = 0.0f;
    unsigned long long kernel_ns = 0;
    for (int i=0; i<niterations; i++)
        for(int ib=0; ib < m; ib += GEMM_BLOCK)
            for(int jb=0; jb < n; jb += GEMM_BLOCK)
                for(int kb=0; kb < k; kb += GEMM_BLOCK)
                {
                    setKernelArgs(hKernel, &m, &n, &k, &ib, &jb, &kb, &hAImage, &hBImage, &hCImage);
                    double host_start = getTimeStamp();
                    appendLaunchKernel(hCommandList, hKernel, &launchArgs, hEvent);
                    zeEventHostSynchronize(hEvent, std::numeric_limits<uint32_t>::max());

                    double host_end = getTimeStamp();
                    thost += (host_end - host_start);
                    ze_kernel_timestamp_result_t timestamp;
                    zeEventQueryKernelTimestamp(hEvent, &timestamp);
                    kernel_ns += (timestamp.context.kernelEnd - timestamp.context.kernelStart);

                    reset(hEvent);
                    reset(hCommandList);

                }
    // average time in msec
    thost = thost * 1000.0f / niterations;
    double tkern = kernel_ns / 1000000.0f / niterations;

    Matrix C_test(C_gold, "C");
    copyToMemory(hCommandList, (void*)&C_test(0,0), hCImage, hEvent);
    zeEventHostSynchronize(hEvent, std::numeric_limits<uint32_t>::max());

    printf("%-18s%.4lf msec\n","kern time:", tkern);
    printf("%-18s%.4lf msec\n","host time:", thost);

    double gflops;
    //gflops = ((2000.0f*m*n*k) / (1.0f*1024*1024*1024)) / tkern;
    //printf("GEN SGEMM (kern-timer): %8.2lf Gflops\n",  gflops);
    gflops = ((2000.0f*m*n*k) / (1.0f*1024*1024*1024)) / thost;
    printf("GEN SGEMM (host-timer): %8.2lf Gflops\n", gflops);

    // We do not initialize result matrix C to zero after each run
    // So check result only when niterations=1; Higher niterations is used
    // to get average performance number.
    bool pass=FALSE;
    if (niterations == 1) {
        if(C_test == C_gold) {
            printf("PASSED\n");
            pass = TRUE;
	    } else
            printf("FAILED\n");
    } else
        printf("Result not checked - make #iterations=1 to check result!\n");
    printf("----------------------------\n");

    destroy(hCImage);
    destroy(hBImage);
    destroy(hAImage);

    destroy(hCommandList);
    destroy(hContext);
    return pass ? 0 : 1;
}


int main(int argc, char** argv)
{
    int m = GEMM_BLOCK;
    int niterations = 1;
    if( argc == 3 ) {
        m = atoi(argv[1]);
        niterations = atoi(argv[2]);
    }

    int success = 0;
    if (niterations == 1)
        success |= run_gemm( m, niterations, 1, 4 );
    else {
        int success = 0;
        success |= run_gemm( m, niterations, 1, 1 );
        success |= run_gemm( m, niterations, 1, 4 );
        success |= run_gemm( m, niterations, 4, 1 );
        success |= run_gemm( m, niterations, 2, 2 );
        success |= run_gemm( m, niterations, 1, 8 );
        success |= run_gemm( m, niterations, 8, 1 );
        success |= run_gemm( m, niterations, 2, 4 );
        success |= run_gemm( m, niterations, 4, 2 );
    }
    return success;
}
