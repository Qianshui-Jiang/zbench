#include <cm/cm.h>


extern "C" _GENX_MAIN_ void
vadd(SurfaceIndex ibuf0,
    SurfaceIndex ibuf1,
    SurfaceIndex obuf)
{
    unsigned tid = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);

    vector<unsigned, 32> in0;
    vector<unsigned, 32> in1;
    read(ibuf0, tid * 32 * sizeof(unsigned), in0);
    read(ibuf1, tid * 32 * sizeof(unsigned), in1);
    vector<unsigned, 32> in2 = in0 + in1;
    write(obuf,  tid * 32 * sizeof(unsigned), in2);
}

