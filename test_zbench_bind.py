import os
import sys
sys.path.append(os.path.dirname(__file__) + "./build/Debug")

import numpy as np

import zbench

if __name__ == "__main__":
    source_bin_path = r"C:\Users\12900K\Documents\Engneeing_works\dml_workspace\zbench\build\Debug"
    # arr1 = np.array([6, 7.5, 8,0, 1])
    # arr2 = np.array([[1,2,3],[4,5,6]])
    # cmt = zbench.test_bind(mode="bench", input="weights", A=16, B=arr1 )
    # print(zbench.add(3, 4))
    # cmt = zbench.test_get_json()
    # print(cmt)
    # print(zbench.test_take_json(cmt))
    # print(zbench.run_sgemm(m=1024, niterations= 1, gy=1, gx=4, 
    #                        bin_file=os.path.join(source_bin_path, "sgemm_genx.bin"),
    #                        fn_name = "sgemm_kernel"
    #                        )
    #       )
    print(zbench.run_bgemm( M = 128, N= 128,K= 128,
                            threadWidth= 4, threadHeight= 4,
                            groupWidth= 1, groupHeight= 1,
                            bin_file= os.path.join(source_bin_path, "bgemm_dpas_genx.bin"), 
                            fn_name= "bgemm_dpas"
                            )
          )