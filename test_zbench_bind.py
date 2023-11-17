import os
import sys
sys.path.append(os.path.dirname(__file__) + "./build/Debug")

import numpy as np

import zbench

if __name__ == "__main__":
    arr1 = np.array([6, 7.5, 8,0, 1])
    arr2 = np.array([[1,2,3],[4,5,6]])
    cmt = zbench.test_bind(mode="bench", input="weights", A=16, B=arr1 )
    print(zbench.add(3, 4))
    cmt = zbench.test_get_json()
    print(cmt)
    print(zbench.test_take_json(cmt))
    
    