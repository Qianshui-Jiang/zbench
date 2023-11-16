import os
import sys
sys.path.append(os.path.dirname(__file__) + "./build/Debug")



import zbench
if __name__ == "__main__":
    print(zbench.add(3, 4))