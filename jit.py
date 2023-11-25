from torch.utils.cpp_extension import load
lltm_cpp = load(name="zbench", sources=["lltm.cpp"], verbose=True)
help(lltm_cpp)
