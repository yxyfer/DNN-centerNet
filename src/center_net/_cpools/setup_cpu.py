from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="cpools",
    ext_modules=[
        CppExtension("top_pool", ["src_cpu/top_pool.cpp"]),
        CppExtension("bottom_pool", ["src_cpu/bottom_pool.cpp"]),
        CppExtension("left_pool", ["src_cpu/left_pool.cpp"]),
        CppExtension("right_pool", ["src_cpu/right_pool.cpp"])
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
