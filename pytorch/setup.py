from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

srcdir = os.path.dirname(os.path.realpath(__file__))

setup(
    name="madrona_pytorch",
    ext_modules=[
        CUDAExtension(
            name="madrona_pytorch",
            sources=[os.path.join(srcdir, "pytorch.cpp")],
            extra_compile_args=[],
            extra_link_args=[],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
