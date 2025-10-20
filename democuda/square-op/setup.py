

from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


setup(
    name="squareCUDA",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="squareCUDA._C",
            sources=["ext.cpp", "square.cu"],
            
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
