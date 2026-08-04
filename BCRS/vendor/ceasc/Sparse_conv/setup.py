from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Bypass CUDA version mismatch check (host nvcc 12.x vs PyTorch 11.3)
BuildExtension._check_cuda_version = lambda self: None

setup(
    name='sparse_conv',
    ext_modules=[
        CUDAExtension('sparse_conv', 
        extra_compile_args={'cxx': [],"nvcc":["--extended-lambda"]},
        sources=[
            'sparse_conv_cuda.cpp',
            'sparse_conv_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
