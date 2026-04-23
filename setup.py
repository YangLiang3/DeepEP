import os
import shutil
import subprocess
import setuptools
import importlib

from pathlib import Path


# Wheel specific: the wheels only include the soname of the host library `libnvshmem_host.so.X`
def get_nvshmem_host_lib_name(base_dir):
    path = Path(base_dir).joinpath('lib')
    for file in path.rglob('libnvshmem_host.so.*'):
        return file.name
    raise ModuleNotFoundError('libnvshmem_host.so not found')


def validate_ishmem_dir(ishmem_dir):
    """Validate that ISHMEM_DIR points to a usable iSHMEM installation or source tree.

    Supported layouts:
    1. Install prefix: ISHMEM_DIR/include/ishmem.h + ISHMEM_DIR/lib/
    2. Source tree:    ISHMEM_DIR/src/ishmem.h + ISHMEM_DIR/build/include/ishmem/config.h
    """
    if not os.path.isdir(ishmem_dir):
        raise RuntimeError(f'ISHMEM_DIR does not exist: {ishmem_dir}')

    include_dirs = []
    lib_dir = None

    # Layout 1: install prefix
    install_header = os.path.join(ishmem_dir, 'include', 'ishmem.h')
    install_lib = os.path.join(ishmem_dir, 'lib')
    if os.path.isfile(install_header):
        include_dirs.append(os.path.join(ishmem_dir, 'include'))
        lib_dir = install_lib
        return include_dirs, lib_dir

    # Layout 2: source tree  (src/ishmem.h + build/include/ for generated headers)
    src_header = os.path.join(ishmem_dir, 'src', 'ishmem.h')
    build_include = os.path.join(ishmem_dir, 'build', 'include')
    if os.path.isfile(src_header):
        # Generated headers (ishmem/config.h) live in build/include/
        if os.path.isdir(build_include):
            include_dirs.append(build_include)
        include_dirs.append(os.path.join(ishmem_dir, 'src'))
        lib_dir = os.path.join(ishmem_dir, 'build', 'lib')
        return include_dirs, lib_dir

    raise RuntimeError(
        f'Cannot find ishmem.h in {ishmem_dir}/include/ or {ishmem_dir}/src/.\n'
        f'Make sure ISHMEM_DIR points to the iSHMEM install prefix or source root.')


def find_icpx():
    """Find icpx (Intel DPC++/C++ Compiler)."""
    icpx = shutil.which('icpx')
    if icpx is None:
        raise RuntimeError(
            'icpx (Intel DPC++/C++ Compiler) not found in PATH.\n'
            'Install the compiler and source the environment:\n'
            '  source /opt/intel/oneapi/setvars.sh')
    return icpx


def setup_sycl_ishmem_build(internode_backend):
    """Configure build for SYCL + iSHMEM backend (Intel GPU)."""
    ishmem_dir = os.getenv('ISHMEM_DIR')
    if ishmem_dir is None:
        raise RuntimeError(
            'ISHMEM_DIR environment variable must be set for ishmem backend.\n'
            'It should point to the iSHMEM installation prefix (with include/ and lib/)\n'
            'or the iSHMEM source root (with src/ishmem.h).')

    ishmem_includes, ishmem_lib = validate_ishmem_dir(ishmem_dir)

    icpx_path = find_icpx()

    # Level Zero
    l0_include = os.getenv('LEVEL_ZERO_INCLUDE_DIR', '/usr/include')
    l0_lib = os.getenv('LEVEL_ZERO_LIB_DIR', '/usr/lib/x86_64-linux-gnu')

    # Force icpx as the compiler
    os.environ['CXX'] = icpx_path
    os.environ['CC'] = icpx_path.replace('icpx', 'icx')

    # SYCL sources — no CUDA files
    sources = ['csrc/sycl_backend/deep_ep_sycl.cpp']
    include_dirs = ['csrc/'] + ishmem_includes + [l0_include]
    library_dirs = [d for d in [ishmem_lib, l0_lib] if d and os.path.isdir(d)]

    cxx_flags = [
        '-fsycl', '-O3',
        '-DSYCL_ISHMEM',
        '-DDISABLE_NVSHMEM',
        '-Wno-deprecated-declarations',
        '-Wno-unused-variable',
        '-Wno-sign-compare',
    ]

    extra_link_args = [
        '-fsycl',
        '-lze_loader',
        f'-Wl,-rpath,{ishmem_lib}',
    ]
    # Link iSHMEM if the library exists
    if os.path.isfile(os.path.join(ishmem_lib, 'libishmem.a')) or \
       os.path.isfile(os.path.join(ishmem_lib, 'libishmem.so')):
        extra_link_args.append('-lishmem')

    extra_compile_args = {'cxx': cxx_flags}

    # Bits of `topk_idx.dtype`
    if "TOPK_IDX_BITS" in os.environ:
        topk_idx_bits = int(os.environ['TOPK_IDX_BITS'])
        cxx_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')

    print(f'Info: SYCL + iSHMEM build path selected.')
    print(f'  icpx: {icpx_path}')
    print(f'  ISHMEM_DIR: {ishmem_dir}')
    print(f'  ISHMEM includes: {ishmem_includes}')
    print(f'  ISHMEM lib: {ishmem_lib}')
    print(f'  Level Zero include: {l0_include}')
    print(f'  Level Zero lib: {l0_lib}')

    return sources, include_dirs, library_dirs, extra_compile_args, extra_link_args


if __name__ == '__main__':
    internode_backend = os.getenv('DEEPEP_INTERNODE_BACKEND', 'nvshmem').strip().lower()
    assert internode_backend in ('nvshmem', 'ishmem', 'none'), \
        f'Unsupported DEEPEP_INTERNODE_BACKEND: {internode_backend}'

    # ==================================================================
    # SYCL + iSHMEM build path (Intel GPU)
    # ==================================================================
    if internode_backend == 'ishmem':
        from torch.utils.cpp_extension import BuildExtension, CppExtension

        sources, include_dirs, library_dirs, extra_compile_args, extra_link_args = \
            setup_sycl_ishmem_build(internode_backend)

        print('\nBuild summary (SYCL + iSHMEM):')
        print(f' > Sources: {sources}')
        print(f' > Includes: {include_dirs}')
        print(f' > Libraries: {library_dirs}')
        print(f' > Compilation flags: {extra_compile_args}')
        print(f' > Link flags: {extra_link_args}')
        print(f' > Internode backend: {internode_backend}')
        print()

        # noinspection PyBroadException
        try:
            cmd = ['git', 'rev-parse', '--short', 'HEAD']
            revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
        except Exception as _:
            revision = ''

        setuptools.setup(name='deep_ep',
                         version='1.2.1' + revision,
                         packages=setuptools.find_packages(include=['deep_ep']),
                         ext_modules=[
                             CppExtension(name='deep_ep_cpp',
                                          include_dirs=include_dirs,
                                          library_dirs=library_dirs,
                                          sources=sources,
                                          extra_compile_args=extra_compile_args,
                                          extra_link_args=extra_link_args)
                         ],
                         cmdclass={'build_ext': BuildExtension})

    # ==================================================================
    # CUDA build path (NVIDIA GPU) — original behavior
    # ==================================================================
    else:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension

        disable_nvshmem = False
        nvshmem_dir = os.getenv('NVSHMEM_DIR', None)
        nvshmem_host_lib = 'libnvshmem_host.so'
        if internode_backend == 'nvshmem':
            if nvshmem_dir is None:
                try:
                    nvshmem_dir = importlib.util.find_spec("nvidia.nvshmem").submodule_search_locations[0]
                    nvshmem_host_lib = get_nvshmem_host_lib_name(nvshmem_dir)
                    import nvidia.nvshmem as nvshmem  # noqa: F401
                except (ModuleNotFoundError, AttributeError, IndexError):
                    print(
                        'Warning: `NVSHMEM_DIR` is not specified, and the NVSHMEM module is not installed. All internode and low-latency features are disabled\n'
                    )
                    disable_nvshmem = True
        else:
            print('Info: DEEPEP_INTERNODE_BACKEND=none selected, building intranode-only mode.\n')
            disable_nvshmem = True

        if not disable_nvshmem:
            assert os.path.exists(nvshmem_dir), f'The specified NVSHMEM directory does not exist: {nvshmem_dir}'

        cxx_flags = ['-O3', '-Wno-deprecated-declarations', '-Wno-unused-variable', '-Wno-sign-compare', '-Wno-reorder', '-Wno-attributes']
        nvcc_flags = ['-O3', '-Xcompiler', '-O3']
        sources = ['csrc/deep_ep.cpp', 'csrc/kernels/runtime.cu', 'csrc/kernels/layout.cu', 'csrc/kernels/intranode.cu']
        include_dirs = ['csrc/']
        library_dirs = []
        nvcc_dlink = []
        extra_link_args = ['-lcuda']

        # NVSHMEM flags
        if disable_nvshmem:
            cxx_flags.append('-DDISABLE_NVSHMEM')
            nvcc_flags.append('-DDISABLE_NVSHMEM')
        else:
            sources.extend(['csrc/kernels/internode.cu', 'csrc/kernels/internode_ll.cu'])
            include_dirs.extend([f'{nvshmem_dir}/include'])
            library_dirs.extend([f'{nvshmem_dir}/lib'])
            nvcc_dlink.extend(['-dlink', f'-L{nvshmem_dir}/lib', '-lnvshmem_device'])
            extra_link_args.extend([f'-l:{nvshmem_host_lib}', '-l:libnvshmem_device.a', f'-Wl,-rpath,{nvshmem_dir}/lib'])

        if int(os.getenv('DISABLE_SM90_FEATURES', 0)):
            # Prefer A100
            os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '8.0')

            # Disable some SM90 features: FP8, launch methods, and TMA
            cxx_flags.append('-DDISABLE_SM90_FEATURES')
            nvcc_flags.append('-DDISABLE_SM90_FEATURES')

            # Disable internode and low-latency kernels
            assert disable_nvshmem
        else:
            # Prefer H800 series
            os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '9.0')

            # CUDA 12 flags
            nvcc_flags.extend(['-rdc=true', '--ptxas-options=--register-usage-level=10'])

        # Disable LD/ST tricks, as some CUDA version does not support `.L1::no_allocate`
        if os.environ['TORCH_CUDA_ARCH_LIST'].strip() != '9.0':
            assert int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', 1)) == 1
            os.environ['DISABLE_AGGRESSIVE_PTX_INSTRS'] = '1'

        # Disable aggressive PTX instructions
        if int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', '1')):
            cxx_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')
            nvcc_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')

        # Bits of `topk_idx.dtype`, choices are 32 and 64
        if "TOPK_IDX_BITS" in os.environ:
            topk_idx_bits = int(os.environ['TOPK_IDX_BITS'])
            cxx_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')
            nvcc_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')

        # Put them together
        extra_compile_args = {
            'cxx': cxx_flags,
            'nvcc': nvcc_flags,
        }
        if len(nvcc_dlink) > 0:
            extra_compile_args['nvcc_dlink'] = nvcc_dlink

        # Summary
        print('Build summary:')
        print(f' > Sources: {sources}')
        print(f' > Includes: {include_dirs}')
        print(f' > Libraries: {library_dirs}')
        print(f' > Compilation flags: {extra_compile_args}')
        print(f' > Link flags: {extra_link_args}')
        print(f' > Arch list: {os.environ["TORCH_CUDA_ARCH_LIST"]}')
        print(f' > Internode backend: {internode_backend}')
        print(f' > NVSHMEM path: {nvshmem_dir}')
        print()

        # noinspection PyBroadException
        try:
            cmd = ['git', 'rev-parse', '--short', 'HEAD']
            revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
        except Exception as _:
            revision = ''

        setuptools.setup(name='deep_ep',
                         version='1.2.1' + revision,
                         packages=setuptools.find_packages(include=['deep_ep']),
                         ext_modules=[
                             CUDAExtension(name='deep_ep_cpp',
                                           include_dirs=include_dirs,
                                           library_dirs=library_dirs,
                                           sources=sources,
                                           extra_compile_args=extra_compile_args,
                                           extra_link_args=extra_link_args)
                         ],
                         cmdclass={'build_ext': BuildExtension})
