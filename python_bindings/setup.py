import os
import platform
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

# Package metadata (name, version, dependencies, ...) lives in pyproject.toml.
# This file only declares the extension module and its custom build logic.

maindir = os.path.join(".", "fastpfor")
library_file = os.path.join(maindir, "libFastPFor.a")
source_files = ['pyfastpfor.cc']

libraries = []
extra_objects = []

if os.path.exists(library_file):
    # if we have a prebuilt library file, use that.
    extra_objects.append(library_file)

else:
    # Otherwise build all the files here directly (excluding test files)
    exclude_files = set("""unit.cpp codecs.cpp partitionbylength.cpp inmemorybenchmark.cpp gapstats.cpp
                           entropy.cpp csv2maropu.cpp benchbitpacking.cpp""".split())

    for root, subdirs, files in os.walk(os.path.join(maindir, "src")):
        source_files.extend(os.path.join(root, f) for f in files
                            if (f.endswith(".cc") or f.endswith(".c") or f.endswith(".cpp")) and f not in exclude_files)

ext_modules = [
    Extension(
        'pyfastpfor',
        source_files,
        include_dirs=[maindir, os.path.join(maindir, "headers")],
        libraries=libraries,
        language='c++',
        extra_objects=extra_objects,
    ),
]

# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def simd_flags(compiler):
    """Return the SIMD/architecture compile flags.

    FastPFor's SIMD code requires SSE4.2 on x86 (provided natively) and NEON on
    ARM (provided through the fastpfor_neon.h shim, and part of the ARMv8-A
    baseline, so no special flag is needed).

    By default we use ``-march=native`` for the best performance, which is the
    right choice for a source install built on the machine that runs it. For
    redistributable wheels this is unsafe (the build machine may support
    instructions the user's CPU lacks), so set ``PYFASTPFOR_PORTABLE=1`` to use
    a portable baseline instead; the CI wheel builds do exactly that.
    """
    portable = os.environ.get('PYFASTPFOR_PORTABLE', '') not in ('', '0', 'false', 'False')
    machine = platform.machine().lower()
    is_x86 = machine in ('x86_64', 'amd64', 'x86', 'i386', 'i686')

    if not portable and has_flag(compiler, '-march=native'):
        return ['-march=native']
    if is_x86 and has_flag(compiler, '-msse4.2'):
        # Portable x86 baseline: SSE4.2 is the minimum FastPFor requires.
        return ['-msse4.2']
    # On ARM/aarch64 NEON is part of the baseline, so no extra flag is needed.
    return []


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.

    #The c++14 is preferred over c++11 (when it is available).
    # This somehow can fail on a Mac with clang
    #"""
    #if has_flag(compiler, '-std=c++14'):
        #return '-std=c++14'
    #elif has_flag(compiler, '-std=c++11'):
    if has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    # Note: language-specific standard flags (-std=c++11 / -std=c99) are NOT
    # listed here. They are applied per source file in _compile_with_std below,
    # because this extension mixes C and C++ sources and a C++ standard flag is
    # rejected by the compiler on C sources (and vice versa).
    c_opts = {
        'msvc': ['/EHsc', '/O2'],
        'unix': ['-O3'],
        #'unix': ['-O0', '-g'],
    }
    link_opts = {
        'unix': [],
        'msvc': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.9']
        link_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.9']
    else:
        link_opts['unix'].append('-pthread')

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = list(self.c_opts.get(ct, []))
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.extend(simd_flags(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO="%s"' % self.distribution.get_version())

        # extend include dirs here (don't assume numpy/pybind11 are installed when first run, since
        # pip could have installed them as part of executing this script
        import pybind11
        import numpy as np
        for ext in self.extensions:
            ext.extra_compile_args.extend(opts)
            ext.extra_link_args.extend(self.link_opts.get(ct, []))
            ext.include_dirs.extend([
                # Path to pybind11 headers
                pybind11.get_include(),
                pybind11.get_include(True),

                # Path to numpy headers
                np.get_include()
            ])

        if ct == 'unix':
            self._patch_compiler_for_mixed_languages()

        build_ext.build_extensions(self)

    def _patch_compiler_for_mixed_languages(self):
        """Apply the right -std flag to each source based on its language.

        distutils applies one set of compile args to every source in an
        extension, but here C++ sources (.cc/.cpp) need -std=c++11 while C
        sources (.c) need -std=c99. We wrap the compiler's _compile method to
        add the appropriate standard flag (and drop C++-only flags on C).
        """
        compiler = self.compiler
        original_compile = compiler._compile
        cxx_std = cpp_flag(compiler)
        cxx_only = ('-stdlib=libc++',)

        def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            postargs = list(extra_postargs)
            if src.endswith(('.cpp', '.cxx', '.cc', '.c++')):
                postargs.append(cxx_std)
            elif src.endswith('.c'):
                postargs = [a for a in postargs if a not in cxx_only]
                postargs.append('-std=c99')
            return original_compile(obj, src, ext, cc_args, postargs, pp_opts)

        compiler._compile = _compile


setup(
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
