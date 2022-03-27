# Adapted from https://github.com/terasakisatoshi/sysimage_creator/
import Pkg

Pkg.add(["PackageCompiler", "IJulia"])

using PackageCompiler

major = VERSION.major
minor = VERSION.minor

sysimage_path = joinpath(@__DIR__, "v$(major).$(minor)-sys.so")

@info "SysImage path: " sysimage_path

PackageCompiler.create_sysimage(
    ["Agents", "DataFrames", "CairoMakie"];
    project=".",
    sysimage_path=sysimage_path,
    cpu_target=PackageCompiler.default_app_cpu_target()
)

nthreads = Sys.CPU_THREADS

@info "Threads: " nthreads

using IJulia
kernelpath = IJulia.installkernel("Julia-sys-$(nthreads)-threads",
    "--project=@.", "--sysimage=$(sysimage_path)";
    specname="julia", env=Dict("JULIA_NUM_THREADS" => "$(nthreads)",)
)

@info "IJulia kernal path: " kernelpath

Pkg.rm("PackageCompiler")
Pkg.gc()
