FROM ghcr.io/sosiristseng/docker-juliabook:latest

WORKDIR /usr/src/app

# Build Julia environment
COPY Project.toml Manifest.toml build_kernel.jl ./
RUN julia --threads=auto --color=yes --project=@. -e 'import Pkg; Pkg.instantiate(); Pkg.resolve(); Pkg.precompile()' && julia --threads=auto --color=yes build_kernel.jl

# Julia command
# CMD ["julia"]

# If using sysimage
CMD ["julia", "--sysimage=sysimage.so"]
