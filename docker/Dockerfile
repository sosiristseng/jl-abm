FROM python:3.10.4-slim

WORKDIR /usr/src/app

# If you use PackageCompiler, you'll need compilers
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Julia
ENV JULIA_PATH /usr/local/julia
ENV PATH $JULIA_PATH/bin:$PATH
COPY --from=julia:1.7.2 ${JULIA_PATH} /usr/local/

# Python deps (jupyter-book and matplotlib)
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip wheel setuptools && pip install --no-cache-dir -r requirements.txt

# Build Julia environment
COPY Project.toml Manifest.toml ./
RUN julia --threads=auto --color=yes -e 'import Pkg; Pkg.add("IJulia")' \
    && julia --threads=auto --color=yes --project=@. -e 'import Pkg; Pkg.instantiate(); Pkg.resolve(); Pkg.precompile()'

# Julia sysimage (optional)
# RUN julia --threads=auto --color=yes -e 'import Pkg; Pkg.add("PackageCompiler"); using PackageCompiler;\
#     create_sysimage(["Example"]; project= ".", sysimage_path="$(pwd())/sysimage.so"); \
#     using IJulia; installkernel("Julia", "--sysimage=$(pwd())/sysimage.so"); \
#     Pkg.remove("PackageCompiler"); Pkg.gc()'

# Julia command
CMD ["julia"]

# If using sysimage
# CMD ["julia", "-J", "sysimage.so"]
