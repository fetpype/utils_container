FROM antsx/ants 

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl tar bzip2 libstdc++6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# Download and install micromamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

# Initialize micromamba

RUN micromamba install -y -n base -c conda-forge python=3.9.0 && \
    micromamba clean --all --yes && \
    echo "source <(micromamba shell hook --shell=bash)" >> ~/.bashrc
# Copy the current directory contents into the container at /app
COPY ./fetpype_utils /app/fetpype_utils
COPY ./pyproject.toml /app/pyproject.toml
WORKDIR /app

# Create a micromamba environment and install dependencies

RUN micromamba run -n base python -m pip install -e .

# Use micromamba shell
#SHELL ["/bin/bash", "-c"]

ENTRYPOINT ["/bin/bash", "-c", "source <(micromamba shell hook --shell=bash) && micromamba activate base && exec \"$@\"", "--"]
