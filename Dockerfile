FROM antsx/ants 

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl tar bzip2 libstdc++6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# Download and install micromamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

# Initialize micromamba
RUN micromamba install -y -n base -c conda-forge python && \
    micromamba clean --all --yes

# Use micromamba shell
SHELL ["micromamba", "run", "-n", "base", "/bin/bash", "-c"]

# Verify micromamba environment
RUN python --version