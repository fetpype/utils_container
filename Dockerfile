FROM antsx/ants

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tar \
    bzip2 \
    libstdc++6 \
    bash \
    ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Micromamba Setup ---
# 1. Define the root prefix location INSIDE the container
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
# Add its bin to PATH for subsequent commands and runtime
ENV PATH=${MAMBA_ROOT_PREFIX}/bin:${PATH}

# 2. Download micromamba executable and place it in the prefix's bin directory
RUN mkdir -p ${MAMBA_ROOT_PREFIX}/bin && \
    echo "Downloading micromamba executable..." && \
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj --strip-components=1 -C ${MAMBA_ROOT_PREFIX}/bin bin/micromamba && \
    chmod +x ${MAMBA_ROOT_PREFIX}/bin/micromamba && \
    echo "Micromamba executable installed in ${MAMBA_ROOT_PREFIX}/bin" && \
    micromamba --version # Verify executable runs

# 3. REMOVED: 'micromamba shell init' step - as it modified .bashrc, not useful here.

# 4. Create/update the 'base' environment (placed under MAMBA_ROOT_PREFIX/envs/base)
ENV CONDA_ENV_NAME=base
RUN echo "Creating/updating environment '${CONDA_ENV_NAME}'..." && \
    micromamba install -y -n ${CONDA_ENV_NAME} -c conda-forge python=3.9.0 && \
    # Clean the environment using micromamba run (simpler than sourcing/activating during build)
    micromamba run -n ${CONDA_ENV_NAME} micromamba clean --all --yes && \
    echo "Environment '${CONDA_ENV_NAME}' created and cleaned."

# --- Application Setup ---
COPY ./fetpype_utils /app/fetpype_utils
COPY ./pyproject.toml /app/pyproject.toml
WORKDIR /app

# Install the application package into the target environment
RUN echo "Installing application package..." && \
    micromamba run -n ${CONDA_ENV_NAME} python -m pip install --no-cache-dir -e . && \
    # Clean pip cache
    rm -rf /root/.cache/pip && \
    echo "Application package installed."

# --- Runtime Environment Activation via ENTRYPOINT ---
# Use 'eval $(micromamba shell hook ...)' to initialize the shell at runtime.
# Explicitly provide the --root-prefix to shell hook.
ENTRYPOINT [ "/bin/bash", "-c", "set -e; \
    echo \"INFO: Evaluating micromamba shell hook...\"; \
    eval \"$(micromamba shell hook --shell bash --root-prefix ${MAMBA_ROOT_PREFIX})\" ; \
    echo \"INFO: Activating environment '${CONDA_ENV_NAME}'...\"; \
    micromamba activate ${CONDA_ENV_NAME}; \
    echo \"INFO: Micromamba environment '${CONDA_ENV_NAME}' activated. PATH is: $PATH\"; \
    echo \"INFO: Running command: $@\"; \
    exec \"$@\"", "--" ]