FROM ubuntu:22.04

# Set non-interactive frontend to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

# Define Binder-required user variables
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER=${NB_USER}
ENV NB_UID=${NB_UID}
ENV HOME=/home/${NB_USER}

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    python3 \
    python3-pip \
    r-base \
    r-base-dev \
    libzmq3-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    libxml2-dev \
    ca-certificates \
    openjdk-11-jre-headless

# Upgrade pip and install base Python dependencies
RUN pip3 install --upgrade pip setuptools wheel

# Install JupyterLab and notebook
RUN pip install --no-cache-dir jupyterlab notebook

# Install IRkernel for R and register the R kernel system-wide
RUN R -e "install.packages('IRkernel', repos='https://cloud.r-project.org'); IRkernel::installspec(user=FALSE)"

# Install h2o R package
RUN R -e "install.packages('h2o', repos='https://cloud.r-project.org')"

# Create jovyan user with UID 1000 (Binder requirement)
RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

# Copy repository contents to the user's home directory
WORKDIR ${HOME}
COPY . ${HOME}

# Change ownership of the directory to the created user
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

# Ensure the PATH includes the correct Python location
ENV PATH="${HOME}/.local/bin:${PATH}"

# Install jumper_ipython_extension and jumper_wrapper_kernel
RUN pip install --no-cache-dir \
    "git+https://github.com/ScaDS/jumper_ipython_extension.git@main" \
    "git+https://github.com/ScaDS/jumper_ipython_extension.git@main#subdirectory=jumper_wrapper_kernel"

# Register the wrapper kernel spec
RUN python3 -m jumper_wrapper_kernel.jumper_wrapper_kernel.install install

# Expose Jupyter Notebook port
EXPOSE 8888

# Set the default command for running Jupyter Notebook
CMD ["jupyter", "notebook", "--NotebookApp.default_url=/lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
