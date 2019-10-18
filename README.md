# ASR

Package of ASR models for my thesis

### Installation

The default method of installation is to run `setup.sh`, which creates a Docker image. The basic requirements
is the Docker system as well as CUDA on the host machine. The Docker should be configured to expose GPU
devices to the container. The experiments were run on Arch Linux with `nvidia-docker` package as the method
to expose the appropriate device files. 

The dockerfile prepares all software dependencies, however, a part of the dependencies are updated
after the running of the container. After the installation finishes, a Jupyter notebook will start and
all required libraries will be in the main directory.

If user wants to install required software, all modules and software packages from files
`template_files/requirements.txt`, `install.sh`, `requirements.txt` should be installed. The installation
method on bare system depends on the package manager and software distribution of the user, therefore installation
via Dockerfile is preferred.

### Obtaining data



### Running the models

By default, the Docker image links the folder of the project to the container in order to persist the results
and facilitate the development of new experiments. The models should be located in `/asr` in form
of the Jupyter notebooks.

To reproduce the results, notebooks should be opened and simply run.

### Updating and testing the framework

The fwks library attached is automatically updated on the run of the Dockerfile. It is hosted on GitHub at
the repository `https://github.com/Zantyr/fwks`
