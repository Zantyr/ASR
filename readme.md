# PICTEC working environment

Installation script: `setup.sh`; Before you proceed, install docker and nvidia docker runtime from official repositories of your system.

This package exposes two default working environments: `stable.Dockerfile` provides out-of-the-box functionality. `custom.Dockerfile` uses
your own tensorflow runtime compiled against CUDA 10.0. For the latter, prepare tensorflow wheel in `link/tensorflow_pkg` folder.
Use it to ensure that the runtime is tied tightly with the platform you're running it on.

Running script:

- in commandline: `sudo docker run --runtime=nvidia -p 8888:8888 -v /var/pictec:/pictec pictec-workspace`
- as a daemon: `sudo docker run --runtime=nvidia -p 8888:8888 -d -v /var/pictec:/pictec pictec-workspace`

The only persistable catalogue is `/pictec` and it is the main catalogue of the notebook app. It should appear as `/var/pictec` in the host filesystem. You may want to remount it by changing parameters in the runtime command.

Default password for the notebook is `PICTECpasswd`. Please change it after you run your container or modify jupyter_notebook_config file before building the container image if you intend to expose it.

If you want to run it at the start-up of the system, prepare proper systemd service file. Example service file in `template_files`.

For throttling of the resources (CPU, RAM), use `--cpu` and `--memory` flags during running the container.

### Library choice

- not applicable yet - libraries are being build

### Q queueing server

Use `Q.commands.from_notebook()` to push the notebook as is as a script to the processing. As for now you have to manually save the notebook in the Jupyter before you dump the code to the script.
Use `Q.commands.list_processes()` to check status of a task. If not present, it has stopped.
Use `Q.commands.archive()` to look at completed tasks.

Q has added logging support. `Q.logging.get_logger()` returns a normal logger when used on a normal task and log to file when pushed into the Q queue. The logs are stored in `/var/log/Q` by default.

Each task has a specific id associated with it. You may use it to create run folder to store data associated with given run. `Q.logging.get_task_id()` provides said id or throws a `RuntimeError` if run outside of a task.

### Support

`pawel.tomasik@pictec.eu`

### Changelog

2018-11-13 prepared two main images of the environment and tested them, updated documentation
