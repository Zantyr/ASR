# For added security, you should as secure key.
# This is on assumption that the server is run in isolation.

# Set ip to '*' to bind on all interfaces (ips) for the public server
c.NotebookApp.ip = '*'
c.NotebookApp.password = u'sha1:a83b085f1ff0:fb3e4379929899c33a60d04aa3bd50a92a7bcf4f'
c.NotebookApp.open_browser = False

# It is a good idea to set a known, fixed port for server access
c.NotebookApp.port = 8888
