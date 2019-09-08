import logging
import socket

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class ResourceManager:
    """
    Server on predefined socket that allows control over all modules
    """
    
    @staticmethod
    def main():
        server_address = './'
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind()
        while True:
            sock.listen(5)
            try:
                # should callback using threadpool...
                conn, client = sock.accept()
                data = connection.recv(16)
                if data:
                    pass
                else:
                    break # should I maintain a connection?
            finally:
                conn.close()
        
    @staticmethod
    def fork():
        """
        Runs a new process for manager
        """

class Application:
    """
    Exposes a pipeline over some application
    
    This is abstract class for application
    Should fork and run resource manager on execution if it cannot find
    Unix Socket for Resource manager 
    """
    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if not os.exists(self.manager_socket_address):
            ResourceManager.fork()
        try:
            sock.connect(self.manager_socket_address)
        except socket.error, msg:
            # should print message and retry - this means we could not create manager
            sys.exit(1)
    
class FlaskApplication(Application):
    """
    Interface for exposing input or output to web-service
    """

class MicrophoneApplication(Application):
    """
    ...
    """
