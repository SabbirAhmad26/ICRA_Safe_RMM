import math
import numpy as np
import os, socket, struct

def camera_handler(image, img_queue):
    """To be used as `camera.listen(camera_handler)`.
    Puts the raw image data into a queue.
    
    :param image: An instance of carla.Image
    :param img_queue: Queue, to which the raw image data will be appended
    
    Example usage:
    camera.listen(lambda image: camera_handler(image, some_image_queue))
    """
    img_queue.put(image.raw_data)


def lidar_handler(point_cloud, pc_queue):
    """To be used as `camera.listen(lidar_handler)`.
    Puts the raw lidar data into a queue.
    
    :param point_cloud: An instance of carla.LidarMeasurement
    :param pc_queue: Queue, to which the raw point cloud will be appended.
    
    Example usage:
    lidar.listen(lambda pc: lidar_handler(pc, some_image_queue))
    """
    pc_queue.put(point_cloud.raw_data)


def process_raw_RGBA(rgb_bytes, H = 375, W = 1242):
    """
    Convert raw BGRA bytes (as given in carla.Image.raw_data) to
    a Numpy array
    
    :param rgb_bytes: Bytestring representing short unsigned ints (0-255),
        where every consecutive 4 bytes represent 4 integers,
        (blue, green, red, alpha)
    :param H: Int representing height of image
    :param W: Int representing width of image
    :returns: Numpy int array of shape (H, W, 4),
        where the four channels are (R, G, B, A)
    """
    # We unpack the img into a list of ints; 'B' signifies uint8
    rgb_data = np.array(np.frombuffer(rgb_bytes, dtype='B').reshape(H,W,4))
    # todo - swapaxes so channels are first, drop 4th channel (alpha)
    # Swap blue and red so we get the RGBA format we want
    rgb_data[:,:,0], rgb_data[:,:,2] = rgb_data[:,:,2], rgb_data[:,:,0].copy()
    return rgb_data


def process_raw_PC(pc_bytes):
    """Converts a carla.LidarMeasurement.raw_data to a usable
    Numpy array of shape (n, 4) point cloud.
    
    :param pc_bytes: Bytestring representing float32s, where
        every consecutive 4 bytes represents 1 float32, and every
        every consecutive 4 floats represent an (x, y, z, i) tuple,
        where (x, y, z) are coordinates and  i  is intensity.
    :returns: Numpy float array of shape (-1, 4)
    """
    return np.array(np.frombuffer(pc_bytes, dtype='<f').reshape(-1,4))


class ImageServer:
    def __init__(self, address = "img.sock", verbose = True, timeout = 60000):
        """
        Wrap the setup of IPC for sending images. Implemented using AF_UNIX
        SOCK_STREAM sockets. The goal is to make it possible to drop-in replace
        this with other forms of IPC whenever needed.
        
        This is intended to be connected to only one socket over the course of
        its life. Does not implement fancy dunders like __del__!
        
        :param address: String; file descriptor for socket
        :param verbose: If True, print extra info
        :method send raw_img:
    
        usage:
        imageserver = ImageServer()
        imageserver.send(some_raw_bytes)
        :param timeout: Float. Time to wait in seconds for blocking actions.
        """
        
        self.V = verbose
        self.address = address
        
        if self.V: print("Checking for existing socket...")
        if os.path.exists(self.address):
            if self.V: print("... It exists! Destructing existing socket...")
            os.remove(self.address)
        elif self.V: print("... No socket found!")
            
        if self.V: print("Creating socket...")
        self.serversocket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.serversocket.settimeout(timeout)
        
        if self.V: print(f"... binding to {self.address} ...")
        self.serversocket.bind(self.address)
        
        if self.V: print("... Bound! Listening...")
        self.serversocket.listen()
        
        if self.V: print("... Found connection! Accepting...")
        self.clientsocket, self.clientaddr = self.serversocket.accept()
        
        if self.V: print(f"... Accepted, Connected to client at addr {self.clientaddr}")
    
    def send(self, raw_img):
        """
        Wraps clientsocket.sendall(raw_img).
        Sends data to the client, to be processed.
        
        :param raw_img: Bytestring
        """
        self.clientsocket.sendall(raw_img)
    
    def recv(self):
        """
        :raises NotImplementedError:
        """
        raise NotImplementedError

    
class PointCloudServer:
    def __init__(self, address = "pc.sock", verbose = True, timeout = 60000):
        """
        Wrap the setup of IPC for sending pointcloud data. Implemented using
        AF_UNIX SOCK_STREAM sockets. The goal is to make it possible to drop-in
        replace this with other forms of IPC whenever needed.
        
        This is intended to be connected to only one socket over the course of
        its life. Does not implement fancy dunders like __del__!
        
        :param address: String; file descriptor for socket
        :param verbose: If True, print extra info
        :param timeout: Float. Time to wait in seconds for blocking actions.
    
        usage:
        pcserver = PointCloudServer()
        pcserver.send(some_raw_bytes)
        """
        
        self.V = verbose
        self.address = address
        
        if self.V: print("Checking for existing socket...")
        if os.path.exists(self.address):
            if self.V: print("... It exists! Destructing existing socket...")
            os.remove(self.address)
        elif self.V: print("... No socket found!")
            
        if self.V: print("Creating socket...")
        self.serversocket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.serversocket.settimeout(timeout)
        
        if self.V: print(f"... binding to {self.address} ...")
        self.serversocket.bind(self.address)
        
        if self.V: print("... Bound! Listening...")
        self.serversocket.listen()
        
        if self.V: print("... Found connection! Accepting...")
        self.clientsocket, self.clientaddr = self.serversocket.accept()
        
        if self.V: print(f"... Accepted, Connected to client at addr {self.clientaddr}")
    
    def send(self, raw_pc, VV = False):
        """
        Wraps clientsocket.sendall(raw_img). Sends data to the client, to be processed.
        
        :param raw_pc: Bytestring
        """
        # first, send the length of the data to be expected
        if VV: print(f"Sending length of data {len(raw_pc)} ...")
        num_bytes = struct.pack('<q', len(raw_pc))
        self.clientsocket.send(num_bytes)
        # then send the bytes on over
        if VV: print(f"... Sending data ...")
        self.clientsocket.send(raw_pc)
        if VV: print(f"... Sent !!!")
    
    def recv(self):
        """
        :raises NotImplementedError:
        """
        raise NotImplementedError


class ImageClient:
    def __init__(self, address = "img.sock", verbose = True, timeout = 60):
        """
        Receives and optionally processes images from ImageClient using sockets.

        Wrapped sockets for modularity of IPCs.
        
        :param address: String; file descriptor for socket
        :param verbose: If True, print extra info during setup
        :param timeout: Float. Time to wait in seconds for blocking actions.
    
        usage:
        imageserver = ImageClient(address = "img.sock")
        processed_image = imageserver.recv(H=375,W=1242,chunksize=2**16)
        raw_img = imageserver.recv(process=False)
        """
        self.V = verbose
        self.address = address
        
        if self.V: print(f"Creating socket...")
        self.socket = socket.socket(family=socket.AF_UNIX, type=socket.SOCK_STREAM)
        self.socket.settimeout(timeout)
        if self.V: print(f"... connecting to socket at {self.address}...")
        connected = False
        while not connected:
            try:
                self.socket.connect(self.address)
                connected = True
            except Exception as e:
                pass
        if self.V: print(f"... Done!")
        
        self.buffer = b''
    
    def recv(self, H=375, W=1242, chunksize=2**16, process=True):
        """
        Receive an Image from the server, building up a buffer, and 
        (optionally) process it.
        
        :param H: Int, image height. Necessary for processing.
        :param W: Int, image width. Necessary for processing.
        :param chunksize: Size of data, passed to recv()
            Optional.
        :param process: If True, process the raw data into an image.
            If not, send the data along unprocessed.
        """
        if process:
            maxsize = H * W * 4
            while len(self.buffer) < maxsize:
                self.buffer = self.buffer  + self.socket.recv(chunksize)

            raw_img = self.buffer[:maxsize]
            self.buffer = self.buffer[maxsize:]
            output = process_raw_RGBA(raw_img, H=H, W=W)
            return output
        else:
            return self.socket.recv(chunksize)
    
    def send(self): 
        raise NotImplementedError


class PointCloudClient:
    def __init__(self, address = "pc.sock", verbose = True, timeout = 60):
        """
        Receives and optionally processes pointcloud data from
        PointCloudServer using sockets.

        Wraped sockets for modularity of IPCs.
        
        Sequence:
        1. receive number of bytes to be expected
        2. receive said number of bytes (the actual pointcloud data)
        
        :param address: String; file descriptor for socket
        :param verbose: If True, print extra info during setup
        :param timeout: Float. Time to wait in seconds for blocking actions.
    
        usage:
        pc_client = PointCloudClient(address = "pc.sock")
        processed_pc = pc_client.recv()
        raw_pc = pc_client.recv(process=False)
        """
        self.V = verbose
        self.address = address
        
        if self.V: print(f"Creating socket...")
        self.socket = socket.socket(family=socket.AF_UNIX, type=socket.SOCK_STREAM)
        self.socket.settimeout(timeout)
        if self.V: print(f"... connecting to socket at {self.address}...")
        connected = False
        while not connected:
            try:
                self.socket.connect(self.address)
                connected = True
            except Exception as e:
                pass
        if self.V: print(f"... Done!")
        
        self.buffer = b''
    
    def send(self):
        raise NotImplementedError
    
    def recv(self, VV = False):
        """
        Receive an Image from the server,
        returning a processed pointcloud.
        """
        # EXPERIMENTAL NEW RECV
        # 1. get maxsize from the first send
        if VV: print("Receiving length...")
        num_bytes = self.socket.recv(8)
        num_bytes = struct.unpack('<q', num_bytes)[0]
        if VV: print(f"... Received length {num_bytes} ...")
        if VV: print(f"... Receiving bytes ...")
        raw_pc = self.socket.recv(num_bytes)
        if VV: print(f"... Received !!!")
        return process_raw_PC(raw_pc)
        
