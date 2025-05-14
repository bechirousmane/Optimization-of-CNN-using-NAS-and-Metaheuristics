import os
import torch
import asyncio
from contextlib import asynccontextmanager

class ResourceManager:
    def __init__(self, use_gpu=True, max_concurrent=None):
        """
            Initializes the ressource manager.
            Args :
                use_gpu : bool, if True, the gpu will be used
                max_concurrent : int, The maximum number of CPUs to use simultaneously
        """
        self.use_gpu = use_gpu
        if use_gpu:
            self.devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        else:
            self.devices = ['cpu'] * (max_concurrent or os.cpu_count())
        
        self.semaphore = asyncio.Semaphore(len(self.devices))

    @asynccontextmanager
    async def acquire(self):
        """
            Gives a free resource and locks it during use.
        """
        await self.semaphore.acquire()
        device = self.devices.pop(0)
        try:
            yield device
        finally:
            self.devices.append(device)
            self.semaphore.release()

