
import torch
from MemoryDiffusion.respace import space_timesteps

ddim_time_steps = torch.tensor(space_timesteps(1000, 100))
t = torch.randint(0, 100, (16, ))

t1 = ddim_time_steps[t]
print(t1)


