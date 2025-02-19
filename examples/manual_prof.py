import torch.profiler
import os

trace_path = "./lightning_logs/trace_profiler_test"
os.makedirs(trace_path, exist_ok=True)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_path),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as profiler:
    for i in range(5):
        print(f"Profiling iteration {i}")
        x = torch.randn(1, 3, 640, 640).cuda()
        y = torch.nn.Conv2d(3, 16, 3, 1, 1).cuda()(x)
        profiler.step()
