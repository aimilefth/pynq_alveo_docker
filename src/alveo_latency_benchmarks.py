import numpy as np
import time
from .alveo_runner import AlveoRunner


def get_average_time_alveo(
    model: AlveoRunner, input_vector: np.ndarray, iterations: int = 1000
) -> float:
    total_time = 0
    # Copy input data to the FPGA buffer and sync it
    model.input_buffer[:] = input_vector
    model.input_buffer.sync_to_device()
    warmup = 10
    for _ in range(warmup):
        _ = model.kernel.call(model.input_buffer, model.output_buffer)
    model.output_buffer.sync_from_device()
    for i in range(iterations):
        start = time.perf_counter()
        _ = model.kernel.call(model.input_buffer, model.output_buffer)
        model.output_buffer.sync_from_device()
        end = time.perf_counter()
        total_time += end - start
    average_time_ms = total_time / iterations * 1000  # (ms)
    return average_time_ms


def get_average_time_alveo_transfers(
    model, input_vector: np.ndarray, iterations: int = 1000
) -> float:
    total_time = 0
    # Copy input data to the FPGA buffer and sync it
    model.input_buffer[:] = input_vector
    model.input_buffer.sync_to_device()
    warmup = 10
    for _ in range(warmup):
        _ = model.kernel.call(model.input_buffer, model.output_buffer)
    model.output_buffer.sync_from_device()
    for i in range(iterations):
        start = time.perf_counter()
        model.input_buffer.sync_to_device()
        _ = model.kernel.call(model.input_buffer, model.output_buffer)
        model.output_buffer.sync_from_device()
        end = time.perf_counter()
        total_time += end - start
    average_time_ms = total_time / iterations * 1000  # (ms)
    return average_time_ms
