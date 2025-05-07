import marimo

__generated_with = "0.8.22"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import json
    import math
    import time

    return json, math, mo, np, plt, time


@app.cell
def __():
    import pynq

    devices = pynq.Device.devices
    for i in range(len(devices)):
        print("{}) {}".format(i, devices[i].name))
    return devices, i, pynq


@app.cell
def __(mo):
    mo.md(r"""## Load model on ALVEO""")
    return


@app.cell
def __():
    from src.alveo_runner import AlveoRunner, AlveoRunnerParameters, convert_types

    return AlveoRunner, AlveoRunnerParameters, convert_types


@app.cell
def __(AlveoRunner, AlveoRunnerParameters, devices):
    XCLBIN_PATH = "/app/mounted_dir/attention_ae.xclbin"
    device = devices[0]
    TIMESTEPS = 12
    FEATURES = 8
    params = AlveoRunnerParameters(
        kernel_name="attention_ae_1",
        input_buffer_elements=TIMESTEPS * FEATURES,
        output_buffer_elements=TIMESTEPS * FEATURES,
    )
    model = AlveoRunner(XCLBIN_PATH, params, device)
    return FEATURES, TIMESTEPS, XCLBIN_PATH, device, model, params


@app.cell
def __(mo):
    mo.md(r"""## Benchmark Latency on ALVEO""")
    return


@app.cell
def __(convert_types, model, np):
    from src.alveo_latency_benchmarks import (
        get_average_time_alveo,
        get_average_time_alveo_transfers,
    )

    input_vector = np.random.random((model.parameters.input_buffer_elements,)).astype(
        convert_types(model.parameters.input_t)
    )
    average_time = get_average_time_alveo(model, input_vector, 1000)
    print(f"Average run: {average_time:.2f} ms")
    average_time_transfers = get_average_time_alveo_transfers(model, input_vector, 1000)
    print(f"Average run with data transfers: {average_time_transfers:.2f} ms")
    return (
        average_time,
        average_time_transfers,
        get_average_time_alveo,
        get_average_time_alveo_transfers,
        input_vector,
    )


@app.cell
def __():
    ## Benchmark Power on ALVEO
    return


@app.cell
def __(model):
    from src.plot_alveo_power import extract_alveo_power_data, plot_alveo_power_data

    data = model.get_power_data(seconds=10, transfers=False)
    print(data)
    alveo_df = extract_alveo_power_data(data)
    plot_alveo_power_data(alveo_df)
    print(f"Average Total Power: {alveo_df['Power'].mean()} W")

    data_tr = model.get_power_data(seconds=10, transfers=True)
    alveo_df_tr = extract_alveo_power_data(data_tr)
    plot_alveo_power_data(alveo_df_tr)
    print(f"Average Total Power with Data Transfers: {alveo_df_tr['Power'].mean()} W")
    return (
        alveo_df,
        alveo_df_tr,
        data,
        data_tr,
        extract_alveo_power_data,
        plot_alveo_power_data,
    )


@app.cell
def __(mo):
    mo.md(r"""## Clean model""")
    return


@app.cell
def __(model):
    model.clean_class()
    return


if __name__ == "__main__":
    app.run()
