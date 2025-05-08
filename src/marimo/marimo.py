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
def __(devices):
    from src.alveo_runner import AlveoRunner, AlveoRunnerParameters, convert_types

    XCLBIN_PATH = "/app/mounted_dir/attention_ae.xclbin"
    device = devices[0]
    TIMESTEPS = 12
    FEATURES = 8
    params = AlveoRunnerParameters(
        input_buffer_elements=TIMESTEPS * FEATURES,
        output_buffer_elements=TIMESTEPS * FEATURES,
    )
    model = AlveoRunner(XCLBIN_PATH, params, device)
    return (
        AlveoRunner,
        AlveoRunnerParameters,
        FEATURES,
        TIMESTEPS,
        XCLBIN_PATH,
        convert_types,
        device,
        model,
        params,
    )


@app.cell
def __(mo):
    mo.md(r"""## Example Run""")
    return


@app.cell
def __(convert_types, model, np):
    ones_vector = np.ones((model.parameters.input_buffer_elements,)).astype(
        convert_types(model.parameters.input_t)
    )
    out_vector = model.run(ones_vector)
    print(out_vector)

    twos_vector = ones_vector * 2
    out_twos_vector = model.timed_run(twos_vector)
    print(out_twos_vector)
    return ones_vector, out_twos_vector, out_vector, twos_vector


@app.cell
def __(mo):
    mo.md(r"""## Benchmark Latency on ALVEO""")
    return


@app.cell
def __(convert_types, model, np):
    input_vector = np.random.random((model.parameters.input_buffer_elements,)).astype(
        convert_types(model.parameters.input_t)
    )
    average_time = model.get_average_time_alveo(input_vector, 1000)
    print(f"Average run: {average_time:.2f} ms")
    average_time_transfers = model.get_average_time_alveo_transfers(input_vector, 1000)
    print(f"Average run with data transfers: {average_time_transfers:.2f} ms")
    return average_time, average_time_transfers, input_vector


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
