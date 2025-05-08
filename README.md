
# PYNQ Alveo Docker for FPGA Benchmarking

This project provides a Dockerized environment for running PYNQ applications on Xilinx Alveo FPGAs. It includes tools for benchmarking kernel latency and power consumption, primarily through an interactive Marimo notebook and a helper class `AlveoRunner`.
This was tested on an Alveo U280, with XRT 2.13.466 version

## Prerequisites

Before you begin, ensure you have the following installed and configured on your **host machine**:

1.  **Docker Engine**: To build and run the Docker container.
2.  **Xilinx Alveo Card**: Physically installed in your machine.
3.  **Xilinx Runtime (XRT)**: Installed and configured on the host system. The Docker container will use this host installation and mounted directories to communicate with the Alveo card. You can verify your XRT installation and card detection with `xbutil scan` on your host.
4.  **Git** (Optional): If you clone this repository.

## Project Structure

```
.
├── docker/                  # Docker-related files
│   ├── Dockerfile           # Defines the Docker image
│   ├── docker_build.sh      # Script to build the Docker image
│   ├── requirements.txt     # Python dependencies for inside the container
│   └── setup.sh             # Script to source XRT/XRM environments inside container
├── docker_run.sh            # Script to run the Docker container (PROVIDED)
├── mounted_dir/             # Directory mounted into the container
│   └── attention_ae.xclbin  # Example Alveo bitstream
├── outputs/                 # Directory for saving benchmark results (e.g., plots, data)
├── src/                     # Python source code
    ├── alveo_power_scraper.py # Scrapes power data using xbutil
    ├── alveo_runner.py      # Main class to interact with Alveo kernel
    ├── marimo/
    │   └── marimo.py        # Marimo notebook for interactive benchmarking
    └── plot_alveo_power.py  # Utility to plot power data
```

## Docker Setup & Building

The Docker image encapsulates all necessary dependencies, including a specific version of XRT, Python libraries, and PYNQ.

1.  **Navigate to the Docker directory**:
    ```bash
    cd docker
    ```

2.  **Build the Docker image**:
    You can use the provided shell script:
    ```bash
    ./docker_build.sh
    ```
    This script will build the image, tag it as `aimilefth/pynq_alveo_docker:coroni`, and attempt to push it (you can remove `--push` from the script if you only want to build locally).

    Alternatively, you can build manually:
    ```bash
    docker build -t aimilefth/pynq_alveo_docker:coroni .
    ```

## Deployment: Running the Container

The provided `docker_run.sh` script in the project root directory simplifies running the container and configuring access to the Alveo FPGA.


1.  **Run the container using the script**:
    ```bash
    bash docker_run.sh
    ```

    **Key features of `docker_run.sh`**:
    *   **Automatic Device Detection**: It automatically finds and passes Alveo management (`/dev/xclmgmt*`) and render (`/dev/dri/renderD*`) devices to the container.
    *   **Privileged Mode**: Runs the container in `--privileged` mode, often required for direct hardware access.
    *   **Host Network**: Uses `--network=host`, allowing easy access to Marimo via `localhost:8080`.
    *   **Essential Xilinx Mounts**: Mounts crucial host directories for Alveo operation:
        *   `/opt/xilinx/dsa`
        *   `/opt/xilinx/overlaybins`
        *   `/etc/xbutler`
    *   **Project Mounts**:
        *   `$(pwd)/src:/app/src` (your source code)
        *   `$(pwd)/mounted_dir:/app/mounted_dir` (for `.xclbin` files)
        *   `$(pwd)/outputs:/app/outputs` (for results)
    *   **Other Mounts**: `/dev/shm`, `/scrape` (purpose of `/scrape` specific to your setup).

    **Important Note on Alveo BDF for Power Scraping**:
    The `docker_run.sh` script does not explicitly pass the Alveo card's BDF (Bus:Device.Function, e.g., `0000:af:00.1`) to the container as an environment variable. The `AlveoRunner` class (specifically its `ALVEOPowerScraper` component) needs this BDF to collect power data.
    *   The Marimo notebook (`src/marimo/marimo.py`) currently uses a hardcoded BDF: `model = AlveoRunner(XCLBIN_PATH, params, device, "0000:af:00.1")`.
    *   **If you have multiple Alveo cards or your target card's BDF is different from `"0000:af:00.1"`, you will need to manually update this BDF string in `src/marimo/marimo.py` for power scraping to work correctly on the intended card.**
    *   You can find your card's BDF by running `xbutil scan` on your host machine.

## Using the Marimo Notebook

Once the container is running (after executing `./docker_run.sh`), it will start the Marimo editor.

1.  **Access Marimo**: Open your web browser and navigate to `http://localhost:8080`.
2.  **Notebook (`src/marimo/marimo.py`)**:
    The notebook provides an interactive way to:
    *   **Load the Alveo model**: Initializes `AlveoRunner` with the `attention_ae.xclbin` bitstream.
        *   As mentioned above, verify and (if necessary) update the hardcoded BDF in this cell if you intend to use power scraping and your card's BDF differs from `"0000:af:00.1"`.
    *   **Example Run**: Demonstrates basic execution of the loaded kernel.
    *   **Benchmark Latency**: Measures average execution time with and without data transfers.
    *   **Benchmark Power**: Collects and plots power consumption data during kernel execution using `ALVEOPowerScraper` and `plot_alveo_power.py`.
    *   **Clean Model**: Frees Alveo resources.

    Simply run the cells in the Marimo notebook to perform these actions. Outputs and plots will be displayed directly in the notebook or saved to the `outputs` directory.

## Key Components Explained

### `AlveoRunner` (`src/alveo_runner.py`)

This class is the primary interface for interacting with the Alveo FPGA kernel.

*   **Initialization**:
    ```python
    from src.alveo_runner import AlveoRunner, AlveoRunnerParameters
    params = AlveoRunnerParameters(input_buffer_elements=N, output_buffer_elements=M)
    model = AlveoRunner(
        bitstream_path="/app/mounted_dir/your_kernel.xclbin",
        parameters=params,
        device=pynq_device_object, # Optional: specific PYNQ device
        device_bus="0000:xx:yy.z"  # Device BDF for power scraping
    )
    ```
    It loads the bitstream, allocates PYNQ buffers for input/output, and optionally initializes `ALVEOPowerScraper` if a `device_bus` (BDF) is provided and valid.

*   **Key Methods**:
    *   `run(input_array)`: Executes the kernel once with `input_array` and returns the output.
    *   `timed_run(input_array)`: Like `run()`, but also measures and prints execution time.
    *   `run_vector(input_vector, output_shape=None)`: Processes larger input vectors by segmenting them to fit buffer sizes.
    *   `get_average_time_alveo(input_vector, iterations, warmup)`: Benchmarks kernel execution latency (no host-FPGA transfer time in loop).
    *   `get_average_time_alveo_transfers(input_vector, iterations, warmup)`: Benchmarks kernel execution latency including host-FPGA transfer time in loop.
    *   `get_power_data(seconds, transfers=False)`: Collects power data for a specified duration while continuously running the kernel. Requires `device_bus` to be correctly set during `AlveoRunner` initialization.
    *   `clean_class()`: Releases PYNQ buffers and frees the overlay.

### `ALVEOPowerScraper` (`src/alveo_power_scraper.py`)

This utility class is responsible for fetching power consumption data from the Alveo card.

*   It uses the `xbutil examine -d <device_bdf> --r electrical` command.
*   It parses the output of `xbutil` to extract various power metrics (Total Power, Power Rails, Voltages, Currents).
*   It handles differences in `xbutil` output format based on the XRT version.
*   It validates the provided device BDF during initialization.

### `plot_alveo_power.py` (`src/plot_alveo_power.py`)

Contains functions to process and visualize the power data collected by `ALVEOPowerScraper`.

*   `extract_alveo_power_data()`: Transforms the raw list of power data dictionaries into a Pandas DataFrame suitable for plotting.
*   `plot_alveo_power_data()`: Generates and displays/saves a plot of power metrics over time.

## Workflow Summary

1.  **Build** the Docker image (`./docker/docker_build.sh`).
2.  **Run** the Docker container using `./docker_run.sh`.
3.  **(If needed for power scraping)**: Identify your Alveo card's BDF (`xbutil scan` on host) and update it in `src/marimo/marimo.py` if it's different from the default `"0000:af:00.1"`.
4.  **Access** the Marimo notebook in your browser (`http://localhost:8080`).
5.  **Execute** cells in the notebook to load the kernel, run benchmarks, and view results.
6.  **Check** the `outputs/` directory for any saved data or plots.
7.  **Clean up** Alveo resources using the "Clean model" cell in Marimo before stopping the container or running other applications.

This setup provides a repeatable and isolated environment for your Alveo development and benchmarking tasks.