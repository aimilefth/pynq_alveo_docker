import sys
import subprocess
import datetime
from typing import Dict, Any, Tuple

# Define the threshold version tuple for <= comparison
LEGACY_THRESHOLD_VERSION = (2, 13)


class ALVEOPowerScraper:
    def __init__(self, device_id: str, xrt_version: str = "2.13") -> None:
        self.device_id = device_id
        self.command = f"xbutil examine -d {device_id} --r electrical"
        try:
            # Parse "2.13" into (2, 13) for comparison
            self.xrt_version_tuple: Tuple[int, ...] = tuple(
                map(int, xrt_version.split("."))
            )
        except ValueError:
            raise ValueError(
                f"xrt_version '{xrt_version}' is not in a valid format like '2.13' or '2.14.1'"
            )

    def get_power(self) -> Dict[str, Any]:
        try:
            if sys.version_info >= (3, 7):
                result = subprocess.run(
                    self.command,
                    check=True,
                    capture_output=True,
                    universal_newlines=True,
                    shell=True,
                )
            else:
                result = subprocess.run(
                    self.command,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    shell=True,
                )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "command '{}' return with error (code {}): {}".format(
                    e.cmd, e.returncode, e.output
                )
            )
        data = self.parse_power_data(result.stdout)
        data["timestamp"] = datetime.datetime.utcnow().isoformat()
        return data

    def parse_power_data(self, output: str) -> Dict[str, Any]:
        lines = output.split("\n")
        data = {}
        power_rails = {}
        data["Max Power"] = int(lines[5].split(":")[1].strip().split()[0])
        data["Power"] = float(lines[6].split(":")[1].strip().split()[0])
        data["Power Warning"] = lines[7].split(":")[1].strip()
        # Determine parsing strategy based on XRT version
        is_legacy_format: bool = self.xrt_version_tuple <= LEGACY_THRESHOLD_VERSION
        if is_legacy_format:
            power_rail_start_index = 10  # Corresponds to `index > 9`
            value_delimiter = "  "  # Double space
        else:
            power_rail_start_index = 11  # Corresponds to `index > 10`
            value_delimiter = ","  # Comma

        for index, line in enumerate(lines):
            if index >= power_rail_start_index and line.strip() != "":
                name = line.split(":")[0].strip()
                voltage_current = line.split(":")[1].strip().split(value_delimiter)
                temp_dir = {"Voltage": float(voltage_current[0].strip().split()[0])}
                if len(voltage_current) > 1:
                    temp_dir["Current"] = float(voltage_current[1].strip().split()[0])
                power_rails[name] = temp_dir
        data["Power Rails"] = power_rails
        return data
