# 🛠️ Installation

## 💻 Environments

Please follow the instructions to install the conda environments and the dependencies of the codebase. We recommend using CUDA 11.x during installations to avoid compatibility issues (remember to replace `11.x` in the following commands with your own CUDA version like `11.4`).

1. Create a new conda environment and activate the environment.
    ```bash
    conda create -n dsp python=3.8
    conda activate dsp
    ```

2. Install necessary dependencies.
    ```bash
    conda install cudatoolkit=11.x
    pip install -r requirements.txt
    ```
## 🦾 Real Robot

**Hardwares**.
- Flexiv Rizon 4 Robotic Arm (or other robotic arms)
- Dahuan AG-95 Gripper (or other grippers)
- Intel RealSense RGB-D Camera (D415/D435/L515)

**Softwares**.
- Ubuntu 20.04 (tested) with previous environment installed.
- If you are using Flexiv Rizon robotic arm, install the [Flexiv RDK](https://rdk.flexiv.com/manual/getting_started.html) to allow the remote control of the arm. Specifically, download [FlexivRDK v0.9](https://github.com/flexivrobotics/flexiv_rdk/releases/tag/v0.9) and copy `lib_py/flexivrdk.cpython-38-[arch].so` to the `device/robot/` directory. Please specify `[arch]` according to your settings. For our platform, `[arch]` is `x86_64-linux-gnu`.
- If you are using Dahuan AG-95 gripper, install the following python packages for communications.
  ```
  pip install pyserial==3.5 modbus_tk==1.1.3 
  ```
- If you are using Intel RealSense RGB-D camera, install the python wrapper `pyrealsense2` of `librealsense` according to [the official installation instructions](https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python#installation).