## Safety Guaranteed Robust Multi-Agent Reinforcement Learning with Hierarchical Control for Connected and Automated Vehicles

#### Presented in ICRA 2025

- Paper: \[[arxiv](https://arxiv.org/abs/2309.11057)\]
- Project Website: \[[link](https://zhili-zh.github.io/projects/project_safe_robust.html)\]

### Carla 0.9.15 Installation

Please check the following link for instruction on Carla Installations.

- https://carla.readthedocs.io/en/0.9.15/start_quickstart/

### Environment Configuration

Create a Conda environment with python 3.10

- ```
  conda create -n safe_rmm python=3.10
  ```

Install PyTorch 2.7 (with cuda 12.8)

- ```bash
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  ```

Install the Carla Python API 0.9.15

- ```
  pip install carla
  ```

Install the other required packages:

* ```
  pip install -r requirements.txt
  ```


### Quick Start to Training
* Having install the quick-start package of Carla 0.9.15, you can launch the CARLA simulator by running the `./CarlaUE4.sh` script (Linux) or `CarlaUE4.exe` (Windows).

  In Windows: 

  * ```
    .\CarlaUE4.exe -ResX=300 -ResY=300 -quality-level=Epic  -carla-rpc-port=2000
    ```

  In Linux (Ubuntu):

  - ```
    nohup ~/<Your_Carla_PATH>/CarlaUE4.sh -RenderOffScreen -quality-level=Low  -carla-rpc-port=2000 &
    ```

* In a separate terminal, run the script under the root directory

  In Windows

  * ```
    .\run_SafeRMM_win.ps1
    ```

  * ```
    .\run_SafeRMM_Linux.sh
    ```

