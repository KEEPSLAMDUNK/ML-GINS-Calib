# ML-GINS-Calib

[![arXiv](https://img.shields.io/badge/arXiv-2507.08349-b31b1b.svg)](https://arxiv.org/abs/2507.08349)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

Joint optimization-based targetless extrinsic calibration for multiple LiDARs and GNSS-aided INS of ground vehicles.

## Overview

ML-GINS-Calib is a calibration method that performs joint optimization for extrinsic calibration between multiple LiDAR sensors and GNSS-aided Inertial Navigation Systems (INS) on ground vehicles. The approach is targetless, eliminating the need for calibration targets.

## Requirements

- Ubuntu 20.04/22.04

## Dependencies

### System Dependencies

```bash
sudo apt update
sudo apt install software-properties-common build-essential cmake git -y
```

### GTSAM Library

```bash
sudo add-apt-repository ppa:borglab/gtsam-release-4.1 -y
sudo apt update
sudo apt install libgtsam-dev libgtsam-unstable-dev -y
```

### Ceres Solver

```bash
sudo apt install libceres-dev -y
```

### Manif Library
```bash
git clone https://github.com/artivis/manif.git
cd manif
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
cd ../..
```

## Building

### Quick Build

```bash
chmod +x scripts/build.sh
./scripts/build.sh
```

### Manual Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Usage

### Basic Usage

```bash
./build/bin/ml_gins_calibrator
```

### Dataset

You can download the experimental dataset from Baidu Pan:

**Link:** https://pan.baidu.com/s/1DVwzhEv6-C0_kyXEfRuYeg?pwd=1uy3  
**Extraction Code:** 1uy3

### Configuration

You can modify the relevant configuration parameters by editing the `config/params.yaml` file:

#### Data Configuration
- `data_dir`: Data file directory path
- `gnss/ins`: GNSS/INS data filename
- `data_mask_path`: Data mask file path
- `gins_installation_height`: GNSS/INS installation height

#### Optimization Parameters
- `optimization_params.num_threads`: Number of optimizer threads
- `optimization_params.lidar_gins_optimization`: First stage LiDAR-GNSS/INS optimization parameters
  - `max_iterations`: Maximum number of iterations (default: 20)
  - `max_correspondence_distance`: Maximum correspondence distance (default: 0.5m)
- `optimization_params.multi_lidar_optimization`: Second stage multi-LiDAR optimization parameters
  - `max_iterations`: Maximum number of iterations (default: 30)
  - `max_correspondence_distance`: Maximum correspondence distance (default: 0.5m)

#### Keyframe Selection Parameters
- `keyframe_selection.rotation_threshold`: Rotation threshold in radians (default: 0.6)
- `keyframe_selection.translation_threshold`: Translation threshold in meters (default: 5.0)

#### Point Cloud Processing Parameters
- `point_cloud.voxel_size`: Voxel grid downsampling size (default: 0.2m)
- `point_cloud.downsample_leaf_size`: Downsampling leaf size (default: 0.1m)
- `point_cloud.ground_distance_threshold`: Ground segmentation distance threshold (default: 0.1m)

#### LiDAR Sensor Configuration
- `lidar_list`: List of active LiDAR sensors
- `init_ext`: Initial extrinsic parameter estimates for each LiDAR relative to IMU
  - `position`: Position parameters (x, y, z)
  - `orientation`: Orientation quaternion (w, x, y, z)

#### Output Configuration
- `output.extrinsic_output_file`: Output file path for extrinsic parameters

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{wang2025joint,
  author    = {Junhui Wang and Yan Qiao and Chao Gao and Naiqi Wu},
  title     = {Joint Optimization-based Targetless Extrinsic Calibration for Multiple LiDARs and GNSS-Aided INS of Ground Vehicles},
  journal   = {arXiv preprint arXiv:2507.08349},
  year      = {2025},
  url       = {https://arxiv.org/abs/2507.08349}
}
```

## Roadmap

- [ ] Code refactoring and optimization
- [x] Upload experimental datasets

## Contact

For questions and support, please open an issue on GitHub.
