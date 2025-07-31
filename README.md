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
./build/bin/ml_gins_calibrator <data_directory> <keyframe_distance>
```

### Example

```bash
./build/bin/ml_gins_calibrator data/ 3
```

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
- [ ] Upload experimental datasets

## Contact

For questions and support, please open an issue on GitHub.
