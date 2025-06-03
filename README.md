# FedPHE: Federated Learning with Homomorphic Encryption

This code applies homomorphic encryption (CKKS scheme) to federated learning. The implementation is adopted from [**Federated-Learning-PyTorch**](https://github.com/c-gabri/Federated-Learning-PyTorch). Clients train and encrypt models locally, while the server aggregates encrypted models without accessing the exact model parameters.

## Abstract

[Insert your paper abstract here]

## Citation

If you find FedPHE useful or relevant to your research, please kindly cite our paper using the following BibTeX:

```bibtex
@inproceedings{[citation_key],
  title={[Paper Title]},
  author={[Author Names]},
  booktitle={[Conference or Journal Name]},
  pages={[Page Numbers]},
  year={[Year]},
  organization={[Organization]}
}
```

## Usage

### Environment

This code has been tested under the following environment:

- **Operating System**: Ubuntu 20.04.2
- **GPU Support**: CUDA + cuDNN
- **Python Version**: 3.7
- **Compiler**: g++ 9.4.0
- **Package Manager**: Anaconda 4.8.2

### Python Requirements

Install all required packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Package versions:**
```
matplotlib==3.5.3
numpy==1.21.6
scikit_learn==1.0.2
timm==0.9.12
torch==1.13.1
torchaudio==0.13.1
torchinfo==1.8.0
torchsummary==1.5.1
torchtext==0.14.1
torchvision==0.14.1
```

### Install OpenFHE

OpenFHE requires compiling both the C++ source code and Python wrapper.

#### Compiling C++ Source Code

Follow the official installation guide: [OpenFHE Linux Installation](https://openfhe-development.readthedocs.io/en/latest/sphinx_rsts/intro/installation/linux.html)

1. **Install system packages:**
   ```bash
   sudo apt-get install build-essential cmake clang libomp5 libomp-dev autoconf
   ```

2. **Configure clang as the default compiler:**
   
   For clang 11:
   ```bash
   export CC=/usr/bin/clang-11
   export CXX=/usr/bin/clang++-11
   ```
   
   For default clang version (e.g., v6 in Ubuntu 20.04):
   ```bash
   export CC=/usr/bin/clang
   export CXX=/usr/bin/clang++
   ```

3. **Clone and checkout specific version:**
   ```bash
   git clone https://github.com/openfheorg/openfhe-development.git
   cd openfhe-development
   git checkout v1.1.2
   ```

4. **Compile and install:**
   ```bash
   mkdir build && cd build
   cmake ..
   make
   make install
   ```

5. **Test installation:**
   ```bash
   # Run unit tests
   sudo make testall
   
   # Run sample code
   bin/examples/pke/simple-integers
   ```

#### Compiling Python Wrapper

Follow the official Python wrapper guide: [OpenFHE Python](https://github.com/openfheorg/openfhe-python)

1. **Clone and checkout specific version:**
   ```bash
   git clone https://github.com/openfheorg/openfhe-python.git
   cd openfhe-python
   git checkout v0.8.5
   ```

2. **Install required Python package:**
   ```bash
   pip install "pybind11[global]"
   ```

3. **Compile and install:**
   ```bash
   mkdir build && cd build
   cmake ..  # Use -DOpenFHE_DIR=/path/to/installed/openfhe if installed elsewhere
   make
   make install  # May require sudo
   ```

4. **Test installation:**
   ```bash
   python -c "__import__('openfhe')"
   ```

5. **Copy compiled library:**
   Copy the compiled `openfhe.cpython-37m-x86_64-linux-gnu.so` file to:
   ```
   <path_to_project>/Federated-Learning-PyTorch_HE_share/src/
   ```

### Install Additional Libraries

```bash
pip install tenseal==0.3.14
pip install Pyfhel==3.4.2
```

## Experiments

### Supported Models
- LeNet
- MobileNet v1, v2, v3
- ResNet-18, ResNet-34, ResNet-50
- EfficientNet-B0, EfficientNet-B5, EfficientNet-B7

### Experimental Settings

#### Speed Tests
- **Clients**: 3
- **Training rounds**: 5
- **Local epochs**: 1
- **Batch size**: 16 (lower for EfficientNet-B7 due to memory requirements)

#### Accuracy Tests
- **Clients**: 3
- **Training rounds**: 10
- **Local epochs**: 5
- **Batch size**: 16 (lower for EfficientNet-B7 due to memory requirements)

## Running the Code

### Prerequisites
Make sure to activate your virtual environment if using Anaconda before executing any commands.

### Basic Execution
```bash
python -u main.py --he_lib TenSeal_CKKS_without_flatten --model lenet 2>&1 | tee training_log.log
```

### Automated Execution Script
Use the provided execution script for automated testing:

```bash
<path_to_project>/Federated-Learning-PyTorch_HE_share/execute.sh
```

This script:
- Executes tests with predefined arguments
- Logs accuracy results to: `<HE_library>_<model_arch>_<ring_dimension>_<scale_bit>_<date>.log`
- Saves execution times to: `save/<HE_library>_<ring_dimension>_<scale_bit>_<datetime>/`

**Example log files:**
- `OpenFHE_mobilenet_v3_4096_14_20240805.log`
- `OpenFHE_CKKS_4096_14_2024-08-12_19-19-24/`

### Exporting Results
Use the provided scripts to export execution times to Excel files:
- `src/export_ckpt_to_excel.py`
- `call_export_ckpt_to_excel.sh`

**Note**: Currently supports settings for 3 clients, 5 training rounds, and 1 local epoch.

## Implementation Notes

- The current implementation only supports selective encryption on the CKKS scheme
- All configuration arguments are defined in `options.py`
- Main execution is handled by `main.py` with setup arguments

## Command Line Arguments

```
usage: python main.py [ARGUMENTS]

Algorithm Arguments:
  --rounds ROUNDS              Number of communication rounds (default: 200)
  --iters ITERS               Number of iterations per round (default: None)
  --num_clients NUM_CLIENTS   Number of clients (default: 100)
  --frac_clients FRAC_CLIENTS Fraction of clients selected per round (default: 0.1)
  --train_bs TRAIN_BS         Client training batch size (default: 50)
  --epochs EPOCHS             Number of client epochs (default: 5)
  --hetero HETERO             Probability of stragglers (default: 0)
  --drop_stragglers           Drop stragglers (default: False)
  --server_lr SERVER_LR       Server learning rate (default: 1)
  --server_momentum           Server momentum for FedAvgM (default: 0)
  --mu MU                     Mu parameter for FedProx (default: 0)
  --centralized               Use centralized algorithm (default: False)
  --fedsgd                    Use FedSGD algorithm (default: False)
  --fedir                     Use FedIR algorithm (default: False)
  --vc_size VC_SIZE          Virtual client size for FedVC (default: None)

Dataset and Split Arguments:
  --dataset {cifar10,fmnist,mnist}  Dataset selection (default: cifar10)
  --dataset_args             Dataset arguments (default: augment=True)
  --frac_valid               Validation fraction (default: 0)
  --iid IID                  Client distribution identicalness (default: inf)
  --balance BALANCE          Client distribution balance (default: inf)

Model, Optimizer, and Scheduler Arguments:
  --model {cnn_cifar10,cnn_mnist,efficientnet,ghostnet,lenet5,lenet5_orig,mlp_mnist,mnasnet,mobilenet_v3}
                             Model selection (default: lenet5)
  --model_args               Model arguments (default: ghost=True,norm=None)
  --optim {adam,sgd}         Optimizer selection (default: sgd)
  --optim_args               Optimizer arguments (default: lr=0.01,momentum=0,weight_decay=4e-4)
  --sched {const,fixed,plateau_loss,step}  Scheduler selection (default: fixed)
  --sched_args               Scheduler arguments (default: None)

Output Arguments:
  --client_stats_every       Client statistics frequency (default: 0)
  --server_stats_every       Server statistics frequency (default: 1)
  --name NAME                Experiment name for logging (default: None)
  --no_log                   Disable logging (default: False)
  --no_save                  Disable checkpoints (default: False)
  --quiet, -q                Reduce output verbosity (default: False)

Other Arguments:
  --test_bs TEST_BS          Test/validation batch size (default: 256)
  --seed SEED                Random seed (default: 0)
  --device {cuda:0,cpu}      Training device (default: cuda:0)
  --resume                   Resume from checkpoint (default: False)
  --help, -h                 Show help message (default: False)

Homomorphic Encryption Arguments:
  --he_lib                   HE library choice: OpenFHE_CKKS, TenSeal_CKKS_without_flatten, Pyfhel_CKKS
  --ring_dim                 Ring dimension: [1024, 2048, 4096, 8192]
  --scale_bit                Scaling bit size: [14, 20, 33, 40, 52]
```

## Acknowledgments

We gratefully acknowledge the following projects and contributors:

- [**Federated-Learning-PyTorch**](https://github.com/c-gabri/Federated-Learning-PyTorch): For the federated learning implementation framework
- [**FedML-HE**](https://arxiv.org/abs/2303.10837): For the innovative algorithm that inspired our "magnitude-based sensitivity" approach

## Authors

This framework was developed by Ren-Yi Huang at the University of South Florida.

## Contact

If you have any questions, please feel free to:
- Open an issue on GitHub
- Contact us via email

## License

This repository is released under the GNU General Public License. See [LICENSE](LICENSE) for additional details.