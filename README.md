# UHRbiqa
This repo is the official implementation of "Super-resolved pseudo references in dual-branch embedding for blind ultra-high-definition image quality assessment"

dataset can be found in link: https://pan.baidu.com/s/1vjKW3GDDGtqI8cMYxT50ZA?pwd=9949 


### Create a new Conda environment

1. First, create and activate a new Conda environment：
   ```bash
   conda create -n SURPRIDE python=3.10
   conda activate SURPRIDE
2. Install frameworks such as PyTorch (it is recommended to refer to the version requirements in requirements.txt and use the recommended commands from the official website)
2. Install the other required packages
   ```bash
   pip install xxx

### How to Run

To execute the code for ablation studies and main experiments, follow the steps below:

#### Run Main Experiments
1. Navigate to the `main_experiments` directory:
   ```bash
   cd main_experiments
2. Run the target script (taking bid.py as an example):
   ```python
   python BIQ.py
