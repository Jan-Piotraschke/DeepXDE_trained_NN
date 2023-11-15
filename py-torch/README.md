# Training and saving Neural Networks using DeepXDE

Create a virtual env  
Install the requirements: ```pip3 install -r requirements.txt```  
Run ```python3 src/floquet_modes.py```

## Run the trained Model

Test your compiled C++ file by running ```./build/main-app model/traced_model.pt```  
Test your Rust code by running ```make run```

## Steps to reproduce using Rust

- install miniforge with homebrew -- See https://naolin.medium.com/conda-on-m1-mac-with-miniforge-bbc4e3924f2b
- create a new conda environment: `conda env create -f environment.yml`
- activate the new environment: `conda activate tch-rs-demo`
- create a symlink in this repo: `ln -sf /opt/homebrew/Caskroom/miniforge/base/envs/tch-rs-demo/lib/python3.10/site-packages/torch/ torch`