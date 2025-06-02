##  Install

###  Setup Python
1. Download and install [Anaconda](https://www.anaconda.com/download/success) to manage python environment. If you are using Linux, you can download it by
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```  

Run `Miniconda3-latest-Linux-x86_64.sh` and install.

2. Create a python 3.12 virtual env 
```
conda create -n madgs312 python=3.12
```

3. Activate this environment
```
conda activate madgs312
```

### Clone Madrona and Genesis
```
mkdir gs_render
cd gs_render
git clone --recurse-submodules git@github.com:yuhongyi/madrona.git
git clone git@github.com:yuhongyi/Genesis.git
```

### Install Madrona
1. Install the following libraries: `sudo apt install libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev mesa-common-dev`

2. Install
```sh
cd  madrona
mkdir  build
cd  build
cmake ..
make -j
cd ..
pip install -e .
cd ..
```

### Install Genesis
1. Install **PyTorch** first following the [official instructions](https://pytorch.org/get-started/locally/).

2. Install Genesis locally
```
cd Genesis
pip install -e ".[dev]"
```

### Render
1. In `gs_render/Genesis`, run
```
python examples/rigid/single_franka_batch_render.py
```

Images will be generated in `image_output`

2. To use ray tracer, change the `use_rasterizer=False` in `single_franka_batch_render.py`
```
        renderer = gs.options.renderers.BatchRenderer(
            use_rasterizer=True,
            batch_render_res=(512, 512),
        )
```