Madrona: A GPU-Accelerated Game Engine for Batch Simulation
===========================================================

Madrona is a prototype game engine for building high-throughput GPU-accelerated _batch simulators_ that execute thousands of virtual worlds concurrently. By leveraging parallelism between and within worlds, simulators built on Madrona are able to execute on the GPU at millions of frames per second, enabling high performance sample generation for training agents using reinforcement learning, or other tasks requiring a high performance simulator in the loop. At its core, Madrona is built around the Entity Component System (ECS) architecture, which the engine uses to provide high-performance C++ interfaces for implementing custom logic and state that the engine can can automatically map to parallel batch execution on the GPU.

**Features**:
* Fully GPU-driven batch ECS implementation for high-throughput execution.
* CPU Backend for debugging & visualization. Simulators can execute on GPU or CPU with no code changes!
* Export ECS Simulation State as PyTorch tensors for efficient interop with learning code.
* (Optional) XPBD rigid body physics for basic 3D collision & contact support.
* (Optional) Simple 3D renderer for visualizing agent behaviors or debugging.

For more background and technical details, please read our paper: [_An Extensible, Data-Oriented Architecture for High-Performance, Many-World Simulation_](https://madrona-engine.github.io/shacklett_siggraph23.pdf), published in Transactions on Graphics / SIGGRAPH 2023.

**Disclaimer**: The Madrona engine is an early-stage research codebase. While we hope to attract interested users / collaborators with this release, there will be missing features / documentation / bugs, as well as breaking API changes as we continue to develop the engine. Please post any issues you find on this github repo.

Dependencies
------------

**Supported Platforms**
* Linux, Ubuntu 18.04 or newer
    * Other distros with equivalent or newer kernel / GLIBC versions will also work
* MacOS 13.x Ventura (or newer)
    * Requires full Xcode 14 install (not just Xcode Command Line Tools)
    * Currently no testing / support for Intel Macs
* Windows 11
    * Requires Visual Studio 16.4 (or newer) with recent Windows SDK

**General Dependencies**
* CMake 3.24 (or newer)
* Python 3.9 (or newer)

**GPU-Backend Dependencies**
* Volta or newer NVIDIA GPU
* CUDA 12.1 or newer (+ appropriate NVIDIA drivers)
* **Linux** (CUDA on Windows lacks certain unified memory features that Madrona requires)

These dependencies are needed for the GPU backend. If they're not present, Madrona's GPU backend will be disabled, but you can still use the CPU backend.

Getting Started
---------------

Madrona is intended to be integrated as a library / submodule of simulators built on top of the engine. Therefore, you should start with one of our example simulators, rather than trying to build this repo directly.

As a starting point for learning how to use the engine, we recommend the [Madrona3DExample project](https://github.com/shacklettbp/madrona_3d_example). This is a simple 3D environment that demonstrates the use of Madrona's ECS APIs, as well as physics and rendering functionality, with a simple task where agents must learn to press buttons and pull blocks to advance through a series of rooms.

For more ML-focused users interested in training agents at high speed, check out the [Madrona RL Environments repo](https://github.com/bsarkar321/madrona_rl_envs) that contains an Overcooked AI implementation where you can train agents in 2 minutes using Google Colab, as well as Hanabi and Cartpole implementations.

If you're interested in building a new simulator on top of Madrona, we recommend forking one of the above projects and adding your own functionality, or forking the [Madrona GridWorld repo](https://github.com/shacklettbp/madrona_gridworld) for an example with very little existing logic to get in your way. Basing your work on one of these repositories will ensure that the CMake build system and python bindings are setup correctly.

**Building:**

Instructions on building and testing the Madrona3DExample simulator are included below for Linux and MacOS:
```bash
git clone --recursive https://github.com/shacklettbp/madrona_3d_example.git
cd madrona_3d_example
pip install -e . 
mkdir build
cd build
cmake ..
make -j # Num Cores To Build With
```

You can then view the environment by running:
```bash
./build/viewer
```

Refer to the simulator's github page for further context / instructions on how to train agents.

**Windows Instructions**:
Windows users should clone the repository as above, and then open the root of the cloned repo in Visual Studio and build with the integrated CMake support. 
By default, Visual Studio has a build directory like `out/build/Release-x64`, depending on your build configuration. This requires changing the `pip install` command above to tell python where the C++ python extensions are located:
```
pip install -e . -Cpackages.madrona_3d_example.ext-out-dir=out/build/Release-x64
```

Code Organization
-----------------
As mentioned above, we recommend starting with the [Madrona3DExample project](https://github.com/shacklettbp/madrona_3d_example) for learning how to use Madrona's ECS APIs, as documentation within Madrona itself is still fairly minimal.

Nevertheless, the following files provide good starting points to start diving into the Madrona codebase:

The `Context` class: [`include/madrona/context.hpp`](https://github.com/shacklettbp/madrona/blob/main/include/madrona/context.hpp#L17) includes the core ECS API entry points for the engine (creating entities, getting components, etc): . Note that the linked file is the header for the CPU backend. The GPU implementation of the same interface lives in [`src/mw/device/include/madrona/context.hpp`](https://github.com/shacklettbp/madrona/blob/main/src/mw/device/include/madrona/context.hpp). Although many of the headers in `include/madrona` are shared between the CPU and GPU backends, the GPU backend prioritizes files in `src/mw/device/include` in order to use GPU specific implementations. This distinction should not be relevant for most users of the engine, as the public interfaces of both backends match.

The `ECSRegistry` class: [`include/madrona/registry.hpp`](https://github.com/shacklettbp/madrona/blob/main/include/madrona/registry.hpp) is where user code registers all the ECS Components and Archetypes that will be used during the simulation. Note that Madrona requires all the used archetypes to be declared up front -- unlike other ECS engines adding and removing components dynamically from entities is not currently supported.

The `TaskGraphBuilder` class: [`include/madrona/taskgraph_builder.hpp`](https://github.com/shacklettbp/madrona/blob/main/include/madrona/taskgraph_builder.hpp) includes the interface for building the taskgraph that will be executed to step the simulation across all worlds.

The `MWCudaExecutor` class: [`include/madrona/mw_gpu.hpp`](https://github.com/shacklettbp/madrona/blob/main/include/madrona/mw_gpu.hpp) is the entry point for the GPU backend.  
The `TaskGraphExecutor` class: [`include/madrona/mw_cpu.hpp`](https://github.com/shacklettbp/madrona/blob/main/include/madrona/mw_gpu.hpp) is the entry point for the CPU backend.

Citation
--------
If you use Madrona in a research project, please cite our SIGGRAPH paper!

```
@article{shacklett23madrona,
    title   = {An Extensible, Data-Oriented Architecture for High-Performance, Many-World Simulation},
    author  = {Brennan Shacklett and Luc Guy Rosenzweig and Zhiqiang Xie and Bidipta Sarkar and Andrew Szot and Erik Wijmans and Vladlen Koltun and Dhruv Batra and Kayvon Fatahalian},
    journal = {Transactions on Graphics},
    year    = {2023}
}
```
