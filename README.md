<h1>
<img src="logo.png" width=600><br>A GPU-Accelerated Game Engine for Batch Simulation
</h1>

Madrona is a prototype game engine for creating high-throughput, GPU-accelerated batch simulators: simulators that run thousands of virtual environment instances, and generate *millions of aggregate simulation steps per second, on a single GPU.* This efficiency is useful for high-performance AI agent training (e.g., via reinforcement learning), or for any task that requires a high-performance environment simulator tightly integrated "in-the-loop" of a broader application.

Madrona uses an [Entity Component System](https://github.com/SanderMertens/ecs-faq) (ECS) architecture. At this time Madrona provides APIs for games to specify custom game play logic and state in C++, and Madrona automatically maps this code to parallel batch execution on the GPU. Implementing a new game (or a new learning environment) in Madrona requires knowledge of data-parallel ECS concepts, but it does not require expert knowledge of GPU programming or GPU performance optimization. 

### Features: ###
* Fully GPU-driven batch ECS implementation for high-throughput execution.
* CPU backend for debugging and visualization. Simulators can execute on GPU or CPU with no code changes.
* Export ECS simulation state as PyTorch tensors for efficient interoperability with learning code.
* (Optional) XPBD rigid body physics for basic 3D collision and contact support.
* (Optional) Simple 3D renderer for visualizing agent behaviors and debugging.

**Disclaimer**: The Madrona engine is a research code base. We hope to attract interested users / collaborators with this release, however there will be missing features / documentation / bugs, as well as breaking API changes as we continue to develop the engine. Please post any issues you find on this github repo.

# Technical Paper

For more background and technical details, please read our paper: [_An Extensible, Data-Oriented Architecture for High-Performance, Many-World Simulation_](https://madrona-engine.github.io/shacklett_siggraph23.pdf), published in Transactions on Graphics / SIGGRAPH 2023.

For general background and tutorials on ECS programming abstractions and the motivation for the ECS design pattern's use in games, we recommend Sander Martens' excellent [ECS FAQ](https://github.com/SanderMertens/ecs-faq).

# Example Madrona-Based Simulators

### [Madrona3DExample](https://github.com/shacklettbp/madrona_3d_example)
* A simple 3D environment that demonstrates the use of Madrona's ECS APIs, as well as physics and rendering functionality, via a simple task where agents must learn to press buttons and pull blocks to advance through a series of rooms.

### [Overcooked AI](https://github.com/bsarkar321/madrona_rl_envs/tree/main/src/overcooked_env#overcooked-environment)
* A high-throughput Madrona rewrite of the [Overcooked AI environment](https://github.com/HumanCompatibleAI/overcooked_ai), a multi-agent learning environment based on the cooperative video game. Check out this repo for a Colab notebook that allows you to train overcooked agents that demonstrate optimal play in 2 minutes.

### [Hide and Seek](https://github.com/shacklettbp/gpu_hideseek)
* A reimplementation of OpenAI's "Hide and Seek" environment from the paper [Emergent Tool Use from Multi-Agent Autocurricula](https://openai.com/research/emergent-tool-use).

### [Hanabi](https://github.com/bsarkar321/madrona_rl_envs/tree/main/src/hanabi_env#hanabi-environment)
  * A Madrona version of the Hanabi card game based on Deepmind's [Hanabi Learning Environment](https://www.deepmind.com/publications/the-hanabi-challenge-a-new-frontier-for-ai-research).

### [Cartpole](https://github.com/bsarkar321/madrona_rl_envs/tree/main/src/cartpole_env#cartpole-environment)
  * The canonical RL training environment.

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

  These dependencies are needed for the GPU backend. If they are not present, Madrona's GPU backend will be disabled, but you can still use the CPU backend.

Getting Started
---------------

Madrona is intended to be integrated as a library / submodule of simulators built on top of the engine. Therefore, you should start with one of our example simulators, rather than trying to build this repo directly.

As a starting point for learning how to use the engine, we recommend the [Madrona3DExample](https://github.com/shacklettbp/madrona_3d_example) project. This is a simple 3D environment that demonstrates the use of Madrona's ECS APIs, as well as physics and rendering functionality, via a simple task where agents must learn to press buttons and pull blocks to advance through a series of rooms.

For ML-focused users interested in training agents at high speed, we recommend you check out the [Madrona RL Environments repo](https://github.com/bsarkar321/madrona_rl_envs) that contains an Overcooked AI implementation where you can train agents in two minutes using Google Colab, as well as Hanabi and Cartpole implementations.

If you're interested in authoring a new simulator on top of Madrona, we recommend forking one of the above projects and adding your own functionality, or forking the [Madrona GridWorld repo](https://github.com/shacklettbp/madrona_gridworld) as an example with very little existing logic to get in your way. Basing your work on one of these repositories will ensure that the CMake build system and python bindings are setup correctly.

### Building: ###

Instructions on building and testing the [Madrona3DExample](https://github.com/shacklettbp/madrona_3d_example) simulator are included below for Linux and MacOS:
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

Please refer to the [Madrona3DExample](https://github.com/shacklettbp/madrona_3d_example) simulator's github page for further context / instructions on how to train agents.

**Windows Instructions**:
Windows users should clone the repository as above, and then open the root of the cloned repo in Visual Studio and build with the integrated CMake support. 
By default, Visual Studio has a build directory like `out/build/Release-x64`, depending on your build configuration. This requires changing the `pip install` command above to tell python where the C++ python extensions are located:
```
pip install -e . -Cpackages.madrona_3d_example.ext-out-dir=out/build/Release-x64
```

Code Organization
-----------------

We recommend starting with the [Madrona3DExample](https://github.com/shacklettbp/madrona_3d_example) project for learning how to use Madrona's ECS APIs, as documentation within Madrona itself is still fairly minimal.

Nevertheless, the following files provide good starting points to start diving into the Madrona codebase:

The `Context` class: [`include/madrona/context.hpp`](https://github.com/shacklettbp/madrona/blob/main/include/madrona/context.hpp#L17) includes the core ECS API entry points for the engine (creating entities, getting components, etc): . Note that the linked file is the header for the CPU backend. The GPU implementation of the same interface lives in [`src/mw/device/include/madrona/context.hpp`](https://github.com/shacklettbp/madrona/blob/main/src/mw/device/include/madrona/context.hpp). Although many of the headers in `include/madrona` are shared between the CPU and GPU backends, the GPU backend prioritizes files in `src/mw/device/include` in order to use GPU specific implementations. This distinction should not be relevant for most users of the engine, as the public interfaces of both backends match.

The `ECSRegistry` class: [`include/madrona/registry.hpp`](https://github.com/shacklettbp/madrona/blob/main/include/madrona/registry.hpp) is where user code registers all the ECS Components and Archetypes that will be used during the simulation. Note that Madrona requires all the used archetypes to be declared up front -- unlike other ECS engines adding and removing components dynamically from entities is not currently supported.

The `TaskGraphBuilder` class: [`include/madrona/taskgraph_builder.hpp`](https://github.com/shacklettbp/madrona/blob/main/include/madrona/taskgraph_builder.hpp) includes the interface for building the task graph that will be executed to step the simulation across all worlds.

The `MWCudaExecutor` class: [`include/madrona/mw_gpu.hpp`](https://github.com/shacklettbp/madrona/blob/main/include/madrona/mw_gpu.hpp) is the entry point for the GPU backend.  

The `TaskGraphExecutor` class: [`include/madrona/mw_cpu.hpp`](https://github.com/shacklettbp/madrona/blob/main/include/madrona/mw_gpu.hpp) is the entry point for the CPU backend.

Citation
--------
If you use Madrona in a research project, please cite our SIGGRAPH 2023 paper:

```
@article{shacklett23madrona,
    title   = {An Extensible, Data-Oriented Architecture for High-Performance, Many-World Simulation},
    author  = {Brennan Shacklett and Luc Guy Rosenzweig and Zhiqiang Xie and Bidipta Sarkar and Andrew Szot and Erik Wijmans and Vladlen Koltun and Dhruv Batra and Kayvon Fatahalian},
    journal = {ACM Trans. Graph.},
    volume = {42},
    number = {4},
    year    = {2023}
}
```
