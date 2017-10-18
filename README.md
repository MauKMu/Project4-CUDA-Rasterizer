CUDA Rasterizer
===============

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Mauricio Mutai
* Tested on: Windows 10, i7-7700HQ @ 2.2280GHz 16GB, GTX 1050Ti 4GB (Personal Computer)

### Overview

#### Introduction

The aim of this project was to implement a simple GPU rasterizer. A rasterizer is a program that takes in 3D vertices that make up models, as well as additional data, such as surface normals and texture maps, and outputs a 2D image representing a view of the models.

The rasterization process involves repeating the same operations on different pieces of data, which makes it a good fit for a parallel program that makes use of a GPU's capabilities.

In addition, this process involves several steps that can be logically separated from one another. Thus, we can build a rasterization pipeline that connects each of the steps in order to produce the final image, while allowing us to modify each step individually.

In my implementation, these steps (or pipeline stages) are:

* Vertex Processing (transforming from 3D into projected 2D space)
* Primitive Assembly (assemble 2D vertices into pieces of geoemetry)
* Back-face Culling (optional -- make next stages ignore pieces of geometry that cannot be rendered due to their orientation)
* Rasterizer (generating "pixel candidates", or fragments, from the 2D data)
* Fragment Shader (coloring fragments)
* Copy to PBO (send image to be displayed)

#### Features

Below are the rasterizer's main features:

* Back-face culling
* UV texture mapping with support for bilinear interpolation and perspective-correct mapping
* Supersample antialiasing
* Basic Lambert shading

### Analysis

#### Scenes used for Analysis

The main scene used for this analysis was `duck.gltf`. This was rendered in two ways. One is the default view that appears when the rasterizer first loads:

![](img/duck-default.PNG)

The other view was obtained by zooming into the duck:

![](img/duck-zoom.PNG)

#### Bare-bones performance overview

Below is a breakdown of the percentage of time spent in each stage of the pipeline for the default duck render:

![](img/percent-bbones-duck-default.png)

As we can see, most of our time is spent on the fragment shader. This makes sense, since each triangle generates many fragments.

### Credits

* [tinygltfloader](https://github.com/syoyo/tinygltfloader) by [@soyoyo](https://github.com/syoyo)
* [glTF Sample Models](https://github.com/KhronosGroup/glTF/blob/master/sampleModels/README.md)
