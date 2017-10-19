CUDA Rasterizer
===============

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Mauricio Mutai
* Tested on: Windows 10, i7-7700HQ @ 2.2280GHz 16GB, GTX 1050Ti 4GB (Personal Computer)

![](img/duck-30s.gif)

## Overview

### Introduction

The aim of this project was to implement a simple GPU rasterizer. A rasterizer is a program that takes in 3D vertices that make up models, as well as additional data, such as surface normals and texture maps, and outputs a 2D image representing a view of the models.

The rasterization process involves repeating the same operations on different pieces of data, which makes it a good fit for a parallel program that makes use of a GPU's capabilities.

In addition, this process involves several steps that can be logically separated from one another. Thus, we can build a rasterization pipeline that connects each of the steps in order to produce the final image, while allowing us to modify each step individually.

In my implementation, these steps (or pipeline stages) are:

* Vertex Processing (transforming from 3D into projected 2D space)
* Primitive Assembly (assemble 2D vertices into pieces of geoemetry, usually triangles)
* Back-face Culling (optional -- make next stages ignore pieces of geometry that cannot be rendered due to their orientation)
* Rasterizer (generating "pixel candidates", or fragments, from the 2D data)
* Fragment Shader (coloring fragments)
* Copy to PBO (send image to be displayed)

### Features

Below are the rasterizer's main features:

* Back-face culling
* UV texture mapping with support for bilinear interpolation and perspective-correct mapping
    * If not texture is present, the normals are used for coloring instead. This can be seen in 'CesiumMilkTruck.gltf'.
* Supersample antialiasing
* Basic Lambert shading
* Render only vertices of triangles (with configurable size)
* Render only edges of triangles (currently supports only fixed color and size)

## Analysis

### Scenes used for Analysis

The main scene used for this analysis was `duck.gltf`. This was rendered in two ways. One is the default view that appears when the rasterizer first loads:

![](img/duck-default.PNG)

The other view was obtained by zooming into the duck:

![](img/duck-zoom.PNG)

### Bare-bones performance overview

Below is a breakdown of the percentage of time spent in each stage of the pipeline for the default duck render:

![](img/percent-bbones-duck-default.png)

As we can see, most of our time is spent on the fragment shader. This makes sense, since each triangle is small relative to the screen, so the rasterizer stage spends relatively less time generating fragments.

Below is the same breakdown, but for the zoomed-in render:

![](img/percent-bbones-duck-zoom.png)

Now, the rasterizer stage takes up much more time than any of the other stages. The triangles are now larger with regards to the screen, the rasterizer spends more time iterating over all the pixels covered by the triangle and generating fragments for them.

Next, we will investigate one optimization I implemented to attempt to speed up my pipeline.

### Back-face culling

#### Overview

Back-face culling is a common optimization in rasterizers. It is based on the idea that triangles have a "correct" side -- that is, they should be visible from one side, but not the other. This information is encoded by the triangle's surface normal. If we look at the triangle such that the normal is pointing towards us, then we are looking at the visible side.

Thus, we can choose to ignore triangles if their normal is not facing us.

This was implemented using `thrust::remove_if` as a stream compaction function.

#### Performance impact

Below is a stage-wise comparison of the default duck render with back-face culling enabled and disabled:

![](img/stages-cull-duck-default.png)

As we would expect, the time spent in the rasterizer is decreased, since it has to process fewer triangles. However, the overhead of performing stream compaction on the list of triangles actually makes the back-face culling version slower overall! However, we can assume the cost of stream compaction will remain roughly the same even if we zoom in on the duck model. So what does the performance look like for that case?

Below is the same comparison, but for the zoomed-in duck render:

![](img/stages-cull-duck-zoom.png)

As we can see, the rasterizer performance has a huge improvement thanks to the culling stage. The cost of culling is now minimal compared to the savings we obtained in the rasterizer stage.

#### Further optimizations

Currently, the check to see if a triangle should be culled is made in the `shouldCull()` function, which is called as part of the stream compaction process. Perhaps this could be moved to the vertex processing stage or primitive assembly stage, although it probably would not create any noticeable improvements.

Unfortunately, the cost of stream compaction means back-face culling does not always improve performance. Essentially, we need to cull away triangles that would generate many fragments to make up for this cost. A faster stream compaction would certainly help to offset this cost.

### UV texture mapping -- perspective-correct interpolation

#### Overview

When projecting 3D vertices onto a 2D space, the "depth" of these vertices gets distorted. This makes it so interpolating values from these coordinates naively (e.g. using only barycentric coordinates) leads to incorrect results. For example, see the following render of `checkerboard.gltf`, which was made using this naive interpolation method:

![](img/persp-wrong.PNG)

The expected result can be seen in this render that uses the perspective-correct interpolation method:

![](img/persp-correct.PNG)

This method basically involves computing a perspective-correct Z (or depth) value using the barycentric coordinates and the inverse of each vertex's depth values, and then multiplying each barycentric coordinate by `(perspective-correct Z) / (each vertex's Z)`.

#### Performance impact

Below is a comparison of the time spent on each pipeline stage for the default duck render, with perspective-correct enabled and disabled:

![](img/stages-persp-duck-default.png)

As we can see, the time spent in the rasterizer increases slightly, but not very significantly. This makes sense, since the rasterizer needs to perform additional work for each fragment in order to calculate the perspective-correct weights.

#### Further optimizations

In order to speed up computations in the rasterizer, the inverse of each vertex's Z value could be computed in the vertex processing stage.

### UV texture mapping -- bilinear filtering

#### Overview

Simply put, bilinear filtering is a way of making texture mapped models have a smoother coloring. With bilinear filtering, we perform bilinear interpolation based on the UV coordinates, such that the final sampled color is based on up to four pixels from the texture, rather than just one. This allows for a smoother transition when there is a change in color in the texture.

Below is a zoomed-in version of `CesiumMilkTruck.gltf` without bilinear filtering enabled:

![](img/bilinear-no.PNG)

Compare to the same scene, rendered with bilinear filtering enabled:

![](img/bilinear-yes.PNG)

The white curve in the Cesium logo looks much smoother when bilinear filtering is turned on.

#### Performance impact

Below is a comparison of the time spent on each pipeline stage for the default duck render, with bilinear filtering enabled and disabled:

![](img/stages-bilinear-duck-default.png)

As we can see, the time spent in the rasterizer increases slightly, moreso than when perspective-correct interpolation is enabled. This is expected, since the rasterizer needs to sample more pixels for each fragment, which increases memory accesses, which are known to be slow and important bottlenecks in GPU programs.

#### Further optimizations

Due to the increase in memory accesses required by this feature, it is likely that performance could be vastly improved if shared memory were used.

### Supersample antialiasing

#### Overview

The basic idea behind antialiasing is that we are generating an image with finite precision out of 3D data with much higher (though technically still finite) granularity (i.e. colors are only precise at the pixel-level, rather than continuously). 

We can mitigate the errors generated by this by rendering larger images, so each pixel covers a smaller area of the scene. However, if we want to produce images of a certain size, we cannot arbitrarily increase the image's dimensions. Instead, we can use antialiasing to effectively render a larger image, but then compress it to the desired size by averaging the pixel colors.

Below is a very zoomed-in duck with no antialiasing:

![](img/aa-none.PNG)

And below is the same scene, but with 4x supersampling antialiasing:

![](img/aa-4x.PNG)

As we can see, the final image looks smoother, with color transitions that are less harsh.

#### Performance impact

Below is a comparison of the time spent on each pipeline stage for the default duck render, with bilinear filtering enabled and disabled:

![](img/stages-aa-duck-default.png)

As one would expect, the rasterizer needs to perform much more work, since it has to iterate over more pixels and generate more fragments. More importantly, since more fragments are generated, the fragment shader has to do significantly more work. 

In addition, in my implementation, the compression from the 4x resolution to the normal resolution is done in the fragment shader. This means this stage needs to perform more memory access in order to read the fragments that correspond to the final pixel in order to compute its final color. This explains the large increase in runtime.

#### Further optimizations

Similar to bilinear filtering, the increase in runtime mainly comes from having more memory accesses, so using shared memory could greatly improve the performance of antialiasing.

### Vertex-only rendering

#### Overview

In this mode, the rasterizer only renders each triangle's vertices, rather than its entire surface. The size of the vertex (that is, how many pixels each vertex should cover) is configurable.

Below is the duck rendered in this mode:

![](img/duck-vertices.PNG)

#### Performance impact

Below is a comparison of the time spent on each pipeline stage for the default duck render, depending on whether we are rendering only vertices:

![](img/stages-verts-duck-default.png)

Interestingly enough, the time spent in the rasterizer increases. This is probably because of the vertex size used (6), which means each vertex will generate 36 fragments (assuming it is not close to the edge of the screen). This increases the time spent interpolating values and atomically checking and writing to the depth buffer.

Indeed, if we decrease the vertex size to 2, we get the following result:

![](img/stages-verts-sizes-duck-default.png)

Which shows the rasterizes runs more quickly for a smaller vertex size.

#### Further optimizations

While not a performance optimization, it would be interesting to scale the vertex size depending on its distance from the camera. Right now, they are all drawn at the same size.

### Edge-only rendering

#### Overview

In this mode, the rasterizer only renders each triangle's edges, rather than its entire surface. Currently, the thickness of the lines and its color is fixed (1 pixel and white, respectively).

Below is the duck rendered in this mode:

![](img/duck-edges.PNG)

#### Performance impact

Below is a comparison of the time spent on each pipeline stage for the default duck render, depending on whether we are rendering only edges:

![](img/stages-edges-duck-default.png)

Here, the time spent in the rasterizer also increases. This is probably due to the way I implemented Bresenham's line-drawing algorithm, which leads to massive warp divergence.

#### Further optimizations

If we found a way to implement the line-drawing algorithm with less warp divergence, we could probably improve performance significantly.

## Enabling/disabling features

The following `#define`s in `rasterize.cu` control whether certain features are available:

```
#define CUDA_MEASURE 1 // if non-zero, repeatedly measures runtime of each stage over 1000 frames and prints average
// ...
#define PERSP_CORRECT 0 // if non-zero, enables perspective-correct interpolation
#define BILINEAR_INTERP 0 // if non-zero, enables bilinear texture filtering
#define BACK_FACE_CULLING 0 // if non-zero, enables back-face culling
#define SSAA_FACTOR 1 // if more than 1, enables (SSAA_FACTOR^2)x SSAA, e.g. 4x SSAA if SSAA_FACTOR is 2
// ...
#define RENDER_FULL_TRIANGLE 0
#define RENDER_VERTICES 1
#define RENDER_EDGES 2

#define RENDER_MODE RENDER_FULL_TRIANGLE // changes render mode according to #defines above

#define VERTEX_RENDER_SIZE 2 // changes dimension of vertex when in vertex-only mode
```

## Credits

* [tinygltfloader](https://github.com/syoyo/tinygltfloader) by [@soyoyo](https://github.com/syoyo)
* [glTF Sample Models](https://github.com/KhronosGroup/glTF/blob/master/sampleModels/README.md)
