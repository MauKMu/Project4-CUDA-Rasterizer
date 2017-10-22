/**
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2016
 * @copyright University of Pennsylvania & Mauricio Mutai
 */

#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include <util/tiny_gltf_loader.h>
#include "rasterizeTools.h"
#include "rasterize.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

#define CUDA_MEASURE 0
#if CUDA_MEASURE == 0
// "undefine" cuda event things
#define cudaEventCreate(x) ((void)(0))
#define cudaEventRecord(x) ((void)(0))
#define cudaEventSynchronize(x) ((void)(0))
#define cudaEventElapsedTime(x, y, z) ((void)(0))
#endif // CUDA_MEASURE == 0
#define PERSP_CORRECT 0
#define BILINEAR_INTERP 0
#define BACK_FACE_CULLING 1
#define SSAA_FACTOR 1
#if SSAA_FACTOR <= 0
#error SSAA_FACTOR must be > 0
#endif

#define RENDER_FULL_TRIANGLE 0
#define RENDER_VERTICES 1
#define RENDER_EDGES 2

#define RENDER_MODE RENDER_FULL_TRIANGLE

#define VERTEX_RENDER_SIZE 2

namespace {

  struct cudaMat4 {
    glm::vec4 x;
    glm::vec4 y;
    glm::vec4 z;
    glm::vec4 w;
  };

  // LOOK: This is a custom function for multiplying cudaMat4 4x4 matrixes with vectors.
  // This is a workaround for GLM matrix multiplication not working properly on pre-Fermi NVIDIA GPUs.
  // Multiplies a cudaMat4 matrix and a vec4 and returns a vec3 clipped from the vec4
  __host__ __device__ glm::vec4 multiplyMV4(cudaMat4 m, glm::vec4 v) {
    glm::vec4 r(1, 1, 1, 1);
    r.x = (m.x.x*v.x) + (m.x.y*v.y) + (m.x.z*v.z) + (m.x.w*v.w);
    r.y = (m.y.x*v.x) + (m.y.y*v.y) + (m.y.z*v.z) + (m.y.w*v.w);
    r.z = (m.z.x*v.x) + (m.z.y*v.y) + (m.z.z*v.z) + (m.z.w*v.w);
    r.w = (m.w.x*v.x) + (m.w.y*v.y) + (m.w.z*v.z) + (m.w.w*v.w);
    return r;
  }

  __host__ __device__
  cudaMat4 glmMat4ToCudaMat4(glm::mat4 a) {
    cudaMat4 m; a = glm::transpose(a);
    m.x = a[0];
    m.y = a[1];
    m.z = a[2];
    m.w = a[3];
    return m;
  }

	typedef unsigned short VertexIndex;
	typedef glm::vec3 VertexAttributePosition;
	typedef glm::vec3 VertexAttributeNormal;
	typedef glm::vec2 VertexAttributeTexcoord;
	typedef unsigned char TextureData;

	typedef unsigned char BufferByte;

	enum PrimitiveType{
		Point = 1,
		Line = 2,
		Triangle = 3
	};

	struct VertexOut {
		glm::vec4 pos;

		// TODO: add new attributes to your VertexOut
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		 glm::vec3 eyePos;	// eye space position used for shading
		 glm::vec3 eyeNor;	// eye space normal used for shading, cuz normal will go wrong after perspective transformation
		// glm::vec3 col;
		 glm::vec2 texcoord0;
		 TextureData* dev_diffuseTex = NULL;
		 int texWidth, texHeight;
		// ...
	};

	struct Primitive {
		PrimitiveType primitiveType = Triangle;	// C++ 11 init
		VertexOut v[3];
	};

	struct Fragment {
		glm::vec3 color;
    glm::ivec2 screenCoord;
    float depth;

		// TODO: add new attributes to your Fragment
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		glm::vec3 eyePos;	// eye space position used for shading
		glm::vec3 eyeNor;
    bool shouldShade = false;
		// VertexAttributeTexcoord texcoord0;
		// TextureData* dev_diffuseTex;
		// ...
	};

	struct PrimitiveDevBufPointers {
		int primitiveMode;	//from tinygltfloader macro
		PrimitiveType primitiveType;
		int numPrimitives;
		int numIndices;
		int numVertices;

		// Vertex In, const after loaded
		VertexIndex* dev_indices;
		VertexAttributePosition* dev_position;
		VertexAttributeNormal* dev_normal;
		VertexAttributeTexcoord* dev_texcoord0;

		// Materials, add more attributes when needed
		TextureData* dev_diffuseTex;
		int diffuseTexWidth;
		int diffuseTexHeight;
		// TextureData* dev_specularTex;
		// TextureData* dev_normalTex;
		// ...

		// Vertex Out, vertex used for rasterization, this is changing every frame
		VertexOut* dev_verticesOut;

		// TODO: add more attributes when needed
	};

}

struct shouldCull
{
  __host__ __device__
  bool operator()(const Primitive& prim) {
    return (glm::dot(glm::normalize(prim.v[0].eyePos), prim.v[0].eyeNor) > 0.0f);
  }
};

static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;


static int width = 0;
static int height = 0;

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;

static int * dev_depth = NULL;	// you might need this buffer when doing depth test
static float * dev_depthValues = NULL; // stores depth values
static int * dev_depthLocks = NULL;	// locks Z-buffer

#if CUDA_MEASURE

float vertProcTimeAcc = 0.0f;
float primAsmTimeAcc = 0.0f;
#if BACK_FACE_CULLING
float cullTimeAcc = 0.0f;
int cullCountAcc = 0;
#endif
float rastTimeAcc = 0.0f;
float fragShaderTimeAcc = 0.0f;
float copyToPBOTimeAcc = 0.0f;

int measureCount = 0;
#define MEASURE_COUNT_MAX 1000

#endif // CUDA_MEASURE

/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__ 
void sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        glm::vec3 color;
        color.x = glm::clamp(image[index].x, 0.0f, 1.0f) * 255.0;
        color.y = glm::clamp(image[index].y, 0.0f, 1.0f) * 255.0;
        color.z = glm::clamp(image[index].z, 0.0f, 1.0f) * 255.0;
        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

/** 
* Writes fragment colors to the framebuffer
*/
__global__
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
      glm::vec3 baseColor;
      glm::vec3 eyePos;
      glm::vec3 eyeNor;
      bool shouldShade = false;
#if SSAA_FACTOR > 1
      int aaIndex = SSAA_FACTOR * index;

      for (int dx = 0; dx < SSAA_FACTOR; dx++) {
        for (int dy = 0; dy < SSAA_FACTOR; dy++) {
          const Fragment &frag = fragmentBuffer[x * SSAA_FACTOR + dx + (y * SSAA_FACTOR + dy) * w * SSAA_FACTOR];
          baseColor += frag.color;
          eyePos += frag.eyePos;
          eyeNor += frag.eyeNor;
          shouldShade |= frag.shouldShade;
        }
      }
      baseColor /= float(SSAA_FACTOR * SSAA_FACTOR);
      //eyePos /= float(SSAA_FACTOR * SSAA_FACTOR);
      //eyeNor /= float(SSAA_FACTOR * SSAA_FACTOR);
      eyeNor = glm::normalize(eyeNor);
#else
      baseColor = fragmentBuffer[index].color;
      eyePos = fragmentBuffer[index].eyePos;
      eyeNor = fragmentBuffer[index].eyeNor;
      shouldShade = fragmentBuffer[index].shouldShade;
#endif
		  // TODO: add your fragment shader code here
      if (shouldShade) {
        float lambert = glm::clamp(glm::dot(glm::normalize(-eyePos), eyeNor), 0.0f, 1.0f);
        framebuffer[index] = baseColor * lambert;
      }
      else {
        framebuffer[index] = baseColor;
      }
    }
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
    width = w * SSAA_FACTOR;
    height = h * SSAA_FACTOR;
    printf("width: %d, height: %d\n", w, h);
	cudaFree(dev_fragmentBuffer);
	cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   w * h * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, w * h * sizeof(glm::vec3));
    
	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width * height * sizeof(int));

  cudaMalloc(&dev_depthValues, width * height * sizeof(float));
  cudaMalloc(&dev_depthLocks, width * height * sizeof(int));

	checkCUDAError("rasterizeInit");
}

__global__
void initDepth(int w, int h, int * depth)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h)
	{
		int index = x + (y * w);
		depth[index] = INT_MAX;
	}
}

__global__
void initDepthValues(int w, int h, float * depthValues)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < w && y < h)
  {
    int index = x + (y * w);
    depthValues[index] = 2.0f;
  }
}

__global__
void initDepthLocks(int w, int h, int * depthLocks)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < w && y < h)
  {
    int index = x + (y * w);
    depthLocks[index] = 0;
  }
}

/**
* kern function with support for stride to sometimes replace cudaMemcpy
* One thread is responsible for copying one component
*/
__global__ 
void _deviceBufferCopy(int N, BufferByte* dev_dst, const BufferByte* dev_src, int n, int byteStride, int byteOffset, int componentTypeByteSize) {
	
	// Attribute (vec3 position)
	// component (3 * float)
	// byte (4 * byte)

	// id of component
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < N) {
		int count = i / n;
		int offset = i - count * n;	// which component of the attribute

		for (int j = 0; j < componentTypeByteSize; j++) {
			
			dev_dst[count * componentTypeByteSize * n 
				+ offset * componentTypeByteSize 
				+ j]

				= 

			dev_src[byteOffset 
				+ count * (byteStride == 0 ? componentTypeByteSize * n : byteStride) 
				+ offset * componentTypeByteSize 
				+ j];
		}
	}
	

}

__global__
void _nodeMatrixTransform(
	int numVertices,
	VertexAttributePosition* position,
	VertexAttributeNormal* normal,
	glm::mat4 MV, glm::mat3 MV_normal) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {
		position[vid] = glm::vec3(MV * glm::vec4(position[vid], 1.0f));
		normal[vid] = glm::normalize(MV_normal * normal[vid]);
	}
}

glm::mat4 getMatrixFromNodeMatrixVector(const tinygltf::Node & n) {
	
	glm::mat4 curMatrix(1.0);

	const std::vector<double> &m = n.matrix;
	if (m.size() > 0) {
		// matrix, copy it

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				curMatrix[i][j] = (float)m.at(4 * i + j);
			}
		}
	} else {
		// no matrix, use rotation, scale, translation

		if (n.translation.size() > 0) {
			curMatrix[3][0] = n.translation[0];
			curMatrix[3][1] = n.translation[1];
			curMatrix[3][2] = n.translation[2];
		}

		if (n.rotation.size() > 0) {
			glm::mat4 R;
			glm::quat q;
			q[0] = n.rotation[0];
			q[1] = n.rotation[1];
			q[2] = n.rotation[2];

			R = glm::mat4_cast(q);
			curMatrix = curMatrix * R;
		}

		if (n.scale.size() > 0) {
			curMatrix = curMatrix * glm::scale(glm::vec3(n.scale[0], n.scale[1], n.scale[2]));
		}
	}

	return curMatrix;
}

void traverseNode (
	std::map<std::string, glm::mat4> & n2m,
	const tinygltf::Scene & scene,
	const std::string & nodeString,
	const glm::mat4 & parentMatrix
	) 
{
	const tinygltf::Node & n = scene.nodes.at(nodeString);
	glm::mat4 M = parentMatrix * getMatrixFromNodeMatrixVector(n);
	n2m.insert(std::pair<std::string, glm::mat4>(nodeString, M));

	auto it = n.children.begin();
	auto itEnd = n.children.end();

	for (; it != itEnd; ++it) {
		traverseNode(n2m, scene, *it, M);
	}
}

void rasterizeSetBuffers(const tinygltf::Scene & scene) {

	totalNumPrimitives = 0;

	std::map<std::string, BufferByte*> bufferViewDevPointers;

	// 1. copy all `bufferViews` to device memory
	{
		std::map<std::string, tinygltf::BufferView>::const_iterator it(
			scene.bufferViews.begin());
		std::map<std::string, tinygltf::BufferView>::const_iterator itEnd(
			scene.bufferViews.end());

		for (; it != itEnd; it++) {
			const std::string key = it->first;
			const tinygltf::BufferView &bufferView = it->second;
			if (bufferView.target == 0) {
				continue; // Unsupported bufferView.
			}

			const tinygltf::Buffer &buffer = scene.buffers.at(bufferView.buffer);

			BufferByte* dev_bufferView;
			cudaMalloc(&dev_bufferView, bufferView.byteLength);
			cudaMemcpy(dev_bufferView, &buffer.data.front() + bufferView.byteOffset, bufferView.byteLength, cudaMemcpyHostToDevice);

			checkCUDAError("Set BufferView Device Mem");

			bufferViewDevPointers.insert(std::make_pair(key, dev_bufferView));

		}
	}



	// 2. for each mesh: 
	//		for each primitive: 
	//			build device buffer of indices, materail, and each attributes
	//			and store these pointers in a map
	{

		std::map<std::string, glm::mat4> nodeString2Matrix;
		auto rootNodeNamesList = scene.scenes.at(scene.defaultScene);

		{
			auto it = rootNodeNamesList.begin();
			auto itEnd = rootNodeNamesList.end();
			for (; it != itEnd; ++it) {
				traverseNode(nodeString2Matrix, scene, *it, glm::mat4(1.0f));
			}
		}


		// parse through node to access mesh

		auto itNode = nodeString2Matrix.begin();
		auto itEndNode = nodeString2Matrix.end();
		for (; itNode != itEndNode; ++itNode) {

			const tinygltf::Node & N = scene.nodes.at(itNode->first);
			const glm::mat4 & matrix = itNode->second;
			const glm::mat3 & matrixNormal = glm::transpose(glm::inverse(glm::mat3(matrix)));

			auto itMeshName = N.meshes.begin();
			auto itEndMeshName = N.meshes.end();

			for (; itMeshName != itEndMeshName; ++itMeshName) {

				const tinygltf::Mesh & mesh = scene.meshes.at(*itMeshName);

				auto res = mesh2PrimitivesMap.insert(std::pair<std::string, std::vector<PrimitiveDevBufPointers>>(mesh.name, std::vector<PrimitiveDevBufPointers>()));
				std::vector<PrimitiveDevBufPointers> & primitiveVector = (res.first)->second;

				// for each primitive
				for (size_t i = 0; i < mesh.primitives.size(); i++) {
					const tinygltf::Primitive &primitive = mesh.primitives[i];

					if (primitive.indices.empty())
						return;

					// TODO: add new attributes for your PrimitiveDevBufPointers when you add new attributes
					VertexIndex* dev_indices = NULL;
					VertexAttributePosition* dev_position = NULL;
					VertexAttributeNormal* dev_normal = NULL;
					VertexAttributeTexcoord* dev_texcoord0 = NULL;

					// ----------Indices-------------

					const tinygltf::Accessor &indexAccessor = scene.accessors.at(primitive.indices);
					const tinygltf::BufferView &bufferView = scene.bufferViews.at(indexAccessor.bufferView);
					BufferByte* dev_bufferView = bufferViewDevPointers.at(indexAccessor.bufferView);

					// assume type is SCALAR for indices
					int n = 1;
					int numIndices = indexAccessor.count;
					int componentTypeByteSize = sizeof(VertexIndex);
					int byteLength = numIndices * n * componentTypeByteSize;

					dim3 numThreadsPerBlock(128);
					dim3 numBlocks((numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					cudaMalloc(&dev_indices, byteLength);
					_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
						numIndices,
						(BufferByte*)dev_indices,
						dev_bufferView,
						n,
						indexAccessor.byteStride,
						indexAccessor.byteOffset,
						componentTypeByteSize);


					checkCUDAError("Set Index Buffer");


					// ---------Primitive Info-------

					// Warning: LINE_STRIP is not supported in tinygltfloader
					int numPrimitives;
					PrimitiveType primitiveType;
					switch (primitive.mode) {
					case TINYGLTF_MODE_TRIANGLES:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices / 3;
						break;
					case TINYGLTF_MODE_TRIANGLE_STRIP:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_TRIANGLE_FAN:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_LINE:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices / 2;
						break;
					case TINYGLTF_MODE_LINE_LOOP:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices + 1;
						break;
					case TINYGLTF_MODE_POINTS:
						primitiveType = PrimitiveType::Point;
						numPrimitives = numIndices;
						break;
					default:
						// output error
						break;
					};


					// ----------Attributes-------------

					auto it(primitive.attributes.begin());
					auto itEnd(primitive.attributes.end());

					int numVertices = 0;
					// for each attribute
					for (; it != itEnd; it++) {
						const tinygltf::Accessor &accessor = scene.accessors.at(it->second);
						const tinygltf::BufferView &bufferView = scene.bufferViews.at(accessor.bufferView);

						int n = 1;
						if (accessor.type == TINYGLTF_TYPE_SCALAR) {
							n = 1;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC2) {
							n = 2;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC3) {
							n = 3;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC4) {
							n = 4;
						}

						BufferByte * dev_bufferView = bufferViewDevPointers.at(accessor.bufferView);
						BufferByte ** dev_attribute = NULL;

						numVertices = accessor.count;
						int componentTypeByteSize;

						// Note: since the type of our attribute array (dev_position) is static (float32)
						// We assume the glTF model attribute type are 5126(FLOAT) here

						if (it->first.compare("POSITION") == 0) {
							componentTypeByteSize = sizeof(VertexAttributePosition) / n;
							dev_attribute = (BufferByte**)&dev_position;
						}
						else if (it->first.compare("NORMAL") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeNormal) / n;
							dev_attribute = (BufferByte**)&dev_normal;
						}
						else if (it->first.compare("TEXCOORD_0") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeTexcoord) / n;
							dev_attribute = (BufferByte**)&dev_texcoord0;
						}

						std::cout << accessor.bufferView << "  -  " << it->second << "  -  " << it->first << '\n';

						dim3 numThreadsPerBlock(128);
						dim3 numBlocks((n * numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
						int byteLength = numVertices * n * componentTypeByteSize;
						cudaMalloc(dev_attribute, byteLength);

						_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
							n * numVertices,
							*dev_attribute,
							dev_bufferView,
							n,
							accessor.byteStride,
							accessor.byteOffset,
							componentTypeByteSize);

						std::string msg = "Set Attribute Buffer: " + it->first;
						checkCUDAError(msg.c_str());
					}

					// malloc for VertexOut
					VertexOut* dev_vertexOut;
					cudaMalloc(&dev_vertexOut, numVertices * sizeof(VertexOut));
					checkCUDAError("Malloc VertexOut Buffer");

					// ----------Materials-------------

					// You can only worry about this part once you started to 
					// implement textures for your rasterizer
					TextureData* dev_diffuseTex = NULL;
					int diffuseTexWidth = 0;
					int diffuseTexHeight = 0;
					if (!primitive.material.empty()) {
						const tinygltf::Material &mat = scene.materials.at(primitive.material);
						printf("material.name = %s\n", mat.name.c_str());

						if (mat.values.find("diffuse") != mat.values.end()) {
							std::string diffuseTexName = mat.values.at("diffuse").string_value;
							if (scene.textures.find(diffuseTexName) != scene.textures.end()) {
								const tinygltf::Texture &tex = scene.textures.at(diffuseTexName);
								if (scene.images.find(tex.source) != scene.images.end()) {
									const tinygltf::Image &image = scene.images.at(tex.source);

									size_t s = image.image.size() * sizeof(TextureData);
									cudaMalloc(&dev_diffuseTex, s);
									cudaMemcpy(dev_diffuseTex, &image.image.at(0), s, cudaMemcpyHostToDevice);
									
									diffuseTexWidth = image.width;
									diffuseTexHeight = image.height;

									checkCUDAError("Set Texture Image data");
								}
							}
						}

						// TODO: write your code for other materails
						// You may have to take a look at tinygltfloader
						// You can also use the above code loading diffuse material as a start point 
					}


					// ---------Node hierarchy transform--------
					cudaDeviceSynchronize();
					
					dim3 numBlocksNodeTransform((numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					_nodeMatrixTransform << <numBlocksNodeTransform, numThreadsPerBlock >> > (
						numVertices,
						dev_position,
						dev_normal,
						matrix,
						matrixNormal);

					checkCUDAError("Node hierarchy transformation");

					// at the end of the for loop of primitive
					// push dev pointers to map
					primitiveVector.push_back(PrimitiveDevBufPointers{
						primitive.mode,
						primitiveType,
						numPrimitives,
						numIndices,
						numVertices,

						dev_indices,
						dev_position,
						dev_normal,
						dev_texcoord0,

						dev_diffuseTex,
						diffuseTexWidth,
						diffuseTexHeight,

						dev_vertexOut	//VertexOut
					});

					totalNumPrimitives += numPrimitives;

				} // for each primitive

			} // for each mesh

		} // for each node

	}
	

	// 3. Malloc for dev_primitives
	{
		cudaMalloc(&dev_primitives, totalNumPrimitives * sizeof(Primitive));
	}
	

	// Finally, cudaFree raw dev_bufferViews
	{

		std::map<std::string, BufferByte*>::const_iterator it(bufferViewDevPointers.begin());
		std::map<std::string, BufferByte*>::const_iterator itEnd(bufferViewDevPointers.end());
			
			//bufferViewDevPointers

		for (; it != itEnd; it++) {
			cudaFree(it->second);
		}

		checkCUDAError("Free BufferView Device Mem");
	}


}



__global__ 
void _vertexTransformAndAssembly(
	int numVertices, 
	PrimitiveDevBufPointers primitive, 
	glm::mat4 MVP, glm::mat4 MV, glm::mat3 MV_normal, 
	int width, int height) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {

		// TODO: Apply vertex transformation here
    glm::vec4 pos = glm::vec4(primitive.dev_position[vid], 1.0f);
    glm::vec3 nor = primitive.dev_normal[vid];
    glm::vec3 eyePos = glm::vec3(MV * pos);
		// Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
    pos = MVP * pos;//multiplyMV4(glmMat4ToCudaMat4(MVP), pos);
    nor = glm::normalize(MV_normal * nor);
		// Then divide the pos by its w element to transform into NDC space
    pos /= pos.w;
		// Finally transform x and y to viewport space
    pos.x = (pos.x + 1.0) * 0.5f * width;
    pos.y = (1.0f - pos.y) * 0.5f * height;

		// TODO: Apply vertex assembly here
		// Assemble all attribute arraies into the primitive array
    primitive.dev_verticesOut[vid].pos = pos;
    primitive.dev_verticesOut[vid].eyeNor = nor;
    primitive.dev_verticesOut[vid].eyePos = eyePos;
    primitive.dev_verticesOut[vid].dev_diffuseTex = primitive.dev_diffuseTex;
    if (primitive.dev_texcoord0 != NULL) {
      primitive.dev_verticesOut[vid].texWidth = primitive.diffuseTexWidth;
      primitive.dev_verticesOut[vid].texHeight = primitive.diffuseTexHeight;
      primitive.dev_verticesOut[vid].texcoord0 = primitive.dev_texcoord0[vid];
    }
	}
}



static int curPrimitiveBeginId = 0;

__global__ 
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives, PrimitiveDevBufPointers primitive) {

	// index id
	int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iid < numIndices) {

		// TODO: uncomment the following code for a start
		// This is primitive assembly for triangles

		int pid;	// id for cur primitives vector
		if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES) {
			pid = iid / (int)primitive.primitiveType;
			dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType]
				= primitive.dev_verticesOut[primitive.dev_indices[iid]];
		}


		// TODO: other primitive types (point, line)
	}
	
}

__device__ float triArea(const glm::vec3& pt0, const glm::vec3& pt1, const glm::vec3& pt2) {
  // don't divide by 2 because all calls should use this and we aren't concerned
  // about the actual value of the area, just the relative values
  return glm::length(glm::cross(pt0 - pt1, pt0 - pt2));
}

__device__ bool isInTriangle(glm::vec3* triPoints, const glm::vec3& pt, float totalArea, float* baryWeights) {
  baryWeights[2] = triArea(pt, triPoints[0], triPoints[1]);
  baryWeights[0] = triArea(pt, triPoints[1], triPoints[2]);
  baryWeights[1] = triArea(pt, triPoints[0], triPoints[2]);
  return (baryWeights[0] + baryWeights[1] + baryWeights[2]) <= totalArea;
}

#if BILINEAR_INTERP
__device__ glm::vec3 colorFromUV(TextureData* texture, glm::vec2 texCoord, int texWidth, int texHeight) {
  glm::vec2 scaledTexCoord = texCoord * glm::vec2(texWidth, texHeight);
  glm::ivec2 intScaledTexCoord = glm::ivec2(scaledTexCoord);
  glm::ivec2 nextScaledTexCoord = glm::clamp(intScaledTexCoord + glm::ivec2(1), glm::ivec2(0), glm::ivec2(texWidth - 1, texHeight - 1));
  int idx = intScaledTexCoord.x + intScaledTexCoord.y * texWidth;
  glm::vec3 col00 = glm::vec3(texture[idx * 3] / 255.0f,
    texture[idx * 3 + 1] / 255.0f,
    texture[idx * 3 + 2] / 255.0f);
  idx = nextScaledTexCoord.x + intScaledTexCoord.y * texWidth;
  glm::vec3 col10 = glm::vec3(texture[idx * 3] / 255.0f,
    texture[idx * 3 + 1] / 255.0f,
    texture[idx * 3 + 2] / 255.0f);
  idx = intScaledTexCoord.x + nextScaledTexCoord.y * texWidth;
  glm::vec3 col01 = glm::vec3(texture[idx * 3] / 255.0f,
    texture[idx * 3 + 1] / 255.0f,
    texture[idx * 3 + 2] / 255.0f);
  idx = nextScaledTexCoord.x + nextScaledTexCoord.y * texWidth;
  glm::vec3 col11 = glm::vec3(texture[idx * 3] / 255.0f,
    texture[idx * 3 + 1] / 255.0f,
    texture[idx * 3 + 2] / 255.0f);
  glm::vec2 diff = scaledTexCoord - glm::vec2(intScaledTexCoord);
  return (1.0f - diff.x) * (1.0f - diff.y) * col00 +
         diff.x * (1.0f - diff.y) * col10 +
         (1.0f - diff.x) * diff.y * col01 +
         diff.x * diff.y * col11;
}
#else
__device__ glm::vec3 colorFromUV(TextureData* texture, glm::vec2 texCoord, int texWidth, int texHeight) {
  int idx = (int)(texCoord.x * texWidth) + (int)(texCoord.y * texHeight) * texWidth;
  glm::vec3 col = glm::vec3(texture[idx * 3] / 255.0f,
    texture[idx * 3 + 1] / 255.0f,
    texture[idx * 3 + 2] / 255.0f);
  return col;
}
#endif
__global__
void rast(Primitive* dev_primitives, int primitivesCount, int w, int h, Fragment *fragmentBuffer, float *dev_depthValues, int *dev_depthLocks) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < primitivesCount) {
    Primitive& prim = dev_primitives[idx];
    // assume triangle
    glm::vec2 bboxMin = glm::max(glm::vec2(0.0f), 
                                 glm::min(glm::vec2(prim.v[0].pos), glm::min(glm::vec2(prim.v[1].pos), glm::vec2(prim.v[2].pos))));
    glm::vec2 bboxMax = glm::min(glm::vec2((float)(w - 1), (float)(h - 1)),
                                 glm::max(glm::vec2(prim.v[0].pos), glm::max(glm::vec2(prim.v[1].pos), glm::vec2(prim.v[2].pos))));
    bboxMin = glm::floor(bboxMin);
    bboxMax = glm::ceil(bboxMax);

    glm::vec3 triPoints[3];
    triPoints[0] = glm::vec3(prim.v[0].pos);
    //triPoints[0].z = 0.0f;
    triPoints[1] = glm::vec3(prim.v[1].pos);
    //triPoints[1].z = 0.0f;
    triPoints[2] = glm::vec3(prim.v[2].pos);
    //triPoints[2].z = 0.0f;

    // make totalArea slightly larger to reduce "shadow acne" due to FP error
    float totalArea = triArea(triPoints[0], triPoints[1], triPoints[2]) * 1.0001f;
    float baryWeights[3];
    glm::vec3 baryCoords;

    bool hasTexture = prim.v[0].dev_diffuseTex != NULL;

#if RENDER_MODE == RENDER_FULL_TRIANGLE
    for (float y = bboxMin.y; y <= bboxMax.y; y += 1.0f) {
      for (float x = bboxMin.x; x <= bboxMax.x; x += 1.0f) {
        baryCoords = calculateBarycentricCoordinate(triPoints, glm::vec2(x, y));
        if (isBarycentricCoordInBounds(baryCoords)) {
          // TODO: persp-correct
          baryWeights[0] = baryCoords[0];
          baryWeights[1] = baryCoords[1];
          baryWeights[2] = baryCoords[2];
#if PERSP_CORRECT

          float z = 1.0f / (baryWeights[0] / prim.v[0].eyePos.z +
                            baryWeights[1] / prim.v[1].eyePos.z +
                            baryWeights[2] / prim.v[2].eyePos.z);
          baryWeights[0] *= z / prim.v[0].eyePos.z;
          baryWeights[1] *= z / prim.v[1].eyePos.z;
          baryWeights[2] *= z / prim.v[2].eyePos.z;

          z = -getZAtCoordinate(baryCoords, triPoints);
#else
          float z = -getZAtCoordinate(baryCoords, triPoints);
#endif
         
          // depth check
          // lock this fragment on the depth buffer
          Fragment frag;
          glm::vec3 nor = baryWeights[0] * prim.v[0].eyeNor +
            baryWeights[1] * prim.v[1].eyeNor +
            baryWeights[2] * prim.v[2].eyeNor;
          // texture mapping
          if (hasTexture) {
            glm::vec2 texCoord = baryWeights[0] * prim.v[0].texcoord0 +
              baryWeights[1] * prim.v[1].texcoord0 +
              baryWeights[2] * prim.v[2].texcoord0;
            // check if UV are in range (may not be if Z value is weird)
            if (texCoord.x < 0.0f || texCoord.x > 1.0f || texCoord.y < 0.0f || texCoord.y > 1.0f) {
              continue;
            }
            frag.shouldShade = true;
            frag.color = colorFromUV(prim.v[0].dev_diffuseTex, texCoord, prim.v[0].texWidth, prim.v[0].texHeight);
          }
          else {
            // color using normal
            
            nor = glm::normalize(nor);
            // check if coords are in range (may not be if Z value is weird)
            if (nor.x < -1.0f || nor.x > 1.0f || nor.y < -1.0f || nor.y > 1.0f || nor.z < -1.0f || nor.z > 1.0f) {
              continue;
            }
            frag.shouldShade = false;
            frag.color = glm::abs(nor);
          }
          frag.eyeNor = nor;
          frag.eyePos = baryWeights[0] * prim.v[0].eyePos +
            baryWeights[1] * prim.v[1].eyePos +
            baryWeights[2] * prim.v[2].eyePos;
          frag.screenCoord = glm::ivec2((int)x, (int)y);
          frag.depth = z;
          //frag.color = glm::vec3(z);

          // add fragment
          int fragIdx = (int)x + (int)y * w;

          // magic to make mutex work
          bool isSet;
          do {
            isSet = (atomicCAS(&dev_depthLocks[fragIdx], 0, 1) == 0);
            if (isSet) {
              // critical section
              if (z < dev_depthValues[fragIdx]) {
                dev_depthValues[fragIdx] = z;
                fragmentBuffer[fragIdx] = frag;
              }
              // unlock fragment
              dev_depthLocks[fragIdx] = 0;//atomicExch(&dev_depthLocks[fragIdx], 0);
            }

          } while (!isSet);
        }
      }
    }
#elif RENDER_MODE == RENDER_VERTICES
    for (int i = 0; i < 3; i++) {
      const glm::vec3& vert = triPoints[i];
      int x = (int)vert.x;
      int y = (int)vert.y;

      if (x >= w || y >= h || x < 0 || y < 0) {
        continue;
      }

      Fragment frag;
      if (hasTexture) {
        frag.shouldShade = true;
        frag.color = colorFromUV(prim.v[0].dev_diffuseTex, prim.v[i].texcoord0, prim.v[i].texWidth, prim.v[i].texHeight);
      }
      else {
        frag.shouldShade = false;
        frag.color = glm::abs(prim.v[i].eyeNor);
      }
      frag.eyePos = prim.v[i].eyePos;
      frag.eyeNor = prim.v[i].eyeNor;

      float z = vert.z;

      int xLimit = glm::clamp(x + VERTEX_RENDER_SIZE, x, w - 1);
      int yLimit = glm::clamp(y + VERTEX_RENDER_SIZE, y, h - 1);
      for (int fragX = x; fragX <= xLimit; fragX++) {
        for (int fragY = y; fragY <= yLimit; fragY++) {
          int fragIdx = fragX + fragY * w;
          bool isSet;
          do {
            isSet = true;// (atomicCAS(&dev_depthLocks[fragIdx], 0, 1) == 0);
            if (isSet) {
              // critical section
              if (z < dev_depthValues[fragIdx]) {
                dev_depthValues[fragIdx] = z;
                fragmentBuffer[fragIdx] = frag;
              }
              // unlock fragment
              dev_depthLocks[fragIdx] = 0;//atomicExch(&dev_depthLocks[fragIdx], 0);
            }
          } while (!isSet);
        }
      }
    }
#elif RENDER_MODE == RENDER_EDGES
    // Bresenham's line algorithm
    Fragment frag;
    frag.shouldShade = false;
    frag.color = glm::vec3(1.0f);
    for (int i = 0; i < 3; i++) {
      glm::ivec2 leftVert;
      glm::ivec2 rightVert;
      int nextIdx = i == 2 ? 0 : i + 1;

      if ((int)triPoints[i].x < 0 || (int)triPoints[i].x >= w || (int)triPoints[nextIdx].x < 0 || (int)triPoints[nextIdx].x >= w ||
          (int)triPoints[i].y < 0 || (int)triPoints[i].y >= h || (int)triPoints[nextIdx].y < 0 || (int)triPoints[nextIdx].y >= h) {
        continue;
      }

      if ((int)triPoints[i].x == (int)triPoints[nextIdx].x) {
        // vertical line
        int yStart = glm::max(0, (int)glm::min(triPoints[i].y, triPoints[nextIdx].y));
        int yEnd = glm::min(h - 1, (int)glm::max(triPoints[i].y, triPoints[nextIdx].y));
        int x = (int)triPoints[i].x;
        // draw(x,y)
        for (int y = yStart; y <= yEnd; y++) {
          int fragIdx = x + y * w;
          fragmentBuffer[fragIdx] = frag;
        }
        continue;
      }
      else if ((int)triPoints[i].y == (int)triPoints[nextIdx].y) {
        // horizontal line
        int xStart = glm::max(0, (int)glm::min(triPoints[i].x, triPoints[nextIdx].x));
        int xEnd = glm::min(w - 1, (int)glm::max(triPoints[i].x, triPoints[nextIdx].x));
        int y = (int)triPoints[i].y;
        // draw(x,y)
        for (int x = xStart; x <= xEnd; x++) {
          int fragIdx = x + y * w;
          fragmentBuffer[fragIdx] = frag;
        }
        continue;
      }
      else if ((int)triPoints[i].x > (int)triPoints[nextIdx].x) {
        rightVert = glm::ivec2(triPoints[i]);
        leftVert = glm::ivec2(triPoints[nextIdx]);
      }
      else {
        leftVert = glm::ivec2(triPoints[i]);
        rightVert = glm::ivec2(triPoints[nextIdx]);
      }
      
      float dErr = abs((triPoints[i].y - triPoints[nextIdx].y) / (triPoints[i].x - triPoints[nextIdx].x));
      bool downward = ((rightVert.y - leftVert.y) < 0);
      int increment = downward ? -1 : 1;

      float accErr = 0.0f;
      int xStart = glm::max(leftVert.x, 0);
      int y = leftVert.y;
      int xEnd = glm::min(rightVert.x, w - 1);
      for (int x = xStart; x <= xEnd; x++) {
        // draw(x,y)
        if (y < 0 || y >= h) {
          break;
        }

        int fragIdx = x + y * w;
        fragmentBuffer[fragIdx] = frag;
        accErr += dErr;
        while (accErr >= 0.5f) {
          y += increment;
          // draw(x, y)
          if (y < 0 || y >= h) {
            break;
          }
          if ((downward && y == rightVert.y - 1) || (!downward && y == rightVert.y + 1)) {
            y -= increment;
          }

          int fragIdx = x + y * w;
          fragmentBuffer[fragIdx] = frag;
          accErr -= 1.0f;
        }
      }
    }
#endif // RENDER_MODE
  }
}

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
		(height - 1) / blockSize2d.y + 1);
    // for frame buffer; will be different is SSAA_FACTOR > 1
    dim3 frameBufferBlockCount2d((width / SSAA_FACTOR - 1) / blockSize2d.x + 1,
      (height / SSAA_FACTOR - 1) / blockSize2d.y + 1);

    // set up CUDA timing events
    cudaEvent_t stageStart, stageEnd;
    cudaEventCreate(&stageStart);
    cudaEventCreate(&stageEnd);

    float vertProcTime = 0.0f;
    float primAsmTime = 0.0f;
#if BACK_FACE_CULLING
    float cullTime;
#endif
    float rastTime;
    float fragShaderTime;
    float copyToPBOTime;

	// Execute your rasterization pipeline here
	// (See README for rasterization pipeline outline.)

	// Vertex Process & primitive assembly
	{
		curPrimitiveBeginId = 0;
		dim3 numThreadsPerBlock(128);

		auto it = mesh2PrimitivesMap.begin();
		auto itEnd = mesh2PrimitivesMap.end();

		for (; it != itEnd; ++it) {
			auto p = (it->second).begin();	// each primitive
			auto pEnd = (it->second).end();
			for (; p != pEnd; ++p) {
        float measurement;
				dim3 numBlocksForVertices((p->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
				dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
        checkCUDAError("pre-Vertex Processing");
        cudaEventRecord(stageStart);
				_vertexTransformAndAssembly << < numBlocksForVertices, numThreadsPerBlock >> >(p->numVertices, *p, MVP, MV, MV_normal, width, height);
        cudaEventRecord(stageEnd);
        checkCUDAError("post-Vertex Processing");
				cudaDeviceSynchronize();
        cudaEventSynchronize(stageEnd);
        cudaEventElapsedTime(&measurement, stageStart, stageEnd);
        vertProcTime += measurement;

        cudaEventRecord(stageStart);
				_primitiveAssembly << < numBlocksForIndices, numThreadsPerBlock >> >
					(p->numIndices, 
					curPrimitiveBeginId, 
					dev_primitives, 
					*p);
        cudaEventRecord(stageEnd);
				checkCUDAError("Primitive Assembly");

        cudaEventSynchronize(stageEnd);
        cudaEventElapsedTime(&measurement, stageStart, stageEnd);
        primAsmTime += measurement;

				curPrimitiveBeginId += p->numPrimitives;
			}
		}

		checkCUDAError("Vertex Processing and Primitive Assembly");
	}
	
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
  initDepthValues << <blockCount2d, blockSize2d >> >(width, height, dev_depthValues);
  cudaMemset(dev_depthLocks, 0, width * height * sizeof(int));
	
  // test screen-space tri
#if 0
  Primitive prim;
  prim.v[0].pos = glm::vec4(100.0f, 100.0f, 0.0f, 1.0f);
  prim.v[1].pos = glm::vec4(800.0f, 100.0f, 0.0f, 1.0f);
  prim.v[2].pos = glm::vec4(200.0f, 800.0f, 0.0f, 1.0f);

  cudaMemcpy(dev_primitives, &prim, sizeof(Primitive), cudaMemcpyHostToDevice);
  curPrimitiveBeginId = 1;
#endif

  // back-face culling
#if BACK_FACE_CULLING
  cudaEventRecord(stageStart);

  Primitive *newEnd = thrust::remove_if(thrust::device, dev_primitives, dev_primitives + curPrimitiveBeginId, shouldCull());
#if CUDA_MEASURE
  cullCountAcc += (dev_primitives + curPrimitiveBeginId) - newEnd;
#endif
  //printf("culled: %d", (dev_primitives + curPrimitiveBeginId) - newEnd);
  curPrimitiveBeginId = newEnd - dev_primitives;

  cudaEventRecord(stageEnd);
  cudaEventSynchronize(stageEnd);

  cudaEventElapsedTime(&cullTime, stageStart, stageEnd);
#endif

	// TODO: rasterize
  checkCUDAError("pre-actual rasterizer");
  cudaEventRecord(stageStart);

  rast << <dim3(curPrimitiveBeginId / 32 + 1), dim3(32) >> > (dev_primitives, curPrimitiveBeginId, width, height, dev_fragmentBuffer, dev_depthValues, dev_depthLocks);
  
  cudaEventRecord(stageEnd);
  cudaEventSynchronize(stageEnd);

  cudaEventElapsedTime(&rastTime, stageStart, stageEnd);
  checkCUDAError("post-actual rasterizer");
#if 0
  float *buf = (float *)malloc(width * height * sizeof(float));
  cudaMemcpy(buf, dev_depthValues, width * height * sizeof(float), cudaMemcpyDeviceToHost);
  int ct = 0;
  for (int i = 0; i < width * height; i++) {
    if (buf[i] <= 0.0f) {
      printf("%.3f ", buf[i]);
      ct++;
    }
  }
  printf("ct: %d", ct);
  printf("\n");
  free(buf);
  while (1);
#endif
    // Copy depthbuffer colors into framebuffer
  cudaEventRecord(stageStart);

	render << <frameBufferBlockCount2d, blockSize2d >> >(width / SSAA_FACTOR, height / SSAA_FACTOR, dev_fragmentBuffer, dev_framebuffer);

  cudaEventRecord(stageEnd);
  cudaEventSynchronize(stageEnd);

  cudaEventElapsedTime(&fragShaderTime, stageStart, stageEnd);

	checkCUDAError("fragment shader");
  cudaEventRecord(stageStart);

  // Copy framebuffer into OpenGL buffer for OpenGL previewing
  sendImageToPBO<<<frameBufferBlockCount2d, blockSize2d >>>(pbo, width / SSAA_FACTOR, height / SSAA_FACTOR, dev_framebuffer);

  cudaEventRecord(stageEnd);
  cudaEventSynchronize(stageEnd);

  cudaEventElapsedTime(&copyToPBOTime, stageStart, stageEnd);
  checkCUDAError("copy render result to pbo");

#if CUDA_MEASURE
  measureCount++;

  vertProcTimeAcc += vertProcTime;
  primAsmTimeAcc += primAsmTime;
#if BACK_FACE_CULLING
  cullTimeAcc += cullTime;
#endif
  rastTimeAcc += rastTime;
  fragShaderTimeAcc += fragShaderTime;
  copyToPBOTimeAcc += copyToPBOTime;

  if (measureCount >= MEASURE_COUNT_MAX) {
    // print measurements
    printf("Vertex Processing:  %.4f\n", vertProcTimeAcc / (float)measureCount);
    printf("Primitive Assembly: %.4f\n", primAsmTimeAcc / (float)measureCount);
#if BACK_FACE_CULLING
    printf("Back-face Culling:  %.4f\n", cullTimeAcc / (float)measureCount);
    printf("Faces Culled:       %.4f\n", (float)(cullCountAcc) / (float)measureCount);
#endif
    printf("Rasterizer:         %.4f\n", rastTimeAcc / (float)measureCount);
    printf("Fragment Shader:    %.4f\n", fragShaderTimeAcc / (float)measureCount);
    printf("Copy to PBO:        %.4f\n", copyToPBOTimeAcc / (float)measureCount);
    printf("\n");

    measureCount = 0;

    vertProcTimeAcc = 0.0f;
    primAsmTimeAcc = 0.0f;
#if BACK_FACE_CULLING
    cullTimeAcc = 0.0f;
    cullCountAcc = 0;
#endif
    rastTimeAcc = 0.0f;
    fragShaderTimeAcc = 0.0f;
    copyToPBOTimeAcc = 0.0f;

  }
#endif

}

/**
 * Called once at the end of the program to free CUDA memory.
 */
void rasterizeFree() {

    // deconstruct primitives attribute/indices device buffer

	auto it(mesh2PrimitivesMap.begin());
	auto itEnd(mesh2PrimitivesMap.end());
	for (; it != itEnd; ++it) {
		for (auto p = it->second.begin(); p != it->second.end(); ++p) {
			cudaFree(p->dev_indices);
			cudaFree(p->dev_position);
			cudaFree(p->dev_normal);
			cudaFree(p->dev_texcoord0);
			cudaFree(p->dev_diffuseTex);

			cudaFree(p->dev_verticesOut);

			
			//TODO: release other attributes and materials
		}
	}

	////////////

    cudaFree(dev_primitives);
    dev_primitives = NULL;

	cudaFree(dev_fragmentBuffer);
	dev_fragmentBuffer = NULL;

    cudaFree(dev_framebuffer);
    dev_framebuffer = NULL;

	cudaFree(dev_depth);
	dev_depth = NULL;

  cudaFree(dev_depthValues);
  dev_depthValues = NULL;

  cudaFree(dev_depthLocks);
  dev_depthLocks = NULL;
    checkCUDAError("rasterize Free");
}
