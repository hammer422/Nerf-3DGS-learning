#include "rasterize.h"

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cub/cub.cuh>
#include <cuda.h>
#include "cuda_runtime.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;



const int NUM_CHANNELS_MY = 3; // Default 3, RGB
const int BLOCK_X_MY = 16;
const int BLOCK_Y_MY = 16;
const int BLOCK_SIZE = BLOCK_X_MY * BLOCK_Y_MY;
// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

////////////////////////////////////////////////////////////
// utils function 
////////////////////////////////////////////////////////////
__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
    // 和原始版本不同，这里假设matrix内存分布是row-major
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z + matrix[3],
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z + matrix[7],
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z + matrix[11],
	};
	return transformed;
}
__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
    // 和原始版本不同，这里假设matrix内存分布是row-major
	float4 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z + matrix[3],
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z + matrix[7],
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z + matrix[11],
		matrix[12] * p.x + matrix[13] * p.y + matrix[14] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ bool in_frustum(
    int idx,
	const float* orig_points,
    const float* viewmatrix,
    bool prefiltered,
	float3& p_view
)
{
    // 获得相机坐标系下的坐标，过滤深度值小于0.2m的高斯球
    // 为什么是0.2？？no idea
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	p_view = transformPoint4x3(p_orig, viewmatrix);

    if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
            // 如果已经预过滤了，还能发现这样的高斯，cuda报错
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

__forceinline__ __device__ float3 project_points(
    int idx,
    const float* orig_points,
    const float* projmatrix
)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

    return p_proj;
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const float* scale, float mod, const float* rot, float* cov3D)
{
    // 输入3个主轴scale, 尺寸大小，四元数，cov3D协方差矩阵
    // Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
    S[0][0] = mod * scale[0];
    S[1][1] = mod * scale[1];
    S[2][2] = mod * scale[2];
    
	// Normalize quaternion to get valid rotation
	float r = rot[0];
	float x = rot[1];
	float y = rot[2];
	float z = rot[3];

    // Compute rotation matrix from quaternion
    // 这里初始化是按col-major初始化的，所以R就是R^Transpose
	glm::mat3 RT = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * RT;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
    // Sigma[0][1]指的是第0列，第1行
    // 实际存储的是下三角，但是因为Sigma是col-major
    // 本质还是存储的上三角矩阵
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(
    const float3& mean,
    float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	// 变换到相机坐标系下
    float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx; // 扩大一点frustum范围
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z; // 当前点P的正切
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);
    
    // 这里和原始不同，因为我们的viewmatrix内存分布是row-major的
    // 这里为了获得R^T
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[1], viewmatrix[2],
		viewmatrix[4], viewmatrix[5], viewmatrix[6],
		viewmatrix[8], viewmatrix[9], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

    glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
    // 输入v:[-1,1],S为图像宽度或高度pixel
	// [-1, 1] -> [-0.5, S-0.5]
	return ((v + 1.0) * S - 1.0) * 0.5;
}
__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X_MY))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y_MY)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X_MY - 1) / BLOCK_X_MY))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y_MY - 1) / BLOCK_Y_MY)))
	};
}
__forceinline__ __device__ void getRect(const float2 p, int2 ext_rect, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - ext_rect.x) / BLOCK_X_MY))),
		min(grid.y, max((int)0, (int)((p.y - ext_rect.y) / BLOCK_Y_MY)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + ext_rect.x + BLOCK_X_MY - 1) / BLOCK_X_MY))),
		min(grid.y, max((int)0, (int)((p.y + ext_rect.y + BLOCK_Y_MY - 1) / BLOCK_Y_MY)))
	};
}

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(
    int idx, int deg, int max_coeffs,
    const float* means_ptr, const float* campos_ptr,
     const float* shs, uint8_t* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
    glm::vec3 pos = glm::vec3(means_ptr[idx*3], means_ptr[idx*3+1], means_ptr[idx*3+2]);
    glm::vec3 campos = glm::vec3(campos_ptr[0], campos_ptr[1], campos_ptr[2]);

	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0) ? 0 : 1;
	clamped[3 * idx + 1] = (result.y < 0) ? 0 : 1;
	clamped[3 * idx + 2] = (result.z < 0) ? 0 : 1;
	return glm::max(result, 0.0f);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	// 每个thread
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		// 起始值，这就是前缀和
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;
		
		// 和当前高斯的关联的tile范围
		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		// 遍历所有接触到的tile
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				// [tile_id | depth] 高32位为tile ID； 低32位为depth
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx; // 高斯的ID
				off++;
			}
		}
	}
}

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}


// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// L为num_rendered
	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32; // 当前高斯相交的tile的tile id

	// 下面的代码总的来说就是在调整ranges的id
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}



template<const int C>
__global__ void rasterizeCUDA_forward_preprocess(
    const int P, const int Degree, const int sh_size,

    const float* means3D,
	const float* opacities,
	const float* scales,
    const float* rotations,
	const float* shs,

    const float* viewmatrix,
	const float* projmatrix,
    const float* cam_pos,
	const float scale_modifier, 
    const float focal_x, 
    const float focal_y,
	const float tan_halffovx, 
	const float tan_halffovy,
    const int image_height,
    const int image_width,
	const bool prefiltered,
	const bool antialiasing,

    const dim3 grid,

    // 以下是会被修改的值
    int* radii,

    float* gemo_depths,
    uint8_t* gemo_clamped,
    float2* gemo_means2D,
    float* gemo_cov3D,
    float4* gemo_conic_opacity,
    float* gemo_rgb,
    uint32_t* gemo_tiles_touched
)
{
    // 一个thread处理一个高斯
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	gemo_tiles_touched[idx] = 0; // 当前高斯关联的tile数量

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, means3D, viewmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
    float3 p_proj = project_points(idx, means3D, projmatrix);

    // 计算3D协方差
    computeCov3D(scales+idx*3, scale_modifier, rotations+4*idx, gemo_cov3D + idx * 6);
    const float* cov3D = gemo_cov3D + idx * 6;

	// Compute 2D screen-space covariance matrix
    float3 p_orig = { means3D[3 * idx], means3D[3 * idx + 1], means3D[3 * idx + 2] };
    float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_halffovx, tan_halffovy, cov3D, viewmatrix);

	constexpr float h_var = 0.3f;
	const float det_cov = cov.x * cov.z - cov.y * cov.y; // 计算原2D高斯的det
	cov.x += h_var; // 对角线加上膨胀 来自Mip-splatting  Cov+sI
	cov.z += h_var; // 对角线加上膨胀 来自Mip-splatting  Cov+sI
	const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
	float h_convolution_scaling = 1.0f; // scale系数 来自Mip-splatting

    // 来自Mip-splatting
	if(antialiasing)
		h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability

	// Invert covariance (EWA algorithm)
	const float det = det_cov_plus_h_cov;

	// 求逆
    if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	// 求协方差的特征值，进而计算半径，
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, image_width), ndc2Pix(p_proj.y, image_height) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

    // 计算颜色值
    glm::vec3 result = computeColorFromSH(idx, Degree, sh_size, means3D, cam_pos, shs, gemo_clamped);
    gemo_rgb[idx * C + 0] = result.x;
    gemo_rgb[idx * C + 1] = result.y;
    gemo_rgb[idx * C + 2] = result.z;


	// Store some useful helper data for the next steps.
	gemo_depths[idx] = p_view.z;
	radii[idx] = my_radius;
	gemo_means2D[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	float opacity = opacities[idx];

	gemo_conic_opacity[idx] = { conic.x, conic.y, conic.z, opacity * h_convolution_scaling };

	gemo_tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}


// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X_MY * BLOCK_Y_MY)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ depths,
	float* __restrict__ invdepth)
{
	// 一个线程负责一个像素
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X_MY - 1) / BLOCK_X_MY;
	uint2 pix_min = { block.group_index().x * BLOCK_X_MY, block.group_index().y * BLOCK_Y_MY };
	uint2 pix_max = { min(pix_min.x + BLOCK_X_MY, W), min(pix_min.y + BLOCK_Y_MY , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	// 获取当前tile对应的range
	// range指示的是与当前tile相交的高斯，可能会有多个
	// 目的是遍历所有和tile有相交的高斯
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	// 因为cuda一个block负责一个tile，一个block一共有BLOCK_SIZE的线程
	// 每一个round可以处理BLOCK_SIZE个当前tile的高斯
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	// 需要处理的数量
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	float expected_invdepth = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		// 整个block都处理完了退出
		// 为什么在循环内加同步判断？因为当前像素的T累积透射率被消耗完了就不需要计算了
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		// 全部拉到shared_mem里面去
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha); // 更新累积透射率
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			if(invdepth)
			expected_invdepth += (1 / depths[collected_id[j]]) * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

		if (invdepth)
		invdepth[pix_id] = expected_invdepth;// 1. / (expected_depth + T * 1e3);
	}
}

////////////////////////////////////////////////////////////
// utils function (end)
////////////////////////////////////////////////////////////




void rasterizeCUDA_forward_invoke(
    const int P, const int Degree, const int sh_size,

    const float* means3D,
	const float* opacities,
	const float* scales,
    const float* rotations,
	const float* sh,

	const float* background,
	const float* viewmatrix,
	const float* projmatrix,
    const float* cam_pos,
	const float scale_modifier, 
	const float tan_halffovx, 
	const float tan_halffovy,
    const int image_height,
    const int image_width,
	const bool prefiltered,
	const bool antialiasing,
	const bool debug,

    // buffers
    // gemotry state
    float* gemo_depths,
    uint8_t* gemo_clamped,
    float2* gemo_means2D,
    float* gemo_cov3D,
    float4* gemo_conic_opacity,
    float* gemo_rgb,
    uint32_t* gemo_tiles_touched,
    size_t& gemo_scan_size,
    uint8_t* gemo_scanning_space,
    uint32_t* gemo_point_offsets,
    // image state
    float* imgstate_accum_alpha,
    uint32_t* imgstate_n_contrib,
    uint2* imgstate_ranges,

    // output
    float* out_color,
	float* depth,
    int* radii
)
{
	const float focal_y = image_height / (2.0f * tan_halffovy);
	const float focal_x = image_width / (2.0f * tan_halffovx);

	dim3 tile_grid((image_width + BLOCK_X_MY - 1) / BLOCK_X_MY, (image_height + BLOCK_Y_MY - 1) / BLOCK_Y_MY, 1);
	dim3 block(BLOCK_X_MY, BLOCK_Y_MY, 1);

    {
        // Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
        const int threads_num = 256;
        rasterizeCUDA_forward_preprocess<NUM_CHANNELS_MY><<<(P+threads_num-1)/threads_num, threads_num>>>(
            P, Degree, sh_size,
            means3D,
            opacities,
            scales,
            rotations,
            sh,

            viewmatrix,
            projmatrix,
            cam_pos,
            scale_modifier, 
            focal_x, 
            focal_y,
            tan_halffovx, 
            tan_halffovy,
            image_height,
            image_width,
            prefiltered,
            antialiasing,

            tile_grid,

            radii,

            gemo_depths,
            gemo_clamped,
            gemo_means2D,
            gemo_cov3D,
            gemo_conic_opacity,
            gemo_rgb,
            gemo_tiles_touched
        );
    }

	// 下面是渲染前的准备工作，看文档的图解析!!!!!

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	// 把前缀和放到point_offsets
	cub::DeviceScan::InclusiveSum(gemo_scanning_space, gemo_scan_size, gemo_tiles_touched, gemo_point_offsets, P);

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	cudaMemcpy(&num_rendered, gemo_point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost);


	// 初始化BinningState
	// 这里都是用num_rendered来初始化，num_rendered指的是每个高斯与tile的相交的个数的总累和
	torch::TensorOptions u8_opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kUInt8);
	torch::TensorOptions u32_opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kUInt32);
	torch::TensorOptions u64_opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kUInt64);

	torch::Tensor binning_point_list = torch::full({num_rendered}, 0, u32_opts).contiguous(); // uint32_t*
	torch::Tensor binning_point_list_unsorted = torch::full({num_rendered}, 0, u32_opts).contiguous(); // uint32_t*
	torch::Tensor binning_point_list_keys = torch::full({num_rendered}, 0, u64_opts).contiguous(); // uint64_t*
	torch::Tensor binning_point_list_keys_unsorted = torch::full({num_rendered}, 0, u64_opts).contiguous(); // uint64_t*
	size_t binning_sorting_size;
	// key输出到binning_point_list_keys, value输出到binning_point_list
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning_sorting_size,
		binning_point_list_keys_unsorted.data_ptr<uint64_t>(), binning_point_list_keys.data_ptr<uint64_t>(),
		binning_point_list_unsorted.data_ptr<uint32_t>(), binning_point_list.data_ptr<uint32_t>(), num_rendered);
	torch::Tensor binning_list_sorting_space = torch::full({num_rendered}, 0, u8_opts).contiguous(); // uint8_t*


	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		gemo_means2D,
		gemo_depths,
		gemo_point_offsets,
		binning_point_list_keys_unsorted.data_ptr<uint64_t>(),
		binning_point_list_unsorted.data_ptr<uint32_t>(),
		radii,
		tile_grid);

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	cub::DeviceRadixSort::SortPairs(
			binning_list_sorting_space.data_ptr<uint8_t>(),
			binning_sorting_size,
			binning_point_list_keys_unsorted.data_ptr<uint64_t>(), binning_point_list_keys.data_ptr<uint64_t>(),
			binning_point_list_unsorted.data_ptr<uint32_t>(), binning_point_list.data_ptr<uint32_t>(),
			num_rendered, 0, 32 + bit);

	// imgstate_ranges的大小原本是image_width * image_height
	cudaMemset(imgstate_ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2));

	// Identify start and end of per-tile workloads in sorted list
	if(num_rendered > 0)
	{
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binning_point_list_keys.data_ptr<uint64_t>(),
			imgstate_ranges);
	}


	// Let each tile blend its range of Gaussians independently in parallel
	{
		renderCUDA<NUM_CHANNELS_MY> <<<tile_grid, block >>> (
			imgstate_ranges,
			binning_point_list.data_ptr<uint32_t>(),
			image_width, image_height,
			gemo_means2D,
			gemo_rgb,
			gemo_conic_opacity,
			imgstate_accum_alpha,
			imgstate_n_contrib,
			background,
			out_color,
			gemo_depths, 
			depth);
	}

}



void
rasterizeCUDA_forward_torch(
    // 注意所有tensor的内存分布都是row-major的！！！！！
    // 训练参数
    const torch::Tensor& means3D, // (N,3) 3D高斯的xyz 世界坐标系下
    const torch::Tensor& opacity, // (N,) 不透明度 
	const torch::Tensor& scales, // (N,3) 三个主轴方向的协方差的scale
	const torch::Tensor& rotations, // (N,4) wxyz四元数，表示协方差的旋转 
    const torch::Tensor& sh, // 球谐函数的系数
    const int degree, // 当前使用的球谐函数的阶数
    // 计算相关
    const torch::Tensor& background, // (3) 背景值 
	const torch::Tensor& viewmatrix, // Tcw，也就是把世界坐标系的点变换到camera坐标系
	const torch::Tensor& projmatrix, // 世界坐标系直接变换到NDC空间 x:[-1,1] y:[-1,1] z:[0,1]
	const torch::Tensor& campos, // 相机在世界坐标系下的平移，也就是Tcw的translation部分
    const float scale_modifier,  // 控制高斯大小的参数，主要用于show
	const float tan_halffovx, 
	const float tan_halffovy,
    const int image_height,
    const int image_width,
	const bool prefiltered,
	const bool antialiasing,
	const bool debug
)
{
    // 确保所有输入tensor都是连续的
    TORCH_CHECK(means3D.is_contiguous());
    TORCH_CHECK(opacity.is_contiguous());
    TORCH_CHECK(scales.is_contiguous());
    TORCH_CHECK(rotations.is_contiguous());
    TORCH_CHECK(background.is_contiguous());
    TORCH_CHECK(sh.is_contiguous());
    TORCH_CHECK(viewmatrix.is_contiguous());
    TORCH_CHECK(projmatrix.is_contiguous());
    TORCH_CHECK(campos.is_contiguous());
    

    const int P = means3D.size(0);
    const int sh_size = sh.size(1);
    // 获得相关options 方便初始化
    auto i32_opts = means3D.options().dtype(torch::kInt32);
    auto float_opts = means3D.options().dtype(torch::kFloat32);
    auto u8_opts = means3D.options().dtype(torch::kUInt8);
    auto u32_opts = means3D.options().dtype(torch::kUInt32);
	auto u64_opts = means3D.options().dtype(torch::kUInt64);
	

    torch::Tensor out_color = torch::full({NUM_CHANNELS_MY, image_height, image_width}, 0.0, float_opts).contiguous(); // 输出的color image
    torch::Tensor out_invdepth = torch::full({1, image_height, image_width}, 0.0, float_opts).contiguous(); // 输出的invdepth
    torch::Tensor radii = torch::full({P}, 0, i32_opts).contiguous(); // 每个高斯的投影到图像上的半径，单位为像素，用于判断可见

    // 和原始开辟buffer存储不同，这里单独开了tensor，为了方便学习
    // 初始化GeometryState
    torch::Tensor gemo_depths = torch::full({P}, 0, float_opts).contiguous(); // float*
    torch::Tensor gemo_clamped = torch::full({P * 3}, 0, u8_opts).contiguous(); // bool*
    torch::Tensor gemo_means2D = torch::full({P * 2}, 0, float_opts).contiguous(); // float2*
    torch::Tensor gemo_cov3D = torch::full({P * 6}, 0, float_opts).contiguous(); // float*
    torch::Tensor gemo_conic_opacity = torch::full({P * 4}, 0, float_opts).contiguous(); // float4*
    torch::Tensor gemo_rgb = torch::full({P * 3}, 0, float_opts).contiguous(); // float*
    torch::Tensor gemo_tiles_touched = torch::full({P}, 0, u32_opts).contiguous(); // uint32_t*
	size_t gemo_scan_size;
    // 这里实际是在给scan_size赋值，计算所需要完成前缀和的内存量
    cub::DeviceScan::InclusiveSum(nullptr, gemo_scan_size, gemo_tiles_touched.data_ptr<uint32_t>(), gemo_tiles_touched.data_ptr<uint32_t>(), P);
    torch::Tensor gemo_scanning_space = torch::full({static_cast<int>(gemo_scan_size)}, 0, u8_opts).contiguous(); // char*
    torch::Tensor gemo_point_offsets = torch::full({P}, 0, u32_opts).contiguous(); // uint32_t*

    // 初始化ImageState
    const int img_state_N = image_width * image_height;
    torch::Tensor imgstate_accum_alpha = torch::full({img_state_N}, 0, float_opts).contiguous(); // float*
    torch::Tensor imgstate_n_contrib = torch::full({img_state_N}, 0, u32_opts).contiguous(); // uint32_t*
    torch::Tensor imgstate_ranges = torch::full({img_state_N * 2}, 0, u32_opts).contiguous(); // uint2*



    if(P > 0)
    {
        rasterizeCUDA_forward_invoke(
            P, degree, sh_size,
            means3D.data_ptr<float>(),
            opacity.data_ptr<float>(),
            scales.data_ptr<float>(),
            rotations.data_ptr<float>(),
            sh.data_ptr<float>(),
            
            background.data_ptr<float>(),
            viewmatrix.data_ptr<float>(),
            projmatrix.data_ptr<float>(),
            campos.data_ptr<float>(),
            scale_modifier,
            tan_halffovx, tan_halffovy,
            image_height, image_width,
            prefiltered, 
            antialiasing,
            debug,

            gemo_depths.data_ptr<float>(),
            gemo_clamped.data_ptr<uint8_t>(),
            gemo_means2D.data_ptr<float2>(),
            gemo_cov3D.data_ptr<float>(),
            gemo_conic_opacity.data_ptr<float4>(),
            gemo_rgb.data_ptr<float>(),
            gemo_tiles_touched.data_ptr<uint32_t>(),
            gemo_scan_size,
            gemo_scanning_space.data_ptr<uint8_t>(),
            gemo_point_offsets.data_ptr<uint32_t>(),

            imgstate_accum_alpha.data_ptr<float>(),
            imgstate_n_contrib.data_ptr<uint32_t>(),
            imgstate_ranges.data_ptr<uint2>(),

            out_color.data_ptr<float>(),
            out_invdepth.data_ptr<float>(),
            radii.data_ptr<int>()
        );
    }



}




