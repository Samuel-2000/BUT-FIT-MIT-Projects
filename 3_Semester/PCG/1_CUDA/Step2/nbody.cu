/**
 * @file      nbody.cu
 *
 * @author    Samuel Kuchta \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            xkucht11@fit.vutbr.cz
 *
 * @brief     PCG Assignment 1
 *
 * @version   2024
 *
 * @date      04 October   2023, 09:00 (created) \n
 */

#include <device_launch_parameters.h>

#include "nbody.cuh"
#include <cfloat>

/* Constants */
constexpr float G                  = 6.67384e-11f;
constexpr float COLLISION_DISTANCE = 0.01f;

/**
 * CUDA kernel to calculate new particles velocity and position
 * @param pIn  - particles in
 * @param pOut - particles out
 * @param N    - Number of particles
 * @param dt   - Size of the time step
 */
__global__ void calculateVelocity(Particles pIn, Particles pOut, const unsigned N, float dt) {
    // Calculate global and local thread IDs
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    
    // Allocate shared memory based on block size
    extern __shared__ float sharedMemory[];
    float* shPosX   = sharedMemory;
    float* shPosY   = &shPosX[blockDim.x];
    float* shPosZ   = &shPosY[blockDim.x];
    float* shVelX   = &shPosZ[blockDim.x];
    float* shVelY   = &shVelX[blockDim.x];
    float* shVelZ   = &shVelY[blockDim.x];
    float* shWeight = &shVelZ[blockDim.x];

    float* const pInPosX   = pIn.posX;
    float* const pInPosY   = pIn.posY;
    float* const pInPosZ   = pIn.posZ;
    float* const pInVelX   = pIn.velX;
    float* const pInVelY   = pIn.velY;
    float* const pInVelZ   = pIn.velZ;
    float* const pInWeight = pIn.weight;

    float* const pOutPosX  = pOut.posX;
    float* const pOutPosY  = pOut.posY;
    float* const pOutPosZ  = pOut.posZ;
    float* const pOutVelX  = pOut.velX;
    float* const pOutVelY  = pOut.velY;
    float* const pOutVelZ  = pOut.velZ;

    float posX = pInPosX[i];
    float posY = pInPosY[i];
    float posZ = pInPosZ[i];

    float velX = pInVelX[i];
    float velY = pInVelY[i];
    float velZ = pInVelZ[i];

    const float weight = pInWeight[i];

    float newVelX = 0.0f;
    float newVelY = 0.0f;
    float newVelZ = 0.0f;

    float newVelX_g = 0.0f;
    float newVelY_g = 0.0f;
    float newVelZ_g = 0.0f;

        // Prefetch first tile
    float prefetchPosX   = pInPosX[tid];
    float prefetchPosY   = pInPosY[tid];
    float prefetchPosZ   = pInPosZ[tid];
    float prefetchVelX   = pInVelX[tid];
    float prefetchVelY   = pInVelY[tid];
    float prefetchVelZ   = pInVelZ[tid];
    float prefetchWeight = pInWeight[tid];

    // Iterate over chunks of particles
    for (unsigned int tile = 0; tile < gridDim.x; tile++) {
        // Move prefetched data to shared memory for the current tile
        shPosX[tid]   = prefetchPosX;
        shPosY[tid]   = prefetchPosY;
        shPosZ[tid]   = prefetchPosZ;
        shVelX[tid]   = prefetchVelX;
        shVelY[tid]   = prefetchVelY;
        shVelZ[tid]   = prefetchVelZ;
        shWeight[tid] = prefetchWeight;

        __syncthreads();

        // Prefetch next tile data
        unsigned int j = (tile + 1) * blockDim.x + tid;
        if (j < N) {
            prefetchPosX   = pInPosX[j];
            prefetchPosY   = pInPosY[j];
            prefetchPosZ   = pInPosZ[j];
            prefetchVelX   = pInVelX[j];
            prefetchVelY   = pInVelY[j];
            prefetchVelZ   = pInVelZ[j];
            prefetchWeight = pInWeight[j];
        }

        for (unsigned int j = 0; j < blockDim.x && tile * blockDim.x + j < N; ++j) {
            const float dx = shPosX[j] - posX;
            const float dy = shPosY[j] - posY;
            const float dz = shPosZ[j] - posZ;

            const float r2 = dx * dx + dy * dy + dz * dz;
            const float r = sqrtf(r2 + FLT_MIN);

            if (r > COLLISION_DISTANCE) {
                const float f = (G * weight * shWeight[j]) / (r2 + FLT_MIN);
                newVelX_g += f * (dx / r);
                newVelY_g += f * (dy / r);
                newVelZ_g += f * (dz / r);
            } else {
                const float weightSum = weight + shWeight[j] + FLT_MIN;
                const float weightDiff = weight - shWeight[j];
                const float p2_w2 = 2 * shWeight[j];

                newVelX += (((weightDiff * velX + p2_w2 * shVelX[j]) / weightSum) - velX);
                newVelY += (((weightDiff * velY + p2_w2 * shVelY[j]) / weightSum) - velY);
                newVelZ += (((weightDiff * velZ + p2_w2 * shVelZ[j]) / weightSum) - velZ);
            }
        }
        __syncthreads();
    }

    newVelX_g *= dt / (weight + FLT_MIN);
    newVelY_g *= dt / (weight + FLT_MIN);
    newVelZ_g *= dt / (weight + FLT_MIN);

    velX = newVelX_g + newVelX;
    velY = newVelY_g + newVelY;
    velZ = newVelZ_g + newVelZ;

    posX += velX * dt;
    posY += velY * dt;
    posZ += velZ * dt;

    pOutPosX[i] = posX;
    pOutPosY[i] = posY;
    pOutPosZ[i] = posZ;

    pOutVelX[i] = velX;
    pOutVelY[i] = velY;
    pOutVelZ[i] = velZ;
}
// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to calculate particles center of mass
 * @param p    - particles
 * @param com  - pointer to a center of mass
 * @param lock - pointer to a user-implemented lock
 * @param N    - Number of particles
 */
__global__ void centerOfMass(Particles p, float4* com, int* lock, const unsigned N)
{

}// end of centerOfMass
//----------------------------------------------------------------------------------------------------------------------

/**
 * CPU implementation of the Center of Mass calculation
 * @param particles - All particles in the system
 * @param N         - Number of particles
 */
__host__ float4 centerOfMassRef(MemDesc& memDesc)
{
  float4 com{};

  for (std::size_t i{}; i < memDesc.getDataSize(); i++)
  {
    const float3 pos = {memDesc.getPosX(i), memDesc.getPosY(i), memDesc.getPosZ(i)};
    const float  w   = memDesc.getWeight(i);

    // Calculate the vector on the line connecting current body and most recent position of center-of-mass
    // Calculate weight ratio only if at least one particle isn't massless
    const float4 d = {pos.x - com.x,
                      pos.y - com.y,
                      pos.z - com.z,
                      ((memDesc.getWeight(i) + com.w) > 0.0f)
                        ? ( memDesc.getWeight(i) / (memDesc.getWeight(i) + com.w))
                        : 0.0f};

    // Update position and weight of the center-of-mass according to the weight ration and vector
    com.x += d.x * d.w;
    com.y += d.y * d.w;
    com.z += d.z * d.w;
    com.w += w;
  }

  return com;
}// enf of centerOfMassRef
//----------------------------------------------------------------------------------------------------------------------
