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
/**
 * CUDA kernel to calculate new particle velocity and position with ternary operators
 * @param pIn  - input particles (initial state)
 * @param pOut - output particles (updated state)
 * @param N    - number of particles
 * @param dt   - size of the time step
 */
__global__ void calculateVelocity(Particles pIn, Particles pOut, const unsigned N, float dt) {
  // Calculate global thread ID for the particle
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Pointer aliasing for faster access
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

  // Load the initial position, velocity, and weight of particle i
  float posX = pInPosX[i];
  float posY = pInPosY[i];
  float posZ = pInPosZ[i];

  float velX = pInVelX[i];
  float velY = pInVelY[i];
  float velZ = pInVelZ[i];

  const float weight = pInWeight[i];

  // Variables to accumulate velocity changes
  float newVelX = 0.0f;
  float newVelY = 0.0f;
  float newVelZ = 0.0f;

  float newVelX_g = 0.0f;
  float newVelY_g = 0.0f;
  float newVelZ_g = 0.0f;

  // Gravitational interaction and collision calculations using a ternary operator
  for (unsigned j = 0; j < N; ++j) {
    if (i == j) continue;

    // Load position, velocity, and weight of particle j
    const float otherPosX   = pInPosX[j];
    const float otherPosY   = pInPosY[j];
    const float otherPosZ   = pInPosZ[j];
    const float otherVelX   = pInVelX[j];
    const float otherVelY   = pInVelY[j];
    const float otherVelZ   = pInVelZ[j];
    const float otherWeight = pInWeight[j];

    // Calculate distance components between particles
    const float dx = otherPosX - posX;
    const float dy = otherPosY - posY;
    const float dz = otherPosZ - posZ;

    const float r2 = dx * dx + dy * dy + dz * dz;
    const float r = sqrtf(r2 + FLT_MIN);

    if(r > COLLISION_DISTANCE) {
      const float f = (G * weight * otherWeight) / (r2 + FLT_MIN);
      newVelX_g += f * (dx / r);
      newVelY_g += f * (dy / r);
      newVelZ_g += f * (dz / r);
    } else {
      const float weightSum = weight + otherWeight + FLT_MIN;
      const float weightDiff = weight - otherWeight;
      const float p2_w2 = 2 * otherWeight;

      newVelX += (((weightDiff * velX + p2_w2 * otherVelX) / weightSum) - velX);
      newVelY += (((weightDiff * velY + p2_w2 * otherVelY) / weightSum) - velY);
      newVelZ += (((weightDiff * velZ + p2_w2 * otherVelZ) / weightSum) - velZ);
    }
    /*
    const float f = (G * weight * otherWeight) / (r2 + FLT_MIN);

    const float g_velx = f * (dx / r);
    const float g_vely = f * (dy / r);
    const float g_velz = f * (dz / r);

    
    const float weightSum = weight + otherWeight;
    const float weightDiff = weight - otherWeight;
    const float p2_w2 = 2 * otherWeight;

    const float col_velx = (((weightDiff * velX + p2_w2 * otherVelX) / weightSum) - velX);
    const float col_vely = (((weightDiff * velY + p2_w2 * otherVelY) / weightSum) - velY);
    const float col_velz = (((weightDiff * velZ + p2_w2 * otherVelZ) / weightSum) - velZ);

    // Apply gravitational force if distance > COLLISION_DISTANCE, else apply collision response
    //newVelX += (r > COLLISION_DISTANCE) ? g_velx : col_velx;
    //newVelY += (r > COLLISION_DISTANCE) ? g_vely : col_vely;
    //newVelZ += (r > COLLISION_DISTANCE) ? g_velz : col_velz;
    
    newVelX_g += (r > COLLISION_DISTANCE) ? g_velx : 0.f;
    newVelY_g += (r > COLLISION_DISTANCE) ? g_vely : 0.f;
    newVelZ_g += (r > COLLISION_DISTANCE) ? g_velz : 0.f;

    newVelX += (r < COLLISION_DISTANCE) ? col_velx : 0.f;
    newVelY += (r < COLLISION_DISTANCE) ? col_vely : 0.f;
    newVelZ += (r < COLLISION_DISTANCE) ? col_velz : 0.f;
    */
  }

  // Scale the velocity changes by dt and weight
  newVelX_g *= dt / (weight + FLT_MIN);
  newVelY_g *= dt / (weight + FLT_MIN);
  newVelZ_g *= dt / (weight + FLT_MIN);

  // Update particle velocity by adding calculated velocity changes
  velX = newVelX_g + newVelX;
  velY = newVelY_g + newVelY;
  velZ = newVelZ_g + newVelZ;

  // Update particle position based on new velocity
  posX += velX * dt;
  posY += velY * dt;
  posZ += velZ * dt;

  // Store the updated position and velocity in the output arrays
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
