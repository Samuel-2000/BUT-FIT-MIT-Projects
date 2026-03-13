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
 * CUDA kernel to calculate gravitation velocity
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
__global__ void calculateGravitationVelocity(Particles p, Velocities tmpVel, const unsigned N, float dt) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Ensure that we don't go beyond the number of particles
  if (i >= N) return;

  // Pointer aliasing for faster access
  float* const pPosX   = p.posX;
  float* const pPosY   = p.posY;
  float* const pPosZ   = p.posZ;
  float* const pWeight = p.weight;

  float* const tmpVelX = tmpVel.x;
  float* const tmpVelY = tmpVel.y;
  float* const tmpVelZ = tmpVel.z;

  float newVelX = 0.0f;
  float newVelY = 0.0f;
  float newVelZ = 0.0f;

  const float posX = pPosX[i];
  const float posY = pPosY[i];
  const float posZ = pPosZ[i];
  const float weight = pWeight[i];

  // Loop over all particles to compute gravitational interaction
  for (unsigned j = 0; j < N; ++j) {
    const float otherPosX = pPosX[j];
    const float otherPosY = pPosY[j];
    const float otherPosZ = pPosZ[j];
    const float otherWeight = pWeight[j];

    // Calculate the distance between particles
    const float dx = otherPosX - posX;
    const float dy = otherPosY - posY;
    const float dz = otherPosZ - posZ;

    const float r2 = dx * dx + dy * dy + dz * dz;
    const float r = sqrtf(r2) + FLT_MIN;

    const float f = (G * weight * otherWeight) / (r2 + FLT_MIN);
    
    // Use ternary operator to avoid thread divergence
    newVelX += (r > COLLISION_DISTANCE) ? (dx / r * f) : 0.0f;
    newVelY += (r > COLLISION_DISTANCE) ? (dy / r * f) : 0.0f;
    newVelZ += (r > COLLISION_DISTANCE) ? (dz / r * f) : 0.0f;
  }

  // Scale velocity change by the time step and mass (weight)
  newVelX *= dt / weight;
  newVelY *= dt / weight;
  newVelZ *= dt / weight;

  // Store the new velocities in the temporary velocity buffer
  tmpVelX[i] = newVelX;
  tmpVelY[i] = newVelY;
  tmpVelZ[i] = newVelZ;

} // end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to calculate collision velocity
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
__global__ void calculateCollisionVelocity(Particles p, Velocities tmpVel, const unsigned N, float dt) {
  // Thread ID for particle i
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Ensure that we don't go beyond the number of particles
  if (i >= N) return;
 
  // Pointer aliasing for faster memory access
  float* const pPosX   = p.posX;
  float* const pPosY   = p.posY;
  float* const pPosZ   = p.posZ;
  float* const pVelX   = p.velX;
  float* const pVelY   = p.velY;
  float* const pVelZ   = p.velZ;
  float* const pWeight = p.weight;

  float* const tmpVelX = tmpVel.x;
  float* const tmpVelY = tmpVel.y;
  float* const tmpVelZ = tmpVel.z;

  // Variables for the new velocity to be added after collision
  float newVelX = 0.0f;
  float newVelY = 0.0f;
  float newVelZ = 0.0f;

  // Read particle data for particle i
  const float posX   = pPosX[i];
  const float posY   = pPosY[i];
  const float posZ   = pPosZ[i];
  const float velX   = pVelX[i];
  const float velY   = pVelY[i];
  const float velZ   = pVelZ[i];
  const float weight = pWeight[i];

  // Loop over all particles to compute collision interaction
  for (unsigned j = 0; j < N; ++j) {
    // Read particle data for particle j
    const float otherPosX   = pPosX[j];
    const float otherPosY   = pPosY[j];
    const float otherPosZ   = pPosZ[j];
    const float otherVelX   = pVelX[j];
    const float otherVelY   = pVelY[j];
    const float otherVelZ   = pVelZ[j];
    const float otherWeight = pWeight[j];

    // Calculate the distance between particles
    const float dx = otherPosX - posX;
    const float dy = otherPosY - posY;
    const float dz = otherPosZ - posZ;

    const float r2 = dx * dx + dy * dy + dz * dz;
    const float r = sqrtf(r2);

    bool colliding = r > 0.0f && r < COLLISION_DISTANCE;

    float weightSum = weight + otherWeight;
    float weightDiff = weight - otherWeight;
    float p2_w2 = 2 * otherWeight;

    // Update velocities based on collision condition using ternary operator to avoid thread divergence
    newVelX += colliding
                ? (((weightDiff * velX + p2_w2 * otherVelX) / (weightSum)) - velX)
                : 0.f;
    newVelY += colliding
                ? (((weightDiff * velY + p2_w2 * otherVelY) / (weightSum)) - velY)
                : 0.f;
    newVelZ += colliding
                ? (((weightDiff * velZ + p2_w2 * otherVelZ) / (weightSum)) - velZ)
                : 0.f;
  }

  // Update the temporary velocity arrays with the computed changes
  tmpVelX[i] += newVelX;
  tmpVelY[i] += newVelY;
  tmpVelZ[i] += newVelZ;
}
// end of calculate_collision_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to update particles
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
/*
__global__ void updateParticles(Particles p, Velocities tmpVel, const unsigned N, float dt) {
  for (unsigned int gID = blockIdx.x * blockDim.x + threadIdx.x; gID < N; gID += blockDim.x * gridDim.x) {
      p.velX[gID] += tmpVel.x[gID];
      p.posX[gID] += p.velX[gID] * dt;

      p.velY[gID] += tmpVel.y[gID];
      p.posY[gID] += p.velY[gID] * dt;

      p.velZ[gID] += tmpVel.z[gID];
      p.posZ[gID] += p.velZ[gID] * dt;
  }
}// end of update_particle
 */

__global__ void updateParticles(Particles p, Velocities tmpVel, const unsigned N, float dt) {
  // Calculate global thread ID
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Ensure the thread is within the particle array bounds
  if (i >= N) return;

  

  // Pointer aliasing for faster access
  float* const pPosX   = p.posX;
  float* const pPosY   = p.posY;
  float* const pPosZ   = p.posZ;
  float* const pVelX   = p.velX;
  float* const pVelY   = p.velY;
  float* const pVelZ   = p.velZ;

  float* const tmpVelX = tmpVel.x;
  float* const tmpVelY = tmpVel.y;
  float* const tmpVelZ = tmpVel.z;

  // Load the current position and velocity of the particle
  float posX = pPosX[i];
  float posY = pPosY[i];
  float posZ = pPosZ[i];

  float velX = pVelX[i];
  float velY = pVelY[i];
  float velZ = pVelZ[i];

  // Load the new velocity increments from the temporary velocity arrays
  const float newVelX = tmpVelX[i];
  const float newVelY = tmpVelY[i];
  const float newVelZ = tmpVelZ[i];

  // Update the particle velocity by adding the temporary velocity
  velX += newVelX;
  velY += newVelY;
  velZ += newVelZ;

  // Update the particle position based on the new velocity
  posX += velX * dt;
  posY += velY * dt;
  posZ += velZ * dt;

  // Store the updated position and velocity back into the particle array
  pPosX[i] = posX;
  pPosY[i] = posY;
  pPosZ[i] = posZ;

  pVelX[i] = velX;
  pVelY[i] = velY;
  pVelZ[i] = velZ;
   
} // end of update_particle

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
