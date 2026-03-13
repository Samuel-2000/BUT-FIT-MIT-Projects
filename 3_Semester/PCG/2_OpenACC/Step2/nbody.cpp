/**
 * @file      nbody.cpp
 *
 * @author    Samuel Kuchta \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            xkucht11@fit.vutbr.cz
 *
 * @brief     PCG Assignment 2
 *
 * @version   2023
 *
 * @date      04 October   2023, 09:00 (created) \n
 */

#include <cfloat>
#include <cmath>

#include "nbody.h"
#include "Vec.h"

/* Constants */
constexpr float G                  = 6.67384e-11f;
constexpr float COLLISION_DISTANCE = 0.01f;

/*********************************************************************************************************************/
/*                TODO: Fullfill Partile's and Velocitie's constructors, destructors and methods                     */
/*                                    for data copies between host and device                                        */
/*********************************************************************************************************************/

/**
 * @brief Constructor
 * @param N - Number of particles
 */
Particles::Particles(const unsigned N) {
  // Memory allocation on CPU
  posX = (float *) malloc(N * sizeof(float));
  posY = (float *) malloc(N * sizeof(float));
  posZ = (float *) malloc(N * sizeof(float));
  velX = (float *) malloc(N * sizeof(float));
  velY = (float *) malloc(N * sizeof(float));
  velZ = (float *) malloc(N * sizeof(float));
  weight = (float *) malloc(N * sizeof(float));

  // Memory allocation on GPU
  #pragma acc enter data copyin(this)
  #pragma acc enter data create(posX[:N])
  #pragma acc enter data create(posY[:N])
  #pragma acc enter data create(posZ[:N])
  #pragma acc enter data create(velX[:N])
  #pragma acc enter data create(velY[:N])
  #pragma acc enter data create(velZ[:N])
  #pragma acc enter data create(weight[:N])

  size = N;
}

/// @brief Destructor
Particles::~Particles() {
  #pragma acc exit data delete(posX)
  #pragma acc exit data delete(posY)
  #pragma acc exit data delete(posZ)
  #pragma acc exit data delete(velX)
  #pragma acc exit data delete(velY)
  #pragma acc exit data delete(velZ)
  #pragma acc exit data delete(weight)
  #pragma acc exit data delete(this)
  free(posX);
  free(posY);
  free(posZ);
  free(velX);
  free(velY);
  free(velZ);
  free(weight);
}

/**
 * @brief Copy particles from host to device
 */
void Particles::copyToDevice() {
  #pragma acc update device(posX[:size])
  #pragma acc update device(posY[:size])
  #pragma acc update device(posZ[:size])
  #pragma acc update device(velX[:size])
  #pragma acc update device(velY[:size])
  #pragma acc update device(velZ[:size])
  #pragma acc update device(weight[:size])
}

/**
 * @brief Copy particles from device to host
 */
void Particles::copyToHost() {
  #pragma acc update self(posX[:size])
  #pragma acc update self(posY[:size])
  #pragma acc update self(posZ[:size])
  #pragma acc update self(velX[:size])
  #pragma acc update self(velY[:size])
  #pragma acc update self(velZ[:size])
}

/*********************************************************************************************************************/

/**
 * Calculate velocity
 * @param pIn  - particles input
 * @param pOut - particles output
 * @param N    - Number of particles
 * @param dt   - Size of the time step
 */
void calculateVelocity(Particles& pIn, Particles& pOut, const unsigned N, float dt) {
  #pragma acc parallel loop gang
  for (unsigned i = 0; i < N; i++) {
    float posX = pIn.posX[i];
    float posY = pIn.posY[i];
    float posZ = pIn.posZ[i];
    float velX = pIn.velX[i];
    float velY = pIn.velY[i];
    float velZ = pIn.velZ[i];
    const float weight = pIn.weight[i];

    float newVelX_g = 0.0f;
    float newVelY_g = 0.0f;
    float newVelZ_g = 0.0f;
    float newVelX = 0.0f;
    float newVelY = 0.0f;
    float newVelZ = 0.0f;

    // Iterate over all particles in a tiled fashion
    #pragma acc loop worker vector
    for (unsigned j = 0; j < N; j++) {
      const float dx = pIn.posX[j] - posX;
      const float dy = pIn.posY[j] - posY;
      const float dz = pIn.posZ[j] - posZ;

      const float r2 = dx * dx + dy * dy + dz * dz;
      const float r = sqrtf(r2 + FLT_MIN);

      if (r > COLLISION_DISTANCE) {
        // Gravitational interaction
        const float f = (G * weight * pIn.weight[j]) / (r2 + FLT_MIN);
        newVelX_g += f * (dx / r);
        newVelY_g += f * (dy / r);
        newVelZ_g += f * (dz / r);
      } else {
        // Collision response
        const float weightSum = weight + pIn.weight[j] + FLT_MIN;
        const float weightDiff = weight - pIn.weight[j];
        const float p2_w2 = 2 * pIn.weight[j];

        newVelX += (((weightDiff * velX + p2_w2 * pIn.velX[j]) / weightSum) - velX);
        newVelY += (((weightDiff * velY + p2_w2 * pIn.velY[j]) / weightSum) - velY);
        newVelZ += (((weightDiff * velZ + p2_w2 * pIn.velZ[j]) / weightSum) - velZ);
      }
    }

    // Update velocities
    newVelX_g *= dt / (weight + FLT_MIN);
    newVelY_g *= dt / (weight + FLT_MIN);
    newVelZ_g *= dt / (weight + FLT_MIN);

    velX += newVelX_g + newVelX;
    velY += newVelY_g + newVelY;
    velZ += newVelZ_g + newVelZ;

    // Update positions
    posX += velX * dt;
    posY += velY * dt;
    posZ += velZ * dt;

    // Store the updated values in the output particle array
    pOut.posX[i] = posX;
    pOut.posY[i] = posY;
    pOut.posZ[i] = posZ;
    pOut.velX[i] = velX;
    pOut.velY[i] = velY;
    pOut.velZ[i] = velZ;
  }
}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * Calculate particles center of mass
 * @param p         - particles
 * @param comBuffer - pointer to a center of mass buffer
 * @param N         - Number of particles
 */
void centerOfMass(Particles& p, float4* comBuffer, const unsigned N) {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float w = 0.0f;

  #pragma acc parallel loop present(p, comBuffer) reduction(+:x,y,z,w)
  for (unsigned i = 0; i < N; i++){
    x += p.posX[i] * p.weight[i];
    y += p.posY[i] * p.weight[i];
    z += p.posZ[i] * p.weight[i];
    w += p.weight[i];
  }

  comBuffer->x = x / w;
  comBuffer->y = y / w;
  comBuffer->z = z / w;
  comBuffer->w = w;

  #pragma acc update device(comBuffer[0:1])

  return;
}// end of centerOfMass
//----------------------------------------------------------------------------------------------------------------------

/**
 * CPU implementation of the Center of Mass calculation
 * @param particles - All particles in the system
 * @param N         - Number of particles
 */
float4 centerOfMassRef(MemDesc& memDesc)
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
