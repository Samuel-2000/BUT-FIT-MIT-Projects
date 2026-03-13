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

/********************************************************************************************************************/
/* TODO: Particles data structure optimized for use on GPU. Use float3 and float4 structures defined in file Vec.h  */
/********************************************************************************************************************/
  #pragma acc update self(posX[:size])
  #pragma acc update self(posY[:size])
  #pragma acc update self(posZ[:size])
  #pragma acc update self(velX[:size])
  #pragma acc update self(velY[:size])
  #pragma acc update self(velZ[:size])
}
/**
 * @brief Constructor
 * @param N - Number of particles
 */
Velocities::Velocities(const unsigned N) {
  // Memory allocation on CPU
  x = (float *) malloc(N * sizeof(float));
  y = (float *) malloc(N * sizeof(float));
  z = (float *) malloc(N * sizeof(float));

  // Memory allocation on GPU
  #pragma acc enter data copyin(this)
  #pragma acc enter data create(x[:N])
  #pragma acc enter data create(y[:N])
  #pragma acc enter data create(z[:N])
  
  size = N;
}

/// @brief Destructor
Velocities::~Velocities() {
  #pragma acc exit data delete(x)
  #pragma acc exit data delete(y)
  #pragma acc exit data delete(z)
  #pragma acc exit data delete(this)
  free(x);
  free(y);
  free(z);
}

/**
 * @brief Copy velocities from host to device
 */
void Velocities::copyToDevice() {
  #pragma acc update device(x[:size])
  #pragma acc update device(y[:size])
  #pragma acc update device(z[:size])
}

/**
 * @brief Copy velocities from device to host
 */
void Velocities::copyToHost() {
  #pragma acc update self(x[:size])
  #pragma acc update self(y[:size])
  #pragma acc update self(z[:size])
}

/*********************************************************************************************************************/

/**
 * Calculate gravitation velocity
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
void calculateGravitationVelocity(Particles& p, Velocities& tmpVel, const unsigned N, float dt)
{
  /*******************************************************************************************************************/
  /*                    TODO: Calculate gravitation velocity, see reference CPU version,                             */
  /*                            you can use overloaded operators defined in Vec.h                                    */
  /*******************************************************************************************************************/
  float* const pPosX   = p.posX;
  float* const pPosY   = p.posY;
  float* const pPosZ   = p.posZ;
  float* const pWeight = p.weight;

  float* const tmpVelX = tmpVel.x;
  float* const tmpVelY = tmpVel.y;
  float* const tmpVelZ = tmpVel.z;

  //#pragma omp parallel for firstprivate(pPosX, pPosY, pPosZ, pWeight, tmpVelX, tmpVelY, tmpVelZ, N, dt)
  #pragma acc parallel loop present(p, tmpVel) gang worker vector
  for (unsigned i = 0u; i < N; ++i)
  {
    float newVelX{};
    float newVelY{};
    float newVelZ{};

    const float posX   = pPosX[i];
    const float posY   = pPosY[i];
    const float posZ   = pPosZ[i];
    const float weight = pWeight[i];

    //#pragma omp simd aligned(pPosX, pPosY, pPosZ, pVelX, pVelY, pVelZ, pWeight, tmpVelX, tmpVelY, tmpVelZ: dataAlignment)
    #pragma acc loop seq
    for (unsigned j = 0u; j < N; ++j)
    {
      const float otherPosX   = pPosX[j];
      const float otherPosY   = pPosY[j];
      const float otherPosZ   = pPosZ[j];
      const float otherWeight = pWeight[j];

      const float dx = otherPosX - posX;
      const float dy = otherPosY - posY;
      const float dz = otherPosZ - posZ;

      const float r2 = dx * dx + dy * dy + dz * dz;
      const float r = std::sqrt(r2);

      const float f = G * weight * otherWeight / r2 + FLT_MIN;

      newVelX += (r > COLLISION_DISTANCE) ? dx / r * f : 0.f;
      newVelY += (r > COLLISION_DISTANCE) ? dy / r * f : 0.f;
      newVelZ += (r > COLLISION_DISTANCE) ? dz / r * f : 0.f;
    }

    newVelX *= dt / weight;
    newVelY *= dt / weight;
    newVelZ *= dt / weight;

    tmpVelX[i] = newVelX;
    tmpVelY[i] = newVelY;
    tmpVelZ[i] = newVelZ;
  }

}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * Calculate collision velocity
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
void calculateCollisionVelocity(Particles& p, Velocities& tmpVel, const unsigned N, float dt)
{
  /*******************************************************************************************************************/
  /*                    TODO: Calculate collision velocity, see reference CPU version,                               */
  /*                            you can use overloaded operators defined in Vec.h                                    */
  /*******************************************************************************************************************/
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

  //#pragma omp parallel for firstprivate(pPosX, pPosY, pPosZ, pVelX, pVelY, pVelZ, pWeight, tmpVelX, tmpVelY, tmpVelZ, N, dt)
  #pragma acc parallel loop present(p, tmpVel) gang worker vector
  for (unsigned i = 0u; i < N; ++i)
  {
    float newVelX{};
    float newVelY{};
    float newVelZ{};

    const float posX   = pPosX[i];
    const float posY   = pPosY[i];
    const float posZ   = pPosZ[i];
    const float velX   = pVelX[i];
    const float velY   = pVelY[i];
    const float velZ   = pVelZ[i];
    const float weight = pWeight[i];

    //#pragma omp simd aligned(pPosX, pPosY, pPosZ, pVelX, pVelY, pVelZ, pWeight, tmpVelX, tmpVelY, tmpVelZ: dataAlignment)
    #pragma acc loop seq
    for (unsigned j = 0u; j < N; ++j)
    {
      const float otherPosX   = pPosX[j];
      const float otherPosY   = pPosY[j];
      const float otherPosZ   = pPosZ[j];
      const float otherVelX   = pVelX[j];
      const float otherVelY   = pVelY[j];
      const float otherVelZ   = pVelZ[j];
      const float otherWeight = pWeight[j];

      const float dx = otherPosX - posX;
      const float dy = otherPosY - posY;
      const float dz = otherPosZ - posZ;

      const float r2 = dx * dx + dy * dy + dz * dz;
      const float r = std::sqrt(r2);

      newVelX += (r > 0.f && r < COLLISION_DISTANCE)
                 ? (((weight * velX - otherWeight * velX + 2.f * otherWeight * otherVelX) / (weight + otherWeight)) - velX)
                 : 0.f;
      newVelY += (r > 0.f && r < COLLISION_DISTANCE)
                 ? (((weight * velY - otherWeight * velY + 2.f * otherWeight * otherVelY) / (weight + otherWeight)) - velY)
                 : 0.f;
      newVelZ += (r > 0.f && r < COLLISION_DISTANCE)
                 ? (((weight * velZ - otherWeight * velZ + 2.f * otherWeight * otherVelZ) / (weight + otherWeight)) - velZ)
                 : 0.f;
    }

    tmpVelX[i] += newVelX;
    tmpVelY[i] += newVelY;
    tmpVelZ[i] += newVelZ;
  }

}// end of calculate_collision_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * Update particles
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
void updateParticles(Particles& p, Velocities& tmpVel, const unsigned N, float dt)
{
  /*******************************************************************************************************************/
  /*                    TODO: Update particles position and velocity, see reference CPU version,                     */
  /*                            you can use overloaded operators defined in Vec.h                                    */
  /*******************************************************************************************************************/
  float* const pPosX   = p.posX;
  float* const pPosY   = p.posY;
  float* const pPosZ   = p.posZ;
  float* const pVelX   = p.velX;
  float* const pVelY   = p.velY;
  float* const pVelZ   = p.velZ;

  float* const tmpVelX = tmpVel.x;
  float* const tmpVelY = tmpVel.y;
  float* const tmpVelZ = tmpVel.z;

  //#pragma omp parallel for simd \
         firstprivate(pPosX, pPosY, pPosZ, pVelX, pVelY, pVelZ, tmpVelX, tmpVelY, tmpVelZ, N, dt) \
         aligned(pPosX, pPosY, pPosZ, pVelX, pVelY, pVelZ, tmpVelX, tmpVelY, tmpVelZ: dataAlignment)
  #pragma acc parallel loop present(p, tmpVel) gang worker vector
  for (unsigned i = 0u; i < N; ++i)
  {
    float posX = pPosX[i];
    float posY = pPosY[i];
    float posZ = pPosZ[i];

    float velX = pVelX[i];
    float velY = pVelY[i];
    float velZ = pVelZ[i];

    const float newVelX = tmpVelX[i];
    const float newVelY = tmpVelY[i];
    const float newVelZ = tmpVelZ[i];

    velX += newVelX;
    velY += newVelY;
    velZ += newVelZ;

    posX += velX * dt;
    posY += velY * dt;
    posZ += velZ * dt;

    pPosX[i] = posX;
    pPosY[i] = posY;
    pPosZ[i] = posZ;

    pVelX[i] = velX;
    pVelY[i] = velY;
    pVelZ[i] = velZ;
  }

}// end of update_particle
//----------------------------------------------------------------------------------------------------------------------

/**
 * Calculate particles center of mass
 * @param p    - particles
 * @param com  - pointer to a center of mass
 * @param lock - pointer to a user-implemented lock
 * @param N    - Number of particles
 */
void centerOfMass(Particles& p, float4& com, int* lock, const unsigned N)
{

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
