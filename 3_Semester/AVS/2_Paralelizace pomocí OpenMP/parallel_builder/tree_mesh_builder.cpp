/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  Samuel Kuchta <xkucht11@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    2.12.2024
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{

}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field) {
	unsigned totalTriangles = 0;
    Vec3_t<float> cubeOffset(0, 0, 0);

    #pragma omp parallel default(none) shared(totalTriangles, field) private(cubeOffset)
    #pragma omp single nowait
    totalTriangles = octree_decomposition(field, cubeOffset, mGridSize);

	return totalTriangles;
}

unsigned TreeMeshBuilder::octree_decomposition(const ParametricScalarField& field, const Vec3_t<float>& position, const unsigned cube_size) {
    // Precompute half cube size
    float half_cube_size = cube_size * 0.5f;

    // Calculate middle point of the cube
    Vec3_t<float> mid_point(
        (position.x + half_cube_size) * mGridResolution,
        (position.y + half_cube_size) * mGridResolution,
        (position.z + half_cube_size) * mGridResolution
    );

    // Calculate threshold for early exit
    float threshold = field.getIsoLevel() + (CONST_SQRT * mGridResolution * cube_size);

    // Early termination check
    if (evaluateFieldAt(mid_point, field) > threshold) {
        return 0;
    }

    if (cube_size <= 1) {  //cutoff
        return buildCube(position, field);  // Base case: build the cube
    }

    unsigned totalTriangles = 0;
    for (size_t i = 0; i < 8; ++i) {  // Loop through child cubes
        #pragma omp task shared(totalTriangles)
        {
            // Compute child position using vertex offsets
            Vec3_t<float> vertexOffset = sc_vertexNormPos[i];
            Vec3_t<float> child_position(
                position.x + vertexOffset.x * half_cube_size,
                position.y + vertexOffset.y * half_cube_size,
                position.z + vertexOffset.z * half_cube_size
            );

            // Recursively evaluate child
            unsigned child_triangles_cnt = octree_decomposition(field, child_position, half_cube_size);

            #pragma omp atomic update
            totalTriangles += child_triangles_cnt;
        }
    }
    #pragma omp taskwait
    return totalTriangles;
}


float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    //#pragma omp smd reduction(min:value) linear(pPoints) simdlen(32)
    for(unsigned i = 0; i < count; ++i) {
        float distanceSquared  = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
        distanceSquared       += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
        distanceSquared       += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);
        value = std::min(value, distanceSquared);
    }

    return sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    #pragma omp critical(emitTriangle)
    mTriangles.push_back(triangle);
}
