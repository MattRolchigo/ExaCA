// Copyright 2021-2023 Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef EXACA_FUNCTIONS_HPP
#define EXACA_FUNCTIONS_HPP

#include <CAtypes.hpp>
#include <Kokkos_Core.hpp>
//*****************************************************************************/
// Inline functions
// Assign octahedron a small initial size, and a center location
// Note that the Y coordinate is relative to the domain origin to keep the coordinate system continuous across ranks
template <typename ViewTypeFloat, typename ViewTypeShort>
KOKKOS_INLINE_FUNCTION void createNewOctahedron(const int index, ViewTypeFloat DiagonalLength, ViewTypeFloat DOCenter,
                                                const int coord_x, const int coord_y, const int y_offset,
                                                const int coord_z, ViewTypeShort SpawnDirection, ViewTypeFloat GrainCenterLocation, ViewTypeFloat FractMaxTipVelocity) {
    DiagonalLength(index) = 0.01;
    DOCenter(3 * index) = coord_x + 0.5;
    DOCenter(3 * index + 1) = coord_y + y_offset + 0.5;
    DOCenter(3 * index + 2) = coord_z + 0.5;
    // Penalization/branching data for octahedron
    SpawnDirection(index) = -1;
    GrainCenterLocation(3 * index) = coord_x + 0.5;
    GrainCenterLocation(3 * index + 1) = coord_y + y_offset + 0.5;
    GrainCenterLocation(3 * index + 2) = coord_z + 0.5;
    FractMaxTipVelocity(index) = 1.0;
}

template <typename ViewTypeFloat, typename ViewTypeShort>
KOKKOS_INLINE_FUNCTION
void calcOctahedronPenalization(const double xp, const double yp, const double zp, int MyOrientation, ViewF GrainUnitVector, const int old_cell_index, const int new_cell_index, ViewTypeShort SpawnDirection, ViewTypeFloat GrainCenterLocation, ViewTypeFloat FractMaxTipVelocity) {

    // Get the x,y,z coordinates of the cell that originally spawned this chain of capture events (considered to be the grain center)
    // Note that this 1D value is relative to the domain's original to keep the coordinate system continuous across ranks
    float grain_center_x = GrainCenterLocation(3 * old_cell_index);
    float grain_center_y = GrainCenterLocation(3 * old_cell_index + 1);
    float grain_center_z = GrainCenterLocation(3 * old_cell_index + 2);
    
    // Orientation of this portion of the solid-liquid interface with respect to the grain center
    float envelope_orientation_x = xp - grain_center_x;
    float envelope_orientation_y = yp - grain_center_y;
    float envelope_orientation_z = zp - grain_center_z;
    float envelope_orientation_mag = sqrtf(envelope_orientation_x * envelope_orientation_x + envelope_orientation_y * envelope_orientation_y + envelope_orientation_z * envelope_orientation_z);
    float envelope_orientation_x_norm = envelope_orientation_x / envelope_orientation_mag;
    float envelope_orientation_y_norm = envelope_orientation_y / envelope_orientation_mag;
    float envelope_orientation_z_norm = envelope_orientation_z / envelope_orientation_mag;

    // Determine the closest 100 direction to the envelope orientation and the index of said angle
    float Angle_envelope[6];
    Angle_envelope[0] = acos(GrainUnitVector(9 * MyOrientation) * envelope_orientation_x_norm + GrainUnitVector(9 * MyOrientation + 1) * envelope_orientation_y_norm + GrainUnitVector(9 * MyOrientation + 2) * envelope_orientation_z_norm);
    Angle_envelope[1] = acos(GrainUnitVector(9 * MyOrientation + 3) * envelope_orientation_x_norm + GrainUnitVector(9 * MyOrientation + 4) * envelope_orientation_y_norm + GrainUnitVector(9 * MyOrientation + 5) * envelope_orientation_z_norm);
    Angle_envelope[2] = acos(GrainUnitVector(9 * MyOrientation + 6) * envelope_orientation_x_norm + GrainUnitVector(9 * MyOrientation + 7) * envelope_orientation_y_norm + GrainUnitVector(9 * MyOrientation + 8) * envelope_orientation_z_norm);
    Angle_envelope[3] = 3.14159 - Angle_envelope[0];
    Angle_envelope[4] = 3.14159 - Angle_envelope[1];
    Angle_envelope[5] = 3.14159 - Angle_envelope[2];
    float min_misorientation = abs(Angle_envelope[0]);
    short spawn_dir_min_misorientation = 0;
    for (short dir=1; dir<6; dir++) {
        if (abs(Angle_envelope[dir]) < min_misorientation) {
            min_misorientation = abs(Angle_envelope[dir]);
            spawn_dir_min_misorientation = dir;
        }
    }
    // Use this minimum angle (converted to radians and normalized by the max possible misorientation) to calculate the fraction of the tip velocity applied to this portion of the interface
    FractMaxTipVelocity(new_cell_index) = 1.0 - sqrt(57.295780 * min_misorientation / 54.7);
    SpawnDirection(new_cell_index) = spawn_dir_min_misorientation;
    // Use old grain center
    GrainCenterLocation(3 * new_cell_index) = grain_center_x;
    GrainCenterLocation(3 * new_cell_index + 1) = grain_center_y;
    GrainCenterLocation(3 * new_cell_index + 2) = grain_center_z;
}

// For the newly active cell located at 1D array position D3D1ConvPosition (3D center coordinate of xp, yp, zp), update
// CritDiagonalLength values for cell capture of neighboring cells. The octahedron has a center located at (cx, cy, cz)
// Note that yp and cy are relative to the domain origin to keep the coordinate system continuous across ranks
template <typename ViewType>
KOKKOS_INLINE_FUNCTION void calcCritDiagonalLength(int index, float xp, float yp, float zp, float cx, float cy,
                                                   float cz, NList NeighborX, NList NeighborY, NList NeighborZ,
                                                   int MyOrientation, ViewType GrainUnitVector,
                                                   ViewType CritDiagonalLength) {
    // Calculate critical octahedron diagonal length to activate nearest neighbor.
    // First, calculate the unique planes (4) associated with all octahedron faces (8)
    // Then just look at distance between face and the point of interest (cell center of
    // neighbor). The critical diagonal length will be the maximum of these (since all other
    // planes will have passed over the point by then
    // ... meaning it must be in the octahedron)
    double Fx[4], Fy[4], Fz[4];

    Fx[0] = GrainUnitVector(9 * MyOrientation) + GrainUnitVector(9 * MyOrientation + 3) +
            GrainUnitVector(9 * MyOrientation + 6);
    Fx[1] = GrainUnitVector(9 * MyOrientation) - GrainUnitVector(9 * MyOrientation + 3) +
            GrainUnitVector(9 * MyOrientation + 6);
    Fx[2] = GrainUnitVector(9 * MyOrientation) + GrainUnitVector(9 * MyOrientation + 3) -
            GrainUnitVector(9 * MyOrientation + 6);
    Fx[3] = GrainUnitVector(9 * MyOrientation) - GrainUnitVector(9 * MyOrientation + 3) -
            GrainUnitVector(9 * MyOrientation + 6);

    Fy[0] = GrainUnitVector(9 * MyOrientation + 1) + GrainUnitVector(9 * MyOrientation + 4) +
            GrainUnitVector(9 * MyOrientation + 7);
    Fy[1] = GrainUnitVector(9 * MyOrientation + 1) - GrainUnitVector(9 * MyOrientation + 4) +
            GrainUnitVector(9 * MyOrientation + 7);
    Fy[2] = GrainUnitVector(9 * MyOrientation + 1) + GrainUnitVector(9 * MyOrientation + 4) -
            GrainUnitVector(9 * MyOrientation + 7);
    Fy[3] = GrainUnitVector(9 * MyOrientation + 1) - GrainUnitVector(9 * MyOrientation + 4) -
            GrainUnitVector(9 * MyOrientation + 7);

    Fz[0] = GrainUnitVector(9 * MyOrientation + 2) + GrainUnitVector(9 * MyOrientation + 5) +
            GrainUnitVector(9 * MyOrientation + 8);
    Fz[1] = GrainUnitVector(9 * MyOrientation + 2) - GrainUnitVector(9 * MyOrientation + 5) +
            GrainUnitVector(9 * MyOrientation + 8);
    Fz[2] = GrainUnitVector(9 * MyOrientation + 2) + GrainUnitVector(9 * MyOrientation + 5) -
            GrainUnitVector(9 * MyOrientation + 8);
    Fz[3] = GrainUnitVector(9 * MyOrientation + 2) - GrainUnitVector(9 * MyOrientation + 5) -
            GrainUnitVector(9 * MyOrientation + 8);

    for (int n = 0; n < 26; n++) {
        float x0 = xp + NeighborX[n] - cx;
        float y0 = yp + NeighborY[n] - cy;
        float z0 = zp + NeighborZ[n] - cz;
        float D0 = x0 * Fx[0] + y0 * Fy[0] + z0 * Fz[0];
        float D1 = x0 * Fx[1] + y0 * Fy[1] + z0 * Fz[1];
        float D2 = x0 * Fx[2] + y0 * Fy[2] + z0 * Fz[2];
        float D3 = x0 * Fx[3] + y0 * Fy[3] + z0 * Fz[3];
        float Dfabs = fmax(fmax(fabs(D0), fabs(D1)), fmax(fabs(D2), fabs(D3)));
        CritDiagonalLength(26 * index + n) = Dfabs;
    }
}

// Get the orientation of a grain from a given grain ID and the number of possible orientations
// By default, start indexing at 0 (GrainID of 1 has Orientation number 0), optionally starting at 1
// GrainID of 0 is a special case - has orientation 0 no matter what (only used when reconstructing grain ID in
// getGrainID)
KOKKOS_INLINE_FUNCTION int getGrainOrientation(int MyGrainID, int NGrainOrientations, bool StartAtZero = true) {
    int MyOrientation;
    if (MyGrainID == 0)
        MyOrientation = 0;
    else {
        MyOrientation = (abs(MyGrainID) - 1) % NGrainOrientations;
        if (!(StartAtZero))
            MyOrientation++;
    }
    return MyOrientation;
}
// Get the repeat number for the orientation of a grain with a given grain ID
// 1, 2, 3... or -1, -2, -3...
KOKKOS_INLINE_FUNCTION int getGrainNumber(int MyGrainID, int NGrainOrientations) {
    int MyGrainNumber = (abs(MyGrainID) - 1) / NGrainOrientations + 1;
    if (MyGrainID < 0)
        MyGrainNumber = -MyGrainNumber;
    return MyGrainNumber;
}
// Get the grain ID from the repeat number of a grain from a given grain ID and the number of possible orientations
KOKKOS_INLINE_FUNCTION int getGrainID(int NGrainOrientations, int MyGrainOrientation, int MyGrainNumber) {
    int MyGrainID = NGrainOrientations * (abs(MyGrainNumber) - 1) + MyGrainOrientation;
    if (MyGrainNumber < 0)
        MyGrainID = -MyGrainID;
    return MyGrainID;
}
// Get the 1D cell coordinate from the x, y, and z cell positions
KOKKOS_INLINE_FUNCTION int get1Dindex(const int coord_x, const int coord_y_local, const int coord_z, const int nx,
                                      const int ny_local) {
    int index = coord_z * nx * ny_local + coord_x * ny_local + coord_y_local;
    return index;
}
// TODO: There is probably some creative way to combine these functions and return one object containing the x, y, and z
// positions Get the z cell position of the cell from the 1D cell coordinate
KOKKOS_INLINE_FUNCTION int getCoordZ(const int index, const int nx, const int ny) {
    int coord_z = index / (nx * ny);
    return coord_z;
}
// Get the y cell position of the cell from the 1D cell coordinate
KOKKOS_INLINE_FUNCTION int getCoordY(const int index, const int nx, const int ny) {
    int Rem = index % (nx * ny);
    int coord_y = Rem % ny;
    return coord_y;
}
// Get the x cell position of the cell from the 1D cell coordinate
KOKKOS_INLINE_FUNCTION int getCoordX(const int index, const int nx, const int ny) {
    int Rem = index % (nx * ny);
    int coord_x = Rem / ny;
    return coord_x;
}

//*****************************************************************************/
int get_nylocal(int p, int ny, int np);
int get_yoffset(int p, int ny, int np);
void AddGhostNodes(int NeighborRank_North, int NeighborRank_South, int &ny_local, int &y_offset);
double MaxVal(double TestVec3[6], int NVals);
void InitialDecomposition(int id, int np, int &NeighborRank_North, int &NeighborRank_South, bool &AtNorthBoundary,
                          bool &AtSouthBoundary);
ViewF_H MisorientationCalc(int NumberOfOrientations, ViewF_H GrainUnitVector, int dir);
float calcVolFractionNucleated(int id, int nx, int ny_local, int DomainSize, ViewS LayerID, ViewI GrainID,
                               bool AtNorthBoundary, bool AtSouthBoundary);

#endif
