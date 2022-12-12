// Copyright 2021-2022 Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef EXACA_FUNCTIONS_HPP
#define EXACA_FUNCTIONS_HPP

#include <CAtypes.hpp>
#include <Kokkos_Core.hpp>
//*****************************************************************************/
// Inline functions

// Get the orientation of a grain from a given grain ID and the number of possible orientations
KOKKOS_INLINE_FUNCTION int getGrainOrientation(int MyGrainID, int NGrainOrientations) {
    int MyOrientation = (abs(MyGrainID) - 1) % NGrainOrientations;
    return MyOrientation;
}
//*****************************************************************************/
int YMPSlicesCalc(int p, int ny, int np);
int YOffsetCalc(int p, int ny, int np);
void AddGhostNodes(int NeighborRank_North, int NeighborRank_South, int &MyYSlices, int &MyYOffset);
double MaxVal(double TestVec3[6], int NVals);
void InitialDecomposition(int id, int np, int &NeighborRank_North, int &NeighborRank_South, bool &AtNorthBoundary,
                          bool &AtSouthBoundary);
ViewF_H MisorientationCalc(int NumberOfOrientations, ViewF_H GrainUnitVector, int dir);
void InterpolateSparseData(ViewD3D &TL, ViewD3D &CR, int nxTemp, int nyTemp, int nzTemp, int HTtoCAratio);
//*****************************************************************************/
// Inline interpolation functions called during InterpolateSparseData parallel loops:

// Interpolate a value at coordinate "DirectionIndex" (between LowIndex and HighIndex, not inclusive), where LowVal is
// the value at coordinate LowIndex and HighVal is the value at coordinate HighIndex, and HTtoCAratio is equivalent to
// HighIndex-LowIndex
KOKKOS_INLINE_FUNCTION double getInterpolatedValue_line(double LowVal, double HighVal, int DirectionIndex, int LowIndex,
                                                        int HighIndex, int HTtoCAratio, double OriginalVal) {

    double InterpolatedValue;
    // Should this value be interpolated or should it be left as OriginalVal? Need values at both points to successfully
    // interpolate along a line First make sure that the points needed for interpolation are in bounds, then check to
    // make sure they have real values
    if (DirectionIndex % HTtoCAratio != 0) {
        if ((LowVal != -1) && (HighVal != -1)) {
            double LowWeight = (double)(HighIndex - DirectionIndex) / (double)(HTtoCAratio);
            double HighWeight = (double)(DirectionIndex - LowIndex) / (double)(HTtoCAratio);
            InterpolatedValue = LowWeight * LowVal + HighWeight * HighVal;
        }
        else
            InterpolatedValue = OriginalVal;
    }
    else
        InterpolatedValue = OriginalVal;
    return InterpolatedValue;
}

// Interpolate a value in the plane of DirectionIndex1 and DirectionIndex2, from values LowVal1, HighVal1, LowVal2, and
// HighVal2
KOKKOS_INLINE_FUNCTION double getInterpolatedValue_plane(double LowVal1, double HighVal1, double LowVal2,
                                                         double HighVal2, int DirectionIndex1, int DirectionIndex2,
                                                         int LowIndex1, int HighIndex1, int LowIndex2, int HighIndex2,
                                                         int HTtoCAratio, double OriginalVal) {

    double Fract = static_cast<double>(1) / static_cast<double>(2);
    double InterpolatedValue;
    // Should this value be interpolated or should it be left as OriginalVal? Need values at all 4 points to
    // successfully interpolate in the plane First make sure that the points needed for interpolation are in bounds,
    // then check to make sure they have real values
    if ((DirectionIndex1 % HTtoCAratio != 0) && (DirectionIndex2 % HTtoCAratio != 0)) {
        if ((LowVal1 != -1) && (HighVal1 != -1) && (LowVal2 != -1) && (HighVal2 != -1)) {
            double LowWeight1 = (double)(HighIndex1 - DirectionIndex1) / (double)(HTtoCAratio);
            double HighWeight1 = (double)(DirectionIndex1 - LowIndex1) / (double)(HTtoCAratio);
            double LowWeight2 = (double)(HighIndex2 - DirectionIndex2) / (double)(HTtoCAratio);
            double HighWeight2 = (double)(DirectionIndex2 - LowIndex2) / (double)(HTtoCAratio);
            InterpolatedValue = Fract * (LowWeight1 * LowVal1 + HighWeight1 * HighVal1) +
                                Fract * (LowWeight2 * LowVal2 + HighWeight2 * HighVal2);
        }
        else
            InterpolatedValue = OriginalVal;
    }
    else
        InterpolatedValue = OriginalVal;
    return InterpolatedValue;
}

// Interpolate a value between planes bounded by LowVal1 and HighVal1 in the DirectionIndex1 direction, LowVal2 and
// HighVal2 in DirectionIndex2, and LowVal3 and HighVal3 in DirectionIndex3
KOKKOS_INLINE_FUNCTION double getInterpolatedValue_volume(double LowVal1, double HighVal1, double LowVal2,
                                                          double HighVal2, double LowVal3, double HighVal3,
                                                          int DirectionIndex1, int DirectionIndex2, int DirectionIndex3,
                                                          int LowIndex1, int HighIndex1, int LowIndex2, int HighIndex2,
                                                          int LowIndex3, int HighIndex3, int HTtoCAratio,
                                                          double OriginalVal) {

    double Fract = static_cast<double>(1) / static_cast<double>(3);
    double InterpolatedValue;
    // Should this value be interpolated or should it be left as OriginalVal? Need values at all 8 points to
    // successfully interpolate in the volume First make sure that the points needed for interpolation are in bounds,
    // then check to make sure they have real values
    if ((DirectionIndex1 % HTtoCAratio != 0) && (DirectionIndex2 % HTtoCAratio != 0) &&
        (DirectionIndex3 % HTtoCAratio != 0)) {
        if ((LowVal1 != -1) && (HighVal1 != -1) && (LowVal2 != -1) && (HighVal2 != -1) && (LowVal3 != -1) &&
            (HighVal3 != -1)) {
            double LowWeight1 = (double)(HighIndex1 - DirectionIndex1) / (double)(HTtoCAratio);
            double HighWeight1 = (double)(DirectionIndex1 - LowIndex1) / (double)(HTtoCAratio);
            double LowWeight2 = (double)(HighIndex2 - DirectionIndex2) / (double)(HTtoCAratio);
            double HighWeight2 = (double)(DirectionIndex2 - LowIndex2) / (double)(HTtoCAratio);
            double LowWeight3 = (double)(HighIndex3 - DirectionIndex3) / (double)(HTtoCAratio);
            double HighWeight3 = (double)(DirectionIndex3 - LowIndex3) / (double)(HTtoCAratio);
            InterpolatedValue = Fract * (LowWeight1 * LowVal1 + HighWeight1 * HighVal1) +
                                Fract * (LowWeight2 * LowVal2 + HighWeight2 * HighVal2) +
                                Fract * (LowWeight3 * LowVal3 + HighWeight3 * HighVal3);
        }
        else
            InterpolatedValue = OriginalVal;
    }
    else
        InterpolatedValue = OriginalVal;
    return InterpolatedValue;
}

#endif
