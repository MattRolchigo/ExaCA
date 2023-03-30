// Copyright 2021-2022 Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef EXACA_GHOST_HPP
#define EXACA_GHOST_HPP

#include "CAtypes.hpp"

#include <Kokkos_Core.hpp>

// Load data (GrainID, DOCenter, DiagonalLength) into ghost nodes if the given RankY is associated with a 1D halo region
KOKKOS_INLINE_FUNCTION void loadghostnodes(const int GhostGID, const float GhostDOCX, const float GhostDOCY,
                                           const float GhostDOCZ, const float GhostDL, const int BufSizeX,
                                           const int MyYSlices, const int RankX, const int RankY, const int RankZ,
                                           const bool AtNorthBoundary, const bool AtSouthBoundary, ViewF2D BufferNorthSend_Octahedron,
                                           ViewF2D BufferSouthSend_Octahedron, ViewI BufferNorthSend_Grain, ViewI BufferSouthSend_Grain) {

    if ((RankY == 1) && (!(AtSouthBoundary))) {
        int GNPosition = RankZ * BufSizeX + RankX;
        BufferSouthSend_Grain(GNPosition) = GhostGID;
        BufferSouthSend_Octahedron(GNPosition, 0) = GhostDOCX;
        BufferSouthSend_Octahedron(GNPosition, 1) = GhostDOCY;
        BufferSouthSend_Octahedron(GNPosition, 2) = GhostDOCZ;
        BufferSouthSend_Octahedron(GNPosition, 3) = GhostDL;
    }
    else if ((RankY == MyYSlices - 2) && (!(AtNorthBoundary))) {
        int GNPosition = RankZ * BufSizeX + RankX;
        BufferNorthSend_Grain(GNPosition) = GhostGID;
        BufferNorthSend_Octahedron(GNPosition, 0) = GhostDOCX;
        BufferNorthSend_Octahedron(GNPosition, 1) = GhostDOCY;
        BufferNorthSend_Octahedron(GNPosition, 2) = GhostDOCZ;
        BufferNorthSend_Octahedron(GNPosition, 3) = GhostDL;
    }
}
void GhostNodes1D(int, int, int NeighborRank_North, int NeighborRank_South, int nx, int MyYSlices, int MyYOffset,
                  NList NeighborX, NList NeighborY, NList NeighborZ, ViewI CellType, ViewF DOCenter, ViewI GrainID,
                  ViewF GrainUnitVector, ViewF DiagonalLength, ViewF CritDiagonalLength, int NGrainOrientations, ViewF2D BufferNorthSend_Octahedron,
                  ViewF2D BufferSouthSend_Octahedron, ViewI BufferNorthSend_Grain, ViewI BufferSouthSend_Grain, ViewF2D BufferNorthRecv_Octahedron,
                  ViewF2D BufferSouthRecv_Octahedron, ViewI BufferNorthRecv_Grain, ViewI BufferSouthRecv_Grain, int BufSizeX, int BufSizeZ, int ZBound_Low);

#endif
