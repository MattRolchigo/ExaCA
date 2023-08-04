// Copyright 2021-2023 Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef EXACA_GHOST_HPP
#define EXACA_GHOST_HPP

#include "CAcelldata.hpp"
#include "CAfunctions.hpp"
#include "CAtypes.hpp"

#include <Kokkos_Core.hpp>

// Load data (GrainID, DOCenter, DiagonalLength) into ghost nodes if the given RankY is associated with a 1D halo region
KOKKOS_INLINE_FUNCTION void load_cell_into_halo(const int GhostGID, const float GhostDOCX, const float GhostDOCY,
                                                const float GhostDOCZ, const float GhostDL, const int RankX,
                                                const int RankZ, const bool AtBoundary, Buffer2D BufferSend,
                                                int NGrainOrientations, int index) {
    if (!(AtBoundary)) {
        BufferSend(index, 0) = static_cast<float>(RankX);
        BufferSend(index, 1) = static_cast<float>(RankZ);
        BufferSend(index, 2) = static_cast<float>(getGrainOrientation(GhostGID, NGrainOrientations, false));
        BufferSend(index, 3) = static_cast<float>(getGrainNumber(GhostGID, NGrainOrientations));
        BufferSend(index, 4) = GhostDOCX;
        BufferSend(index, 5) = GhostDOCY;
        BufferSend(index, 6) = GhostDOCZ;
        BufferSend(index, 7) = GhostDL;
    }
}
void ResetSendBuffers(int BufSize, Buffer2D BufferNorthSend, Buffer2D BufferSouthSend);
int ResizeBuffers(Buffer2D &BufferNorthSend, Buffer2D &BufferSouthSend, Buffer2D &BufferNorthRecv,
                  Buffer2D &BufferSouthRecv, int required_buffer_size_global, int NumCellsBufferPadding = 25);
void ResetBufferCapacity(Buffer2D &BufferNorthSend, Buffer2D &BufferSouthSend, Buffer2D &BufferNorthRecv,
                         Buffer2D &BufferSouthRecv, int NewBufSize);
void FillBuffers(ViewI numSteerCommNorth, ViewI SteeringVectorCommNorth, ViewI numSteerCommSouth,
                 ViewI SteeringVectorCommSouth, int numSteerCommMax, int nx, int MyYSlices, int,
                 CellData<device_memory_space> &cellData, Buffer2D BufferNorthSend, Buffer2D BufferSouthSend,
                 bool AtNorthBoundary, bool AtSouthBoundary, ViewF DOCenter, ViewF DiagonalLength,
                 int NGrainOrientations);
void LoadGhostNodes(ViewI_H numSteerCommNorth_Host, ViewI numSteerCommNorth, ViewI SteeringVectorCommNorth,
                    ViewI_H numSteerCommSouth_Host, ViewI numSteerCommSouth, ViewI SteeringVectorCommSouth, int nx,
                    int MyYSlices, int id, CellData<device_memory_space> &cellData, Buffer2D BufferNorthSend,
                    Buffer2D BufferSouthSend, Buffer2D BufferNorthRecv, Buffer2D BufferSouthRecv, bool AtNorthBoundary,
                    bool AtSouthBoundary, ViewF DOCenter, ViewF DiagonalLength, int NGrainOrientations, int BufSize);
void GhostNodes1D(int, int, int NeighborRank_North, int NeighborRank_South, int nx, int MyYSlices, int MyYOffset,
                  NList NeighborX, NList NeighborY, NList NeighborZ, CellData<device_memory_space> &cellData,
                  ViewF DOCenter, ViewF GrainUnitVector, ViewF DiagonalLength, ViewF CritDiagonalLength,
                  int NGrainOrientations, Buffer2D BufferNorthSend, Buffer2D BufferSouthSend, Buffer2D BufferNorthRecv,
                  Buffer2D BufferSouthRecv, int BufSize, int ZBound_Low);

#endif
