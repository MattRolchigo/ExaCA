// Copyright 2021-2022 Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "CAghostnodes.hpp"
#include "CAfunctions.hpp"
#include "CAupdate.hpp"
#include "mpi.h"

#include <cmath>
#include <vector>

//*****************************************************************************/
// 1D domain decomposition: update ghost nodes with new cell data from Nucleation and CellCapture routines
void GhostNodes1D(int, int, int NeighborRank_North, int NeighborRank_South, int nx, int MyYSlices, int MyYOffset,
                  NList NeighborX, NList NeighborY, NList NeighborZ, ViewI CellType, ViewF DOCenter, ViewI GrainID,
                  ViewF GrainUnitVector, ViewF DiagonalLength, ViewF CritDiagonalLength, int NGrainOrientations, ViewF2D BufferNorthSend_Octahedron,
                  ViewF2D BufferSouthSend_Octahedron, ViewI BufferNorthSend_Grain, ViewI BufferSouthSend_Grain, ViewF2D BufferNorthRecv_Octahedron,
                  ViewF2D BufferSouthRecv_Octahedron, ViewI BufferNorthRecv_Grain, ViewI BufferSouthRecv_Grain, int BufSizeX, int BufSizeZ, int ZBound_Low) {

    std::vector<MPI_Request> SendRequests(4, MPI_REQUEST_NULL);
    std::vector<MPI_Request> RecvRequests(4, MPI_REQUEST_NULL);

    // Send data to each other rank (MPI_Isend)
    MPI_Isend(BufferSouthSend_Octahedron.data(), 4 * BufSizeX * BufSizeZ, MPI_FLOAT, NeighborRank_South, 0, MPI_COMM_WORLD,
              &SendRequests[0]);
    MPI_Isend(BufferNorthSend_Octahedron.data(), 4 * BufSizeX * BufSizeZ, MPI_FLOAT, NeighborRank_North, 0, MPI_COMM_WORLD,
              &SendRequests[1]);
    MPI_Isend(BufferSouthSend_Grain.data(), BufSizeX * BufSizeZ, MPI_INT, NeighborRank_South, 1, MPI_COMM_WORLD,
              &SendRequests[2]);
    MPI_Isend(BufferNorthSend_Grain.data(), BufSizeX * BufSizeZ, MPI_INT, NeighborRank_North, 1, MPI_COMM_WORLD,
              &SendRequests[3]);

    // Receive buffers for all neighbors (MPI_Irecv)
    MPI_Irecv(BufferSouthRecv_Octahedron.data(), 4 * BufSizeX * BufSizeZ, MPI_FLOAT, NeighborRank_South, 0, MPI_COMM_WORLD,
              &RecvRequests[0]);
    MPI_Irecv(BufferNorthRecv_Octahedron.data(), 4 * BufSizeX * BufSizeZ, MPI_FLOAT, NeighborRank_North, 0, MPI_COMM_WORLD,
              &RecvRequests[1]);
    MPI_Irecv(BufferSouthRecv_Grain.data(), BufSizeX * BufSizeZ, MPI_INT, NeighborRank_South, 1, MPI_COMM_WORLD,
              &RecvRequests[0]);
    MPI_Irecv(BufferNorthRecv_Grain.data(), BufSizeX * BufSizeZ, MPI_INT, NeighborRank_North, 1, MPI_COMM_WORLD,
              &RecvRequests[1]);

    // Wait on sends and receives
    MPI_Waitall(4, RecvRequests.data(), MPI_STATUS_IGNORE);

    // Unpack
    int RecvBufSize = BufSizeX * BufSizeZ;
    Kokkos::parallel_for(
        "BufferUnpack", RecvBufSize, KOKKOS_LAMBDA(const int &BufPosition) {
            for (int unpack_index=0; unpack_index<2; unpack_index++) {
                int RankX, RankY, RankZ, NewGrainID;
                long int CellLocation;
                float DOCenterX, DOCenterY, DOCenterZ, NewDiagonalLength;
                bool Place = false;
                RankZ = BufPosition / BufSizeX;
                RankX = BufPosition % BufSizeX;
                // Which rank was the data received from?
                if ((unpack_index == 0) && (NeighborRank_South != MPI_PROC_NULL)) {
                    // Data receieved from South
                    RankY = 0;
                    CellLocation = RankZ * nx * MyYSlices + MyYSlices * RankX + RankY;
                    int GlobalCellLocation = CellLocation + ZBound_Low * nx * MyYSlices;
                    if ((BufferSouthRecv_Octahedron(BufPosition, 3) > 0) && (CellType(GlobalCellLocation) == Liquid)) {
                        Place = true;
                        NewGrainID = BufferSouthRecv_Grain(BufPosition);
                        DOCenterX = BufferSouthRecv_Octahedron(BufPosition, 0);
                        DOCenterY = BufferSouthRecv_Octahedron(BufPosition, 1);
                        DOCenterZ = BufferSouthRecv_Octahedron(BufPosition, 2);
                        NewDiagonalLength = BufferSouthRecv_Octahedron(BufPosition, 3);
                    }
                }
                else if ((unpack_index == 1) && (NeighborRank_North != MPI_PROC_NULL)) {
                    // Data received from North
                    RankY = MyYSlices - 1;
                    CellLocation = RankZ * nx * MyYSlices + MyYSlices * RankX + RankY;
                    int GlobalCellLocation = CellLocation + ZBound_Low * nx * MyYSlices;
                    if ((BufferNorthRecv_Octahedron(BufPosition, 3) > 0) && (CellType(GlobalCellLocation) == Liquid)) {
                        Place = true;
                        NewGrainID = BufferNorthRecv_Grain(BufPosition);
                        DOCenterX = BufferNorthRecv_Octahedron(BufPosition, 0);
                        DOCenterY = BufferNorthRecv_Octahedron(BufPosition, 1);
                        DOCenterZ = BufferNorthRecv_Octahedron(BufPosition, 2);
                        NewDiagonalLength = BufferNorthRecv_Octahedron(BufPosition, 3);
                    }
                }
                if (Place) {
                    int GlobalZ = RankZ + ZBound_Low;
                    int GlobalCellLocation = GlobalZ * nx * MyYSlices + RankX * MyYSlices + RankY;
                    
                    // Update this ghost node cell's information with data from other rank
                    GrainID(GlobalCellLocation) = NewGrainID;
                    DOCenter((long int)(3) * CellLocation) = DOCenterX;
                    DOCenter((long int)(3) * CellLocation + (long int)(1)) = DOCenterY;
                    DOCenter((long int)(3) * CellLocation + (long int)(2)) = DOCenterZ;
                    int MyOrientation = getGrainOrientation(GrainID(GlobalCellLocation), NGrainOrientations);
                    DiagonalLength(CellLocation) = NewDiagonalLength;
                    // Global coordinates of cell center
                    double xp = RankX + 0.5;
                    double yp = RankY + MyYOffset + 0.5;
                    double zp = GlobalZ + 0.5;
                    // Calculate critical values at which this active cell leads to the activation of a neighboring
                    // liquid cell
                    calcCritDiagonalLength(CellLocation, xp, yp, zp, DOCenterX, DOCenterY, DOCenterZ, NeighborX,
                                           NeighborY, NeighborZ, MyOrientation, GrainUnitVector,
                                           CritDiagonalLength);
                    CellType(GlobalCellLocation) = Active;
                }
            }
        });
    Kokkos::fence();
}
