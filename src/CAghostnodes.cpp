// Copyright 2021-2023 Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "CAghostnodes.hpp"
#include "CAfunctions.hpp"
#include "CAupdate.hpp"
#include "mpi.h"

#include <algorithm>
#include <cmath>
#include <vector>

// Set first index in send buffers to -1 (placeholder) for all cells in the buffer, and reset the counts of number of
// cells contained in buffers to 0s
void ResetSendBuffers(int BufSize, Buffer2D BufferNorthSend, Buffer2D BufferSouthSend) {

    Kokkos::parallel_for(
        "BufferReset", BufSize, KOKKOS_LAMBDA(const int &i) {
            BufferNorthSend(i, 0) = -1.0;
            BufferSouthSend(i, 0) = -1.0;
        });
}

// Resize the send/recv buffers resize the buffers
// if necessary, returning the new buffer size
int ResizeBuffers(Buffer2D &BufferNorthSend, Buffer2D &BufferSouthSend, Buffer2D &BufferNorthRecv,
                  Buffer2D &BufferSouthRecv, int required_buffer_size_global, int NumCellsBufferPadding) {

    // Increase buffer size to fit all data
    // Add numcells_buffer_padding (defaults to 25) cells as additional padding
    int NewBufSize = required_buffer_size_global + NumCellsBufferPadding;
    Kokkos::realloc(BufferNorthSend, NewBufSize, 8);
    Kokkos::realloc(BufferSouthSend, NewBufSize, 8);
    Kokkos::realloc(BufferNorthRecv, NewBufSize, 8);
    Kokkos::realloc(BufferSouthRecv, NewBufSize, 8);
    return NewBufSize;
}

// Reset the buffer sizes to a set value (defaulting to 25, which was the initial size) preserving the existing values
void ResetBufferCapacity(Buffer2D &BufferNorthSend, Buffer2D &BufferSouthSend, Buffer2D &BufferNorthRecv,
                         Buffer2D &BufferSouthRecv, int NewBufSize) {
    Kokkos::resize(BufferNorthSend, NewBufSize, 8);
    Kokkos::resize(BufferSouthSend, NewBufSize, 8);
    Kokkos::resize(BufferNorthRecv, NewBufSize, 8);
    Kokkos::resize(BufferSouthRecv, NewBufSize, 8);
}

// Fill the buffers as necessary
// GhostLiquid cells - loaded with "empty"/zeros for values to indicate that the cell has no grain data
// GhostActive cells - loaded with grain ID, octahedron center, diagonal length

void FillBuffers(ViewI numSteerCommNorth, ViewI SteeringVectorCommNorth, ViewI numSteerCommSouth,
                 ViewI SteeringVectorCommSouth, int numSteerCommMax, int nx, int MyYSlices, int,
                 CellData<device_memory_space> &cellData, Buffer2D BufferNorthSend, Buffer2D BufferSouthSend,
                 bool AtNorthBoundary, bool AtSouthBoundary, ViewF DOCenter, ViewF DiagonalLength,
                 int NGrainOrientations) {

    auto CellType = cellData.getCellTypeSubview();
    auto GrainID = cellData.getGrainIDSubview();
    Kokkos::parallel_for(
        "FillBuffers", numSteerCommMax, KOKKOS_LAMBDA(const int &num) {
            if (num < numSteerCommNorth(0)) {
                int CellCoordinate1D = SteeringVectorCommNorth(num);
                // Cells of interest for the CA - marked cells at edge of halos in Y
                int RankZ = CellCoordinate1D / (nx * MyYSlices);
                int Rem = CellCoordinate1D % (nx * MyYSlices);
                int RankX = Rem / MyYSlices;
                if (CellType(CellCoordinate1D) == GhostActive) {
                    int GhostGID = GrainID(CellCoordinate1D);
                    float GhostDOCX = DOCenter(3 * CellCoordinate1D);
                    float GhostDOCY = DOCenter(3 * CellCoordinate1D + 1);
                    float GhostDOCZ = DOCenter(3 * CellCoordinate1D + 2);
                    float GhostDL = DiagonalLength(CellCoordinate1D);
                    // Data loaded into the ghost nodes is for the cell that was just captured
                    load_cell_into_halo(GhostGID, GhostDOCX, GhostDOCY, GhostDOCZ, GhostDL, RankX, RankZ,
                                        AtNorthBoundary, BufferNorthSend, NGrainOrientations, num);
                    // Cell activation is now finished - cell type can be changed to Active
                    CellType(CellCoordinate1D) = Active;
                }
                else {
                    // Dummy values for first 4 arguments (Grain ID and octahedron center coordinates), 0 for
                    // diagonal length
                    load_cell_into_halo(-1, -1.0, -1.0, -1.0, 0.0, RankX, RankZ, AtNorthBoundary, BufferNorthSend,
                                        NGrainOrientations, num);
                    // Cell melting is now finished - cell type can be changed to Liquid
                    CellType(CellCoordinate1D) = Liquid;
                }
            }
            if (num < numSteerCommSouth(0)) {
                int CellCoordinate1D = SteeringVectorCommSouth(num);
                // Cells of interest for the CA - marked cells at edge of halos in Y
                int RankZ = CellCoordinate1D / (nx * MyYSlices);
                int Rem = CellCoordinate1D % (nx * MyYSlices);
                int RankX = Rem / MyYSlices;
                if (CellType(CellCoordinate1D) == GhostActive) {
                    int GhostGID = GrainID(CellCoordinate1D);
                    float GhostDOCX = DOCenter(3 * CellCoordinate1D);
                    float GhostDOCY = DOCenter(3 * CellCoordinate1D + 1);
                    float GhostDOCZ = DOCenter(3 * CellCoordinate1D + 2);
                    float GhostDL = DiagonalLength(CellCoordinate1D);
                    // Data loaded into the ghost nodes is for the cell that was just captured
                    load_cell_into_halo(GhostGID, GhostDOCX, GhostDOCY, GhostDOCZ, GhostDL, RankX, RankZ,
                                        AtSouthBoundary, BufferSouthSend, NGrainOrientations, num);
                    // Cell activation is now finished - cell type can be changed to Active
                    CellType(CellCoordinate1D) = Active;
                }
                else {
                    // Dummy values for first 4 arguments (Grain ID and octahedron center coordinates), 0 for
                    // diagonal length
                    load_cell_into_halo(-1, -1.0, -1.0, -1.0, 0.0, RankX, RankZ, AtSouthBoundary, BufferSouthSend,
                                        NGrainOrientations, num);
                    // Cell melting is now finished - cell type can be changed to Liquid
                    CellType(CellCoordinate1D) = Liquid;
                }
            }
        });
    Kokkos::fence();
}

void LoadGhostNodes(ViewI_H numSteerCommNorth_Host, ViewI numSteerCommNorth, ViewI SteeringVectorCommNorth,
                    ViewI_H numSteerCommSouth_Host, ViewI numSteerCommSouth, ViewI SteeringVectorCommSouth, int nx,
                    int MyYSlices, int id, CellData<device_memory_space> &cellData, Buffer2D BufferNorthSend,
                    Buffer2D BufferSouthSend, Buffer2D BufferNorthRecv, Buffer2D BufferSouthRecv, bool AtNorthBoundary,
                    bool AtSouthBoundary, ViewF DOCenter, ViewF DiagonalLength, int NGrainOrientations, int BufSize) {

    // Check if buffers are large enough to store the comm steering vector data, resizing them if needed
    int required_buffer_size_local = max(numSteerCommNorth_Host(0), numSteerCommSouth_Host(0));
    int required_buffer_size_global;
    MPI_Allreduce(&required_buffer_size_local, &required_buffer_size_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (required_buffer_size_global > BufSize) {
        int OldBufSize = BufSize;
        BufSize = ResizeBuffers(BufferNorthSend, BufferSouthSend, BufferNorthRecv, BufferSouthRecv,
                                required_buffer_size_global);
        if (id == 0)
            std::cout << "Resized number of cells stored in send/recv buffers from " << OldBufSize << " to " << BufSize
                      << std::endl;
    }

    // Fill the send buffers using cells marked with ghost types in CellCapture
    FillBuffers(numSteerCommNorth, SteeringVectorCommNorth, numSteerCommSouth, SteeringVectorCommSouth,
                required_buffer_size_local, nx, MyYSlices, id, cellData, BufferNorthSend, BufferSouthSend,
                AtNorthBoundary, AtSouthBoundary, DOCenter, DiagonalLength, NGrainOrientations);

    numSteerCommNorth_Host(0) = 0;
    numSteerCommSouth_Host(0) = 0;
    Kokkos::deep_copy(numSteerCommNorth, numSteerCommNorth_Host);
    Kokkos::deep_copy(numSteerCommSouth, numSteerCommSouth_Host);
}
//*****************************************************************************/
// 1D domain decomposition: update ghost nodes with new cell data from Nucleation and CellCapture routines
void GhostNodes1D(int, int, int NeighborRank_North, int NeighborRank_South, int nx, int MyYSlices, int MyYOffset,
                  NList NeighborX, NList NeighborY, NList NeighborZ, CellData<device_memory_space> &cellData,
                  ViewF DOCenter, ViewF GrainUnitVector, ViewF DiagonalLength, ViewF CritDiagonalLength,
                  int NGrainOrientations, Buffer2D BufferNorthSend, Buffer2D BufferSouthSend, Buffer2D BufferNorthRecv,
                  Buffer2D BufferSouthRecv, int BufSize, int ZBound_Low) {

    std::vector<MPI_Request> SendRequests(2, MPI_REQUEST_NULL);
    std::vector<MPI_Request> RecvRequests(2, MPI_REQUEST_NULL);

    // Send data to each other rank (MPI_Isend)
    MPI_Isend(BufferSouthSend.data(), 8 * BufSize, MPI_FLOAT, NeighborRank_South, 0, MPI_COMM_WORLD, &SendRequests[0]);
    MPI_Isend(BufferNorthSend.data(), 8 * BufSize, MPI_FLOAT, NeighborRank_North, 0, MPI_COMM_WORLD, &SendRequests[1]);

    // Receive buffers for all neighbors (MPI_Irecv)
    MPI_Irecv(BufferSouthRecv.data(), 8 * BufSize, MPI_FLOAT, NeighborRank_South, 0, MPI_COMM_WORLD, &RecvRequests[0]);
    MPI_Irecv(BufferNorthRecv.data(), 8 * BufSize, MPI_FLOAT, NeighborRank_North, 0, MPI_COMM_WORLD, &RecvRequests[1]);

    // unpack in any order
    bool unpack_complete = false;
    auto CellType = cellData.getCellTypeSubview();
    auto GrainID = cellData.getGrainIDSubview();
    while (!unpack_complete) {
        // Get the next buffer to unpack from rank "unpack_index"
        int unpack_index = MPI_UNDEFINED;
        MPI_Waitany(2, RecvRequests.data(), &unpack_index, MPI_STATUS_IGNORE);
        // If there are no more buffers to unpack, leave the while loop
        if (MPI_UNDEFINED == unpack_index) {
            unpack_complete = true;
        }
        // Otherwise unpack the next buffer.
        else {
            Kokkos::parallel_for(
                "BufferUnpack", BufSize, KOKKOS_LAMBDA(const int &BufPosition) {
                    int RankX, RankY, RankZ, NewGrainID;
                    long int CellLocation;
                    float DOCenterX, DOCenterY, DOCenterZ, NewDiagonalLength;
                    bool Place = false;
                    // Which rank was the data received from? Is there valid data at this position in the buffer (i.e.,
                    // not set to -1.0)?
                    if ((unpack_index == 0) && (BufferSouthRecv(BufPosition, 0) != -1.0) &&
                        (NeighborRank_South != MPI_PROC_NULL)) {
                        // Data receieved from South
                        RankX = static_cast<int>(BufferSouthRecv(BufPosition, 0));
                        RankY = 0;
                        RankZ = static_cast<int>(BufferSouthRecv(BufPosition, 1));
                        CellLocation = RankZ * nx * MyYSlices + MyYSlices * RankX + RankY;
                        // Two possibilities: buffer data with non-zero diagonal length was loaded, and a liquid cell
                        // may have to be updated to active - or zero diagonal length data was loaded, and an active
                        // cell may have to be updated to liquid
                        if (CellType(CellLocation) == Liquid) {
                            Place = true;
                            int MyGrainOrientation = static_cast<int>(BufferSouthRecv(BufPosition, 2));
                            int MyGrainNumber = static_cast<int>(BufferSouthRecv(BufPosition, 3));
                            NewGrainID = getGrainID(NGrainOrientations, MyGrainOrientation, MyGrainNumber);
                            DOCenterX = BufferSouthRecv(BufPosition, 4);
                            DOCenterY = BufferSouthRecv(BufPosition, 5);
                            DOCenterZ = BufferSouthRecv(BufPosition, 6);
                            NewDiagonalLength = BufferSouthRecv(BufPosition, 7);
                        }
                        else if ((CellType(CellLocation) == Active) && (BufferSouthRecv(BufPosition, 7) == 0.0)) {
                            CellType(CellLocation) = Liquid;
                        }
                    }
                    else if ((unpack_index == 1) && (BufferNorthRecv(BufPosition, 0) != -1.0) &&
                             (NeighborRank_North != MPI_PROC_NULL)) {
                        // Data received from North
                        RankX = static_cast<int>(BufferNorthRecv(BufPosition, 0));
                        RankY = MyYSlices - 1;
                        RankZ = static_cast<int>(BufferNorthRecv(BufPosition, 1));
                        CellLocation = RankZ * nx * MyYSlices + MyYSlices * RankX + RankY;
                        // Two possibilities: buffer data with non-zero diagonal length was loaded, and a liquid cell
                        // may have to be updated to active - or zero diagonal length data was loaded, and an active
                        // cell may have to be updated to liquid
                        if (CellType(CellLocation) == Liquid) {
                            Place = true;
                            int MyGrainOrientation = static_cast<int>(BufferNorthRecv(BufPosition, 2));
                            int MyGrainNumber = static_cast<int>(BufferNorthRecv(BufPosition, 3));
                            NewGrainID = getGrainID(NGrainOrientations, MyGrainOrientation, MyGrainNumber);
                            DOCenterX = BufferNorthRecv(BufPosition, 4);
                            DOCenterY = BufferNorthRecv(BufPosition, 5);
                            DOCenterZ = BufferNorthRecv(BufPosition, 6);
                            NewDiagonalLength = BufferNorthRecv(BufPosition, 7);
                        }
                        else if ((CellType(CellLocation) == Active) && (BufferNorthRecv(BufPosition, 7) == 0.0)) {
                            CellType(CellLocation) = Liquid;
                        }
                    }
                    if (Place) {
                        int GlobalZ = RankZ + ZBound_Low;
                        // Update this ghost node cell's information with data from other rank
                        GrainID(CellLocation) = NewGrainID;
                        DOCenter((long int)(3) * CellLocation) = DOCenterX;
                        DOCenter((long int)(3) * CellLocation + (long int)(1)) = DOCenterY;
                        DOCenter((long int)(3) * CellLocation + (long int)(2)) = DOCenterZ;
                        int MyOrientation = getGrainOrientation(GrainID(CellLocation), NGrainOrientations);
                        DiagonalLength(CellLocation) = static_cast<float>(NewDiagonalLength);
                        // Global coordinates of cell center
                        double xp = RankX + 0.5;
                        double yp = RankY + MyYOffset + 0.5;
                        double zp = GlobalZ + 0.5;
                        // Calculate critical values at which this active cell leads to the activation of a neighboring
                        // liquid cell
                        calcCritDiagonalLength(CellLocation, xp, yp, zp, DOCenterX, DOCenterY, DOCenterZ, NeighborX,
                                               NeighborY, NeighborZ, MyOrientation, GrainUnitVector,
                                               CritDiagonalLength);
                        CellType(CellLocation) = Active;
                    }
                });
        }
    }

    // Wait on send requests
    MPI_Waitall(2, SendRequests.data(), MPI_STATUSES_IGNORE);
    Kokkos::fence();

    // Reset send buffer data to -1 (used as placeholder)
    // Reset comm steering vector sizes to 0
    ResetSendBuffers(BufSize, BufferNorthSend, BufferSouthSend);
}
