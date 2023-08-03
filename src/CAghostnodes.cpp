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
void ResetSendBuffers(int BufSize, Buffer2D BufferNorthSend, Buffer2D BufferSouthSend, ViewI SendSizeNorth,
                      ViewI SendSizeSouth) {

    Kokkos::parallel_for(
        "BufferReset", BufSize, KOKKOS_LAMBDA(const int &i) {
            BufferNorthSend(i, 0) = -1.0;
            BufferSouthSend(i, 0) = -1.0;
        });
    Kokkos::parallel_for(
        "HaloCountReset", 1, KOKKOS_LAMBDA(const int) {
            SendSizeNorth(0) = 0;
            SendSizeSouth(0) = 0;
        });
}

// Count the number of cells' in halo regions where the data did not fit into the send buffers, and resize the buffers
// if necessary, returning the new buffer size
int ResizeBuffers(Buffer2D &BufferNorthSend, Buffer2D &BufferSouthSend, Buffer2D &BufferNorthRecv,
                  Buffer2D &BufferSouthRecv, ViewI SendSizeNorth, ViewI SendSizeSouth, ViewI_H SendSizeNorth_Host,
                  ViewI_H SendSizeSouth_Host, int OldBufSize, int NumCellsBufferPadding) {

    int NewBufSize;
    SendSizeNorth_Host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), SendSizeNorth);
    SendSizeSouth_Host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), SendSizeSouth);
    int max_count_local = max(SendSizeNorth_Host(0), SendSizeSouth_Host(0));
    int max_count_global;
    MPI_Allreduce(&max_count_local, &max_count_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (max_count_global > OldBufSize) {
        // Increase buffer size to fit all data
        // Add numcells_buffer_padding (defaults to 25) cells as additional padding
        NewBufSize = max_count_global + NumCellsBufferPadding;
        Kokkos::resize(BufferNorthSend, NewBufSize, 8);
        Kokkos::resize(BufferSouthSend, NewBufSize, 8);
        Kokkos::resize(BufferNorthRecv, NewBufSize, 8);
        Kokkos::resize(BufferSouthRecv, NewBufSize, 8);
        // Reset count variables on device to the old buffer size
        Kokkos::parallel_for(
            "ResetCounts", 1, KOKKOS_LAMBDA(const int &) {
                SendSizeNorth(0) = OldBufSize;
                SendSizeSouth(0) = OldBufSize;
            });
    }
    else
        NewBufSize = OldBufSize;
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

// Fill the buffers as necessary starting from the old count size
// This is called either for refilling the buffers starting after CellCapture, or starting from a failed loading state
// due to buffer capacity

// If NOT being called from the failed loading state:
// GhostLiquid cells - loaded with "empty"/zeros for values to indicate that the cell has no grain data
// GhostActive cells - loaded with grain ID, octahedron center, diagonal length
// If the buffer load event was successful, change cell type to Liquid or Active
// If the buffer load event was unsuccessful, change cell type to LiquidFailedBufferLoad or ActiveFailedBufferLoad

// If being called from the failed loading state (the buffer capacity now being larger):
// LiquidFailedBufferLoad cells - attempt again to load with "empty"/zeros for values to indicate that the cell has no
// grain data ActiveFailedBufferLoad cells - attempt again to load with grain ID, octahedron center, diagonal length If
// the buffer load event was successful, change cell type to Liquid or Active If the buffer load event was unsuccessful,
// throw an error: the buffer should've been large enough to hold this event
void FillBuffers(int nx, int nzActive, int MyYSlices, int, CellData<device_memory_space> &cellData,
                 Buffer2D BufferNorthSend, Buffer2D BufferSouthSend, ViewI SendSizeNorth, ViewI SendSizeSouth,
                 bool AtNorthBoundary, bool AtSouthBoundary, ViewF DOCenter, ViewF DiagonalLength,
                 int NGrainOrientations, int BufSize, bool loading_from_failed) {

    auto CellType = cellData.getCellTypeSubview();
    auto GrainID = cellData.getGrainIDSubview();
    int LoadActiveType, LoadLiquidType;
    if (loading_from_failed) {
        LoadActiveType = ActiveFailedBufferLoad;
        LoadLiquidType = LiquidFailedBufferLoad;
    }
    else {
        LoadActiveType = GhostActive;
        LoadLiquidType = GhostLiquid;
    }
    Kokkos::parallel_for(
        "FillBuffers", nx, KOKKOS_LAMBDA(const int &i) {
            for (int k = 0; k < nzActive; k++) {
                int j_loc[2] = {1, MyYSlices - 2};
                for (int idx = 0; idx < 2; idx++) {
                    int j = j_loc[idx];
                    int CellCoordinate1D = k * nx * MyYSlices + i * MyYSlices + j;
                    if (CellType(CellCoordinate1D) == LoadActiveType) {
                        int GhostGID = GrainID(CellCoordinate1D);
                        float GhostDOCX = DOCenter(3 * CellCoordinate1D);
                        float GhostDOCY = DOCenter(3 * CellCoordinate1D + 1);
                        float GhostDOCZ = DOCenter(3 * CellCoordinate1D + 2);
                        float GhostDL = DiagonalLength(CellCoordinate1D);
                        // Data loaded into the ghost nodes is for the cell that was just captured
                        bool DataFitsInBuffer =
                            load_cell_into_halo(GhostGID, GhostDOCX, GhostDOCY, GhostDOCZ, GhostDL, SendSizeNorth,
                                                SendSizeSouth, MyYSlices, i, j, k, AtNorthBoundary, AtSouthBoundary,
                                                BufferSouthSend, BufferNorthSend, NGrainOrientations, BufSize);
                        if (!(DataFitsInBuffer)) {
                            // This cell's data did not fit in the buffer with current size BufSize - mark with
                            // temporary type for future reloading or warn of potential data loss at MPI processor
                            // boundaries
                            if (loading_from_failed)
                                printf("Error: Send/recv buffer resize failed to include all necessary data, predicted "
                                       "results at MPI processor boundaries may be inaccurate\n");
                            else
                                CellType(CellCoordinate1D) = ActiveFailedBufferLoad;
                        }
                        else {
                            // Cell activation is now finished - cell type can be changed to Active
                            CellType(CellCoordinate1D) = Active;
                        }
                    }
                    else if (CellType(CellCoordinate1D) == LoadLiquidType) {
                        // Dummy values for first 4 arguments (Grain ID and octahedron center coordinates), 0 for
                        // diagonal length
                        bool DataFitsInBuffer =
                            load_cell_into_halo(-1, -1.0, -1.0, -1.0, 0.0, SendSizeNorth, SendSizeSouth, MyYSlices, i,
                                                j, k, AtNorthBoundary, AtSouthBoundary, BufferSouthSend,
                                                BufferNorthSend, NGrainOrientations, BufSize);
                        if (!(DataFitsInBuffer)) {
                            // This cell's data did not fit in the buffer with current size BufSize - mark with
                            // temporary type for future reloading or warn of potential data loss at MPI processor
                            // boundaries
                            if (loading_from_failed)
                                printf("Error: Send/recv buffer resize failed to include all necessary data, predicted "
                                       "results at MPI processor boundaries may be inaccurate\n");
                            else
                                CellType(CellCoordinate1D) = LiquidFailedBufferLoad;
                        }
                        else {
                            // Cell melting is now finished - cell type can be changed to Liquid
                            CellType(CellCoordinate1D) = Liquid;
                        }
                    }
                }
            }
        });
    Kokkos::fence();
}

void LoadGhostNodes(int nx, int nzActive, int MyYSlices, int id, CellData<device_memory_space> &cellData,
                    Buffer2D BufferNorthSend, Buffer2D BufferSouthSend, ViewI SendSizeNorth, ViewI_H SendSizeNorth_Host,
                    ViewI SendSizeSouth, ViewI_H SendSizeSouth_Host, Buffer2D BufferNorthRecv, Buffer2D BufferSouthRecv,
                    bool AtNorthBoundary, bool AtSouthBoundary, ViewF DOCenter, ViewF DiagonalLength,
                    int NGrainOrientations, int BufSize) {

    // First, attempt to fill the send buffers using cells marked with ghost types in CellCapture
    FillBuffers(nx, nzActive, MyYSlices, id, cellData, BufferNorthSend, BufferSouthSend, SendSizeNorth, SendSizeSouth,
                AtNorthBoundary, AtSouthBoundary, DOCenter, DiagonalLength, NGrainOrientations, BufSize, false);

    // Count the number of cells' in halo regions where the data did not fit into the send buffers
    // Reduce across all ranks, as the same BufSize should be maintained across all ranks
    // If any rank overflowed its buffer size, resize all buffers to the new size plus 10% padding
    int OldBufSize = BufSize;
    BufSize = ResizeBuffers(BufferNorthSend, BufferSouthSend, BufferNorthRecv, BufferSouthRecv, SendSizeNorth,
                            SendSizeSouth, SendSizeNorth_Host, SendSizeSouth_Host, OldBufSize);
    if (OldBufSize != BufSize) {
        if (id == 0)
            std::cout << "Resized number of cells stored in send/recv buffers from " << OldBufSize << " to " << BufSize
                      << std::endl;
        // Attempt to fill buffers again, starting from the failed state
        FillBuffers(nx, nzActive, MyYSlices, id, cellData, BufferNorthSend, BufferSouthSend, SendSizeNorth,
                    SendSizeSouth, AtNorthBoundary, AtSouthBoundary, DOCenter, DiagonalLength, NGrainOrientations,
                    BufSize, true);
    }
}
//*****************************************************************************/
// 1D domain decomposition: update ghost nodes with new cell data from Nucleation and CellCapture routines
void GhostNodes1D(int, int, int NeighborRank_North, int NeighborRank_South, int nx, int MyYSlices, int MyYOffset,
                  NList NeighborX, NList NeighborY, NList NeighborZ, CellData<device_memory_space> &cellData,
                  ViewF DOCenter, ViewF GrainUnitVector, ViewF DiagonalLength, ViewF CritDiagonalLength,
                  int NGrainOrientations, Buffer2D BufferNorthSend, Buffer2D BufferSouthSend, Buffer2D BufferNorthRecv,
                  Buffer2D BufferSouthRecv, int BufSize, int ZBound_Low, ViewI SendSizeNorth, ViewI SendSizeSouth) {

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

    // Reset send buffer data to -1 (used as placeholder) and reset the number of cells stored in the buffers to 0
    ResetSendBuffers(BufSize, BufferNorthSend, BufferSouthSend, SendSizeNorth, SendSizeSouth);
    // Wait on send requests
    MPI_Waitall(2, SendRequests.data(), MPI_STATUSES_IGNORE);
    Kokkos::fence();
}
