// Copyright 2021-2022 Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include <Kokkos_Core.hpp>

#include "CAfunctions.hpp"
#include "CAinitialize.hpp"
#include "CAparsefiles.hpp"
#include "CAtypes.hpp"

#include <gtest/gtest.h>

#include "mpi.h"

#include <fstream>
#include <string>
#include <vector>

namespace Test {
//---------------------------------------------------------------------------//
void testOrientationInit_Vectors() {

    int id;
    // Get individual process ID
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    int ValsPerLine = 9;
    int NGrainOrientations = 0;
    std::string GrainOrientationFile = checkFileInstalled("GrainOrientationVectors.csv", id);

    // View for storing orientation data
    ViewF GrainOrientationData(Kokkos::ViewAllocateWithoutInitializing("GrainOrientationData"), 0);

    // Call OrientationInit - without optional final argument
    OrientationInit(id, NGrainOrientations, GrainOrientationData, GrainOrientationFile);

    // Copy orientation data back to the host
    ViewF_H GrainOrientationData_Host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), GrainOrientationData);

    // Check results
    EXPECT_EQ(NGrainOrientations, 10000);

    std::vector<float> ExpectedGrainOrientations = {0.848294,  0.493303,  0.19248,  -0.522525, 0.720911,  0.455253,
                                                    0.0858167, -0.486765, 0.869308, 0.685431,  0.188182,  0.7034,
                                                    -0.468504, 0.85348,   0.228203, -0.557394, -0.485963, 0.673166};
    for (int n = 0; n < 2 * ValsPerLine; n++) {
        EXPECT_FLOAT_EQ(GrainOrientationData_Host(n), ExpectedGrainOrientations[n]);
    }
}

void testOrientationInit_Angles() {

    int id;
    // Get individual process ID
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    int ValsPerLine = 3;
    int NGrainOrientations = 0;
    std::string GrainOrientationFile = checkFileInstalled("GrainOrientationEulerAnglesBungeZXZ.csv", id);

    // View for storing orientation data
    ViewF GrainOrientationData(Kokkos::ViewAllocateWithoutInitializing("GrainOrientationData"), 0);

    // Call OrientationInit - with optional final argument
    OrientationInit(id, NGrainOrientations, GrainOrientationData, GrainOrientationFile, ValsPerLine);

    // Copy orientation data back to the host
    ViewF_H GrainOrientationData_Host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), GrainOrientationData);

    // Check results
    EXPECT_EQ(NGrainOrientations, 10000);

    // Check first two orientations
    std::vector<float> ExpectedGrainOrientations = {9.99854, 29.62172, 22.91854, 311.08350, 47.68814, 72.02547};
    for (int n = 0; n < 2 * ValsPerLine; n++) {
        EXPECT_FLOAT_EQ(GrainOrientationData_Host(n), ExpectedGrainOrientations[n]);
    }
}

void testInterpolateSparseData() {

    // Domain size
    int nx = 10;
    int ny = 10;
    int nz = 7;
    int HTtoCAratio = 3;

    // Views to hold data - cells start with values of -1
    ViewD3D_H TL_Host(Kokkos::ViewAllocateWithoutInitializing("TL"), nz, nx, ny);
    Kokkos::deep_copy(TL_Host, -1);
    ViewD3D_H CR_Host(Kokkos::ViewAllocateWithoutInitializing("CR"), nz, nx, ny);
    Kokkos::deep_copy(CR_Host, -1);

    // Seed TL data at coordinates of 0, 3, 6, and 9 for interpolation
    // Seeded TL data for a point (k,i,j) is equal to one more than the sum of i, j, and k

    // These values ensure that the temperature data in a volume bounded by 8 points will be interpolated
    TL_Host(0, 0, 0) = 1;
    TL_Host(3, 0, 0) = 4;
    TL_Host(0, 3, 0) = 4;
    TL_Host(0, 0, 3) = 4;
    TL_Host(3, 3, 0) = 7;
    TL_Host(3, 0, 3) = 7;
    TL_Host(0, 3, 3) = 7;
    TL_Host(3, 3, 3) = 10;

    // These values ensure that the temperature data bounded by 4 points in an XY plane will be interpolated
    TL_Host(0, 6, 0) = 7;
    TL_Host(0, 6, 3) = 10;

    // These values ensure that the temperature data bounded by 4 points in a YZ plane will be interpolated
    TL_Host(0, 3, 6) = 10;
    TL_Host(3, 3, 6) = 13;

    // This value ensures that temperature data bounded by 4 points in an XZ plane will be interpolated
    TL_Host(3, 6, 3) = 13;

    // This value ensures that the temperature data bounded by 2 points in a line in X will be interpolated
    TL_Host(0, 9, 3) = 13;

    // This value ensures that the temperature data bounded by 2 points in a line in Y will be interpolated
    TL_Host(3, 3, 9) = 16;

    // This value ensures that the temperature data bounded by 2 points in a line in Z will be interpolated
    TL_Host(6, 3, 3) = 13;

    // CR data: leave as -1 where no TL data exists, set to 1 where TL data does exist
    for (int k = 0; k < nz; k++) {
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                if (TL_Host(k, i, j) != -1)
                    CR_Host(k, i, j) = 1;
            }
        }
    }

    // Copy data to device
    ViewD3D TL = Kokkos::create_mirror_view_and_copy(device_memory_space(), TL_Host);
    ViewD3D CR = Kokkos::create_mirror_view_and_copy(device_memory_space(), CR_Host);

    // Run interpolation
    InterpolateSparseData(TL, CR, nx, ny, nz, HTtoCAratio);

    // Expected values - start with -1s
    ViewD3D_H TL_Expected_Host("TL_Expected_Host", nz, nx, ny);
    Kokkos::deep_copy(TL_Expected_Host, -1);
    ViewD3D_H CR_Expected_Host("CR_Expected_Host", nz, nx, ny);
    Kokkos::deep_copy(CR_Expected_Host, -1);

    // These cells should have interpolated CR data (which will be all 1s, since no other CR values were given):
    // Bounded by 8 points
    for (int k = 0; k <= 3; k++) {
        for (int i = 0; i <= 3; i++) {
            for (int j = 0; j <= 3; j++) {
                CR_Expected_Host(k, i, j) = 1;
            }
        }
    }
    // Bounded by 4 points (XY plane)
    for (int i = 3; i <= 6; i++) {
        for (int j = 0; j <= 3; j++) {
            CR_Expected_Host(0, i, j) = 1;
        }
    }
    // Bounded by 4 points (YZ plane)
    for (int k = 0; k <= 3; k++) {
        for (int j = 3; j <= 6; j++) {
            CR_Expected_Host(k, 3, j) = 1;
        }
    }
    // Bounded by 4 points (XZ plane)
    for (int k = 0; k <= 3; k++) {
        for (int i = 3; i <= 6; i++) {
            CR_Expected_Host(k, i, 3) = 1;
        }
    }
    // Bounded by 2 points (X line)
    for (int i = 6; i <= 9; i++) {
        CR_Expected_Host(0, i, 3) = 1;
    }
    // Bounded by 2 points (Y line)
    for (int j = 6; j <= 9; j++) {
        CR_Expected_Host(3, 3, j) = 1;
    }
    // Bounded by 2 points (Z line)
    for (int k = 3; k <= 6; k++) {
        CR_Expected_Host(k, 3, 3) = 1;
    }

    // The same cells that have interpolated CR data should have interpolated TL data
    for (int k = 0; k < nz; k++) {
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                if (CR_Expected_Host(k, i, j) == 1)
                    TL_Expected_Host(k, i, j) = k + i + j + 1;
            }
        }
    }

    // Copy device TL and CR back to host and check against expected values
    TL_Host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), TL);
    CR_Host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), CR);
    for (int k = 0; k < nz; k++) {
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                std::cout << "k = " << k << ", i = " << i << ", j = " << j << std::endl;
                EXPECT_DOUBLE_EQ(TL_Host(k, i, j), TL_Expected_Host(k, i, j));
                EXPECT_DOUBLE_EQ(CR_Host(k, i, j), CR_Expected_Host(k, i, j));
            }
        }
    }
}
//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST(TEST_CATEGORY, orientation_init_tests) {
    testOrientationInit_Vectors();
    testOrientationInit_Angles();
}
TEST(TEST_CATEGORY, temperature_init_tests) { testInterpolateSparseData(); }
} // end namespace Test
