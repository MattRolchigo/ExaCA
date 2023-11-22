// Copyright 2021-2023 Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include <Kokkos_Core.hpp>

#include "CAconfig.hpp"
#include "CAinitialize.hpp"
#include "CAinterfacialresponse.hpp"
#include "CAparsefiles.hpp"
#include "CAprint.hpp"

#include <gtest/gtest.h>

#include "mpi.h"

#include <fstream>
#include <string>
#include <vector>

namespace Test {
//---------------------------------------------------------------------------//
// file_read_tests
//---------------------------------------------------------------------------//
void testReadWrite(bool PrintReadBinary) {

    // Make lists of some int and float data
    int IntData[5] = {-2, 0, 2, 4, 6};
    float FloatData[5] = {-1.0, 0.0, 1.0, 2.0, 3.0};

    // Write data as binary to be used as input
    std::ofstream TestIntData;
    std::ofstream TestFloatData;
    if (PrintReadBinary) {
        TestIntData.open("TestIntData.txt", std::ios::out | std::ios::binary);
        TestFloatData.open("TestFloatData.txt", std::ios::out | std::ios::binary);
    }
    else {
        TestIntData.open("TestIntData.txt");
        TestFloatData.open("TestFloatData.txt");
    }
    for (int n = 0; n < 5; n++) {
        // Write to files
        WriteData(TestIntData, IntData[n], PrintReadBinary, true);
        WriteData(TestFloatData, FloatData[n], PrintReadBinary, true);
    }
    TestIntData.close();
    TestFloatData.close();

    // Read data and convert back to ints and floats, compare to original values
    std::ifstream TestIntDataRead;
    TestIntDataRead.open("TestIntData.txt");
    std::ifstream TestFloatDataRead;
    TestFloatDataRead.open("TestFloatData.txt");
    // For reading ASCII data, obtain the lines from the files first, then parse the string stream at the spaces
    if (PrintReadBinary) {
        for (int n = 0; n < 5; n++) {
            int IntToCompare = ReadBinaryData<int>(TestIntDataRead, true);
            float FloatToCompare = ReadBinaryData<float>(TestFloatDataRead, true);
            // Compare to expected values
            EXPECT_EQ(IntToCompare, IntData[n]);
            EXPECT_FLOAT_EQ(FloatToCompare, FloatData[n]);
        }
    }
    else {
        std::string intline, floatline;
        getline(TestIntDataRead, intline);
        getline(TestFloatDataRead, floatline);
        std::istringstream intss(intline);
        std::istringstream floatss(floatline);
        for (int n = 0; n < 5; n++) {
            // Get values from string stream
            int IntToCompare = ParseASCIIData<float>(intss);
            float FloatToCompare = ParseASCIIData<float>(floatss);
            // Compare to expected values
            EXPECT_EQ(IntToCompare, IntData[n]);
            EXPECT_FLOAT_EQ(FloatToCompare, FloatData[n]);
        }
    }
}

//---------------------------------------------------------------------------//
// activedomainsizecalc
//---------------------------------------------------------------------------//
void calcz_layer_bottom() {

    int LayerHeight = 10;
    int NumberOfLayers = 10;
    double deltax = 1 * pow(10, -6);
    double ZMin = -0.5 * pow(10, -6);
    double *ZMinLayer = new double[NumberOfLayers];
    for (int layernumber = 0; layernumber < NumberOfLayers; layernumber++) {
        // Set ZMinLayer for each layer to be offset by LayerHeight cells from the previous one (lets solution for
        // both problem types by the same)
        ZMinLayer[layernumber] = ZMin + layernumber * LayerHeight * deltax;
        // Call function for each layernumber, and for simulation types "S" and "R"
        int z_layer_bottom_S =
            calc_z_layer_bottom("S", layernumber * LayerHeight, layernumber, ZMinLayer, ZMin, deltax);
        EXPECT_EQ(z_layer_bottom_S, LayerHeight * layernumber);
        int z_layer_bottom_R = calc_z_layer_bottom("R", LayerHeight, layernumber, ZMinLayer, ZMin, deltax);
        EXPECT_EQ(z_layer_bottom_R, LayerHeight * layernumber);
    }
}

void calcz_layer_top() {

    // A separate function is now used for ZBound_High calculation
    int SpotRadius = 100;
    int LayerHeight = 10;
    double ZMin = 0.5 * pow(10, -6);
    double deltax = 1.0 * pow(10, -6);
    int nz = 101;
    int NumberOfLayers = 10;
    double *ZMaxLayer = new double[NumberOfLayers];
    for (int layernumber = 0; layernumber < NumberOfLayers; layernumber++) {
        // Set ZMaxLayer for each layer to be offset by LayerHeight cells from the previous one, with layer 0 having a
        // ZMax value of ZMin + SpotRadius (lets solution for both problem types be the same)
        ZMaxLayer[layernumber] = ZMin + SpotRadius * deltax + layernumber * LayerHeight * deltax;
        // Call function for each layernumber, and for simulation types "S" and "R"
        int z_layer_top_S =
            calc_z_layer_top("S", SpotRadius, layernumber * LayerHeight, layernumber, ZMin, deltax, nz, ZMaxLayer);
        EXPECT_EQ(z_layer_top_S, SpotRadius + LayerHeight * layernumber);
        int z_layer_top_R =
            calc_z_layer_top("R", SpotRadius, layernumber * LayerHeight, layernumber, ZMin, deltax, nz, ZMaxLayer);
        EXPECT_EQ(z_layer_top_R, SpotRadius + LayerHeight * layernumber);
        // For simulation type C, should be independent of layernumber
        int z_layer_top_C = calc_z_layer_top("C", SpotRadius, LayerHeight, layernumber, ZMin, deltax, nz, ZMaxLayer);
        EXPECT_EQ(z_layer_top_C, nz - 1);
    }
}

void testcalc_nz_layer() {

    int id = 0;
    int z_layer_bottom = 5;
    int NumberOfLayers = 10;
    for (int layernumber = 0; layernumber < NumberOfLayers; layernumber++) {
        int z_layer_top = 6 + layernumber;
        int nz_layer = calc_nz_layer(z_layer_bottom, z_layer_top, id, layernumber);
        EXPECT_EQ(nz_layer, 2 + layernumber);
    }
}

void testcalcLayerDomainSize() {

    int nx = 5;
    int ny_local = 4;
    int nz_layer = 10;
    int DomainSize = calcLayerDomainSize(nx, ny_local, nz_layer);
    EXPECT_EQ(DomainSize, 10 * 5 * 4);
}

//---------------------------------------------------------------------------//
// bounds_init_test
//---------------------------------------------------------------------------//
void testFindXYZBounds(bool TestBinaryInputRead, int LayerVariability) {

    int NumberOfLayers = 6;
    // Repeat this for having 1, 2, and 3 temperature data files
    for (int TempFilesInSeries = 1; TempFilesInSeries <= 3; TempFilesInSeries++) {
        // default inputs struct with default values - manually set non-default substrateInputs values
        Inputs inputs;
        inputs.SimulationType = "R";
        inputs.temperature.TempFilesInSeries = TempFilesInSeries;
        std::string Ext;
        if (TestBinaryInputRead)
            Ext = ".catemp";
        else
            Ext = ".txt";

        // Write fake OpenFOAM data - temperature data should be of type double
        double deltax = 1 * pow(10, -6);
        // only x,y,z data should be read, tm, tl, cr should not affect result
        // For additional temperature files, have more data in Z but have the same top surface in Z = 0 for each file
        for (int n = 0; n < TempFilesInSeries; n++) {
            std::string TestFilename = "TestData_" + std::to_string(n) + Ext;
            inputs.temperature.temp_paths.push_back(TestFilename);
            std::ofstream TestData;
            if (TestBinaryInputRead)
                TestData.open(TestFilename, std::ios::out | std::ios::binary);
            else {
                TestData.open(TestFilename);
                TestData << "x, y, z, tm, tl, cr" << std::endl;
            }
            int DatasizeZ = 2 + (2 * n);
            // Without layer offsets applied, TestData_1 spans Z = 0, -1, -2 microns
            // Without layer offsets applied, TestData_2 spans Z = 0, -1, -2, -3, -4 microns
            // Without layer offsets applied, TestData_3 spans Z = 0, -1, -2, -3, -4, -5, -6 microns
            for (int k = 0; k <= DatasizeZ; k++) {
                for (int j = 0; j < 3; j++) {
                    for (int i = 0; i < 4; i++) {
                        if (TestBinaryInputRead) {
                            WriteData(TestData, static_cast<double>(i * deltax), TestBinaryInputRead);
                            WriteData(TestData, static_cast<double>(j * deltax), TestBinaryInputRead);
                            WriteData(TestData, static_cast<double>(-(DatasizeZ - k) * deltax), TestBinaryInputRead);
                            WriteData(TestData, static_cast<double>(-1.0), TestBinaryInputRead);
                            WriteData(TestData, static_cast<double>(-1.0), TestBinaryInputRead);
                            WriteData(TestData, static_cast<double>(-1.0), TestBinaryInputRead);
                        }
                        else
                            TestData << i * deltax << "," << j * deltax << "," << -(DatasizeZ - k) * deltax << ","
                                     << static_cast<double>(-1.0) << "," << static_cast<double>(-1.0) << ","
                                     << static_cast<double>(-1.0) << std::endl;
                    }
                }
            }
            TestData.close();
        }
        int LayerHeight = 3;

        // Values to be calculated in FindXYZBounds
        int nx, ny, nz;
        double XMin, XMax, YMin, YMax, ZMin, ZMax;
        double *ZMinLayer = new double[NumberOfLayers];
        double *ZMaxLayer = new double[NumberOfLayers];

        // Variable layer height - first layer has no height offset
        ViewI_H LayerHeightList(Kokkos::ViewAllocateWithoutInitializing("LayerHeightList"), NumberOfLayers);
        LayerHeightList(0) = 0;
        LayerHeightList(1) = LayerHeight - LayerVariability;
        LayerHeightList(2) = LayerHeight + LayerVariability;
        LayerHeightList(3) = LayerHeight;
        LayerHeightList(4) = LayerHeight + LayerVariability;
        LayerHeightList(5) = LayerHeight - LayerVariability;
        ViewI_H CumLayerHeightList = getCumLayerHeights(LayerHeightList, NumberOfLayers);
        FindXYZBounds(0, deltax, nx, ny, nz, XMin, XMax, YMin, YMax, ZMin, ZMax, ZMinLayer, ZMaxLayer, NumberOfLayers,
                      LayerHeightList, CumLayerHeightList, inputs);

        // Size of overall domain in the lateral dimensions
        EXPECT_EQ(nx, 4);
        EXPECT_EQ(ny, 3);
        EXPECT_DOUBLE_EQ(XMin, 0.0);
        EXPECT_DOUBLE_EQ(YMin, 0.0);
        EXPECT_DOUBLE_EQ(XMax, 3 * deltax);
        EXPECT_DOUBLE_EQ(YMax, 2 * deltax);

        // Bounds for each individual layer
        // TempFilesInSeries = 1 (repeat TestData_1): layer bottoms should be -2, 1 - LayerVariability, 4, 7, 10 +
        // LayerVariability, 13 TempFilesInSeries = 2 (repeat TestData_1 and TestData_2): layer bottoms should be -2, -1
        // - LayerVariability, 4, 5, 10 + LayerVariability, 11 TempFilesInSeries = 3 (repeat TestData_1, TestData_2, and
        // TestData_3): layer bottoms should be -2, -1 - LayerVariability, 0, 7, 8 + LayerVariability, 9 Layer tops
        // should be 0, 3 - LayerVariability, 6, 9, 12 + LayerVariability, 15 for all TempFilesInSeries
        EXPECT_DOUBLE_EQ(ZMinLayer[0], -2 * deltax);
        if (TempFilesInSeries == 1) {
            EXPECT_DOUBLE_EQ(ZMinLayer[1], (1 - LayerVariability) * deltax);
            EXPECT_DOUBLE_EQ(ZMinLayer[2], 4 * deltax);
            EXPECT_DOUBLE_EQ(ZMinLayer[3], 7 * deltax);
            EXPECT_DOUBLE_EQ(ZMinLayer[4], (10 + LayerVariability) * deltax);
            EXPECT_DOUBLE_EQ(ZMinLayer[5], 13 * deltax);
        }
        else if (TempFilesInSeries == 2) {
            EXPECT_DOUBLE_EQ(ZMinLayer[1], (-1 - LayerVariability) * deltax);
            EXPECT_DOUBLE_EQ(ZMinLayer[2], 4 * deltax);
            EXPECT_DOUBLE_EQ(ZMinLayer[3], 5 * deltax);
            EXPECT_DOUBLE_EQ(ZMinLayer[4], (10 + LayerVariability) * deltax);
            EXPECT_DOUBLE_EQ(ZMinLayer[5], 11 * deltax);
        }
        else {
            EXPECT_DOUBLE_EQ(ZMinLayer[1], (-1 - LayerVariability) * deltax);
            EXPECT_DOUBLE_EQ(ZMinLayer[2], 0);
            EXPECT_DOUBLE_EQ(ZMinLayer[3], 7 * deltax);
            EXPECT_DOUBLE_EQ(ZMinLayer[4], (8 + LayerVariability) * deltax);
            EXPECT_DOUBLE_EQ(ZMinLayer[5], 9 * deltax);
        }
        EXPECT_DOUBLE_EQ(ZMaxLayer[0], 0);
        EXPECT_DOUBLE_EQ(ZMaxLayer[1], (3 - LayerVariability) * deltax);
        EXPECT_DOUBLE_EQ(ZMaxLayer[2], 6 * deltax);
        EXPECT_DOUBLE_EQ(ZMaxLayer[3], 9 * deltax);
        EXPECT_DOUBLE_EQ(ZMaxLayer[4], (12 + LayerVariability) * deltax);
        EXPECT_DOUBLE_EQ(ZMaxLayer[5], 15 * deltax);

        // ZMin is at either -2, or if LayerVariability > 1, ZMin is at -1 - LayerVariability
        if (TempFilesInSeries == 1)
            EXPECT_DOUBLE_EQ(ZMin, -2 * deltax);
        else {
            double ZMin_expected = fmin(ZMin, deltax * (-1 - LayerVariability));
            if (LayerVariability > 1)
                EXPECT_DOUBLE_EQ(ZMin, ZMin_expected);
            else
                EXPECT_DOUBLE_EQ(ZMin, -2 * deltax);
        }
        EXPECT_DOUBLE_EQ(ZMax, 15 * deltax);

        // nz from ZMin and ZMax
        int nz_expected = round((ZMax - ZMin) / deltax) + 1;
        EXPECT_EQ(nz, nz_expected);
    }
}

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
//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST(TEST_CATEGORY, fileread_test) {
    // test functions for reading and writing data as binary (true) and ASCII (false)
    testReadWrite(true);
    testReadWrite(false);
}
TEST(TEST_CATEGORY, activedomainsizecalc) {
    calcz_layer_bottom();
    calcz_layer_top();
    testcalc_nz_layer();
    testcalcLayerDomainSize();
}
TEST(TEST_CATEGORY, bounds_init_test) {
    // reading temperature files to obtain xyz bounds, using binary/non-binary format and with variable layer heights
    testFindXYZBounds(false, 0);
    testFindXYZBounds(false, 1);
    testFindXYZBounds(false, 2);
    testFindXYZBounds(true, 0);
}
TEST(TEST_CATEGORY, orientation_init_tests) {
    testOrientationInit_Vectors();
    testOrientationInit_Angles();
}
} // end namespace Test
