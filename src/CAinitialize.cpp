// Copyright 2021-2023 Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "CAinitialize.hpp"

#include "CAconfig.hpp"
#include "CAfunctions.hpp"
#include "CAghostnodes.hpp"
#include "CAparsefiles.hpp"
#include "CAupdate.hpp"

#include "mpi.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <regex>

// Initializes input parameters, mesh, temperature field, and grain structures for CA simulations

// Read ExaCA input file (JSON format)
void InputReadFromFile(int id, std::string InputFile, std::string &SimulationType, double &deltax, double &NMax,
                       double &dTN, double &dTsigma, std::string &GrainOrientationFile, int &TempFilesInSeries,
                       std::vector<std::string> &temp_paths, double &HT_deltax, double &deltat, int &NumberOfLayers,
                       int &LayerHeight, std::string &MaterialFileName, std::string &SubstrateFileName,
                       float &SubstrateGrainSpacing, bool &UseSubstrateFile, double &G, double &R, int &nx, int &ny,
                       int &nz, double &FractSurfaceSitesActive, int &NSpotsX, int &NSpotsY, int &SpotOffset,
                       int &SpotRadius, double &RNGSeed, bool &BaseplateThroughPowder, double &PowderActiveFraction,
                       bool &LayerwiseTempRead, bool &PowderFirstLayer, Print &print) {

    std::ifstream InputData(InputFile);
    nlohmann::json inputdata = nlohmann::json::parse(InputData);

    // General inputs
    SimulationType = inputdata["SimulationType"];
    // "C": constrained (directional) solidification
    // "S": array of overlapping hemispherical spots
    // "R": time-temperature history comes from external files
    // Check if simulation type includes remelting ("M" suffix to input problem type) - all simulations now use
    // remelting, so in the absence of this suffix, print warning that the problem will use remelting DirSoldification
    // problems now include remelting logic
    if (SimulationType == "RM") {
        SimulationType = "R";
    }
    else if (SimulationType == "SM") {
        SimulationType = "S";
    }
    else if ((SimulationType == "S") || (SimulationType == "R")) {
        if (id == 0) {
            if (SimulationType == "S")
                std::cout << "Warning: While the specified problem type did not include remelting, all simulations now "
                             "include remelting"
                          << std::endl;
        }
    }
    // Input files that should be present for all problem types
    std::string MaterialFileName_Read = inputdata["MaterialFileName"];
    std::string GrainOrientationFile_Read = inputdata["GrainOrientationFile"];
    // Path to file of materials constants based on install/source location
    MaterialFileName = checkFileInstalled(MaterialFileName_Read, id);
    checkFileNotEmpty(MaterialFileName);
    // Path to file of grain orientations based on install/source location
    GrainOrientationFile = checkFileInstalled(GrainOrientationFile_Read, id);
    checkFileNotEmpty(GrainOrientationFile);
    // Seed for random number generator (defaults to 0 if not given)
    if (inputdata.contains("RandomSeed"))
        RNGSeed = inputdata["RandomSeed"];
    else
        RNGSeed = 0.0;

    // Domain inputs:
    // Cell size - given in meters, stored in micrometers
    deltax = inputdata["Domain"]["CellSize"];
    deltax = deltax * pow(10, -6);
    // Time step - given in seconds, stored in microseconds
    deltat = inputdata["Domain"]["TimeStep"];
    deltat = deltat * pow(10, -6);
    if (SimulationType == "C") {
        // Domain size, in cells
        nx = inputdata["Domain"]["Nx"];
        ny = inputdata["Domain"]["Ny"];
        nz = inputdata["Domain"]["Nz"];
        NumberOfLayers = 1;
        LayerHeight = nz;
    }
    else {
        // Number of layers, layer height are needed for problem types S and R
        NumberOfLayers = inputdata["Domain"]["NumberOfLayers"];
        LayerHeight = inputdata["Domain"]["LayerOffset"];
        // Type S needs spot information, which is then used to compute the domain bounds
        if (SimulationType == "S") {
            NSpotsX = inputdata["Domain"]["NSpotsX"];
            NSpotsY = inputdata["Domain"]["NSpotsY"];
            // Radius and offset are given in micrometers, convert to cells
            SpotRadius = inputdata["Domain"]["RSpots"];
            SpotRadius = SpotRadius * pow(10, -6) / deltax;
            SpotOffset = inputdata["Domain"]["SpotOffset"];
            SpotOffset = SpotOffset * pow(10, -6) / deltax;
            // Calculate nx, ny, and nz based on spot array pattern and number of layers
            nz = SpotRadius + 1 + (NumberOfLayers - 1) * LayerHeight;
            nx = 2 * SpotRadius + 1 + SpotOffset * (NSpotsX - 1);
            ny = 2 * SpotRadius + 1 + SpotOffset * (NSpotsY - 1);
        }
    }

    // Nucleation inputs:
    // Nucleation density (normalized by 10^12 m^-3), mean nucleation undercooling/st dev undercooling(K)
    NMax = inputdata["Nucleation"]["Density"];
    NMax = NMax * pow(10, 12);
    dTN = inputdata["Nucleation"]["MeanUndercooling"];
    dTsigma = inputdata["Nucleation"]["StDev"];

    // Temperature inputs:
    if (SimulationType == "R") {
        // Temperature data resolution - default to using CA cell size if the assumed temperature data resolution if not
        // given
        if (inputdata["TemperatureData"].contains("HeatTransferCellSize")) {
            HT_deltax = inputdata["TemperatureData"]["HeatTransferCellSize"];
            // Value is given in micrometers, convert to meters
            HT_deltax = HT_deltax * pow(10, -6);
        }
        else
            HT_deltax = deltax;
        if (HT_deltax != deltax)
            throw std::runtime_error("Error: CA cell size and input temperature data resolution must be equivalent");
        // Read all temperature files at once (default), or one at a time?
        if (inputdata["TemperatureData"].contains("LayerwiseTempRead")) {
            LayerwiseTempRead = inputdata["TemperatureData"]["LayerwiseTempRead"];
        }
        else
            LayerwiseTempRead = false;
        // Get the paths/number of/names of the temperature data files used
        TempFilesInSeries = inputdata["TemperatureData"]["TemperatureFiles"].size();
        if (TempFilesInSeries == 0)
            throw std::runtime_error("Error: No temperature files listed in the temperature instructions file");
        else {
            for (int filename = 0; filename < TempFilesInSeries; filename++)
                temp_paths.push_back(inputdata["TemperatureData"]["TemperatureFiles"][filename]);
        }
    }
    else {
        // Temperature data uses fixed thermal gradient (K/m) and cooling rate (K/s)
        G = inputdata["TemperatureData"]["G"];
        R = inputdata["TemperatureData"]["R"];
    }

    // Substrate inputs:
    if (SimulationType == "C") {
        // Fraction of sites at bottom surface active
        FractSurfaceSitesActive = inputdata["Substrate"]["FractionSurfaceSitesActive"];
    }
    else {
        // Substrate data - should data come from an initial size or a file?
        if ((inputdata["Substrate"].contains("SubstrateFilename")) && (inputdata["Substrate"].contains("MeanSize")))
            throw std::runtime_error("Error: only one of substrate grain size and substrate structure filename should "
                                     "be provided in the input file");
        else if (inputdata["Substrate"].contains("SubstrateFilename")) {
            SubstrateFileName = inputdata["Substrate"]["SubstrateFilename"];
            UseSubstrateFile = true;
        }
        else if (inputdata["Substrate"].contains("MeanSize")) {
            SubstrateGrainSpacing = inputdata["Substrate"]["MeanSize"];
            UseSubstrateFile = false;
        }
        // Should the baseplate microstructure be extended through the powder layers? Default is false
        if (inputdata["Substrate"].contains("ExtendSubstrateThroughPower"))
            BaseplateThroughPowder = inputdata["Substrate"]["ExtendSubstrateThroughPower"];
        else
            BaseplateThroughPowder = false;
        if (inputdata["Substrate"].contains("PowderDensity")) {
            // powder density is given as a density per unit volume, normalized by 10^12 m^-3 --> convert this into a
            // density of sites active on the CA grid (0 to 1)
            PowderActiveFraction = inputdata["Substrate"]["PowderDensity"];
            PowderActiveFraction = PowderActiveFraction * pow(10, 12) * pow(deltax, 3);
            if ((PowderActiveFraction < 0.0) || (PowderActiveFraction > 1.0))
                throw std::runtime_error("Error: Density of powder surface sites active must be larger than 0 and less "
                                         "than 1/(CA cell volume)");
        }
        else
            PowderActiveFraction = 1.0; // defaults to a unique grain at each site in the powder layers
        // Should a powder layer be initialized at the top of the baseplate for the first layer? (Defaults to false,
        // where the baseplate spans all of layer 0, and only starting with layer 1 is a powder layer present)
        if (inputdata["Substrate"].contains("PowderFirstLayer"))
            PowderFirstLayer = inputdata["Substrate"]["PowderFirstLayer"];
        else
            PowderFirstLayer = false;
        if ((BaseplateThroughPowder) && ((PowderFirstLayer) || (inputdata["Substrate"].contains("PowderDensity"))))
            throw std::runtime_error(
                "Error: if the option to extend the baseplate through the powder layers is toggled, options regarding "
                "the powder layer (PowderFirstLayer/PowderDensity cannot be given");
    }

    // Printing inputs:
    print.getPrintDataFromInputFile(inputdata, id, deltat);

    // Print information to console about the input file data read
    if (id == 0) {
        std::cout << "Material simulated is " << MaterialFileName << std::endl;
        std::cout << "CA cell size is " << deltax * pow(10, 6) << " microns" << std::endl;
        std::cout << "Nucleation density is " << NMax << " per m^3" << std::endl;
        std::cout << "Mean nucleation undercooling is " << dTN << " K, standard deviation of distribution is "
                  << dTsigma << "K" << std::endl;
        if (SimulationType == "C") {
            std::cout << "CA Simulation using a unidirectional, fixed thermal gradient of " << G
                      << " K/m and a cooling rate of " << R << " K/s" << std::endl;
            std::cout << "The time step is " << deltat * pow(10, 6) << " microseconds" << std::endl;
            std::cout << "The fraction of CA cells at the bottom surface that are active is " << FractSurfaceSitesActive
                      << std::endl;
        }
        else if (SimulationType == "S") {
            std::cout << "CA Simulation using a radial, fixed thermal gradient of " << G
                      << " K/m as a series of hemispherical spots, and a cooling rate of " << R << " K/s" << std::endl;
            std::cout << "A total of " << NumberOfLayers << " spots per layer, with layers offset by " << LayerHeight
                      << " CA cells will be simulated" << std::endl;
            std::cout << "The time step is " << deltat * pow(10, 6) << " microseconds" << std::endl;
        }
        else if (SimulationType == "R") {
            std::cout << "CA Simulation using temperature data from file(s)" << std::endl;
            std::cout << "The time step is " << deltat << " seconds" << std::endl;
            std::cout << "The first temperature data file to be read is " << temp_paths[0] << ", and there are "
                      << TempFilesInSeries << " in the series" << std::endl;
            std::cout << "A total of " << NumberOfLayers << " layers of solidification offset by " << LayerHeight
                      << " CA cells will be simulated" << std::endl;
        }
    }
}

void checkPowderOverflow(int nx, int ny, int LayerHeight, int NumberOfLayers, bool BaseplateThroughPowder,
                         double PowderActiveFraction) {

    // Check to make sure powder grain density is compatible with the number of powder sites
    // If this problem type includes a powder layer of some grain density, ensure that integer overflow won't occur when
    // assigning powder layer GrainIDs
    if (!(BaseplateThroughPowder)) {
        long int NumCellsPowderLayers =
            (long int)(nx) * (long int)(ny) * (long int)(LayerHeight) * (long int)(NumberOfLayers - 1);
        long int NumAssignedCellsPowderLayers =
            std::lround(round(static_cast<double>(NumCellsPowderLayers) * PowderActiveFraction));
        if (NumAssignedCellsPowderLayers > INT_MAX)
            throw std::runtime_error("Error: A smaller value for powder density is required to avoid potential integer "
                                     "overflow when assigning powder layer GrainID");
    }
}

//*****************************************************************************/
// Intialize neighbor list structures (NeighborX, NeighborY, NeighborZ)
void NeighborListInit(NList &NeighborX, NList &NeighborY, NList &NeighborZ) {

    // Assignment of neighbors around a cell "X" is as follows (in order of closest to furthest from cell "X")
    // Neighbors 0 through 8 are in the -Y direction
    // Neighbors 9 through 16 are in the XY plane with cell X
    // Neighbors 17 through 25 are in the +Y direction
    NeighborX = {0, 1, -1, 0, 0, -1, 1, -1, 1, 0, 0, 1, -1, 1, -1, 1, -1, 0, 1, -1, 0, 0, 1, -1, 1, -1};
    NeighborY = {-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    NeighborZ = {0, 0, 0, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 0, 0, 0, 0, 0, 1, -1, 1, 1, -1, -1};
}

// Check if the temperature data is in ASCII or binary format
bool checkTemperatureFileFormat(std::string tempfile_thislayer) {
    bool BinaryInputData;
    std::size_t found = tempfile_thislayer.find(".catemp");
    if (found == std::string::npos)
        BinaryInputData = false;
    else
        BinaryInputData = true;
    return BinaryInputData;
}

// Obtain the physical XYZ bounds of the domain, using either domain size from the input file, or reading temperature
// data files and parsing the coordinates
void FindXYZBounds(std::string SimulationType, int id, double &deltax, int &nx, int &ny, int &nz,
                   std::vector<std::string> &temp_paths, double &XMin, double &XMax, double &YMin, double &YMax,
                   double &ZMin, double &ZMax, int &LayerHeight, int NumberOfLayers, int TempFilesInSeries,
                   double *ZMinLayer, double *ZMaxLayer, int SpotRadius, double &XMin_Temp) {

    if (SimulationType == "R") {
        // Two passes through reading temperature data files- the first pass only reads the headers to
        // determine units and X/Y/Z bounds of the simulaton domain. Using the X/Y/Z bounds of the simulation domain,
        // nx, ny, and nz can be calculated and the domain decomposed among MPI processes. The maximum number of
        // remelting events in the simulation can also be calculated. The second pass reads the actual X/Y/Z/liquidus
        // time/cooling rate data and each rank stores the data relevant to itself in "RawData" - this is done in the
        // subroutine "ReadTemperatureData"
        XMin = std::numeric_limits<double>::max();
        YMin = std::numeric_limits<double>::max();
        ZMin = std::numeric_limits<double>::max();
        XMax = std::numeric_limits<double>::lowest();
        YMax = std::numeric_limits<double>::lowest();
        ZMax = std::numeric_limits<double>::lowest();

        // Read the first temperature file, first line to determine if the "new" OpenFOAM output format (with a 1 line
        // header) is used, or whether the "old" OpenFOAM header (which contains information like the X/Y/Z bounds of
        // the simulation domain) is
        std::ifstream FirstTemperatureFile;
        FirstTemperatureFile.open(temp_paths[0]);
        std::string FirstLineFirstFile;
        getline(FirstTemperatureFile, FirstLineFirstFile);
        std::size_t found = FirstLineFirstFile.find("Number of temperature data points");
        if (found != std::string::npos) {
            // Old temperature data format detected - no longer supported by ExaCA
            std::string error = "Error: Old header and temperature file format no longer supported";
            throw std::runtime_error(error);
        }

        // Read all data files to determine the domain bounds, max number of remelting events
        // for simulations with remelting
        int LayersToRead = std::min(NumberOfLayers, TempFilesInSeries); // was given in input file
        for (int LayerReadCount = 1; LayerReadCount <= LayersToRead; LayerReadCount++) {

            std::string tempfile_thislayer = temp_paths[LayerReadCount - 1];
            // Get min and max x coordinates in this file, which can be a binary or ASCII input file
            // binary file type uses extension .catemp, all other file types assumed to be comma-separated ASCII input
            bool BinaryInputData = checkTemperatureFileFormat(tempfile_thislayer);
            // { Xmin, Xmax, Ymin, Ymax, Zmin, Zmax }
            std::array<double, 6> XYZMinMax_ThisLayer =
                parseTemperatureCoordinateMinMax(tempfile_thislayer, BinaryInputData);

            // Based on the input file's layer offset, adjust ZMin/ZMax from the temperature data coordinate
            // system to the multilayer CA coordinate system Check to see in the XYZ bounds for this layer are
            // also limiting for the entire multilayer CA coordinate system
            XYZMinMax_ThisLayer[4] += deltax * LayerHeight * (LayerReadCount - 1);
            XYZMinMax_ThisLayer[5] += deltax * LayerHeight * (LayerReadCount - 1);
            if (XYZMinMax_ThisLayer[0] < XMin)
                XMin = XYZMinMax_ThisLayer[0];
            if (XYZMinMax_ThisLayer[1] > XMax)
                XMax = XYZMinMax_ThisLayer[1];
            if (XYZMinMax_ThisLayer[2] < YMin)
                YMin = XYZMinMax_ThisLayer[2];
            if (XYZMinMax_ThisLayer[3] > YMax)
                YMax = XYZMinMax_ThisLayer[3];
            if (XYZMinMax_ThisLayer[4] < ZMin)
                ZMin = XYZMinMax_ThisLayer[4];
            if (XYZMinMax_ThisLayer[5] > ZMax)
                ZMax = XYZMinMax_ThisLayer[5];
            ZMinLayer[LayerReadCount - 1] = XYZMinMax_ThisLayer[4];
            ZMaxLayer[LayerReadCount - 1] = XYZMinMax_ThisLayer[5];
            if (id == 0)
                std::cout << "Layer = " << LayerReadCount << " Z Bounds are " << XYZMinMax_ThisLayer[4] << " "
                          << XYZMinMax_ThisLayer[5] << std::endl;
        }
        // Extend domain in Z (build) direction if the number of layers are simulated is greater than the number
        // of temperature files read
        if (NumberOfLayers > TempFilesInSeries) {
            for (int LayerReadCount = TempFilesInSeries; LayerReadCount < NumberOfLayers; LayerReadCount++) {
                if (TempFilesInSeries == 1) {
                    // Only one temperature file was read, so the upper Z bound should account for an additional
                    // "NumberOfLayers-1" worth of data Since all layers have the same temperature data, each
                    // layer's "ZMinLayer" is just translated from that of the first layer
                    ZMinLayer[LayerReadCount] = ZMinLayer[LayerReadCount - 1] + deltax * LayerHeight;
                    ZMaxLayer[LayerReadCount] = ZMaxLayer[LayerReadCount - 1] + deltax * LayerHeight;
                    ZMax += deltax * LayerHeight;
                }
                else {
                    // "TempFilesInSeries" temperature files was read, so the upper Z bound should account for
                    // an additional "NumberOfLayers-TempFilesInSeries" worth of data
                    int RepeatedFile = (LayerReadCount) % TempFilesInSeries;
                    int RepeatUnit = LayerReadCount / TempFilesInSeries;
                    ZMinLayer[LayerReadCount] =
                        ZMinLayer[RepeatedFile] + RepeatUnit * TempFilesInSeries * deltax * LayerHeight;
                    ZMaxLayer[LayerReadCount] =
                        ZMaxLayer[RepeatedFile] + RepeatUnit * TempFilesInSeries * deltax * LayerHeight;
                    ZMax += deltax * LayerHeight;
                }
            }
        }

        // Now at the conclusion of "Loop 0", the decomposition can be performed as the domain bounds are known
        // (all header lines from all files have been read)
        // CA cells in each direction span from the lower to the higher bound of the temperature data - without wall
        // cells or padding around the simulation edges
        nx = round((XMax - XMin) / deltax) + 1;
        ny = round((YMax - YMin) / deltax) + 1;
        nz = round((ZMax - ZMin) / deltax) + 1;
    }
    else {
        // Using fixed G/R values to set temperature field - no temperature data to read
        // Setting physical domain bounds for consistency with problems that use input temperature data
        // Let XMin/YMin/ZMin be equal to 0, XMax/YMax/ZMax equal to nx*deltax, ny*deltax, nz*deltax
        XMin = 0.0;
        YMin = 0.0;
        ZMin = 0.0;
        XMax = nx * deltax;
        YMax = ny * deltax;
        ZMax = nz * deltax;
        // If this is a spot melt problem, also set the ZMin/ZMax for each layer
        for (int n = 0; n < NumberOfLayers; n++) {
            ZMinLayer[n] = deltax * (LayerHeight * n);
            ZMaxLayer[n] = deltax * (SpotRadius + LayerHeight * n);
        }
    }
    
    // From the limits, take the domain in X and recenter it so that XMin = 0.0
    XMin_Temp = XMin;
    XMin = 0.0;
    XMax = XMax - XMin_Temp;
    // Size of domain in Y is the same as in X
    YMin = 0.0;
    YMax = XMax;
    // Recompute ny
    ny = nx;
    if (id == 0) {
        std::cout << "Domain size: " << nx << " by " << ny << " by " << nz << std::endl;
        std::cout << "X Limits of domain: " << XMin << " and " << XMax << std::endl;
        std::cout << "Y Limits of domain: " << YMin << " and " << YMax << std::endl;
        std::cout << "Z Limits of domain: " << ZMin << " and " << ZMax << std::endl;
        std::cout << "================================================================" << std::endl;
    }
}

// Decompose the domain into subdomains on each MPI rank: Calculate MyYSlices and MyYOffset for each rank, where each
// subdomain contains "MyYSlices" in Y, offset from the full domain origin by "MyYOffset" cells in Y
void DomainDecomposition(int id, int np, int &MyYSlices, int &MyYOffset, int &NeighborRank_North,
                         int &NeighborRank_South, int &nx, int &ny, int &nz, long int &LocalDomainSize,
                         bool &AtNorthBoundary, bool &AtSouthBoundary) {

    // Compare total MPI ranks to total Y cells.
    if (np > ny)
        throw std::runtime_error("Error: Cannot run with more MPI ranks than cells in Y (decomposition direction).");

    // Determine which subdomains are at which locations on the grid relative to the others
    InitialDecomposition(id, np, NeighborRank_North, NeighborRank_South, AtNorthBoundary, AtSouthBoundary);
    // Determine, for each MPI process id, the local grid size in x and y (and the offsets in x and y relative to the
    // overall simulation domain)
    MyYOffset = YOffsetCalc(id, ny, np);
    MyYSlices = YMPSlicesCalc(id, ny, np);

    // Add ghost nodes at subdomain overlaps
    AddGhostNodes(NeighborRank_North, NeighborRank_South, MyYSlices, MyYOffset);

    LocalDomainSize = nx * MyYSlices * nz; // Number of cells on this MPI rank
}

// Read in temperature data from files, stored in the host view "RawData", with the appropriate MPI ranks storing the
// appropriate data
void ReadTemperatureData(int id, double &deltax, double HT_deltax, int &HTtoCAratio, int MyYSlices, int MyYOffset,
                         double YMin, double YMax, std::vector<std::string> &temp_paths, int NumberOfLayers, int TempFilesInSeries,
                         int *FirstValue, int *LastValue, bool LayerwiseTempRead, int layernumber,
                         ViewD_H &RawTemperatureData, double XMin_Temp, double XMax) {

    double HTtoCAratio_unrounded = HT_deltax / deltax;
    double HTtoCAratio_floor = floor(HTtoCAratio_unrounded);
    if (((HTtoCAratio_unrounded - HTtoCAratio_floor) > 0.0005) && (id == 0)) {
        std::string error = "Error: Temperature data point spacing not evenly divisible by CA cell size";
        throw std::runtime_error(error);
    }
    else if (((HTtoCAratio_unrounded - HTtoCAratio_floor) > 0.000001) && (id == 0)) {
        std::cout << "Note: Adjusting cell size from " << deltax << " to " << HT_deltax / HTtoCAratio_floor
                  << " to "
                     "ensure even divisibility of CA cell size into temperature data spacing"
                  << std::endl;
    }
    // Adjust deltax to exact value based on temperature data spacing and ratio between heat transport/CA cell sizes
    deltax = HT_deltax / HTtoCAratio_floor;
    HTtoCAratio = round(HT_deltax / deltax); // OpenFOAM/CA cell size ratio
    // If HTtoCAratio > 1, an interpolation of input temperature data is needed
    // The Y bounds are the region (for this MPI rank) of the physical domain that needs to be
    // read extends past the actual spatial extent of the local domain for purposes of interpolating
    // from HT_deltax to deltax
    int LowerYBound = MyYOffset - (MyYOffset % HTtoCAratio);
    int UpperYBound;
    if (HTtoCAratio == 1)
        UpperYBound = MyYOffset + MyYSlices - 1;
    else
        UpperYBound = MyYOffset + MyYSlices - 1 + HTtoCAratio - (MyYOffset + MyYSlices - 1) % HTtoCAratio;

    // Store raw data relevant to each rank in the vector structure RawData
    // Two passes through reading temperature data files- this is the second pass, reading the actual X/Y/Z/liquidus
    // time/cooling rate data and each rank stores the data relevant to itself in "RawData". With remelting
    // (SimulationType == "RM"), this is the same except that some X/Y/Z coordinates may be repeated in a file, and
    // a "melting time" value is stored in addition to liquidus time and cooling rate
    int NumberOfTemperatureDataPoints = 0;
    // Second pass through the files - ignore header line
    int FirstLayerToRead, LastLayerToRead;
    if (LayerwiseTempRead) {
        FirstLayerToRead = layernumber;
        LastLayerToRead = layernumber;
    }
    else {
        FirstLayerToRead = 0;
        LastLayerToRead = std::min(NumberOfLayers, TempFilesInSeries) - 1;
    }
    // Which temperature files should be read? Just the one file for layer "layernumber", or all of them?
    for (int LayerReadCount = FirstLayerToRead; LayerReadCount <= LastLayerToRead; LayerReadCount++) {

        std::string tempfile_thislayer;
        if (LayerwiseTempRead) {
            int LayerInSeries = layernumber % TempFilesInSeries;
            tempfile_thislayer = temp_paths[LayerInSeries];
        }
        else
            tempfile_thislayer = temp_paths[LayerReadCount];

        FirstValue[LayerReadCount] = NumberOfTemperatureDataPoints;
        // Read and parse temperature file for either binary or ASCII, storing the appropriate values on each MPI rank
        // within RawData and incrementing NumberOfTemperatureDataPoints appropriately
        bool BinaryInputData = checkTemperatureFileFormat(tempfile_thislayer);
        parseTemperatureData(tempfile_thislayer, YMin, YMax, deltax, LowerYBound, UpperYBound, NumberOfTemperatureDataPoints,
                             BinaryInputData, RawTemperatureData, XMin_Temp, XMax, layernumber);
        LastValue[LayerReadCount] = NumberOfTemperatureDataPoints;
    } // End loop over all files read for all layers
    Kokkos::resize(RawTemperatureData, NumberOfTemperatureDataPoints);
    // Determine start values for each layer's data within "RawData", if all layers were read
    if (!(LayerwiseTempRead)) {
        if (NumberOfLayers > TempFilesInSeries) {
            for (int LayerReadCount = TempFilesInSeries; LayerReadCount < NumberOfLayers; LayerReadCount++) {
                if (TempFilesInSeries == 1) {
                    // Since all layers have the same temperature data, each layer's "ZMinLayer" is just
                    // translated from that of the first layer
                    FirstValue[LayerReadCount] = FirstValue[LayerReadCount - 1];
                    LastValue[LayerReadCount] = LastValue[LayerReadCount - 1];
                }
                else {
                    // All layers have different temperature data but in a repeating pattern
                    int RepeatedFile = (LayerReadCount) % TempFilesInSeries;
                    FirstValue[LayerReadCount] = FirstValue[RepeatedFile];
                    LastValue[LayerReadCount] = LastValue[RepeatedFile];
                }
            }
        }
    }
}

//*****************************************************************************/
// Get the Z coordinate of the lower bound of iteration
int calcZBound_Low(std::string SimulationType, int LayerHeight, int layernumber, double *ZMinLayer, double ZMin,
                   double deltax) {

    int ZBound_Low = -1; // assign dummy initial value
    if (SimulationType == "C") {
        // Not a multilayer problem, top of "layer" is the top of the overall simulation domain
        ZBound_Low = 0;
    }
    else if (SimulationType == "S") {
        // lower bound of domain is an integer multiple of the layer spacing, since the temperature field is the
        // same for every layer
        ZBound_Low = LayerHeight * layernumber;
    }
    else if (SimulationType == "R") {
        // lower bound of domain is based on the data read from the file(s)
        ZBound_Low = round((ZMinLayer[layernumber] - ZMin) / deltax);
    }
    if (ZBound_Low == -1)
        throw std::runtime_error("Error: ZBound_Low went uninitialized, problem type must be C, S, or R");
    return ZBound_Low;
}
//*****************************************************************************/
// Get the Z coordinate of the upper bound of iteration
int calcZBound_High(std::string SimulationType, int SpotRadius, int LayerHeight, int layernumber, double ZMin,
                    double deltax, int nz, double *ZMaxLayer) {

    int ZBound_High = -1; // assign dummy initial value
    if (SimulationType == "C") {
        // Not a multilayer problem, top of "layer" is the top of the overall simulation domain
        ZBound_High = nz - 1;
    }
    else if (SimulationType == "S") {
        // Top of layer is equal to the spot radius for a problem of hemispherical spot solidification, plus an offset
        // depending on the layer number
        ZBound_High = SpotRadius + LayerHeight * layernumber;
    }
    else if (SimulationType == "R") {
        // Top of layer comes from the layer's file data (implicitly assumes bottom of layer 0 is the bottom of the
        // overall domain - this should be fixed in the future for edge cases where this isn't true)
        ZBound_High = round((ZMaxLayer[layernumber] - ZMin) / deltax);
    }
    if (ZBound_High == -1)
        throw std::runtime_error("Error: ZBound_High went uninitialized, problem type must be C, S, or R");
    return ZBound_High;
}
//*****************************************************************************/
// Calculate the size of the active domain in Z
int calcnzActive(int ZBound_Low, int ZBound_High, int id, int layernumber) {
    int nzActive = ZBound_High - ZBound_Low + 1;
    if (id == 0)
        std::cout << "Layer " << layernumber << "'s active domain is from Z = " << ZBound_Low << " through "
                  << ZBound_High << " (" << nzActive << ") cells" << std::endl;
    return nzActive;
}
//*****************************************************************************/
// Calculate the size of the domain, as a number of cells
int calcLocalActiveDomainSize(int nx, int MyYSlices, int nzActive) {
    int LocalActiveDomainSize = nx * MyYSlices * nzActive;
    return LocalActiveDomainSize;
}
//*****************************************************************************/
// Initialize temperature data for a constrained solidification test problem
void TempInit_DirSolidification(double G, double R, int, int &nx, int &MyYSlices, double deltax, double deltat, int,
                                int LocalDomainSize, ViewI &CritTimeStep, ViewF &UndercoolingChange,
                                ViewI &NumberOfSolidificationEvents, ViewI &SolidificationEventCounter,
                                ViewI &MeltTimeStep, ViewI MaxSolidificationEvents, ViewF3D &LayerTimeTempHistory) {

    // TODO: When all simulations use remelting, set size of these outside of this subroutine since all problems do it
    Kokkos::realloc(NumberOfSolidificationEvents, LocalDomainSize);
    Kokkos::realloc(SolidificationEventCounter, LocalDomainSize);
    Kokkos::realloc(MeltTimeStep, LocalDomainSize);
    Kokkos::realloc(LayerTimeTempHistory, LocalDomainSize, 1, 3);

    // Initialize temperature field in Z direction with thermal gradient G set in input file
    // Cells at the bottom surface (Z = 0) are at the liquidus at time step 0 (no wall cells at the bottom boundary)
    Kokkos::parallel_for(
        "TempInitDirS", LocalDomainSize, KOKKOS_LAMBDA(const int &Coordinate1D) {
            int ZCoordinate = Coordinate1D / (nx * MyYSlices);
            // All cells past melting time step
            LayerTimeTempHistory(Coordinate1D, 0, 0) = -1;
            MeltTimeStep(Coordinate1D) = -1;
            // Cells reach liquidus at a time dependent on their Z coordinate
            LayerTimeTempHistory(Coordinate1D, 0, 1) = static_cast<int>((ZCoordinate * G * deltax) / (R * deltat));
            CritTimeStep(Coordinate1D) = static_cast<int>((ZCoordinate * G * deltax) / (R * deltat));
            // Cells cool at a constant rate
            LayerTimeTempHistory(Coordinate1D, 0, 2) = R * deltat;
            UndercoolingChange(Coordinate1D) = R * deltat;
            // All cells solidify once
            MaxSolidificationEvents(0) = 1;
            SolidificationEventCounter(Coordinate1D) = 0;
            NumberOfSolidificationEvents(Coordinate1D) = 1;
        });
}

// For an overlapping spot melt pattern, determine the maximum number of times a cell will melt/solidify as part of a
// layer
int calcMaxSolidificationEventsSpot(int nx, int MyYSlices, int NumberOfSpots, int NSpotsX, int SpotRadius,
                                    int SpotOffset, int MyYOffset) {

    ViewI2D_H MaxSolidificationEvents_Temp("SEvents_Temp", nx, MyYSlices);
    for (int n = 0; n < NumberOfSpots; n++) {
        int XSpotPos = SpotRadius + (n % NSpotsX) * SpotOffset;
        int YSpotPos = SpotRadius + (n / NSpotsX) * SpotOffset;
        for (int i = 0; i < nx; i++) {
            float DistX = (float)(XSpotPos - i);
            for (int j = 0; j < MyYSlices; j++) {
                int YGlobal = j + MyYOffset;
                float DistY = (float)(YSpotPos - YGlobal);
                float TotDist = sqrt(DistX * DistX + DistY * DistY);
                if (TotDist <= SpotRadius) {
                    MaxSolidificationEvents_Temp(i, j)++;
                }
            }
        }
    }
    int TempMax = 0;
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < MyYSlices; j++) {
            if (MaxSolidificationEvents_Temp(i, j) > TempMax) {
                TempMax = MaxSolidificationEvents_Temp(i, j);
            }
        }
    }
    // Max solidification events should be the same on each rank (and each layer, since the pattern for all layers is
    // identical)
    int GlobalMaxSEvents;
    MPI_Allreduce(&TempMax, &GlobalMaxSEvents, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    return GlobalMaxSEvents;
}

// Initialize temperature data for an array of overlapping spot melts (done at the start of each layer)
void TempInit_Spot(int layernumber, double G, double R, std::string, int id, int &nx, int &MyYSlices, int &MyYOffset,
                   double deltax, double deltat, int ZBound_Low, int, int LocalActiveDomainSize, int LocalDomainSize,
                   ViewI &CritTimeStep, ViewF &UndercoolingChange, ViewF &UndercoolingCurrent, int,
                   double FreezingRange, int NSpotsX, int NSpotsY, int SpotRadius, int SpotOffset,
                   ViewF3D &LayerTimeTempHistory, ViewI &NumberOfSolidificationEvents, ViewI &MeltTimeStep,
                   ViewI &MaxSolidificationEvents, ViewI &SolidificationEventCounter) {

    int NumberOfSpots = NSpotsX * NSpotsY;

    // Temporary host view for the maximum number of times a cell in a given layer will solidify
    ViewI_H MaxSolidificationEvents_Host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), MaxSolidificationEvents);
    MaxSolidificationEvents_Host(layernumber) =
        calcMaxSolidificationEventsSpot(nx, MyYSlices, NumberOfSpots, NSpotsX, SpotRadius, SpotOffset, MyYOffset);

    // These views are initialized to zeros on the host (requires knowing MaxSolidificationEvents first), filled with
    // data, and then copied to the device for layer "layernumber"
    ViewF3D_H LayerTimeTempHistory_Host("TimeTempHistory_H", LocalActiveDomainSize,
                                        MaxSolidificationEvents_Host(layernumber), 3);
    ViewI_H NumberOfSolidificationEvents_Host("NumSEvents_H", LocalActiveDomainSize);

    // Resize device views for active domain size if initializing first layer (don't resize after that, as all layers
    // are the same)
    if (layernumber == 0) {
        Kokkos::resize(LayerTimeTempHistory, LocalActiveDomainSize, MaxSolidificationEvents_Host(0), 3);
        Kokkos::resize(NumberOfSolidificationEvents, LocalActiveDomainSize);
        Kokkos::resize(SolidificationEventCounter, LocalActiveDomainSize);
        Kokkos::resize(MeltTimeStep, LocalDomainSize);
    }

    // Temporary host views for storing initialized temperature data for active region data structures
    // No resize of device views necessary, as multilayer spot melt simulations have the same active domain size for all
    // layers These views are copied to the host, updated for layer "layernumber", and later copied back to the device
    ViewI_H MeltTimeStep_Host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), MeltTimeStep);
    ViewI_H CritTimeStep_Host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), CritTimeStep);
    ViewF_H UndercoolingChange_Host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), UndercoolingChange);

    // Outer edges of spots are initialized at the liquidus temperature
    // Spots cool at constant rate R, spot thermal gradient = G
    // Time between "start" of next spot is the time it takes for the previous spot
    // to have entirely gone below the solidus temperature
    float IsothermVelocity = (R / G) * deltat / deltax;                                  // in cells per time step
    int TimeBetweenSpots = SpotRadius / IsothermVelocity + (FreezingRange / R) / deltat; // in time steps

    if (id == 0)
        std::cout << "Initializing temperature field for " << NumberOfSpots << " spots on layer " << layernumber
                  << ", each of which takes approximately " << TimeBetweenSpots << " time steps to solidify"
                  << std::endl;

    for (int n = 0; n < NumberOfSpots; n++) {
        if (id == 0)
            std::cout << "Initializing spot " << n << " on layer " << layernumber << std::endl;
        // Initialize LayerTimeTempHistory data values for this spot/this layer - relative to the layer bottom
        int XSpotPos = SpotRadius + (n % NSpotsX) * SpotOffset;
        int YSpotPos = SpotRadius + (n / NSpotsX) * SpotOffset;
        for (int k = 0; k <= SpotRadius; k++) {
            // Distance of this cell from the spot center
            float DistZ = (float)(SpotRadius - k);
            for (int i = 0; i < nx; i++) {
                float DistX = (float)(XSpotPos - i);
                for (int j = 0; j < MyYSlices; j++) {
                    int YGlobal = j + MyYOffset;
                    float DistY = (float)(YSpotPos - YGlobal);
                    float TotDist = sqrt(DistX * DistX + DistY * DistY + DistZ * DistZ);
                    if (TotDist <= SpotRadius) {
                        int D3D1ConvPosition = k * nx * MyYSlices + i * MyYSlices + j;
                        // Melt time
                        LayerTimeTempHistory_Host(D3D1ConvPosition, NumberOfSolidificationEvents_Host(D3D1ConvPosition),
                                                  0) = 1 + TimeBetweenSpots * n;
                        // Liquidus time
                        LayerTimeTempHistory_Host(D3D1ConvPosition, NumberOfSolidificationEvents_Host(D3D1ConvPosition),
                                                  1) =
                            1 + (int)(((float)(SpotRadius)-TotDist) / IsothermVelocity) + TimeBetweenSpots * n;
                        // Cooling rate
                        LayerTimeTempHistory_Host(D3D1ConvPosition, NumberOfSolidificationEvents_Host(D3D1ConvPosition),
                                                  2) = R * deltat;
                        NumberOfSolidificationEvents_Host(D3D1ConvPosition)++;
                    }
                }
            }
        }
    }

    // Initialize data for first melt-solidification event for all cells with data in this layer
    for (int k = 0; k <= SpotRadius; k++) {
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < MyYSlices; j++) {
                int D3D1ConvPosition = k * nx * MyYSlices + i * MyYSlices + j;
                int GlobalD3D1ConvPosition = (k + ZBound_Low) * nx * MyYSlices + i * MyYSlices + j;
                if (NumberOfSolidificationEvents_Host(D3D1ConvPosition) > 0) {
                    MeltTimeStep_Host(GlobalD3D1ConvPosition) = LayerTimeTempHistory_Host(D3D1ConvPosition, 0, 0);
                    CritTimeStep_Host(GlobalD3D1ConvPosition) = LayerTimeTempHistory_Host(D3D1ConvPosition, 0, 1);
                    UndercoolingChange_Host(GlobalD3D1ConvPosition) = LayerTimeTempHistory_Host(D3D1ConvPosition, 0, 2);
                }
                else {
                    MeltTimeStep_Host(GlobalD3D1ConvPosition) = 0;
                    CritTimeStep_Host(GlobalD3D1ConvPosition) = 0;
                    UndercoolingChange_Host(GlobalD3D1ConvPosition) = 0.0;
                }
            }
        }
    }

    // Initial undercooling of all cells is 0, solidification event counter is 0 at the start of each layer
    Kokkos::deep_copy(UndercoolingCurrent, 0.0);
    Kokkos::deep_copy(SolidificationEventCounter, 0.0);

    // Copy host view data back to device
    MaxSolidificationEvents = Kokkos::create_mirror_view_and_copy(device_memory_space(), MaxSolidificationEvents_Host);
    MeltTimeStep = Kokkos::create_mirror_view_and_copy(device_memory_space(), MeltTimeStep_Host);
    CritTimeStep = Kokkos::create_mirror_view_and_copy(device_memory_space(), CritTimeStep_Host);
    UndercoolingChange = Kokkos::create_mirror_view_and_copy(device_memory_space(), UndercoolingChange_Host);
    LayerTimeTempHistory = Kokkos::create_mirror_view_and_copy(device_memory_space(), LayerTimeTempHistory_Host);
    NumberOfSolidificationEvents =
        Kokkos::create_mirror_view_and_copy(device_memory_space(), NumberOfSolidificationEvents_Host);
    MPI_Barrier(MPI_COMM_WORLD);
    if (id == 0)
        std::cout << "Spot melt temperature field for layer " << layernumber
                  << " initialized; each cell will solidify up to " << MaxSolidificationEvents_Host(layernumber)
                  << " times" << std::endl;
}

// Read data from storage, and calculate the normalized x value of the data point
int getTempCoordX(int i, double XMin, double deltax, const ViewD_H RawTemperatureData) {
    int XInt = round((RawTemperatureData(i) - XMin) / deltax);
    return XInt;
}
// Read data from storage, and calculate the normalized y value of the data point
int getTempCoordY(int i, double YMin, double deltax, const ViewD_H RawTemperatureData) {
    int YInt = round((RawTemperatureData(i + 1) - YMin) / deltax);
    return YInt;
}
// Read data from storage, and calculate the normalized z value of the data point
int getTempCoordZ(int i, double deltax, const ViewD_H RawTemperatureData, int LayerHeight, int LayerCounter,
                  double *ZMinLayer) {
    int ZInt =
        round((RawTemperatureData(i + 2) + deltax * LayerHeight * LayerCounter - ZMinLayer[LayerCounter]) / deltax);
    return ZInt;
}
// Read data from storage, obtain melting time
double getTempCoordTM(int i, const ViewD_H RawTemperatureData) {
    double TMelting = RawTemperatureData(i + 3);
    return TMelting;
}
// Read data from storage, obtain liquidus time
double getTempCoordTL(int i, const ViewD_H RawTemperatureData) {
    double TLiquidus = RawTemperatureData(i + 4);
    return TLiquidus;
}
// Read data from storage, obtain cooling rate
double getTempCoordCR(int i, const ViewD_H RawTemperatureData) {
    double CoolingRate = RawTemperatureData(i + 5);
    return CoolingRate;
}

// Calculate the number of times that a cell in layer "layernumber" undergoes melting/solidification, and store in
// MaxSolidificationEvents_Host
void calcMaxSolidificationEventsR(int id, int layernumber, int TempFilesInSeries, ViewI_H MaxSolidificationEvents_Host,
                                  int StartRange, int EndRange, ViewD_H RawTemperatureData, double XMin, double YMin,
                                  double deltax, double *ZMinLayer, int LayerHeight, int nx, int MyYSlices,
                                  int MyYOffset, int LocalActiveDomainSize) {

    if (layernumber > TempFilesInSeries) {
        // Use the value from a previously checked layer, since the time-temperature history is reused
        if (TempFilesInSeries == 1) {
            // All layers have the same temperature data, MaxSolidificationEvents for this layer is the same as the last
            MaxSolidificationEvents_Host(layernumber) = MaxSolidificationEvents_Host(layernumber - 1);
        }
        else {
            // All layers have different temperature data but in a repeating pattern
            int RepeatedFile = layernumber % TempFilesInSeries;
            MaxSolidificationEvents_Host(layernumber) = MaxSolidificationEvents_Host(RepeatedFile);
        }
    }
    else {
        // Need to calculate MaxSolidificationEvents(layernumber) from the values in RawData
        // Init to 0
        ViewI_H TempMeltCount("TempMeltCount", LocalActiveDomainSize);

        for (int i = StartRange; i < EndRange; i += 6) {

            // Get the integer X, Y, Z coordinates associated with this data point
            int XInt = getTempCoordX(i, XMin, deltax, RawTemperatureData);
            int YInt = getTempCoordY(i, YMin, deltax, RawTemperatureData);
            int ZInt = getTempCoordZ(i, deltax, RawTemperatureData, LayerHeight, layernumber, ZMinLayer);
            // Convert to 1D coordinate in the current layer's domain
            int D3D1ConvPosition = ZInt * nx * MyYSlices + XInt * MyYSlices + (YInt - MyYOffset);
            TempMeltCount(D3D1ConvPosition)++;
        }
        int MaxCount = 0;
        for (int i = 0; i < LocalActiveDomainSize; i++) {
            if (TempMeltCount(i) > MaxCount)
                MaxCount = TempMeltCount(i);
        }
        int MaxCountGlobal;
        MPI_Allreduce(&MaxCount, &MaxCountGlobal, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        MaxSolidificationEvents_Host(layernumber) = MaxCountGlobal;
    }
    if (id == 0)
        std::cout << "The maximum number of melting/solidification events during layer " << layernumber << " is "
                  << MaxSolidificationEvents_Host(layernumber) << std::endl;
}

// Initialize temperature fields for this layer if remelting is considered and data comes from files
void TempInit_ReadData(int layernumber, int id, int nx, int MyYSlices, int, int LocalActiveDomainSize,
                       int LocalDomainSize, int MyYOffset, double &deltax, double deltat, double FreezingRange,
                       ViewF3D &LayerTimeTempHistory, ViewI &NumberOfSolidificationEvents,
                       ViewI &MaxSolidificationEvents, ViewI &MeltTimeStep, ViewI &CritTimeStep,
                       ViewF &UndercoolingChange, ViewF &UndercoolingCurrent, double XMin, double YMin,
                       double *ZMinLayer, int LayerHeight, int nzActive, int ZBound_Low, int *FinishTimeStep,
                       int *FirstValue, int *LastValue, ViewD_H RawTemperatureData, ViewI &SolidificationEventCounter,
                       int TempFilesInSeries) {

    // Data was already read into the "RawData" temporary data structure
    // Determine which section of "RawData" is relevant for this layer of the overall domain
    int StartRange = FirstValue[layernumber];
    int EndRange = LastValue[layernumber];

    // Resize device views to have sizes compatible with the temporary host views
    // Copy MaxSolidificationEvents back to the host as part of resizing LayerTimeTempHistory
    ViewI_H MaxSolidificationEvents_Host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), MaxSolidificationEvents);

    // Get the maximum number of times a cell in layer "layernumber" will undergo melting/solidification
    // Store in the host view "MaxSolidificationEvents_Host"
    calcMaxSolidificationEventsR(id, layernumber, TempFilesInSeries, MaxSolidificationEvents_Host, StartRange, EndRange,
                                 RawTemperatureData, XMin, YMin, deltax, ZMinLayer, LayerHeight, nx, MyYSlices,
                                 MyYOffset, LocalActiveDomainSize);
    // With MaxSolidificationEvents_Host(layernumber) known, can resize LayerTimeTempHistory
    Kokkos::resize(LayerTimeTempHistory, LocalActiveDomainSize, MaxSolidificationEvents_Host(layernumber), 3);
    Kokkos::resize(NumberOfSolidificationEvents, LocalActiveDomainSize);
    Kokkos::resize(SolidificationEventCounter, LocalActiveDomainSize);
    if (layernumber == 0) {
        // Only needs to be resized during initialization of the first layer, as LocalDomainSize is constant while
        // LocalActiveDomainSize is not
        Kokkos::resize(MeltTimeStep, LocalDomainSize);
    }

    // These views are copied to the host, updated for layer "layernumber", and later copied back to the device
    ViewI_H MeltTimeStep_Host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), MeltTimeStep);
    ViewI_H CritTimeStep_Host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), CritTimeStep);
    ViewF_H UndercoolingChange_Host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), UndercoolingChange);

    // These views are initialized to zeros on the host, filled with data, and then copied to the device for layer
    // "layernumber"
    ViewF3D_H LayerTimeTempHistory_Host("TimeTempHistory_H", LocalActiveDomainSize,
                                        MaxSolidificationEvents_Host(layernumber), 3);
    ViewI_H NumberOfSolidificationEvents_Host("NumSEvents_H", LocalActiveDomainSize);

    double LargestTime = 0;
    double LargestTime_Global = 0;
    if (id == 0)
        std::cout << "Range of raw data for layer " << layernumber << " on rank 0 is " << StartRange << " to "
                  << EndRange << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = StartRange; i < EndRange; i += 6) {

        // Get the integer X, Y, Z coordinates associated with this data point, along with the associated TM, TL, CR
        // values
        int XInt = getTempCoordX(i, XMin, deltax, RawTemperatureData);
        int YInt = getTempCoordY(i, YMin, deltax, RawTemperatureData);
        int ZInt = getTempCoordZ(i, deltax, RawTemperatureData, LayerHeight, layernumber, ZMinLayer);
        double TMelting = getTempCoordTM(i, RawTemperatureData);
        double TLiquidus = getTempCoordTL(i, RawTemperatureData);
        double CoolingRate = getTempCoordCR(i, RawTemperatureData);

        // 1D cell coordinate on this MPI rank's domain
        int D3D1ConvPosition = ZInt * nx * MyYSlices + XInt * MyYSlices + (YInt - MyYOffset);
        // Store TM, TL, CR values for this solidification event in LayerTimeTempHistory
        LayerTimeTempHistory_Host(D3D1ConvPosition, NumberOfSolidificationEvents_Host(D3D1ConvPosition), 0) =
            round(TMelting / deltat) + 1;
        LayerTimeTempHistory_Host(D3D1ConvPosition, NumberOfSolidificationEvents_Host(D3D1ConvPosition), 1) =
            round(TLiquidus / deltat) + 1;
        LayerTimeTempHistory_Host(D3D1ConvPosition, NumberOfSolidificationEvents_Host(D3D1ConvPosition), 2) =
            std::abs(CoolingRate) * deltat;
        // Increment number of solidification events for this cell
        NumberOfSolidificationEvents_Host(D3D1ConvPosition)++;
        // Estimate of the time step where the last possible solidification is expected to occur
        double SolidusTime = TLiquidus + FreezingRange / CoolingRate;
        if (SolidusTime > LargestTime)
            LargestTime = SolidusTime;
    }
    MPI_Allreduce(&LargestTime, &LargestTime_Global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (id == 0)
        std::cout << "Largest time globally for layer " << layernumber << " is " << LargestTime_Global << std::endl;
    FinishTimeStep[layernumber] = round((LargestTime_Global) / deltat);
    if (id == 0)
        std::cout << " Layer " << layernumber << " FINISH TIME STEP IS " << FinishTimeStep[layernumber] << std::endl;
    if (id == 0)
        std::cout << "Layer " << layernumber << " temperatures read" << std::endl;

    // Reorder solidification events in LayerTimeTempHistory(location,event number,component) so that they are in order
    // based on the melting time values (component = 0)
    for (int n = 0; n < LocalActiveDomainSize; n++) {
        if (NumberOfSolidificationEvents_Host(n) > 0) {
            for (int i = 0; i < NumberOfSolidificationEvents_Host(n) - 1; i++) {
                for (int j = (i + 1); j < NumberOfSolidificationEvents_Host(n); j++) {
                    if (LayerTimeTempHistory_Host(n, i, 0) > LayerTimeTempHistory_Host(n, j, 0)) {
                        // Swap these two points - melting event "j" happens before event "i"
                        float OldMeltVal = LayerTimeTempHistory_Host(n, i, 0);
                        float OldLiqVal = LayerTimeTempHistory_Host(n, i, 1);
                        float OldCRVal = LayerTimeTempHistory_Host(n, i, 2);
                        LayerTimeTempHistory_Host(n, i, 0) = LayerTimeTempHistory_Host(n, j, 0);
                        LayerTimeTempHistory_Host(n, i, 1) = LayerTimeTempHistory_Host(n, j, 1);
                        LayerTimeTempHistory_Host(n, i, 2) = LayerTimeTempHistory_Host(n, j, 2);
                        LayerTimeTempHistory_Host(n, j, 0) = OldMeltVal;
                        LayerTimeTempHistory_Host(n, j, 1) = OldLiqVal;
                        LayerTimeTempHistory_Host(n, j, 2) = OldCRVal;
                    }
                }
            }
        }
    }
    // If a cell melts twice before reaching the liquidus temperature, this is a double counted solidification
    // event and should be removed
    for (int n = 0; n < LocalActiveDomainSize; n++) {
        if (NumberOfSolidificationEvents_Host(n) > 1) {
            for (int i = 0; i < NumberOfSolidificationEvents_Host(n) - 1; i++) {
                if (LayerTimeTempHistory_Host(n, i + 1, 0) < LayerTimeTempHistory_Host(n, i, 1)) {
                    std::cout << "Cell " << n << " removing anomalous event " << i + 1 << " out of "
                              << NumberOfSolidificationEvents_Host(n) - 1 << std::endl;
                    // Keep whichever event has the larger liquidus time
                    if (LayerTimeTempHistory_Host(n, i + 1, 1) > LayerTimeTempHistory_Host(n, i, 1)) {
                        LayerTimeTempHistory_Host(n, i, 0) = LayerTimeTempHistory_Host(n, i + 1, 0);
                        LayerTimeTempHistory_Host(n, i, 1) = LayerTimeTempHistory_Host(n, i + 1, 1);
                        LayerTimeTempHistory_Host(n, i, 2) = LayerTimeTempHistory_Host(n, i + 1, 2);
                    }
                    LayerTimeTempHistory_Host(n, i + 1, 0) = 0.0;
                    LayerTimeTempHistory_Host(n, i + 1, 1) = 0.0;
                    LayerTimeTempHistory_Host(n, i + 1, 2) = 0.0;
                    // Reshuffle other solidification events over if needed
                    for (int ii = (i + 1); ii < NumberOfSolidificationEvents_Host(n) - 1; ii++) {
                        LayerTimeTempHistory_Host(n, ii, 0) = LayerTimeTempHistory_Host(n, ii + 1, 0);
                        LayerTimeTempHistory_Host(n, ii, 1) = LayerTimeTempHistory_Host(n, ii + 1, 1);
                        LayerTimeTempHistory_Host(n, ii, 2) = LayerTimeTempHistory_Host(n, ii + 1, 2);
                    }
                    NumberOfSolidificationEvents_Host(n)--;
                }
            }
        }
    }
    // First melt-solidification event from LayerTimeTempHistory to happen is initialized
    for (int k = 0; k < nzActive; k++) {
        int GlobalZ = k + ZBound_Low;
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < MyYSlices; j++) {
                int D3D1ConvPosition = k * nx * MyYSlices + i * MyYSlices + j;
                int GlobalD3D1ConvPosition = GlobalZ * nx * MyYSlices + i * MyYSlices + j;
                if (LayerTimeTempHistory_Host(D3D1ConvPosition, 0, 0) > 0) {
                    // This cell undergoes solidification in layer "layernumber" at least once
                    MeltTimeStep_Host(GlobalD3D1ConvPosition) =
                        (int)(LayerTimeTempHistory_Host(D3D1ConvPosition, 0, 0));
                    CritTimeStep_Host(GlobalD3D1ConvPosition) =
                        (int)(LayerTimeTempHistory_Host(D3D1ConvPosition, 0, 1));
                    UndercoolingChange_Host(GlobalD3D1ConvPosition) = LayerTimeTempHistory_Host(D3D1ConvPosition, 0, 2);
                }
                else {
                    // This cell does not undergo solidification in layer "layernumber"
                    // Set liquidus time step values to a time step that will not be reached, as these cells do not
                    // traverse the liquidus (don't melt, don't resolidify)
                    MeltTimeStep_Host(GlobalD3D1ConvPosition) = -1;
                    CritTimeStep_Host(GlobalD3D1ConvPosition) = -1;
                    UndercoolingChange_Host(GlobalD3D1ConvPosition) = 0.0;
                }
            }
        }
    }

    // Initial undercooling of all cells is 0, solidification event counter is 0 at the start of each layer
    Kokkos::deep_copy(UndercoolingCurrent, 0.0);
    Kokkos::deep_copy(SolidificationEventCounter, 0.0);

    // Copy host view data back to device
    MaxSolidificationEvents = Kokkos::create_mirror_view_and_copy(device_memory_space(), MaxSolidificationEvents_Host);
    MeltTimeStep = Kokkos::create_mirror_view_and_copy(device_memory_space(), MeltTimeStep_Host);
    CritTimeStep = Kokkos::create_mirror_view_and_copy(device_memory_space(), CritTimeStep_Host);
    UndercoolingChange = Kokkos::create_mirror_view_and_copy(device_memory_space(), UndercoolingChange_Host);
    LayerTimeTempHistory = Kokkos::create_mirror_view_and_copy(device_memory_space(), LayerTimeTempHistory_Host);
    NumberOfSolidificationEvents =
        Kokkos::create_mirror_view_and_copy(device_memory_space(), NumberOfSolidificationEvents_Host);

    if (id == 0)
        std::cout << "Layer " << layernumber << " temperature field is from Z = " << ZBound_Low << " through "
                  << nzActive + ZBound_Low - 1 << " of the global domain" << std::endl;
}

//*****************************************************************************/
void ZeroResetViews(int LocalActiveDomainSize, ViewF &DiagonalLength, ViewF &CritDiagonalLength, ViewF &DOCenter,
                    ViewI &SteeringVector) {

    // Realloc steering vector as LocalActiveDomainSize may have changed (old values aren't needed)
    Kokkos::realloc(SteeringVector, LocalActiveDomainSize);

    // Realloc active cell data structure and halo regions on device (old values not needed)
    Kokkos::realloc(DiagonalLength, LocalActiveDomainSize);
    Kokkos::realloc(DOCenter, 3 * LocalActiveDomainSize);
    Kokkos::realloc(CritDiagonalLength, 26 * LocalActiveDomainSize);

    // Reset active cell data structures on device
    Kokkos::deep_copy(DiagonalLength, 0);
    Kokkos::deep_copy(DOCenter, 0);
    Kokkos::deep_copy(CritDiagonalLength, 0);
}
