// Copyright 2021 Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "runCA.hpp"

#include "CAfunctions.hpp"
#include "CAghostnodes.hpp"
#include "CAinitialize.hpp"
#include "CAprint.hpp"
#include "CAtypes.hpp"
#include "CAupdate.hpp"
#include "mpi.h"

#include <string>
#include <vector>

void RunExaCA(int id, int np, std::string InputFile) {

    double NuclTime = 0.0, CaptureTime = 0.0, GhostTime = 0.0;
    double StartNuclTime, StartCaptureTime, StartGhostTime;
    double StartTime = MPI_Wtime();

    int nx, ny, nz, DecompositionStrategy, NumberOfLayers, LayerHeight, TempFilesInSeries, NSpotsX, NSpotsY, SpotOffset,
        SpotRadius, PrintDebug, TimeSeriesInc;
    unsigned int NumberOfTemperatureDataPoints = 0; // Initialized to 0 - updated if/when temperature files are read
    bool PrintMisorientation, PrintFullOutput, RemeltingYN, UseSubstrateFile, PrintTimeSeries,
        PrintIdleTimeSeriesFrames;
    float SubstrateGrainSpacing;
    double HT_deltax, deltax, deltat, FractSurfaceSitesActive, G, R, AConst, BConst, CConst, DConst, FreezingRange,
        NMax, dTN, dTsigma;
    std::string SubstrateFileName, temppath, tempfile, SimulationType, OutputFile, GrainOrientationFile, PathToOutput;
    std::vector<std::string> temp_paths;

    // Read input file given on the command line
    InputReadFromFile(id, InputFile, SimulationType, DecompositionStrategy, AConst, BConst, CConst, DConst,
                      FreezingRange, deltax, NMax, dTN, dTsigma, OutputFile, GrainOrientationFile, temppath, tempfile,
                      TempFilesInSeries, temp_paths, HT_deltax, RemeltingYN, deltat, NumberOfLayers,
                      LayerHeight, SubstrateFileName, SubstrateGrainSpacing, UseSubstrateFile, G, R, nx, ny, nz,
                      FractSurfaceSitesActive, PathToOutput, PrintDebug, PrintMisorientation, PrintFullOutput, NSpotsX,
                      NSpotsY, SpotOffset, SpotRadius, PrintTimeSeries, TimeSeriesInc, PrintIdleTimeSeriesFrames);

    // Grid decomposition
    int ProcessorsInXDirection, ProcessorsInYDirection;
    // Variables characterizing local processor grids relative to global domain
    int MyXSlices, MyXOffset, MyYSlices, MyYOffset;
    long int LocalDomainSize;
    // Variables characterizing process IDs of neighboring MPI ranks on the grid
    // Positive X/Negative X directions are West/East, Positive Y/NegativeY directions are North/South
    int NeighborRank_North, NeighborRank_South, NeighborRank_East, NeighborRank_West, NeighborRank_NorthEast,
        NeighborRank_NorthWest, NeighborRank_SouthEast, NeighborRank_SouthWest;
    // Neighbor lists for cells
    ViewI_H NeighborX_H(Kokkos::ViewAllocateWithoutInitializing("NeighborX"), 26);
    ViewI_H NeighborY_H(Kokkos::ViewAllocateWithoutInitializing("NeighborY"), 26);
    ViewI_H NeighborZ_H(Kokkos::ViewAllocateWithoutInitializing("NeighborZ"), 26);
    ViewI_H OppositeNeighbor_H(Kokkos::ViewAllocateWithoutInitializing("OppositeNeighbor"), 26);
    ViewI2D_H ItList_H(Kokkos::ViewAllocateWithoutInitializing("ItList"), 9, 26);
    // ****** Parameters used when reading and using temperature data from files ******
    float XMin, YMin, ZMin, XMax, YMax, ZMax;     // Simulation bounds in X, Y, and Z
    float *ZMinLayer = new float[NumberOfLayers]; // Lower Z bound for input temperature data for each layer
    float *ZMaxLayer = new float[NumberOfLayers]; // Upper Z bound for input temperature data for each layer
    int *FinishTimeStep =
        new int[NumberOfLayers]; // Last time step for each layer before all cells are below the solidus temperature

    // Data structure for storing raw temperature data from file(s)
    // Store data as double - needed for small time steps to resolve local differences in solidification conditions
    // With no remelting, each data point has 5 values (X, Y, Z coordinates, liquidus time, and either solidus time OR
    // cooling rate) Initial estimate for size
    std::vector<double> RawData(1000000);
    // Maximum number of times a cell in a given layer undergoes solidification
    ViewI_H MaxSolidificationEvents_H(Kokkos::ViewAllocateWithoutInitializing("NumberOfRemeltEvents"), NumberOfLayers);

    // Contains "NumberOfLayers" values corresponding to the location within "RawData" of the first/last data element in
    // each temperature file
    int *FirstValue = new int[NumberOfLayers];
    int *LastValue = new int[NumberOfLayers];

    // Intialize neighbor list structures (NeighborX, NeighborY, NeighborZ, OppositeNeighbor, and ItList)
    NeighborListInit(NeighborX_H, NeighborY_H, NeighborZ_H, OppositeNeighbor_H, ItList_H);

    // Obtain the physical XYZ bounds of the domain, using either domain size from the input file, or reading
    // temperature data files and parsing the coordinates
    FindXYZBounds(SimulationType, id, deltax, nx, ny, nz, temp_paths, XMin, XMax, YMin, YMax, ZMin, ZMax, LayerHeight,
                  NumberOfLayers, TempFilesInSeries, ZMinLayer, ZMaxLayer);

    // Decompose the domain into subdomains on each MPI rank: Each subdomain contains "MyXSlices" cells in X, and
    // "MyYSlices" in Y. Each subdomain is offset from the full domain origin by "MyXOffset" cells in X, and "MyYOffset"
    // cells in Y
    DomainDecomposition(DecompositionStrategy, id, np, MyXSlices, MyYSlices, MyXOffset, MyYOffset, NeighborRank_North,
                        NeighborRank_South, NeighborRank_East, NeighborRank_West, NeighborRank_NorthEast,
                        NeighborRank_NorthWest, NeighborRank_SouthEast, NeighborRank_SouthWest, nx, ny, nz,
                        ProcessorsInXDirection, ProcessorsInYDirection, LocalDomainSize);

    // Read in temperature data from files, stored in "RawData", with the appropriate MPI ranks storing the appropriate
    // data based on the domain decomposition. Also obtain the maximum number of times a cell will solidify for each
    // layer
    //if ((SimulationType == "R") || (SimulationType == "RM"))
        ReadTemperatureData(id, RemeltingYN, MaxSolidificationEvents_H, MyXSlices, MyYSlices, MyXOffset, MyYOffset,
                            deltax, HT_deltax, XMin, YMin, temp_paths, FreezingRange, LayerHeight,
                            NumberOfLayers, TempFilesInSeries, NumberOfTemperatureDataPoints, ZMinLayer, ZMaxLayer,
                            FirstValue, LastValue, RawData);
    //else if (SimulationType == "SM")
    //    MaxSolidificationEventSpotCount(nx, ny, NSpotsX, NSpotsY, SpotRadius, SpotOffset, NumberOfLayers,
    //                                    MaxSolidificationEvents_H);

    // If performing a spot melt pattern simulation with remelting, the geometry of the spots can be used to determine
    // the maximium number of times a cell will solidify (each layer has the same spot pattern, so this will be the same
    // for each layer)
    MPI_Barrier(MPI_COMM_WORLD);
    if (id == 0)
        std::cout << "Mesh initialized and (if being used), temperature data read" << std::endl;

    // Temperature fields characterized by these variables:
    // For simulations that include remelting (initialize 0 size, resize only if they will be used):
    // A view that holds melting time step, liquidus time step, and cooling rate/time step data for all cells this layer
    ViewF3D_H LayerTimeTempHistory_H(Kokkos::ViewAllocateWithoutInitializing("MaxSEvents"), 0, 0, 0);
    // The number of times that each CA cell will undergo solidification during this layer
    ViewI_H NumberOfSolidificationEvents_H(Kokkos::ViewAllocateWithoutInitializing("NumSEvents"), 0);
    // A counter for the number of times each CA cell has undergone solidification so far this layer
    ViewI_H SolidificationEventCounter_H(Kokkos::ViewAllocateWithoutInitializing("SEventCounter"), 0);
    // The next time that each cell will melt during this layer
    ViewI_H MeltTimeStep_H(Kokkos::ViewAllocateWithoutInitializing("MeltTimeStep"), 0);
    
    int ZBound_Low = 0;
    int ZBound_High = round((ZMaxLayer[0] - ZMinLayer[0]) / deltax);
    int nzActive = ZBound_High - ZBound_Low + 1;
    
    int LocalActiveDomainSize = MyXSlices * MyYSlices * nzActive; // Number of active cells on this MPI rank
    //if (RemeltingYN) {
        // resize these views to hold data if remelting is being considered
        Kokkos::resize(LayerTimeTempHistory_H, LocalActiveDomainSize, MaxSolidificationEvents_H(0), 3);
        Kokkos::resize(NumberOfSolidificationEvents_H, LocalActiveDomainSize);
        Kokkos::resize(SolidificationEventCounter_H, LocalActiveDomainSize);
        Kokkos::resize(MeltTimeStep_H, LocalDomainSize);
    //}

    // For all simulations:
    // The next time each cell will cool below the liquidus during this layer
    ViewI_H CritTimeStep_H("CritTimeStep", LocalDomainSize);
    // Marks which layer a cell's final solidification is associated with
    ViewI_H LayerID_H(Kokkos::ViewAllocateWithoutInitializing("LayerID"), LocalDomainSize);
    // The rate of cooling from the liquidus temperature for each cell for this solidification event
    ViewF_H UndercoolingChange_H("UndercoolingChange", LocalDomainSize);
    // The undercooling of each cell (> 0 if actively cooling from the liquidus, = 0 otherwise)
    // ViewF_H UndercoolingCurrent_H(Kokkos::ViewAllocateWithoutInitializing("UndercoolingCurrent"), LocalDomainSize);
    bool *Melted = new bool[LocalDomainSize];
    ViewI_H CellType_H(Kokkos::ViewAllocateWithoutInitializing("CellType"), LocalDomainSize);

    // Initialize the temperature fields
//    if (SimulationType == "R") {
//        // input temperature data from files using reduced/sparse data format
//        TempInit_Reduced(id, MyXSlices, MyYSlices, MyXOffset, MyYOffset, deltax, HT_deltax, deltat, nz,
//                         CritTimeStep_H, UndercoolingChange_H, UndercoolingCurrent_H, XMin, YMin, ZMin, Melted,
//                         ZMinLayer, ZMaxLayer, LayerHeight, NumberOfLayers, FinishTimeStep, FreezingRange, LayerID_H,
//                         FirstValue, LastValue, RawData);
//    }
//    else if (SimulationType == "RM") {
        // input temperature data from files using extended data format
        TempInit_Remelt(0, id, MyXSlices, MyYSlices, nz, MyXOffset, MyYOffset, deltax, deltat, FreezingRange,
                        LayerTimeTempHistory_H, MaxSolidificationEvents_H, NumberOfSolidificationEvents_H,
                        SolidificationEventCounter_H, MeltTimeStep_H, CritTimeStep_H, UndercoolingChange_H,
                        XMin, YMin, Melted, ZMinLayer, LayerHeight, nzActive, ZBound_Low,
                        FinishTimeStep, LayerID_H, FirstValue, LastValue, RawData, CellType_H);
//    }
//    else if ((SimulationType == "S") || (SimulationType == "SM")) {
//        // spot melt array test problem
//        TempInit_SpotMelt(RemeltingYN, 0, G, R, SimulationType, id, MyXSlices, MyYSlices, MyXOffset, MyYOffset, deltax,
//                          deltat, nz, MeltTimeStep_H, CritTimeStep_H, UndercoolingChange_H, UndercoolingCurrent_H,
//                          Melted, LayerHeight, NumberOfLayers, nzActive, ZBound_Low, ZBound_High, FreezingRange,  LayerID_H, NSpotsX, NSpotsY, SpotRadius,
//                          SpotOffset, LayerTimeTempHistory_H, NumberOfSolidificationEvents_H,
//                          SolidificationEventCounter_H);
//    }
//    else if (SimulationType == "C") {
//        // directional/constrained solidification test problem
//        TempInit_DirSolidification(G, R, id, MyXSlices, MyYSlices, deltax, deltat, nz,
//                                   CritTimeStep_H, UndercoolingChange_H, UndercoolingCurrent_H, Melted, nzActive, ZBound_Low, ZBound_High, LayerID_H);
//    }
    // Delete temporary data structure for temperature data read if remelting isn't considered
    // With remelting, each layer's temperature data is initialized at the end of the previous layer, so RawData
    // will need to be accessed again for multilayer simulations
//    if ((NumberOfLayers == 1) || (!(RemeltingYN)))
//        RawData.clear();
    MPI_Barrier(MPI_COMM_WORLD);
    if (id == 0)
        std::cout << "Done with temperature field initialization, active domain size is " << nzActive << " out of "
                  << nz << " cells in the Z direction" << std::endl;

    int NGrainOrientations = 10000; // Number of grain orientations considered in the simulation
    ViewF_H GrainUnitVector_H(Kokkos::ViewAllocateWithoutInitializing("GrainUnitVector"), 9 * NGrainOrientations);
    ViewI_H GrainOrientation_H(Kokkos::ViewAllocateWithoutInitializing("GrainOrientation"), NGrainOrientations);

    // Initialize grain orientations
    OrientationInit(id, NGrainOrientations, GrainOrientation_H, GrainUnitVector_H, GrainOrientationFile);
    MPI_Barrier(MPI_COMM_WORLD);
    if (id == 0)
        std::cout << "Done with orientation initialization " << std::endl;

    // CA cell variables
    ViewI_H GrainID_H(Kokkos::ViewAllocateWithoutInitializing("GrainID"), LocalDomainSize);
//    ViewI_H CellType_H(Kokkos::ViewAllocateWithoutInitializing("CellType"), LocalDomainSize);

    // Variables characterizing the active cell region within each rank's grid
    ViewF_H DiagonalLength_H(Kokkos::ViewAllocateWithoutInitializing("DiagonalLength"), LocalActiveDomainSize);
    ViewF_H CritDiagonalLength_H(Kokkos::ViewAllocateWithoutInitializing("CritDiagonalLength"),
                                 26 * LocalActiveDomainSize);
    ViewF_H DOCenter_H(Kokkos::ViewAllocateWithoutInitializing("DOCenter"), 3 * LocalActiveDomainSize);
    //ViewI_H CaptureTimeStep_H(Kokkos::ViewAllocateWithoutInitializing("CritDiagonalLength"),
     //                            26 * LocalActiveDomainSize);
    // Initialize the grain structure - for either a constrained solidification problem, using a substrate from a file,
    // or generating a substrate using the existing CA algorithm
    int PossibleNuclei_ThisRank, NextLayer_FirstNucleatedGrainID;
    // Normalize solidification parameters
    AConst = AConst * deltat / deltax;
    BConst = BConst * deltat / deltax;
    CConst = CConst * deltat / deltax;
//    if (SimulationType == "C") {
//        SubstrateInit_ConstrainedGrowth(FractSurfaceSitesActive, MyXSlices, MyYSlices, nx, ny, nz, MyXOffset, MyYOffset,
//                                        id, np, CellType_H, GrainID_H);
//    }
//    else {
//        if (UseSubstrateFile)
//            SubstrateInit_FromFile(SubstrateFileName, nz, MyXSlices, MyYSlices, MyXOffset, MyYOffset, id, GrainID_H);
//        else
            SubstrateInit_FromGrainSpacing(SubstrateGrainSpacing, nx, ny, nz, nzActive, MyXSlices,
                                           MyYSlices, MyXOffset, MyYOffset, LocalActiveDomainSize, id, np, deltax,
                                           GrainID_H);
//        if ((SimulationType != "RM") && (SimulationType != "SM"))
//            ActiveCellInit(id, MyXSlices, MyYSlices, nz, CellType_H, CritTimeStep_H, NeighborX_H, NeighborY_H, NeighborZ_H);
//    }

    // Nuclei data structures - initial estimates for size, to be resized later when "PossibleNuclei_ThisRank" is known
    // on each MPI rank
    ViewI_H NucleiGrainID_H(Kokkos::ViewAllocateWithoutInitializing("NucleiGrainID"), LocalActiveDomainSize);
    ViewI_H NucleationTimes_H(Kokkos::ViewAllocateWithoutInitializing("NucleationTimes"), LocalActiveDomainSize);
    ViewI_H NucleiLocation_H(Kokkos::ViewAllocateWithoutInitializing("NucleiLocation"), LocalActiveDomainSize);
    //if ((SimulationType == "RM") || (SimulationType == "SM")) {
        GrainNucleiInitRemelt(0, MyXSlices, MyYSlices, nzActive, id, np, CellType_H, CritTimeStep_H,
                              NumberOfSolidificationEvents_H, LayerTimeTempHistory_H, deltax, NMax, dTN, dTsigma,
                              NextLayer_FirstNucleatedGrainID, PossibleNuclei_ThisRank, NucleationTimes_H,
                              NucleiLocation_H, NucleiGrainID_H, ZBound_Low);
        // Initialize active cell data structures to zero
//      Kokkos::deep_copy(DiagonalLength_H, 0);
        Kokkos::deep_copy(DOCenter_H, 0);
//        Kokkos::deep_copy(CritDiagonalLength_H, 0);
//    }
//    else {
//        GrainInit(-1, NGrainOrientations, DecompositionStrategy, nx, ny, nz, LocalActiveDomainSize, MyXSlices, MyYSlices, MyXOffset, MyYOffset, id, np, NeighborRank_North, NeighborRank_South, NeighborRank_East, NeighborRank_West,
//                  NeighborRank_NorthEast, NeighborRank_NorthWest, NeighborRank_SouthEast, NeighborRank_SouthWest, ItList_H,
//                  NeighborX_H, NeighborY_H, NeighborZ_H, GrainOrientation_H, GrainUnitVector_H, DiagonalLength_H,
//                  CellType_H, GrainID_H, CritDiagonalLength_H, DOCenter_H, CritTimeStep_H, deltax, NMax,
//                  NextLayer_FirstNucleatedGrainID, PossibleNuclei_ThisRank, ZBound_High, ZBound_Low, UndercoolingChange_H, CaptureTimeStep_H, AConst, BConst, CConst, DConst);
//        NucleiInit(DecompositionStrategy, MyXSlices, MyYSlices, nz, id, dTN, dTsigma, NeighborRank_North,
//                   NeighborRank_South, NeighborRank_East, NeighborRank_West, NeighborRank_NorthEast,
//                   NeighborRank_NorthWest, NeighborRank_SouthEast, NeighborRank_SouthWest, NucleiLocation_H,
//                   NucleationTimes_H, CellType_H, GrainID_H, CritTimeStep_H, UndercoolingChange_H);
//    }

    Kokkos::resize(NucleationTimes_H, PossibleNuclei_ThisRank);
    Kokkos::resize(NucleiLocation_H, PossibleNuclei_ThisRank);
    Kokkos::resize(NucleiGrainID_H, PossibleNuclei_ThisRank);

    MPI_Barrier(MPI_COMM_WORLD);
    if (id == 0)
        std::cout << "Grain structure and nucleation initialized" << std::endl;
    int cycle;

    // Buffers for ghost node data (fixed size)
    int BufSizeX, BufSizeY;
   // if (DecompositionStrategy == 1) {
        BufSizeX = MyXSlices;
        BufSizeY = 0;
   // }
//    else {
//        BufSizeX = MyXSlices - 2;
//        BufSizeY = MyYSlices - 2;
//    }
    
    Buffer2D BufferSouthSend("BufferSouthSend", BufSizeX * nzActive, 5);
    Buffer2D BufferNorthSend("BufferNorthSend",  BufSizeX * nzActive, 5);
    Buffer2D BufferEastSend("BufferEastSend", BufSizeY * nzActive, 5);
    Buffer2D BufferWestSend("BufferWestSend", BufSizeY * nzActive, 5);
    Buffer2D BufferNorthEastSend("BufferNorthEastSend", nzActive, 5);
    Buffer2D BufferNorthWestSend("BufferNorthWestSend", nzActive, 5);
    Buffer2D BufferSouthEastSend("BufferSouthEastSend", nzActive, 5);
    Buffer2D BufferSouthWestSend("BufferSouthWestSend", nzActive, 5);
    Buffer2D BufferSouthRecv("BufferSouthRecv",  BufSizeX * nzActive, 5);
    Buffer2D BufferNorthRecv("BufferNorthRecv",  BufSizeX * nzActive, 5);
    Buffer2D BufferEastRecv("BufferEastRecv", BufSizeY * nzActive, 5);
    Buffer2D BufferWestRecv("BufferWestRecv", BufSizeY * nzActive, 5);
    Buffer2D BufferNorthEastRecv("BufferNorthEastRecv", nzActive, 5);
    Buffer2D BufferNorthWestRecv("BufferNorthWestRecv", nzActive, 5);
    Buffer2D BufferSouthEastRecv("BufferSouthEastRecv", nzActive, 5);
    Buffer2D BufferSouthWestRecv("BufferSouthWestRecv", nzActive, 5);

    // Copy view data to GPU
    using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
    ViewI GrainID_G = Kokkos::create_mirror_view_and_copy(memory_space(), GrainID_H);
    ViewI CellType_G = Kokkos::create_mirror_view_and_copy(memory_space(), CellType_H);
    ViewF DiagonalLength_G = Kokkos::create_mirror_view_and_copy(memory_space(), DiagonalLength_H);
    ViewF CritDiagonalLength_G = Kokkos::create_mirror_view_and_copy(memory_space(), CritDiagonalLength_H);
    //ViewI CaptureTimeStep_G = Kokkos::create_mirror_view_and_copy(memory_space(), CaptureTimeStep_H);

    ViewF DOCenter_G = Kokkos::create_mirror_view_and_copy(memory_space(), DOCenter_H);
    ViewI CritTimeStep_G = Kokkos::create_mirror_view_and_copy(memory_space(), CritTimeStep_H);
    ViewI LayerID_G = Kokkos::create_mirror_view_and_copy(memory_space(), LayerID_H);
    ViewF UndercoolingChange_G = Kokkos::create_mirror_view_and_copy(memory_space(), UndercoolingChange_H);
    // ViewF UndercoolingCurrent_G = Kokkos::create_mirror_view_and_copy(memory_space(), UndercoolingCurrent_H);
    ViewI NucleiLocation_G = Kokkos::create_mirror_view_and_copy(memory_space(), NucleiLocation_H);
    ViewI NucleationTimes_G = Kokkos::create_mirror_view_and_copy(memory_space(), NucleationTimes_H);
    ViewI NucleiGrainID_G = Kokkos::create_mirror_view_and_copy(memory_space(), NucleiGrainID_H);
    ViewI NeighborX_G = Kokkos::create_mirror_view_and_copy(memory_space(), NeighborX_H);
    ViewI NeighborY_G = Kokkos::create_mirror_view_and_copy(memory_space(), NeighborY_H);
    ViewI NeighborZ_G = Kokkos::create_mirror_view_and_copy(memory_space(), NeighborZ_H);
    ViewI2D ItList_G = Kokkos::create_mirror_view_and_copy(memory_space(), ItList_H);
    ViewI OppositeNeighbor_G = Kokkos::create_mirror_view_and_copy(memory_space(), OppositeNeighbor_H);
    ViewI GrainOrientation_G = Kokkos::create_mirror_view_and_copy(memory_space(), GrainOrientation_H);
    ViewF GrainUnitVector_G = Kokkos::create_mirror_view_and_copy(memory_space(), GrainUnitVector_H);
    ViewI MaxSolidificationEvents_G = Kokkos::create_mirror_view_and_copy(memory_space(), MaxSolidificationEvents_H);
    ViewI NumberOfSolidificationEvents_G =
        Kokkos::create_mirror_view_and_copy(memory_space(), NumberOfSolidificationEvents_H);
    ViewF3D LayerTimeTempHistory_G = Kokkos::create_mirror_view_and_copy(memory_space(), LayerTimeTempHistory_H);
    ViewI SolidificationEventCounter_G =
        Kokkos::create_mirror_view_and_copy(memory_space(), SolidificationEventCounter_H);
    ViewI MeltTimeStep_G = Kokkos::create_mirror_view_and_copy(memory_space(), MeltTimeStep_H);

    // Steering Vector
    ViewI SteeringVector(Kokkos::ViewAllocateWithoutInitializing("SteeringVector"), LocalActiveDomainSize);
    ViewI_H numSteer_H(Kokkos::ViewAllocateWithoutInitializing("SteeringVectorSize"), 1);
    numSteer_H(0) = 0;
    ViewI numSteer_G = Kokkos::create_mirror_view_and_copy(memory_space(), numSteer_H);

    if (np > 1) {
        // Ghost nodes for initial microstructure state
        GhostExchange(id, -1, NeighborRank_North, NeighborRank_South, MyXSlices,
                      MyYSlices, ZBound_Low, nzActive, GrainID_G, CellType_G,
                      DOCenter_G, DiagonalLength_G, CritDiagonalLength_G, BufSizeX);
    }

    // If specified, print initial values in some views for debugging purposes
    double InitTime = MPI_Wtime() - StartTime;
    if (id == 0)
        std::cout << "Data initialized: Time spent: " << InitTime << " s" << std::endl;
    if (PrintDebug) {
        PrintExaCAData(id, -1, np, nx, ny, nz, MyXSlices, MyYSlices, MyXOffset, MyYOffset, ProcessorsInXDirection, ProcessorsInYDirection,
                       GrainID_H, GrainOrientation_H, CritTimeStep_H, GrainUnitVector_H, LayerID_H, CellType_H,
                       UndercoolingChange_H, OutputFile, DecompositionStrategy,
                       NGrainOrientations, Melted, PathToOutput, PrintDebug, false, false, false, 0, ZBound_Low,
                       nzActive, deltax, XMin, YMin, ZMin);
        MPI_Barrier(MPI_COMM_WORLD);
        if (id == 0)
            std::cout << "Initialization data file(s) printed" << std::endl;
    }
    cycle = 0;
    int IntermediateFileCounter = 0;
    for (int layernumber = 0; layernumber < NumberOfLayers; layernumber++) {

        int nn = 0; // Counter for the number of nucleation events
        int XSwitch = 0;
        double LayerTime1 = MPI_Wtime();

        // Loop continues until all liquid cells claimed by solid grains
        do {
            //if (id == 0) std::cout << cycle << std::endl;
            // Start of time step - check and see if intermediate system output is to be printed to files
            if ((PrintTimeSeries) && (cycle % TimeSeriesInc == 0)) {
                // Print current state of ExaCA simulation (up to and including the current layer's data)
                Kokkos::deep_copy(GrainID_H, GrainID_G);
                Kokkos::deep_copy(CellType_H, CellType_G);
                PrintExaCAData(id, layernumber, np, nx, ny, nz, MyXSlices, MyYSlices, MyXOffset, MyYOffset, ProcessorsInXDirection,
                               ProcessorsInYDirection, GrainID_H, GrainOrientation_H, CritTimeStep_H, GrainUnitVector_H,
                               LayerID_H, CellType_H, UndercoolingChange_H, OutputFile,
                               DecompositionStrategy, NGrainOrientations, Melted, PathToOutput, 0, false, false, true,
                               IntermediateFileCounter, ZBound_Low, nzActive, deltax, XMin, YMin, ZMin);
                IntermediateFileCounter++;
            }
            cycle++;

            // Update cells on GPU - undercooling and diagonal length updates, nucleation
            StartNuclTime = MPI_Wtime();
            Nucleation(MyXSlices, MyYSlices, cycle, nn, CellType_G, NucleiLocation_G,
                       NucleationTimes_G, NucleiGrainID_G, GrainID_G, PossibleNuclei_ThisRank);
            NuclTime += MPI_Wtime() - StartNuclTime;

            // Update cells on GPU - new active cells, solidification of old active cells
            StartCaptureTime = MPI_Wtime();
//            if (!(RemeltingYN))
//                CellCapture_NoRM(np, cycle, DecompositionStrategy, LocalActiveDomainSize, LocalDomainSize,
//                            MyXSlices, MyYSlices, AConst, BConst, CConst, DConst, MyXOffset, MyYOffset,
//                            NeighborX_G, NeighborY_G, NeighborZ_G, OppositeNeighbor_G, CritTimeStep_G, UndercoolingCurrent_G,
//                            UndercoolingChange_G, GrainUnitVector_G, CritDiagonalLength_G, DiagonalLength_G,
//                            GrainOrientation_G, CellType_G, DOCenter_G, GrainID_G, NGrainOrientations, BufferWestSend,
//                            BufferEastSend, BufferNorthSend, BufferSouthSend, BufferNorthEastSend, BufferNorthWestSend,
//                            BufferSouthEastSend, BufferSouthWestSend, BufSizeX, BufSizeY, ZBound_Low, nzActive, nz,
//                            SteeringVector, numSteer_G, numSteer_H, CaptureTimeStep_G);
//            else
                CellCapture_RM(cycle, LocalActiveDomainSize, id,
                            MyXSlices, MyYSlices, nx, ny, AConst, BConst, CConst, DConst, MyXOffset, MyYOffset,
                            NeighborX_G, NeighborY_G, NeighborZ_G, CritTimeStep_G,
                            UndercoolingChange_G, GrainUnitVector_G, GrainOrientation_G, CellType_G, DOCenter_G, GrainID_G, NGrainOrientations, ZBound_Low, nzActive, nz, SteeringVector, numSteer_G, numSteer_H, MeltTimeStep_G,
                            SolidificationEventCounter_G, NumberOfSolidificationEvents_G, LayerTimeTempHistory_G, DiagonalLength_G, CritDiagonalLength_G, BufferWestSend,
                            BufferEastSend, BufferNorthSend, BufferSouthSend, BufferNorthEastSend, BufferNorthWestSend,
                            BufferSouthEastSend, BufferSouthWestSend, np, DecompositionStrategy, BufSizeX, BufSizeY);
            CaptureTime += MPI_Wtime() - StartCaptureTime;

            if (np > 1) {
                // Update ghost nodes
                StartGhostTime = MPI_Wtime();
               // GhostExchange(id, cycle, NeighborRank_North, NeighborRank_South, MyXSlices,
               //               MyYSlices, ZBound_Low, nzActive, GrainID_G, CellType_G,
               //               DOCenter_G, DiagonalLength_G, CritDiagonalLength_G, BufferSouthSend, BufferNorthSend, BufferSouthRecv, BufferNorthRecv);
//                if (DecompositionStrategy == 1)
                    GhostNodes1D(cycle, id, NeighborRank_North, NeighborRank_South, MyXSlices, MyYSlices, MyXOffset,
                                 MyYOffset, NeighborX_G, NeighborY_G, NeighborZ_G, CellType_G, DOCenter_G, GrainID_G,
                                 GrainUnitVector_G, GrainOrientation_G, DiagonalLength_G, CritDiagonalLength_G,
                                 NGrainOrientations, BufferNorthSend, BufferSouthSend, BufferNorthRecv, BufferSouthRecv,
                                 BufSizeX, BufSizeY, nzActive, ZBound_Low);
//                else
//                    GhostNodes2D(cycle, id, NeighborRank_North, NeighborRank_South, NeighborRank_East,
//                                 NeighborRank_West, NeighborRank_NorthEast, NeighborRank_NorthWest,
//                                 NeighborRank_SouthEast, NeighborRank_SouthWest, MyXSlices, MyYSlices, MyXOffset,
//                                 MyYOffset, NeighborX_G, NeighborY_G, NeighborZ_G, CellType_G, DOCenter_G, GrainID_G,
//                                 GrainUnitVector_G, GrainOrientation_G, DiagonalLength_G, CritDiagonalLength_G,
//                                 NGrainOrientations, BufferWestSend, BufferEastSend, BufferNorthSend, BufferSouthSend,
//                                 BufferNorthEastSend, BufferNorthWestSend, BufferSouthEastSend, BufferSouthWestSend,
//                                 BufferWestRecv, BufferEastRecv, BufferNorthRecv, BufferSouthRecv, BufferNorthEastRecv,
//                                 BufferNorthWestRecv, BufferSouthEastRecv, BufferSouthWestRecv, BufSizeX, BufSizeY,
//                                 nzActive, ZBound_Low);
                GhostTime += MPI_Wtime() - StartGhostTime;
            }

            if (cycle % 1000 == 0) {
                //if (RemeltingYN)
                    IntermediateOutputAndCheck_Remelt(
                        id, np, cycle, MyXSlices, MyYSlices, MyXOffset, MyYOffset, LocalActiveDomainSize, nx, ny, nz, nzActive, deltax, XMin,
                        YMin, ZMin, DecompositionStrategy, ProcessorsInXDirection, ProcessorsInYDirection, nn, XSwitch,
                        CellType_G, CellType_H, CritTimeStep_H, MeltTimeStep_G, GrainID_G, GrainID_H, layernumber,
                        NumberOfLayers, ZBound_Low, NGrainOrientations, Melted, LayerID_G, LayerID_H,
                        GrainOrientation_H, GrainUnitVector_H, UndercoolingChange_H,
                        PathToOutput, OutputFile, PrintIdleTimeSeriesFrames, TimeSeriesInc, IntermediateFileCounter);
//                else
//                    IntermediateOutputAndCheck(
//                    id, np, cycle, MyXSlices, MyYSlices, MyXOffset, MyYOffset, NeighborRank_North, NeighborRank_South, NeighborRank_West, NeighborRank_East, LocalDomainSize, LocalActiveDomainSize, nx, ny, nz, nzActive,
//                    deltax, XMin, YMin, ZMin, DecompositionStrategy, ProcessorsInXDirection, ProcessorsInYDirection, nn,
//                    XSwitch, CellType_G, CellType_H, CritTimeStep_G, CritTimeStep_H, GrainID_G, GrainID_H,
//                    SimulationType, FinishTimeStep, layernumber, NumberOfLayers, ZBound_Low, NGrainOrientations, Melted,
//                    LayerID_G, LayerID_H, GrainOrientation_H, GrainUnitVector_H, UndercoolingChange_H,
//                    UndercoolingCurrent_H, PathToOutput, OutputFile, PrintIdleTimeSeriesFrames, TimeSeriesInc,
//                    IntermediateFileCounter);
            }
            
        } while (XSwitch == 0);

        if (layernumber != NumberOfLayers - 1) {
            // Copy cell type and grain ID back to host for initialization of next layer
            Kokkos::deep_copy(CellType_H, CellType_G);
            Kokkos::deep_copy(GrainID_H, GrainID_G);
            
            // Reset intermediate file counter to zero if printing video files
            if (PrintTimeSeries)
                IntermediateFileCounter = 0;

            ZBound_Low = round((ZMinLayer[layernumber+1] - ZMin) / deltax);
            ZBound_High = round((ZMaxLayer[layernumber+1] - ZMin) / deltax);
            nzActive = ZBound_High - ZBound_Low + 1;
            LocalActiveDomainSize = MyXSlices * MyYSlices * nzActive;
            if (id == 0)
                std::cout << "Initializing layer " << layernumber+1 << " Z = " << ZBound_Low << " through Z = " << ZBound_High << std::endl;
            // Determine new active cell domain size and offset from bottom of global domain
//            DomainShiftAndResize(SimulationType, id, MyXSlices, MyYSlices, ZBound_Low, ZBound_High, ZMin, ZMinLayer,
//                                 ZMaxLayer, deltax, nzActive, nz, SpotRadius, LocalActiveDomainSize, layernumber + 1,
//                                 LayerHeight, CellType_G, layernumber+1);

            // Resize steering vector as LocalActiveDomainSize may have changed
            Kokkos::resize(SteeringVector, LocalActiveDomainSize);

            // Resize active cell data structures
            Kokkos::resize(DiagonalLength_G, LocalActiveDomainSize);
            Kokkos::resize(DOCenter_G, 3 * LocalActiveDomainSize);
            Kokkos::resize(CritDiagonalLength_G, 26 * LocalActiveDomainSize);

            Kokkos::resize(BufferNorthSend, BufSizeX * nzActive, 5);
            Kokkos::resize(BufferSouthSend, BufSizeX * nzActive, 5);
            Kokkos::resize(BufferEastSend, BufSizeY * nzActive, 5);
            Kokkos::resize(BufferWestSend, BufSizeY * nzActive, 5);
            Kokkos::resize(BufferNorthEastSend, nzActive, 5);
            Kokkos::resize(BufferNorthWestSend, nzActive, 5);
            Kokkos::resize(BufferSouthEastSend, nzActive, 5);
            Kokkos::resize(BufferSouthWestSend, nzActive, 5);

            Kokkos::resize(BufferNorthRecv, BufSizeX * nzActive, 5);
            Kokkos::resize(BufferSouthRecv, BufSizeX * nzActive, 5);
            Kokkos::resize(BufferEastRecv, BufSizeY * nzActive, 5);
            Kokkos::resize(BufferWestRecv, BufSizeY * nzActive, 5);
            Kokkos::resize(BufferNorthEastRecv, nzActive, 5);
            Kokkos::resize(BufferNorthWestRecv, nzActive, 5);
            Kokkos::resize(BufferSouthEastRecv, nzActive, 5);
            Kokkos::resize(BufferSouthWestRecv, nzActive, 5);

            // Re-zero views for active cell data structures/buffers on GPU
            Kokkos::deep_copy(DiagonalLength_G, 0.0);
            Kokkos::deep_copy(CritDiagonalLength_G, 0.0);
            Kokkos::deep_copy(DOCenter_G, 0.0);
            Kokkos::deep_copy(BufferSouthSend, 0.0);
            Kokkos::deep_copy(BufferSouthRecv, 0.0);
            Kokkos::deep_copy(BufferNorthSend, 0.0);
            Kokkos::deep_copy(BufferNorthRecv, 0.0);
//            ZeroInitViews(DOCenter_G, BufferWestSend,
//                          BufferEastSend, BufferNorthSend, BufferSouthSend, BufferNorthEastSend, BufferNorthWestSend,
//                          BufferSouthEastSend, BufferSouthWestSend, BufferWestRecv, BufferEastRecv, BufferNorthRecv,
//                          BufferSouthRecv, BufferNorthEastRecv, BufferNorthWestRecv, BufferSouthEastRecv,
//                          BufferSouthWestRecv);

//            if (RemeltingYN) {
                // Also resize and zero init temperature data structures exclusive to runs with remelting
            Kokkos::resize(NumberOfSolidificationEvents_H, LocalActiveDomainSize);
            Kokkos::resize(SolidificationEventCounter_H, LocalActiveDomainSize);
            Kokkos::resize(LayerTimeTempHistory_H, LocalActiveDomainSize,
                           MaxSolidificationEvents_H(layernumber + 1), 3);
            Kokkos::resize(NumberOfSolidificationEvents_G, LocalActiveDomainSize);
            Kokkos::resize(SolidificationEventCounter_G, LocalActiveDomainSize);
            Kokkos::resize(LayerTimeTempHistory_G, LocalActiveDomainSize,
                           MaxSolidificationEvents_H(layernumber + 1), 3);

            Kokkos::deep_copy(MeltTimeStep_H, 0.0);
            Kokkos::deep_copy(CritTimeStep_H, 0.0);
            Kokkos::deep_copy(UndercoolingChange_H, 0.0);
            Kokkos::deep_copy(NumberOfSolidificationEvents_H, 0.0);
            Kokkos::deep_copy(SolidificationEventCounter_H, 0.0);
            Kokkos::deep_copy(LayerTimeTempHistory_H, 0.0);

            //std::cout << "Number of solidification events = " << NumberOfSolidificationEvents_H(2 * MyXSlices * MyYSlices + 248 * MyYSlices)  << std::endl;


            // Initialize temperature field for the next layer
            // if (SimulationType == "RM")
            TempInit_Remelt(layernumber + 1, id, MyXSlices, MyYSlices, nz, MyXOffset, MyYOffset, deltax, deltat,
                            FreezingRange, LayerTimeTempHistory_H, MaxSolidificationEvents_H,
                            NumberOfSolidificationEvents_H, SolidificationEventCounter_H, MeltTimeStep_H,
                            CritTimeStep_H, UndercoolingChange_H, XMin, YMin, Melted,
                            ZMinLayer, LayerHeight, nzActive, ZBound_Low, FinishTimeStep, LayerID_H, FirstValue,
                            LastValue, RawData, CellType_H);
//            if (id == 0) std::cout << "Cell Type for cell is: " << CellType_H((ZBound_Low + 3) * MyXSlices * MyYSlices + 400 * MyYSlices + 134) << " Crit time step for cell is " << CritTimeStep_H((ZBound_Low + 3) * MyXSlices * MyYSlices + 400 * MyYSlices + 134) << " Number of solidification events for cell : " << NumberOfSolidificationEvents_H(3 * MyXSlices * MyYSlices + 400 * MyYSlices + 134) << std::endl;
//                else if (SimulationType == "SM")
//                    TempInit_SpotMelt(RemeltingYN, layernumber + 1, G, R, SimulationType, id, MyXSlices, MyYSlices, MyXOffset, MyYOffset, deltax,
//                                      deltat, nz, MeltTimeStep_H, CritTimeStep_H, UndercoolingChange_H, UndercoolingCurrent_H,
//                                      Melted, LayerHeight, NumberOfLayers, nzActive, ZBound_Low, ZBound_High, FreezingRange,  LayerID_H, NSpotsX, NSpotsY, SpotRadius,
//                                      SpotOffset, LayerTimeTempHistory_H, NumberOfSolidificationEvents_H,
//                                      SolidificationEventCounter_H);

                // Re-initialize solid cells (part of this layer that will melt/solidify) and wall cells (cells that are
                // ignored in this layer) Estimate number of nuclei on each rank (resize later when
                // "PossibleNuclei_ThisRank" is known)
                Kokkos::resize(NucleationTimes_H, LocalActiveDomainSize);
                Kokkos::resize(NucleiLocation_H, LocalActiveDomainSize);
                Kokkos::resize(NucleiGrainID_H, LocalActiveDomainSize);

                GrainNucleiInitRemelt(layernumber + 1, MyXSlices, MyYSlices, nzActive, id, np, CellType_H,
                                      CritTimeStep_H, NumberOfSolidificationEvents_H, LayerTimeTempHistory_H, deltax,
                                      NMax, dTN, dTsigma, NextLayer_FirstNucleatedGrainID, PossibleNuclei_ThisRank,
                                      NucleationTimes_H, NucleiLocation_H, NucleiGrainID_H, ZBound_Low);
                Kokkos::resize(NucleationTimes_H, PossibleNuclei_ThisRank);
                Kokkos::resize(NucleiLocation_H, PossibleNuclei_ThisRank);
                Kokkos::resize(NucleiGrainID_H, PossibleNuclei_ThisRank);
                Kokkos::resize(NucleationTimes_G, PossibleNuclei_ThisRank);
                Kokkos::resize(NucleiLocation_G, PossibleNuclei_ThisRank);
                Kokkos::resize(NucleiGrainID_G, PossibleNuclei_ThisRank);

                // Deep copy updated views from host
                Kokkos::deep_copy(CellType_G, CellType_H);
                Kokkos::deep_copy(CritTimeStep_G, CritTimeStep_H);
                Kokkos::deep_copy(MeltTimeStep_G, MeltTimeStep_H);
                Kokkos::deep_copy(UndercoolingChange_G, UndercoolingChange_H);
                //Kokkos::deep_copy(UndercoolingCurrent_G, UndercoolingCurrent_H);
                Kokkos::deep_copy(LayerTimeTempHistory_G, LayerTimeTempHistory_H);
                Kokkos::deep_copy(LayerID_G, LayerID_H);
                Kokkos::deep_copy(NumberOfSolidificationEvents_G, NumberOfSolidificationEvents_H);
                Kokkos::deep_copy(SolidificationEventCounter_G, SolidificationEventCounter_H);
                Kokkos::deep_copy(NucleationTimes_G, NucleationTimes_H);
                Kokkos::deep_copy(NucleiLocation_G, NucleiLocation_H);
                Kokkos::deep_copy(NucleiGrainID_G, NucleiGrainID_H);
//            }
//            else {
//                // Update active cell data structures for simulation of next layer
//                LayerSetup(MyXSlices, MyYSlices, MyXOffset, MyYOffset, LocalActiveDomainSize, GrainOrientation_G,
//                           NGrainOrientations, GrainUnitVector_G, NeighborX_G, NeighborY_G, NeighborZ_G,
//                           DiagonalLength_G, CellType_G, GrainID_G, CritDiagonalLength_G, DOCenter_G, ZBound_Low);
//            }
            MPI_Barrier(MPI_COMM_WORLD);
            if (id == 0)
                std::cout << "Resize executed and new layer setup, GN dimensions are " << BufSizeX << " " << BufSizeY
                          << " " << nzActive << std::endl;

            // Update ghost nodes for grain locations and attributes
            MPI_Barrier(MPI_COMM_WORLD);
            if (np > 1) {
                GhostExchange(id, -1, NeighborRank_North, NeighborRank_South, MyXSlices,
                              MyYSlices, ZBound_Low, nzActive, GrainID_G, CellType_G,
                              DOCenter_G, DiagonalLength_G, CritDiagonalLength_G, BufSizeX);
            }
            // XSwitch = 0;
            MPI_Barrier(MPI_COMM_WORLD);
            double LayerTime2 = MPI_Wtime();
            cycle = 0;
            if (id == 0)
                std::cout << "Time for layer number " << layernumber << " was " << LayerTime2 - LayerTime1
                          << " s, starting layer " << layernumber + 1 << std::endl;
        }
        else {
            MPI_Barrier(MPI_COMM_WORLD);
            double LayerTime2 = MPI_Wtime();
            if (id == 0)
                std::cout << "Time for final layer was " << LayerTime2 - LayerTime1 << " s" << std::endl;
        }
    }

    double RunTime = MPI_Wtime() - InitTime;

    // Copy GPU results for GrainID back to CPU for printing to file(s)
    Kokkos::deep_copy(GrainID_H, GrainID_G);
    Kokkos::deep_copy(CellType_H, CellType_G);

    MPI_Barrier(MPI_COMM_WORLD);
    if ((PrintMisorientation) || (PrintFullOutput)) {
        if (id == 0)
            std::cout << "Collecting data on rank 0 and printing to files" << std::endl;
        PrintExaCAData(id, NumberOfLayers - 1, np, nx, ny, nz, MyXSlices, MyYSlices, MyXOffset, MyYOffset, ProcessorsInXDirection,
                       ProcessorsInYDirection, GrainID_H, GrainOrientation_H, CritTimeStep_H, GrainUnitVector_H,
                       LayerID_H, CellType_H, UndercoolingChange_H, OutputFile,
                       DecompositionStrategy, NGrainOrientations, Melted, PathToOutput, 0, PrintMisorientation,
                       PrintFullOutput, false, 0, ZBound_Low, nzActive, deltax, XMin, YMin, ZMin);
    }
    else {
        if (id == 0)
            std::cout << "No output files to be printed, exiting program" << std::endl;
    }

    double OutTime = MPI_Wtime() - RunTime - InitTime;
    double InitMaxTime, InitMinTime, OutMaxTime, OutMinTime = 0.0;
    double NuclMaxTime, NuclMinTime, CaptureMaxTime, CaptureMinTime, GhostMaxTime, GhostMinTime = 0.0;
    MPI_Allreduce(&InitTime, &InitMaxTime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&InitTime, &InitMinTime, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&NuclTime, &NuclMaxTime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&NuclTime, &NuclMinTime, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&CaptureTime, &CaptureMaxTime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&CaptureTime, &CaptureMinTime, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&GhostTime, &GhostMaxTime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&GhostTime, &GhostMinTime, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&OutTime, &OutMaxTime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&OutTime, &OutMinTime, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    PrintExaCALog(id, np, InputFile, SimulationType, DecompositionStrategy, MyXSlices, MyYSlices, MyXOffset, MyYOffset,
                  AConst, BConst, CConst, DConst, FreezingRange, deltax, NMax, dTN, dTsigma, tempfile,
                  TempFilesInSeries, HT_deltax, RemeltingYN, deltat, NumberOfLayers, LayerHeight, SubstrateFileName,
                  SubstrateGrainSpacing, UseSubstrateFile, G, R, nx, ny, nz, FractSurfaceSitesActive, PathToOutput,
                  NSpotsX, NSpotsY, SpotOffset, SpotRadius, OutputFile, InitTime, RunTime, OutTime, cycle, InitMaxTime,
                  InitMinTime, NuclMaxTime, NuclMinTime, CaptureMaxTime, CaptureMinTime, GhostMaxTime, GhostMinTime,
                  OutMaxTime, OutMinTime);
}
