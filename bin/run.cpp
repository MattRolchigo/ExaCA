// Copyright Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "ExaCA.hpp"

#include <Kokkos_Core.hpp>

#include "mpi.h"

#include <stdexcept>
#include <string>

int main(int argc, char *argv[]) {
    // Initialize MPI
    int id, np;
    MPI_Init(&argc, &argv);
    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
    {
        using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

        // Get number of processes
        MPI_Comm_size(MPI_COMM_WORLD, &np);
        // Get individual process ID
        MPI_Comm_rank(MPI_COMM_WORLD, &id);

        if (id == 0) {
            std::cout << "ExaCA version: " << version() << " \nExaCA commit:  " << gitCommitHash()
                      << "\nKokkos version: " << kokkosVersion() << std::endl;
            Kokkos::DefaultExecutionSpace().print_configuration(std::cout);
            std::cout << "Number of MPI ranks = " << np << std::endl;
        }
        if (argc < 2) {
            throw std::runtime_error("Error: Must provide path to input file on the command line.");
        }
        else {

            // Create timers
            Timers timers(id);
            timers.startInit();

            // Run CA code using reduced temperature data format
            std::string input_file = argv[1];
            // Read input file
            Inputs inputs(id, input_file);

            // Setup local and global grids, decomposing domain (needed to construct temperature)
            Grid grid(inputs.simulation_type, id, np, inputs.domain.number_of_layers, inputs.domain,
                      inputs.temperature);
            // Temperature fields characterized by data in this structure
            Temperature<memory_space> temperature;

            CellData<memory_space> celldata(grid.domain_size, grid.domain_size_all_layers, inputs.substrate);
            int baseplate_size_z = celldata.getBaseplateSizeZ(id, grid);
            celldata.initBaseplateGrainID(id, grid, inputs.rng_seed, baseplate_size_z);
            
            // Initialize printing struct from inputs
            Print print(grid, np, inputs.print);
            
            std::string vtk_filename = "FinchExaCASubstrate.vtk";
            std::ofstream subfilestream;
            print.writeHeader(subfilestream, vtk_filename, grid, true);
            auto grain_id_whole_domain = print.collectViewData(id, np, grid, false, MPI_SHORT, celldata.grain_id_all_layers);
            print.printViewData(id, subfilestream, grid, false, "int", "GrainID",
                                grain_id_whole_domain);
            subfilestream.close();
        }
    }
    // Finalize Kokkos
    Kokkos::finalize();
    // Finalize MPI
    MPI_Finalize();
    return 0;
}
