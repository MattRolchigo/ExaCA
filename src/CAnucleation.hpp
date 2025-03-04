// Copyright Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef EXACA_NUCLEATION_HPP
#define EXACA_NUCLEATION_HPP

#include "CAcelldata.hpp"
#include "CAgrid.hpp"
#include "CAinputs.hpp"
#include "CAinterface.hpp"
#include "CAtemperature.hpp"

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

// Data regarding nucleation events in the domain
template <typename MemorySpace>
struct Nucleation {

    using memory_space = MemorySpace;
    using view_type_int = Kokkos::View<int *, memory_space>;
    using view_type_int_host = typename view_type_int::HostMirror;
    using view_type_float = Kokkos::View<float *, memory_space>;
    using view_type_float_host = typename view_type_float::HostMirror;

    // Using the default exec space for this memory space.
    using execution_space = typename memory_space::execution_space;

    // Four counters tracked here:
    // 1. nuclei_whole_domain - tracks all nuclei (regardless of whether an event would be possible based on the layer
    // ID and cell type), used for Grain ID assignment to ensure that no Grain ID get reused - same on all MPI ranks
    // 2. possible_nuclei - the subset of Nuclei_ThisLayer that are located within the bounds of a
    // given MPI rank that may possibly occur (nuclei locations are associated with a liquid cell with a layer ID that
    // matches this layer number). Starts at 0 each layer
    // 3. nucleation_counter - the number of nucleation events that have actually either failed or succeeded on a given
    // MPI rank. Starts at 0 each layer
    // 4. successful_nucleation_counter - the number of nucleation events that have successfully occurred at a given
    // point in the simulation of a layer. Starts at 0 each layer.
    int nuclei_whole_domain, possible_nuclei, nucleation_counter;
    double d_exclusion;
    int d_exclusion_round;
    view_type_int successful_nucleation_counter;

    // The time steps at which nucleation events will occur in the given layer, on the host
    view_type_int_host nucleation_times_host;
    // The locations, initial octahedron sizes, and grain IDs of the potential nuclei in the layer
    view_type_int nuclei_locations;
    view_type_float nuclei_oct_sizes;
    view_type_int nuclei_grain_id;
    // Nucleation inputs from file
    NucleationInputs _inputs;

    // Constructor - initialize CA views using input for initial guess at number of possible events
    // Default is that no nucleation has occurred to this point, optional input argument to start with the counter at a
    // specific value
    Nucleation(const int estimated_nuclei_this_rank_this_layer, const NucleationInputs inputs,
               const int num_prior_nuclei = 0)
        : successful_nucleation_counter(view_type_int(Kokkos::ViewAllocateWithoutInitializing("NucleiLocations"), 1))
        , nucleation_times_host(view_type_int_host(Kokkos::ViewAllocateWithoutInitializing("NucleationTimes_Host"),
                                                   estimated_nuclei_this_rank_this_layer))
        , nuclei_locations(view_type_int(Kokkos::ViewAllocateWithoutInitializing("NucleiLocations"),
                                         estimated_nuclei_this_rank_this_layer))
        , nuclei_oct_sizes(view_type_float(Kokkos::ViewAllocateWithoutInitializing("NucleiOctSize"),
                                     estimated_nuclei_this_rank_this_layer))
        , nuclei_grain_id(view_type_int(Kokkos::ViewAllocateWithoutInitializing("NucleiGrainID"),
                                        estimated_nuclei_this_rank_this_layer))
        , _inputs(inputs) {
        nuclei_whole_domain = num_prior_nuclei;
        resetNucleiCounters(); // start counters at 0
    }

    // Reset the nuclei counters prior to nuclei initialization for a layer
    void resetNucleiCounters() {
        // Init appropriate counters for the layer to 0 - possible nuclei for this layer will be calculated in this
        // function
        possible_nuclei = 0;
        nucleation_counter = 0;
        Kokkos::deep_copy(successful_nucleation_counter, 0);
    }

    // Initialize nucleation site locations, GrainID values, and time at which nucleation events will potentially occur,
    // accounting for multiple possible nucleation events in cells that melt and solidify multiple times
    template <class... Params>
    void placeNuclei(const Temperature<memory_space> &temperature, Interface<memory_space> &interface, Orientation<memory_space> &orientation, const unsigned long rng_seed, const int layernumber,
                     const Grid &grid, const InterfacialResponseFunction &irf, const int id) {

        // TODO: convert this subroutine into kokkos kernels, rather than copying data back to the host, and nucleation
        // data back to the device again. This is currently performed on the device due to heavy usage of standard
        // library algorithm functions Copy temperature data into temporary host views for this subroutine
        auto max_solidification_events_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), temperature.max_solidification_events);
//        auto number_of_solidification_events_host =
//            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), temperature.number_of_solidification_events);
//        auto liquidus_time_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), temperature.liquidus_time);
//        auto cooling_rate_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), temperature.cooling_rate);
//        auto grain_unit_vector_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), orientation.grain_unit_vector);

        // Threshold for nucleated grains
        const float misorientation_angle_threshold = 59.0; //20.0;
        // Region around a nucleus that must be clear of non-liquid cells for it to form
        d_exclusion = 1.5 * pow(10,-6) / grid.deltax;
        float _d_exclusion = d_exclusion;
        d_exclusion_round = static_cast<int>(Kokkos::ceil(d_exclusion));
        // Use new RNG seed for each layer
        std::mt19937_64 generator(rng_seed + static_cast<unsigned long>(layernumber));
        // Uniform distributions for nuclei location assignment - associate each nucleation event with an XYZ coordinate
        // in meters
        std::uniform_real_distribution<double> nucleation_site_dist_x(grid.x_min, grid.x_max);
        std::uniform_real_distribution<double> nucleation_site_dist_y(grid.y_min, grid.y_max);
        std::uniform_real_distribution<double> nucleation_site_dist_z(grid.z_min_layer(layernumber),
                                                                      grid.z_max_layer(layernumber));
        // Gaussian distribution of nucleation undercooling
        std::normal_distribution<double> nucleation_undercooling_dist(_inputs.dtn, _inputs.dtsigma);

        // Max number of nucleated grains in this layer
        // Use long int in intermediate steps calculating the number of nucleated grains, though the number should be
        // small enough to be stored as an int
        const double domain_volume = (grid.x_max - grid.x_min) * (grid.y_max - grid.y_min) *
                                     (grid.z_max_layer(layernumber) - grid.z_min_layer(layernumber));
        // If each cell underwent solidification 1x, the number of potential nuclei in the layer
        const long int nuclei_this_layer_single_long = std::lround(_inputs.n_max * domain_volume);
        // Multiplier for the number of nucleation events per layer, based on the max number of solidification events
        long int nuclei_multiplier_long = static_cast<long int>(max_solidification_events_host(layernumber));
        long int nuclei_this_layer_long = nuclei_this_layer_single_long * nuclei_multiplier_long;
        // Vector and view sizes should be type int for indexing and grain ID assignment purposes -
        // Nuclei_ThisLayer_long should be less than INT_MAX
        if (nuclei_this_layer_long > INT_MAX)
            throw std::runtime_error("Error: Number of potential nucleation sites in the system exceeds the number of "
                                     "valid GrainID; either nucleation density, the number of melt-solidification "
                                     "events in the temperature data, or the domain size should be reduced");
        int nuclei_this_layer_single = static_cast<int>(nuclei_this_layer_single_long);
        int nuclei_this_layer = static_cast<int>(nuclei_this_layer_long);
        int nuclei_multiplier = static_cast<int>(nuclei_multiplier_long);

        // Nuclei Grain ID are assigned to avoid reusing values from previous layers
        std::vector<int> nuclei_grain_id_whole_domain_v(nuclei_this_layer);
        std::vector<double> nuclei_undercooling_whole_domain_v(nuclei_this_layer);
        // Views for storing potential nucleated grain coordinates
        view_type_int_host nuclei_x_whole_domain_host(Kokkos::ViewAllocateWithoutInitializing("NucleiX"), nuclei_this_layer);
        view_type_int_host nuclei_y_whole_domain_host(Kokkos::ViewAllocateWithoutInitializing("NucleiY"), nuclei_this_layer);
        view_type_int_host nuclei_z_whole_domain_host(Kokkos::ViewAllocateWithoutInitializing("NucleiZ"), nuclei_this_layer);

        for (int meltevent = 0; meltevent < nuclei_multiplier; meltevent++) {
            for (int n = 0; n < nuclei_this_layer_single; n++) {
                int n_event = meltevent * nuclei_this_layer_single + n;
                // Generate possible nuclei locations
                double nuclei_x_unrounded = nucleation_site_dist_x(generator);
                double nuclei_y_unrounded = nucleation_site_dist_y(generator);
                double nuclei_z_unrounded = nucleation_site_dist_z(generator);
                // Associate these locations with a specific cell on the grid
                nuclei_x_whole_domain_host(n_event) = Kokkos::round((nuclei_x_unrounded - grid.x_min) / grid.deltax);
                nuclei_y_whole_domain_host(n_event) = Kokkos::round((nuclei_y_unrounded - grid.y_min) / grid.deltax);
                nuclei_z_whole_domain_host(n_event) = Kokkos::round((nuclei_z_unrounded - grid.z_min_layer[layernumber]) / grid.deltax);
                // Assign each nuclei a Grain ID (negative values used for nucleated grains) and an undercooling
                nuclei_grain_id_whole_domain_v[n_event] =
                    -(nuclei_whole_domain + n_event + 1); // avoid using grain ID 0
                nuclei_undercooling_whole_domain_v[n_event] = nucleation_undercooling_dist(generator);
            }
        }

        // Shuffle these vectors to make sure the same grain IDs and undercooling don't end up in the same spots each
        // layer
        std::shuffle(nuclei_grain_id_whole_domain_v.begin(), nuclei_grain_id_whole_domain_v.end(), generator);
        std::shuffle(nuclei_undercooling_whole_domain_v.begin(), nuclei_undercooling_whole_domain_v.end(), generator);
        if ((id == 0) && (nuclei_this_layer > 0))
            std::cout << "Range of Grain IDs from which layer " << layernumber << " nucleation events were selected: -"
                      << nuclei_whole_domain + 1 << " through -" << nuclei_whole_domain + nuclei_this_layer
                      << std::endl;
        // Update number of nuclei counter for whole domain based on the number of nuclei in this layer
        nuclei_whole_domain += nuclei_this_layer;
        
        view_type_float_host misorientation_z = orientation.misorientationCalc(2);

        // Create temporary views to store nucleation locations, grain ID data initialized on the host - estimating the number of possible nuclei this layer
        // nucleation_times_host are stored using a host view that is passed to the nucleateGrain subroutine and later
        // used
        view_type_int_host nuclei_grain_id_whole_domain_host(Kokkos::ViewAllocateWithoutInitializing("NucleiGrainIDWD_Host"), nuclei_this_layer);
        view_type_float_host nuclei_undercooling_whole_domain_host(Kokkos::ViewAllocateWithoutInitializing("NucleiUndercoolingWD_Host"), nuclei_this_layer);
        for (int n=0; n<nuclei_this_layer; n++) {
            nuclei_grain_id_whole_domain_host(n) = nuclei_grain_id_whole_domain_v[n];
            nuclei_undercooling_whole_domain_host(n) = nuclei_undercooling_whole_domain_v[n];
        }

        // Device views - some are copied from host, others are filled in the following parallel_for. Local copies of views stored in class for lambda capture
        auto nuclei_x_whole_domain = Kokkos::create_mirror_view_and_copy(memory_space(), nuclei_x_whole_domain_host);
        auto nuclei_y_whole_domain = Kokkos::create_mirror_view_and_copy(memory_space(), nuclei_y_whole_domain_host);
        auto nuclei_z_whole_domain = Kokkos::create_mirror_view_and_copy(memory_space(), nuclei_z_whole_domain_host);
        auto nuclei_undercooling_whole_domain = Kokkos::create_mirror_view_and_copy(memory_space(), nuclei_undercooling_whole_domain_host);
        auto nuclei_grain_id_whole_domain = Kokkos::create_mirror_view_and_copy(memory_space(), nuclei_grain_id_whole_domain_host);
        // Views for storing nucleation data on this rank only - use nuclei_this_layer as an initial max size
        Kokkos::resize(nuclei_locations, nuclei_this_layer);
        Kokkos::resize(nuclei_oct_sizes, nuclei_this_layer);
        Kokkos::resize(nuclei_grain_id, nuclei_this_layer);
        auto _nuclei_locations = nuclei_locations;
        auto _nuclei_oct_sizes = nuclei_oct_sizes;
        auto _nuclei_grain_id = nuclei_grain_id;
        view_type_int nucleation_times(Kokkos::ViewAllocateWithoutInitializing("NucleationTimes"), nuclei_this_layer);
        view_type_int possible_nuclei_view("PossibleNuclei", 1);

        // Loop through nuclei for this layer - each MPI rank storing the nucleation events that are possible (i.e,
        // nucleation event is associated with a CA cell on that MPI rank's subdomain, the cell is liquid type, and the
        // cell is associated with the current layer of the multilayer problem) Don't put nuclei in "ghost" cells -
        // those nucleation events occur on other ranks and the existing halo exchange functionality will handle this
        Kokkos::parallel_for(
            "PlaceNuclei", nuclei_this_layer_single, KOKKOS_LAMBDA(const int &n) {
                for (int meltevent = 0; meltevent < nuclei_multiplier; meltevent++) {
                    // event "n_event" is somewhere in the simulation domain - only consider storing it on this rank if within the local Y bounds
                    int n_event = meltevent * nuclei_this_layer_single + n;
                    if (((nuclei_y_whole_domain(n_event) > grid.y_offset) || (grid.at_south_boundary)) &&
                        ((nuclei_y_whole_domain(n_event) < grid.y_offset + grid.ny_local - 1) || (grid.at_north_boundary))) {
                        // Convert 3D location (using global X and Y coordinates) into a 1D location (using local X and Y
                        // coordinates) for the possible nucleation event, both as relative to the bottom of this layer
                        int nucleus_location_this_layer =
                        grid.get1DIndex(nuclei_x_whole_domain(n_event), nuclei_y_whole_domain(n_event) - grid.y_offset, nuclei_z_whole_domain(n_event));
                        // Criteria for placing a nucleus - whether or not this nuclei is associated with a solidification
                        // event
                        if (meltevent < temperature.number_of_solidification_events(nucleus_location_this_layer)) {
                            // Increment counter on this MPI rank
                            const int possible_nuclei_local = Kokkos::atomic_fetch_add(&possible_nuclei_view(0), 1);
                            // Nucleation event is possible - cell undergoes solidification at least once, this nucleation
                            // event is associated with one of the time periods during which the associated cell undergoes
                            // solidification
                            double liq_time_this_event =
                            static_cast<double>(temperature.liquidus_time(nucleus_location_this_layer, meltevent, 1));
                            double cooling_rate_this_event =
                            static_cast<double>(temperature.cooling_rate(nucleus_location_this_layer, meltevent));
                            double time_to_nuc_und =
                            liq_time_this_event + nuclei_undercooling_whole_domain(n_event) / cooling_rate_this_event;
                            if (liq_time_this_event > time_to_nuc_und)
                                time_to_nuc_und = liq_time_this_event;
                            // At the time of the nucleation event, what is the coldest neighboring cell? Defaults to current cell location for edge case where no neighboring cells are colder
//                            int coldest_neighbor_location = nucleus_location_this_layer;
//                            int coldest_neighbor_index = -1;
//                            float coldest_neighbor_undercooling = nuclei_undercooling_whole_domain(n_event);
//                            float coldest_neighbor_cooling_rate = cooling_rate_this_event;
                            float liquidus_time_neighbors[6];
                            bool liquidus_times_exist = true;
                            for (int l = 0; l < 6; l++) {
                                // Local coordinates of adjacent cell center
                                const int neighbor_coord_x = nuclei_x_whole_domain(n_event) + interface.neighbor_x[l];
                                const int neighbor_coord_y = nuclei_y_whole_domain(n_event) - grid.y_offset + interface.neighbor_y[l];
                                const int neighbor_coord_z = nuclei_z_whole_domain(n_event) + interface.neighbor_z[l];
                                // Check if neighbor is in bounds
                                const int neighbor_index =
                                grid.getNeighbor1DIndex(neighbor_coord_x, neighbor_coord_y, neighbor_coord_z);
                                if (neighbor_index != -1) {
                                    // Find liquidus time for neighbor that is the closest to "time_to_nuc_und"
                                    bool liquidus_time_l_exists = false;
                                    const int num_solidification_events_neighbor = temperature.number_of_solidification_events(neighbor_index);
                                    for (int melt_event_neighbor=0; melt_event_neighbor<num_solidification_events_neighbor; melt_event_neighbor++) {
                                        float liquidus_time_neighbor = temperature.liquidus_time(neighbor_index,melt_event_neighbor,1);
                                        if (Kokkos::abs(liq_time_this_event - liquidus_time_neighbor) <= 2000) {
                                            liquidus_time_neighbors[l] = liquidus_time_neighbor;
                                            liquidus_time_l_exists = true;
                                        }
                                    }
                                    if (!liquidus_time_l_exists) {
                                        liquidus_times_exist = false;
                                        l = 6;
                                    }
                                }
                                else {
                                    liquidus_times_exist = false;
                                    l = 6;
                                }
                            }
                            // Nucleation data stored on this MPI rank (index views using the atomically incremented nucleation counter "possible_nuclei_local")
                            // What size does this nucleated grain need to reach to extend a distance "diag_nucleation" in the direction of the liquidus time gradient? If no liquidus time gradient (at a melt pool edge), set the nucleation undercooling to the value calculated from the Gaussian distribution and the initial size to the default value
                           if (liquidus_times_exist) {
                               const float grad_x = (liquidus_time_neighbors[0] - liquidus_time_neighbors[3]) / 2.0;
                               const float grad_y = (liquidus_time_neighbors[1] - liquidus_time_neighbors[4]) / 2.0;
                               const float grad_z = (liquidus_time_neighbors[2] - liquidus_time_neighbors[5]) / 2.0;
                               const float grad_mag = Kokkos::sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z);
                               // Unit vector in direction of liquidus time gradient
                               const float x_capt = grad_x / grad_mag;
                               const float y_capt = grad_y / grad_mag;
                               const float z_capt = grad_z / grad_mag;
//                               printf("Cell w/ melt event %d at %d, %d, %d has liquidus time gradient in direction %f, %f, %f\n",meltevent,nuclei_x_whole_domain(n_event),nuclei_y_whole_domain(n_event),nuclei_z_whole_domain(n_event),x_capt,y_capt,z_capt);
                               const int my_orientation = getGrainOrientation(nuclei_grain_id_whole_domain(n_event), orientation.n_grain_orientations);
                               // x_dist, y_dist, z_dist represents a location d_exclusion cell lengths away in the negative liquidus time gradient direction
//                               const float x_dist = - x_capt * _d_exclusion;
//                               const float y_dist = - y_capt * _d_exclusion;
//                               const float z_dist = - z_capt * _d_exclusion;
                               // Misorientation between the thermal gradient direction and this grain's closest oriented <100>
                               float misorientation_angle_min = 54.7356;
                               for (int ll = 0; ll < 3; ll++) {
                                   float misorientation =
                                       Kokkos::abs((180 / M_PI) * Kokkos::acos(Kokkos::abs(orientation.grain_unit_vector(9 * my_orientation + 3 * ll) * x_capt + orientation.grain_unit_vector(9 * my_orientation + 3 * ll + 1) * y_capt + orientation.grain_unit_vector(9 * my_orientation + 3 * ll + 2) * z_capt)));
                                   if (misorientation < misorientation_angle_min) {
                                       misorientation_angle_min = misorientation;
                                   }
                               }
                               // Octahedron size needed to capture a cell at location x_dist, y_dist, z_dist
                               //const float capture_size = interface.calcCritDiagonalLength(x_dist, y_dist, z_dist, my_orientation, orientation.grain_unit_vector);
                               _nuclei_oct_sizes(possible_nuclei_local) = interface._init_oct_size; //capture_size;
                               // What time is needed to reach this critical size, based on the nucleation undercooling at the cooling rate of the cell? Use the estimated undercooling at x_dist, y_dist, z_dist and the local undercooling, but only use the local cooling rate to avoid noise
                               //float undercooling_cell = nuclei_undercooling_whole_domain(n_event);
                               //float undercooling_xyzdist = nuclei_undercooling_whole_domain(n_event) + _d_exclusion * grad_mag;
                               //float cooling_rate_cell = cooling_rate_this_event;
                               if (misorientation_angle_min < misorientation_angle_threshold)
                                   nucleation_times(possible_nuclei_local) = time_to_nuc_und;
                               else
                                   nucleation_times(possible_nuclei_local) = 900000000;
                               //interface.getCaptureTime(undercooling_cell, undercooling_xyzdist, time_to_nuc_und, cooling_rate_cell, capture_size, irf);
                               //printf("Nucleus at %d, %d, %d, Capture size %f, angle z %f, growth lag %f\n",nuclei_x_whole_domain(n_event),nuclei_y_whole_domain(n_event),nuclei_z_whole_domain(n_event),capture_size,misorientation_z(my_orientation),nucleation_times(possible_nuclei_local)-time_to_nuc_und);

                           }
                           else {
                               // Nucleation happens as normal
                               _nuclei_oct_sizes(possible_nuclei_local) = interface._init_oct_size;
                               nucleation_times(possible_nuclei_local) = time_to_nuc_und;
                           }
                            // Assign the nucleation event a grain ID
                            _nuclei_grain_id(possible_nuclei_local) = nuclei_grain_id_whole_domain(n_event);
                            // Assign the nucleation event to a cell
                            _nuclei_locations(possible_nuclei_local) = nucleus_location_this_layer;
                        }
                    }
                }
        });

        // How many nucleation events are actually possible (associated with a cell in this layer that will undergo
        // solidification)?
        auto possible_nuclei_view_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), possible_nuclei_view);
        possible_nuclei = possible_nuclei_view_host(0);
        int possible_nuclei_all_ranks_this_layer;
        MPI_Reduce(&possible_nuclei, &possible_nuclei_all_ranks_this_layer, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (id == 0)
            std::cout << "Number of potential nucleation events in layer " << layernumber << " : "
                      << possible_nuclei_all_ranks_this_layer << std::endl;

        // Now that the number of nucleation events on each rank is known, resize the device views and copy back to host vectors
        Kokkos::resize(nuclei_locations, possible_nuclei);
        Kokkos::resize(nuclei_grain_id, possible_nuclei);
        Kokkos::resize(nucleation_times, possible_nuclei);
        Kokkos::resize(nuclei_oct_sizes, possible_nuclei);
        auto nuclei_locations_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nuclei_locations);
        auto nuclei_oct_sizes_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nuclei_oct_sizes);
        nucleation_times_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nucleation_times);
        auto nuclei_grain_id_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nuclei_grain_id);

        std::vector<int> nuclei_grain_id_myrank_v(possible_nuclei), nuclei_locations_myrank_v(possible_nuclei);
        // Store nucleation times as doubles to correctly order events in times, then convert to time steps later
        std::vector<double> nucleation_times_myrank_v(possible_nuclei);
        // Store initial octahedron sizes in cases of successful nucleation events
        std::vector<float> nuclei_oct_sizes_myrank_v(possible_nuclei);

        for (int n=0; n<possible_nuclei; n++) {
            nuclei_grain_id_myrank_v[n] = nuclei_grain_id_host(n);
            nuclei_locations_myrank_v[n] = nuclei_locations_host(n);
            nuclei_oct_sizes_myrank_v[n] = nuclei_oct_sizes_host(n);
            nucleation_times_myrank_v[n] = nucleation_times_host(n);
        }

        // Sort the list of time steps at which nucleation occurs, keeping the time steps paired with the corresponding
        // locations for nucleation events and grain IDs
        std::vector<std::tuple<double, int, int, float>> nucleation_time_loc_id;
        nucleation_time_loc_id.reserve(possible_nuclei);
        for (int n = 0; n < possible_nuclei; n++) {
            nucleation_time_loc_id.push_back(std::make_tuple(nucleation_times_myrank_v[n], nuclei_locations_myrank_v[n],
                                                             nuclei_grain_id_myrank_v[n], nuclei_oct_sizes_myrank_v[n]));
        }
        // Sorting from low to high
        std::sort(nucleation_time_loc_id.begin(), nucleation_time_loc_id.end());

        // Copy back to host views
        for (int n = 0; n < possible_nuclei; n++) {
            double nucleation_time = std::get<0>(nucleation_time_loc_id[n]);
            // Convert nucleation time to a time step
            nucleation_times_host(n) = Kokkos::round(nucleation_time);
            nuclei_locations_host(n) = std::get<1>(nucleation_time_loc_id[n]);
            nuclei_grain_id_host(n) = std::get<2>(nucleation_time_loc_id[n]);
            nuclei_oct_sizes_host(n) = std::get<3>(nucleation_time_loc_id[n]);
        }
        // Copy nucleation data to the device
        nuclei_locations = Kokkos::create_mirror_view_and_copy(memory_space(), nuclei_locations_host);
        nuclei_grain_id = Kokkos::create_mirror_view_and_copy(memory_space(), nuclei_grain_id_host);
        nuclei_oct_sizes = Kokkos::create_mirror_view_and_copy(memory_space(), nuclei_oct_sizes_host);
        MPI_Barrier(MPI_COMM_WORLD);
        if (id == 0)
            std::cout << "Nuclei initialized" << std::endl;
    }

    // Check for nucleation events on this time step, updating the corresponding cell appropriately for any successful
    // nucleation event
    void nucleateGrain(const int cycle, const Grid &grid, CellData<memory_space> &celldata,
                       Interface<memory_space> interface) {

        auto grain_id = celldata.getGrainIDSubview(grid);

        // Is there nucleation left in this layer to check?
        if (nucleation_counter < possible_nuclei) {
            // Is there at least one potential nucleation event on this rank, at this time step?
            if (cycle == nucleation_times_host(nucleation_counter)) {
                bool nucleation_check = true;
                int first_event = nucleation_counter; // first potential nucleation event to check
                // Are there any other nucleation events this time step to check?
                while (nucleation_check) {
                    nucleation_counter++;
                    // If the previous nucleation event was the last one for this layer of the simulation, exit loop
                    if (nucleation_counter == possible_nuclei)
                        break;
                    // If the next nucleation event corresponds to a future time step, finish check
                    if (cycle != nucleation_times_host(nucleation_counter))
                        nucleation_check = false;
                }
                int last_event = nucleation_counter;
                // parallel_for checks each potential nucleation event this time step (first_event, up to but not
                // including last_event)
                auto nuclei_locations_local = nuclei_locations;
                auto nuclei_grain_id_local = nuclei_grain_id;

                // Launch kokkos kernel - check if the corresponding CA cell location is liquid
                auto policy = Kokkos::RangePolicy<execution_space>(first_event, last_event);
                Kokkos::parallel_for("NucleiUpdateLoop", policy,
                    KOKKOS_LAMBDA(const int nucleation_counter_device) {
                        int nucleation_event_location = nuclei_locations_local(nucleation_counter_device);
                        // Whether or not this nucleation event is successful depends on the cells around it - checked as part of cellCapture loop over steering vector
                        int update_val = FutureActive;
                        int old_val = Liquid;
                        int old_cell_type_value = Kokkos::atomic_compare_exchange(
                            &celldata.cell_type(nucleation_event_location), old_val, update_val);
                        if (old_cell_type_value == Liquid) {
                            // Successful nucleation event - atomic update of cell type, proceeded if the atomic
                            // exchange is successful (cell was liquid) Add future active cell location to steering
                            // vector, temporarily store nucleation event number in grain ID view for later reference
                            Kokkos::atomic_fetch_add(&successful_nucleation_counter(0), 1);
                            grain_id(nucleation_event_location) = nuclei_grain_id(nucleation_counter_device);
                            interface.steering_vector(Kokkos::atomic_fetch_add(&interface.num_steer(0), 1)) =
                                nucleation_event_location;
                            
//                           printf("Potential nucleated grain at location %d, ct %d, id %d\n",nucleation_event_location,nucleation_counter_device,nuclei_grain_id_local(nucleation_counter_device));
                        }
                    });
            }
        }
    }
    
    // Get value of counter for this MPI rank
    int getSuccessfulNucleationCounter() {
        auto successful_nucleation_counter_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), successful_nucleation_counter);
        const int successful_nuc_events_this_rank = successful_nucleation_counter_host(0);
        return successful_nuc_events_this_rank;
    }
};

#endif
