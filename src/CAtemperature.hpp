// Copyright 2021-2023 Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef EXACA_TEMPS_HPP
#define EXACA_TEMPS_HPP

#include "CAgrid.hpp"
#include "CAinputs.hpp"
#include "CAparsefiles.hpp"
#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Reduced form of the time-temperature history and temperature field variables used by ExaCA
template <typename MemorySpace>
struct Temperature {

    using memory_space = MemorySpace;
    using view_type_int = Kokkos::View<int *, memory_space>;
    using view_type_float = Kokkos::View<float *, memory_space>;
    using view_type_double = Kokkos::View<double *, memory_space>;
    using view_type_double_2d = Kokkos::View<double **, memory_space>;
    using view_type_float_2d = Kokkos::View<float **, memory_space>;
    using view_type_int_host = typename view_type_int::HostMirror;
    using view_type_float_host = typename view_type_float::HostMirror;
    using view_type_double_host = typename view_type_double::HostMirror;
    using view_type_double_2d_host = typename view_type_double_2d::HostMirror;
    using view_type_float_2d_host = typename view_type_float_2d::HostMirror;
    using view_type_coupled = Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace>;

    // Using the default exec space for this memory space.
    using execution_space = typename memory_space::execution_space;

    // Maximum number of times a cell on this layer undergoes melting/solidification
    int max_num_solidification_events = 1;
    // Number of times a given cell melts/solidifies (stored on host only)
    view_type_int_host number_of_solidification_events;
    // Index in the list of events associated with the current solidification event occuring in a given cell
    view_type_int current_solidification_event;
    // Index in the list of events associated with the last solidification of a given cell
    view_type_int last_solidification_event;
    // List of melt-solidification events, each with a tm (melting time), tl (time cell cools below liquidus), and cr
    // (cooling rate from liquidus temperature) value
    view_type_float_2d layer_time_temp_history;
    // The current undercooling of a CA cell (if superheated liquid or hasn't undergone solidification yet, equals 0)
    // Also maintained for the full multilayer domain
    view_type_float undercooling_current_all_layers, undercooling_current;
    // Data structure for storing raw temperature data from file(s)
    // Store data as double - needed for small time steps to resolve local differences in solidification conditions
    // Each data point has 6 values (X, Y, Z coordinates, melting time, liquidus time, and cooling rate)
    const int num_temperature_components = 6;
    view_type_double_2d_host raw_temperature_data;
    // These contain "number_of_layers" values corresponding to the location within "raw_temperature_data" of the first
    // data element in each temperature file, if used
    view_type_int_host first_value, last_value;
    // Undercooling when a solidification first began in a cell (optional to store based on selected inputs)
    bool _store_solidification_start;
    view_type_float undercooling_solidification_start_all_layers, undercooling_solidification_start;
    // Temperature field inputs from file
    TemperatureInputs _inputs;

    // Constructor creates views with size based on the grid inputs - each cell assumed to solidify once by default,
    // layer_time_temp_history modified to account for multiple events if needed undercooling_current and
    // solidification_event_counter are default initialized to zeros
    Temperature(const Grid &grid, TemperatureInputs inputs, const bool store_solidification_start = false)
        : number_of_solidification_events(view_type_int_host("number_of_solidification_events", grid.domain_size))
        , current_solidification_event(
              view_type_int(Kokkos::ViewAllocateWithoutInitializing("current_solidification_event"), grid.domain_size))
        , last_solidification_event(
              view_type_int(Kokkos::ViewAllocateWithoutInitializing("last_solidification_event"), grid.domain_size))
        , layer_time_temp_history(view_type_float_2d(Kokkos::ViewAllocateWithoutInitializing("layer_time_temp_history"),
                                                     grid.domain_size, 3))
        , undercooling_current_all_layers(view_type_float("undercooling_current", grid.domain_size_all_layers))
        , raw_temperature_data(view_type_double_2d_host(Kokkos::ViewAllocateWithoutInitializing("raw_temperature_data"),
                                                        grid.domain_size, num_temperature_components))
        , first_value(view_type_int_host(Kokkos::ViewAllocateWithoutInitializing("first_value"), grid.number_of_layers))
        , last_value(view_type_int_host(Kokkos::ViewAllocateWithoutInitializing("last_value"), grid.number_of_layers))
        , _store_solidification_start(store_solidification_start)
        , _inputs(inputs) {

        if (_store_solidification_start) {
            // Default init starting undercooling in cells to zero
            undercooling_solidification_start_all_layers =
                view_type_float("undercooling_solidification_start", grid.domain_size_all_layers);
            getCurrentLayerStartingUndercooling(grid.layer_range);
        }
        getCurrentLayerUndercooling(grid.layer_range);
    }

    // Constructor using in-memory temperature data from external source.
    Temperature(const int id, const int np, const Grid &grid, TemperatureInputs inputs,
                view_type_coupled input_temperature_data, const bool store_solidification_start = false)
        : number_of_solidification_events(view_type_int_host("number_of_solidification_events", grid.domain_size))
        , current_solidification_event(
              view_type_int(Kokkos::ViewAllocateWithoutInitializing("current_solidification_event"), grid.domain_size))
        , last_solidification_event(
              view_type_int(Kokkos::ViewAllocateWithoutInitializing("last_solidification_event"), grid.domain_size))
        , layer_time_temp_history(view_type_float_2d(Kokkos::ViewAllocateWithoutInitializing("layer_time_temp_history"),
                                                     grid.domain_size, 3))
        , undercooling_current_all_layers(view_type_float("undercooling_current", grid.domain_size_all_layers))
        , raw_temperature_data(view_type_double_2d_host(Kokkos::ViewAllocateWithoutInitializing("raw_temperature_data"),
                                                        grid.domain_size, num_temperature_components))
        , first_value(view_type_int_host(Kokkos::ViewAllocateWithoutInitializing("first_value"), grid.number_of_layers))
        , last_value(view_type_int_host(Kokkos::ViewAllocateWithoutInitializing("last_value"), grid.number_of_layers))
        , _store_solidification_start(store_solidification_start)
        , _inputs(inputs) {

        copyTemperatureData(id, np, grid, input_temperature_data);
        if (_store_solidification_start) {
            // Default init starting undercooling in cells to zero
            undercooling_solidification_start_all_layers =
                view_type_float("undercooling_solidification_start", grid.domain_size_all_layers);
            getCurrentLayerStartingUndercooling(grid.layer_range);
        }
        getCurrentLayerUndercooling(grid.layer_range);
    }

    // Copy data from external source onto the appropriate MPI ranks
    void copyTemperatureData(const int id, const int np, const Grid &grid, view_type_coupled input_temperature_data,
                             const int resize_padding = 100) {

        // Take first num_temperature_components columns of input_temperature_data
        int finch_data_size = input_temperature_data.extent(0);
        int finch_temp_components = input_temperature_data.extent(1);
        std::cout << "Rank " << id << " has " << finch_data_size << " events from the Finch simulation" << std::endl;

        // First, store data with Y coordinates in bounds for this rank in raw_temperature_data
        int temperature_point_counter = 0;
        for (int n = 0; n < finch_data_size; n++) {
            int coord_y = (input_temperature_data(n, 1) - grid.y_min) / grid.deltax;
            if ((coord_y >= grid.y_offset) && (coord_y < grid.y_offset + grid.ny_local)) {
                for (int comp = 0; comp < num_temperature_components; comp++)
                    raw_temperature_data(temperature_point_counter, comp) = input_temperature_data(n, comp);
                // Increment counter for each point stored on this rank
                temperature_point_counter++;
            }
        }

        // Communication pattern - sending to right, receiving from left
        int left, right;
        if (id == 0)
            left = np - 1;
        else
            left = id - 1;
        if (id == np - 1)
            right = 0;
        else
            right = id + 1;

        // Send and recieve data so each rank parses all finch data points
        int send_data_size = finch_data_size;
        for (int i = 0; i < np - 1; i++) {
            // Get size for sending/receiving
            int recv_data_size;
            MPI_Request send_request_size, recv_request_size;
            MPI_Isend(&send_data_size, 1, MPI_INT, right, 0, MPI_COMM_WORLD, &send_request_size);
            MPI_Irecv(&recv_data_size, 1, MPI_INT, left, 0, MPI_COMM_WORLD, &recv_request_size);
            MPI_Wait(&send_request_size, MPI_STATUS_IGNORE);
            MPI_Wait(&recv_request_size, MPI_STATUS_IGNORE);
            // Allocate view for received data
            view_type_coupled finch_data_recv(Kokkos::ViewAllocateWithoutInitializing("finch_data_recv"),
                                              recv_data_size, finch_temp_components);

            // Send data to the right, recieve data from the left - if needed, increase size of data stored on this rank
            // to accomodate received data
            MPI_Request send_request_data, recv_request_data;
            MPI_Isend(input_temperature_data.data(), send_data_size * finch_temp_components, MPI_DOUBLE, right, 1,
                      MPI_COMM_WORLD, &send_request_data);
            MPI_Irecv(finch_data_recv.data(), recv_data_size * finch_temp_components, MPI_DOUBLE, left, 1,
                      MPI_COMM_WORLD, &recv_request_data);
            int current_size = raw_temperature_data.extent(0);
            if (temperature_point_counter + recv_data_size >= current_size)
                Kokkos::resize(raw_temperature_data, temperature_point_counter + recv_data_size + resize_padding,
                               num_temperature_components);
            MPI_Wait(&send_request_data, MPI_STATUS_IGNORE);
            MPI_Wait(&recv_request_data, MPI_STATUS_IGNORE);

            // Unpack the appropriate received data into raw_temperature_data
            for (int n = 0; n < recv_data_size; n++) {
                int coord_y = (finch_data_recv(n, 1) - grid.y_min) / grid.deltax;
                if ((coord_y >= grid.y_offset) && (coord_y < grid.y_offset + grid.ny_local)) {
                    for (int comp = 0; comp < num_temperature_components; comp++)
                        raw_temperature_data(temperature_point_counter, comp) = finch_data_recv(n, comp);
                    // Increment counter for each point stored on this rank
                    temperature_point_counter++;
                }
            }

            // Replace send buffer with the received data
            input_temperature_data = finch_data_recv;
            send_data_size = recv_data_size;
        }
        // Resize with the number of temperature data points on this rank now known
        Kokkos::resize(raw_temperature_data, temperature_point_counter, num_temperature_components);
        for (int n = 0; n < grid.number_of_layers; n++) {
            first_value(n) = 0;
            last_value(n) = temperature_point_counter;
        }
        std::cout << "Rank " << id << " has " << temperature_point_counter << " events to simulate with ExaCA"
                  << std::endl;
    }

    // Read and parse the temperature file (double precision values in a comma-separated, ASCII format with a header
    // line - or a binary string of double precision values), storing the x, y, z, tm, tl, cr values in the RawData
    // vector. Each rank only contains the points corresponding to cells within the associated Y bounds.
    // number_of_temperature_data_points is incremented on each rank as data is added to RawData
    void parseTemperatureData(const std::string tempfile_thislayer, const double y_min, const double deltax,
                              const int lower_y_bound, const int upper_y_bound, int &number_of_temperature_data_points,
                              const bool binary_input_data, const int temperature_buffer_increment = 100000) {

        std::ifstream temperature_filestream;
        temperature_filestream.open(tempfile_thislayer);
        if (binary_input_data) {
            while (!temperature_filestream.eof()) {
                double x_temperature_point = readBinaryData<double>(temperature_filestream);
                double y_temperature_point = readBinaryData<double>(temperature_filestream);
                // If no data was extracted from the stream, the end of the file was reached
                if (!(temperature_filestream))
                    break;
                // Check the y value from parsed_line, to check if this point is stored on this rank
                // Check the CA grid positions of the data point to see which rank(s) should store it
                int y_int = Kokkos::round((y_temperature_point - y_min) / deltax);
                if ((y_int >= lower_y_bound) && (y_int <= upper_y_bound)) {
                    // This data point is inside the bounds of interest for this MPI rank
                    // Store the x and y values in RawData
                    raw_temperature_data(number_of_temperature_data_points, 0) = x_temperature_point;
                    raw_temperature_data(number_of_temperature_data_points, 1) = y_temperature_point;
                    // Parse the remaining 4 components (z, tm, tl, cr) from the line and store in RawData
                    for (int component = 2; component < num_temperature_components; component++) {
                        raw_temperature_data(number_of_temperature_data_points, component) =
                            readBinaryData<double>(temperature_filestream);
                    }
                    number_of_temperature_data_points++;
                    int raw_temperature_data_extent = raw_temperature_data.extent(0);
                    // Adjust size of RawData if it is near full
                    if (number_of_temperature_data_points >= raw_temperature_data_extent) {
                        Kokkos::resize(raw_temperature_data, raw_temperature_data_extent + temperature_buffer_increment,
                                       num_temperature_components);
                    }
                }
                else {
                    // This data point is inside the bounds of interest for this MPI rank
                    // ignore the z, tm, tl, cr values associated with it
                    unsigned char temp[4 * sizeof(double)];
                    temperature_filestream.read(reinterpret_cast<char *>(temp), 4 * sizeof(double));
                }
            }
        }
        else {
            // Get number of columns in this temperature file
            std::string header_line;
            getline(temperature_filestream, header_line);
            int vals_per_line = checkForHeaderValues(header_line);
            while (!temperature_filestream.eof()) {
                std::vector<std::string> parsed_line(
                    num_temperature_components); // Each line has an x, y, z, tm, tl, cr
                std::string read_line;
                if (!getline(temperature_filestream, read_line))
                    break;
                // Only parse the first 6 columns of the temperature data
                splitString(read_line, parsed_line, vals_per_line);
                // Check the y value from parsed_line, to check if this point is stored on this rank
                double y_temperature_point = getInputDouble(parsed_line[1]);
                // Check the CA grid positions of the data point to see which rank(s) should store it
                int y_int = Kokkos::round((y_temperature_point - y_min) / deltax);
                if ((y_int >= lower_y_bound) && (y_int <= upper_y_bound)) {
                    // This data point is inside the bounds of interest for this MPI rank: Store the x, z, tm, tl, and
                    // cr vals inside of RawData, incrementing with each value added
                    for (int component = 0; component < num_temperature_components; component++) {
                        raw_temperature_data(number_of_temperature_data_points, component) =
                            getInputDouble(parsed_line[component]);
                    }
                    number_of_temperature_data_points++;
                    // Adjust size of RawData if it is near full
                    int raw_temperature_data_extent = raw_temperature_data.extent(0);
                    // Adjust size of RawData if it is near full
                    if (number_of_temperature_data_points >= raw_temperature_data_extent) {
                        Kokkos::resize(raw_temperature_data, raw_temperature_data_extent + temperature_buffer_increment,
                                       num_temperature_components);
                    }
                }
            }
        }
    }

    // Read in temperature data from files, stored in the host view "RawData", with the appropriate MPI ranks storing
    // the appropriate data
    void readTemperatureData(int id, const Grid &grid, int layernumber) {

        // Y coordinates of this rank's data, inclusive and including ghost nodes
        int lower_y_bound = grid.y_offset;
        int upper_y_bound = grid.y_offset + grid.ny_local - 1;

        std::cout << "On MPI rank " << id << ", the Y bounds (in cells) are [" << lower_y_bound << "," << upper_y_bound
                  << "]" << std::endl;
        // Store raw data relevant to each rank in the vector structure RawData
        // Two passes through reading temperature data files- this is the second pass, reading the actual X/Y/Z/liquidus
        // time/cooling rate data and each rank stores the data relevant to itself in "RawData". With remelting
        // (simulation_type == "RM"), this is the same except that some X/Y/Z coordinates may be repeated in a file, and
        // a "melting time" value is stored in addition to liquidus time and cooling rate
        int number_of_temperature_data_points = 0;
        // Second pass through the files - ignore header line
        int first_layer_to_read, last_layer_to_read;
        if (_inputs.layerwise_temp_read) {
            first_layer_to_read = layernumber;
            last_layer_to_read = layernumber;
        }
        else {
            first_layer_to_read = 0;
            last_layer_to_read = std::min(grid.number_of_layers, _inputs.temp_files_in_series) - 1;
        }
        // Which temperature files should be read? Just the one file for layer "layernumber", or all of them?
        for (int layer_read_count = first_layer_to_read; layer_read_count <= last_layer_to_read; layer_read_count++) {

            std::string tempfile_thislayer;
            if (_inputs.layerwise_temp_read) {
                int LayerInSeries = layernumber % _inputs.temp_files_in_series;
                tempfile_thislayer = _inputs.temp_paths[LayerInSeries];
            }
            else
                tempfile_thislayer = _inputs.temp_paths[layer_read_count];

            first_value(layer_read_count) = number_of_temperature_data_points;
            // Read and parse temperature file for either binary or ASCII, storing the appropriate values on each MPI
            // rank within RawData and incrementing number_of_temperature_data_points appropriately
            bool binary_input_data = checkTemperatureFileFormat(tempfile_thislayer);
            parseTemperatureData(tempfile_thislayer, grid.y_min, grid.deltax, lower_y_bound, upper_y_bound,
                                 number_of_temperature_data_points, binary_input_data);
            last_value(layer_read_count) = number_of_temperature_data_points;
        } // End loop over all files read for all layers
        Kokkos::resize(raw_temperature_data, number_of_temperature_data_points, num_temperature_components);
        // Determine start values for each layer's data within "RawData", if all layers were read
        if (!(_inputs.layerwise_temp_read)) {
            if (grid.number_of_layers > _inputs.temp_files_in_series) {
                for (int layer_read_count = _inputs.temp_files_in_series; layer_read_count < grid.number_of_layers;
                     layer_read_count++) {
                    if (_inputs.temp_files_in_series == 1) {
                        // Since all layers have the same temperature data, each layer's "ZMinLayer" is just
                        // translated from that of the first layer
                        first_value(layer_read_count) = first_value(layer_read_count - 1);
                        last_value(layer_read_count) = last_value(layer_read_count - 1);
                    }
                    else {
                        // All layers have different temperature data but in a repeating pattern
                        int repeated_file = (layer_read_count) % _inputs.temp_files_in_series;
                        first_value(layer_read_count) = first_value(repeated_file);
                        last_value(layer_read_count) = last_value(repeated_file);
                    }
                }
            }
        }
    }

    // Initialize temperature data with a fixed thermal gradient in Z (can also be zero) for constrained/single grain
    // problem types
    void initialize(const int id, const std::string simulation_type, const Grid &grid, const double deltat) {

        // Check for valid simulation type.
        validSimulationType(simulation_type);

        // Initialize temperature field in Z direction with thermal gradient G set in input file
        // Liquidus front (InitUndercooling = 0) is at domain bottom for directional solidification, is at domain center
        // (with custom InitUndercooling value) for single grain solidification
        int location_init_undercooling, location_liquidus_isotherm;
        if (simulation_type == "Directional")
            location_init_undercooling = 0;
        else
            location_init_undercooling = Kokkos::floorf(static_cast<float>(grid.nz) / 2.0);

        // If thermal gradient is 0, liquidus isotherm does not exist - initialize to nz to avoid divide by zero error
        // and ensure all cells are initialized as undercooled (i.e., at Z coordinates less than nz)
        if (_inputs.G == 0)
            location_liquidus_isotherm = grid.nz;
        else
            location_liquidus_isotherm =
                location_init_undercooling + Kokkos::round(_inputs.init_undercooling / (_inputs.G * grid.deltax));

        // Each cell solidifies once
        Kokkos::deep_copy(number_of_solidification_events, 1);

        // Local copies for lambda capture.
        auto _layer_time_temp_history = layer_time_temp_history;
        auto _undercooling_current = undercooling_current;
        auto _current_solidification_event = current_solidification_event;
        auto _last_solidification_event = last_solidification_event;
        const double _init_undercooling = _inputs.init_undercooling;
        const double _G = _inputs.G;
        const double _R = _inputs.R;
        auto policy = Kokkos::RangePolicy<execution_space>(0, grid.domain_size);
        Kokkos::parallel_for(
            "TempInitG", policy, KOKKOS_LAMBDA(const int &index) {
                // Each cell solidifies once
                _current_solidification_event(index) = index;
                _last_solidification_event(index) = index + 1;

                // All cells past melting time step
                _layer_time_temp_history(index, 0) = -1;
                // Negative dist_from_liquidus and dist_from_init_undercooling values for cells below the liquidus
                // isotherm
                const int coord_z = grid.getCoordZ(index);
                const int dist_from_liquidus = coord_z - location_liquidus_isotherm;
                // Cells reach liquidus at a time dependent on their Z coordinate
                // Cells with negative liquidus time values are already undercooled, should have positive undercooling
                // and negative liquidus time step
                if (dist_from_liquidus < 0) {
                    _layer_time_temp_history(index, 1) = -1;
                    const int dist_from_init_undercooling = coord_z - location_init_undercooling;
                    _undercooling_current(index) = _init_undercooling - dist_from_init_undercooling * _G * grid.deltax;
                }
                else {
                    // Cells with positive liquidus time values are not yet tracked - leave current undercooling as
                    // default zeros and set liquidus time step. R_local will never be zero here, as all cells at a
                    // fixed undercooling must be below the liquidus (distFromLiquidus < 0)
                    _layer_time_temp_history(index, 1) = dist_from_liquidus * _G * grid.deltax / (_R * deltat);
                }
                // Cells cool at a constant rate
                _layer_time_temp_history(index, 2) = _R * deltat;
            });
        if (id == 0) {
            std::cout << "Temperature field initialized for unidirectional solidification with G = " << _G
                      << " K/m, initial undercooling at Z = " << location_init_undercooling << " of "
                      << _inputs.init_undercooling << " K below the liquidus" << std::endl;
            std::cout << "Done with temperature field initialization" << std::endl;
        }
    }

    // Initialize temperature data for a hemispherical spot melt (single layer simulation)
    void initialize(const int id, const Grid &grid, const double freezing_range, double deltat, double spot_radius) {

        // Each cell solidifies 0 or 1 time
        auto number_of_solidification_events_device =
            Kokkos::create_mirror_view_and_copy(memory_space(), number_of_solidification_events);

        // Outer edges of spots are initialized at the liquidus temperature
        // Spots cool at constant rate R, spot thermal gradient = G
        float isotherm_velocity = (_inputs.R / _inputs.G) * deltat / grid.deltax; // in cells per time step
        int spot_time_est = spot_radius / isotherm_velocity + (freezing_range / _inputs.R) / deltat; // in time steps
        // Spot center location - center of domain in X and Y, top of domain in Z
        float spot_center_x = spot_radius + 1;
        float spot_center_y = spot_radius + 1;
        float spot_center_z = grid.nz - 0.5;

        if (id == 0)
            std::cout << "Initializing temperature field for the hemispherical spot, which will take approximately "
                      << spot_time_est << " time steps to fully solidify" << std::endl;

        // Initialize layer_time_temp_history data values for this spot
        auto _layer_time_temp_history = layer_time_temp_history;
        auto _current_solidification_event = current_solidification_event;
        auto _last_solidification_event = last_solidification_event;
        view_type_int _number_of_solidification_events("n_s_events_local", grid.domain_size);
        double _R = _inputs.R;
        // event count view init to zero
        view_type_int event_count("event_count", 1);

        auto md_policy =
            Kokkos::MDRangePolicy<execution_space, Kokkos::Rank<3, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>(
                {0, 0, 0}, {grid.nz, grid.nx, grid.ny_local});
        Kokkos::parallel_for(
            "SpotTemperatureInit", md_policy, KOKKOS_LAMBDA(const int coord_z, const int coord_x, const int coord_y) {
                // 1D cell index
                const int index = grid.get1DIndex(coord_x, coord_y, coord_z);
                // Distance of this cell from the spot center
                const float dist_z = spot_center_z - coord_z;
                const float dist_x = spot_center_x - coord_x;
                const int coord_y_global = coord_y + grid.y_offset;
                const float dist_y = spot_center_y - coord_y_global;
                const float tot_dist = Kokkos::hypot(dist_x, dist_y, dist_z);
                if (tot_dist <= spot_radius) {
                    // This cell melts and resolidifies - increment counter
                    int current_count = Kokkos::atomic_fetch_add(&event_count(0), 1);
                    number_of_solidification_events_device(index) = 1;
                    _current_solidification_event(index) = current_count;
                    _last_solidification_event(index) = current_count + 1;
                    _number_of_solidification_events(index) = 1;
                    // Melt time step (first time step)
                    _layer_time_temp_history(current_count, 0) = 1;
                    // Liquidus time step (related to distance to spot edge)
                    _layer_time_temp_history(current_count, 1) =
                        Kokkos::round((spot_radius - tot_dist) / isotherm_velocity) + 1;
                    // Cooling rate per time step
                    _layer_time_temp_history(current_count, 2) = _R * deltat;
                }
            });
        MPI_Barrier(MPI_COMM_WORLD);
        auto event_count_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), event_count);
        number_of_solidification_events =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), _number_of_solidification_events);
        Kokkos::resize(layer_time_temp_history, event_count_host(0), 3);
        if (id == 0)
            std::cout << "Spot melt temperature field initialized" << std::endl;
    }

    // Set the maximum number of times a cell undergoes solidification in a layer
    void setMaxNumSolidificationEvents(const int id, const int domain_size,
                                       view_type_int &number_of_solidification_events_device) {
        auto policy = Kokkos::RangePolicy<execution_space>(0, domain_size);
        max_num_solidification_events = 0;
        Kokkos::parallel_reduce(
            "getMaxSEvents", policy,
            KOKKOS_LAMBDA(const int index, int &loc_max) {
                if (number_of_solidification_events_device(index) > loc_max)
                    loc_max = number_of_solidification_events_device(index);
            },
            max_num_solidification_events);
        if (id == 0)
            std::cout << "The maximum number of times a cell will melt and resolidify on this layer is "
                      << max_num_solidification_events << std::endl;
    }

    // Initialize temperature fields for layer "layernumber" in case where temperature data comes from file(s)
    void initialize(const int layernumber, const int id, const Grid &grid, const double deltat) {

        // Data was already read into the "raw_temperature_data" data structure
        // Determine which section of "raw_temperature_data" is relevant for this layer of the overall domain
        int start_range = first_value[layernumber];
        int end_range = last_value[layernumber];

        // From raw_temperature_data, get the number of times each cell melts/solidifies
        view_type_double_2d raw_temperature_data_device =
            Kokkos::create_mirror_view_and_copy(memory_space(), raw_temperature_data);
        view_type_double z_min_layer = Kokkos::create_mirror_view_and_copy(memory_space(), grid.z_min_layer);
        // Temporary view for the number of times each cell solidifies - default init to zeros
        view_type_int number_of_solidification_events_device("num_solidification_events_dev", grid.domain_size);
        // Reallocate views for solidification event counting
        Kokkos::realloc(current_solidification_event, grid.domain_size);
        Kokkos::realloc(last_solidification_event, grid.domain_size);
        Kokkos::deep_copy(current_solidification_event, 0);
        Kokkos::deep_copy(last_solidification_event, 0);

        if (id == 0)
            std::cout << "Range of raw data for layer " << layernumber << " on rank 0 is " << start_range << " to "
                      << end_range << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        auto events_policy = Kokkos::RangePolicy<execution_space>(start_range, end_range);
        Kokkos::parallel_for(
            "CalcNumSEvents", events_policy, KOKKOS_LAMBDA(const int &event) {
                // Get the integer X, Y, Z coordinates associated with this data point
                const int coord_x = Kokkos::round((raw_temperature_data_device(event, 0) - grid.x_min) / grid.deltax);
                const int coord_y =
                    Kokkos::round((raw_temperature_data_device(event, 1) - grid.y_min) / grid.deltax) - grid.y_offset;
                const int coord_z = Kokkos::round((raw_temperature_data_device(event, 2) +
                                                   grid.deltax * static_cast<double>(grid.layer_height * layernumber) -
                                                   z_min_layer(layernumber)) /
                                                  grid.deltax);

                // 1D cell coordinate on this MPI rank's domain
                const int index = grid.get1DIndex(coord_x, coord_y, coord_z);

                // Increment number of solidification events for this cell
                Kokkos::atomic_increment(&number_of_solidification_events_device(index));
            });
        Kokkos::fence();

        // Local copy for lambda capture
        auto _current_solidification_event = current_solidification_event;
        auto _last_solidification_event = last_solidification_event;
        // Get the initial and final event number for each event in the list (offsets for event number in
        // layer_time_temp_history)
        auto policy = Kokkos::RangePolicy<execution_space>(0, grid.domain_size);
        int tot_num_events;
        Kokkos::parallel_scan(
            "GetCurrentEventNum", policy,
            KOKKOS_LAMBDA(const int &index, int &partial_sum, bool is_final) {
                if (is_final)
                    _current_solidification_event(index) = partial_sum;
                partial_sum += number_of_solidification_events_device(index);
                if (is_final)
                    _last_solidification_event(index) = partial_sum;
            },
            tot_num_events);
        Kokkos::fence();

        // Place solidification events into the appropriate positions of layer_time_temp_history, which is resized with
        // the number of events to simulate now known
        Kokkos::resize(layer_time_temp_history, tot_num_events, 3);
        auto _layer_time_temp_history = layer_time_temp_history;

        view_type_int current_solidification_event_temp("current_solidification_event_temp", grid.domain_size);
        Kokkos::deep_copy(current_solidification_event_temp, current_solidification_event);
        Kokkos::parallel_for(
            "LoadSolidificationEvents", events_policy, KOKKOS_LAMBDA(const int &event) {
                // Get the integer X, Y, Z coordinates associated with this data point
                const int coord_x = Kokkos::round((raw_temperature_data_device(event, 0) - grid.x_min) / grid.deltax);
                const int coord_y =
                    Kokkos::round((raw_temperature_data_device(event, 1) - grid.y_min) / grid.deltax) - grid.y_offset;
                const int coord_z = Kokkos::round((raw_temperature_data_device(event, 2) +
                                                   grid.deltax * static_cast<double>(grid.layer_height * layernumber) -
                                                   z_min_layer(layernumber)) /
                                                  grid.deltax);

                // 1D cell coordinate on this MPI rank's domain
                const int index = grid.get1DIndex(coord_x, coord_y, coord_z);

                // Increment number of solidification events for this cell and get the old value
                const int cell_event_count = Kokkos::atomic_fetch_add(&current_solidification_event_temp(index), 1);

                // Event data to store
                const double t_melting = raw_temperature_data_device(event, 3);
                const double t_liquidus = raw_temperature_data_device(event, 4);
                const double cooling_rate = raw_temperature_data_device(event, 5);

                // Store event data in layer_time_temp_history
                _layer_time_temp_history(cell_event_count, 0) = Kokkos::round(t_melting / deltat) + 1;
                _layer_time_temp_history(cell_event_count, 1) = Kokkos::round(t_liquidus / deltat) + 1;
                _layer_time_temp_history(cell_event_count, 2) = std::abs(cooling_rate) * deltat;
            });
        Kokkos::fence();
        MPI_Barrier(MPI_COMM_WORLD);
        if (id == 0)
            std::cout << "Layer " << layernumber << " temperatures stored" << std::endl;

        // Reorder solidification events in layer_time_temp_history based on the melting time values (component = 0)
        Kokkos::parallel_for(
            "OrderEvents", policy, KOKKOS_LAMBDA(const int &index) {
                int n_solidification_events_cell = number_of_solidification_events_device(index);
                if (n_solidification_events_cell > 1) {
                    for (int i = 0; i < n_solidification_events_cell - 1; i++) {
                        for (int j = (i + 1); j < n_solidification_events_cell; j++) {
                            const int event_a = _current_solidification_event(index) + i;
                            const int event_b = _current_solidification_event(index) + j;
                            if (_layer_time_temp_history(event_a, 0) > _layer_time_temp_history(event_b, 0)) {
                                // Swap these two points - melting event "b" happens before event "a"
                                const float old_melt_val = _layer_time_temp_history(event_a, 0);
                                const float old_liq_val = _layer_time_temp_history(event_a, 1);
                                const float old_cr_val = _layer_time_temp_history(event_a, 2);
                                _layer_time_temp_history(event_a, 0) = _layer_time_temp_history(event_b, 0);
                                _layer_time_temp_history(event_a, 1) = _layer_time_temp_history(event_b, 1);
                                _layer_time_temp_history(event_a, 2) = _layer_time_temp_history(event_b, 2);
                                _layer_time_temp_history(event_b, 0) = old_melt_val;
                                _layer_time_temp_history(event_b, 1) = old_liq_val;
                                _layer_time_temp_history(event_b, 2) = old_cr_val;
                            }
                        }
                    }
                }
            });

        // If a cell melts twice before reaching the liquidus temperature, this is a double counted solidification
        // event and should be ignored (move to back the index of the last event and decrement the value for
        // last_solidification_event so that it does not get iterated over
        // TODO: Store removed solidification event data to print to files for future reference
        Kokkos::parallel_for(
            "RemoveDoubleEvents", policy, KOKKOS_LAMBDA(const int &index) {
                int n_solidification_events_cell = number_of_solidification_events_device(index);
                if (n_solidification_events_cell > 1) {
                    for (int i = 0; i < n_solidification_events_cell - 1; i++) {
                        float event_liq = _layer_time_temp_history(_current_solidification_event(index) + i, 1);
                        float next_event_melt =
                            _layer_time_temp_history(_current_solidification_event(index) + i + 1, 0);
                        if (next_event_melt < event_liq) {
                            // Keep whichever event has the larger liquidus time - ignore the removed event for now and
                            // move other event data into the next location in layer_time_temp_history. In the future,
                            // the ignored event data could be printed to files for debugging
                            float next_event_liq =
                                _layer_time_temp_history(_current_solidification_event(index) + i + 1, 1);
                            if (next_event_liq > event_liq) {
                                _layer_time_temp_history(_current_solidification_event(index) + i, 0) =
                                    _layer_time_temp_history(_current_solidification_event(index) + i + 1, 0);
                                _layer_time_temp_history(_current_solidification_event(index) + i, 1) =
                                    _layer_time_temp_history(_current_solidification_event(index) + i + 1, 1);
                                _layer_time_temp_history(_current_solidification_event(index) + i, 2) =
                                    _layer_time_temp_history(_current_solidification_event(index) + i + 1, 2);
                            }
                            for (int ii = (i + 1); ii < n_solidification_events_cell - 1; ii++) {
                                _layer_time_temp_history(_current_solidification_event(index) + i + 1, 0) =
                                    _layer_time_temp_history(_current_solidification_event(index) + ii + 1, 0);
                                _layer_time_temp_history(_current_solidification_event(index) + i + 1, 1) =
                                    _layer_time_temp_history(_current_solidification_event(index) + ii + 1, 1);
                                _layer_time_temp_history(_current_solidification_event(index) + i + 1, 2) =
                                    _layer_time_temp_history(_current_solidification_event(index) + ii + 1, 2);
                            }
                            // Substract one from the number of solidification events, the local
                            // n_solidification_events_cell, and last_solidification_event
                            number_of_solidification_events_device(index)--;
                            n_solidification_events_cell--;
                            _last_solidification_event(index)--;
                        }
                    }
                }
            });

        // Set the max number of solidification events undergone by any one cell, and copy view of number of
        // solidification events to host (device copy not used outside this function)
        setMaxNumSolidificationEvents(id, grid.domain_size, number_of_solidification_events_device);
        number_of_solidification_events =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), number_of_solidification_events_device);
        if (id == 0) {
            std::cout << "Layer " << layernumber << " temperature field is from Z = " << grid.z_layer_bottom
                      << " through " << grid.nz_layer + grid.z_layer_bottom - 1 << " of the global domain" << std::endl;
            std::cout << "Done with temperature field initialization" << std::endl;
        }
    }

    // Get the subview associated with the undercooling of cells in the current layer. Do not reset the undercooling of
    // cells from the prior layer to zero as this information will be stored for a potential print (and a cell that
    // remelts in the current layer will have its undercooling reset to 0 and recalculated)
    void getCurrentLayerUndercooling(std::pair<int, int> layer_range) {
        undercooling_current = Kokkos::subview(undercooling_current_all_layers, layer_range);
    }

    // (Optional based on selected inputs) Get the subview associated with the initial undercooling of cells during
    // solidification start in the current layer. Do not reset the undercooling of cells from the prior layer to zero as
    // this information will be stored for a potential print (and a cell that remelts in the current layer will have its
    // undercooling reset to 0 and recalculated)
    void getCurrentLayerStartingUndercooling(std::pair<int, int> layer_range) {
        undercooling_solidification_start = Kokkos::subview(undercooling_solidification_start_all_layers, layer_range);
    }

    // For each Z coordinate, find the smallest undercooling at which solidification started and finished, writing this
    // data to an output file
    view_type_float_2d_host getFrontUndercoolingStartFinish(const int id, const Grid &grid) {
        view_type_float start_solidification_z(Kokkos::ViewAllocateWithoutInitializing("start_solidification_z"),
                                               grid.nz);
        view_type_float end_solidification_z(Kokkos::ViewAllocateWithoutInitializing("end_solidification_z"), grid.nz);
        auto _undercooling_solidification_start = undercooling_solidification_start;
        auto _undercooling_current = undercooling_current;
        auto policy = Kokkos::RangePolicy<execution_space>(0, grid.nz);
        Kokkos::parallel_for(
            "GetMinUndercooling", policy, KOKKOS_LAMBDA(const int &coord_z) {
                float min_start_undercooling = Kokkos::Experimental::finite_max_v<float>;
                float min_end_undercooling = Kokkos::Experimental::finite_max_v<float>;
                for (int coord_x = 0; coord_x < grid.nx; coord_x++) {
                    for (int coord_y = 1; coord_y < grid.ny_local - 1; coord_y++) {
                        int index = grid.get1DIndex(coord_x, coord_y, coord_z);
                        if (_undercooling_solidification_start(index) < min_start_undercooling)
                            min_start_undercooling = _undercooling_solidification_start(index);
                        if (_undercooling_current(index) < min_end_undercooling)
                            min_end_undercooling = _undercooling_current(index);
                    }
                }
                start_solidification_z(coord_z) = min_start_undercooling;
                end_solidification_z(coord_z) = min_end_undercooling;
            });

        // Rank 0 - collect min values from all ranks to get the global start/end solidification undercoolings
        view_type_float start_solidification_z_reduced(
            Kokkos::ViewAllocateWithoutInitializing("start_solidification_z_red"), grid.nz);
        view_type_float end_solidification_z_reduced(
            Kokkos::ViewAllocateWithoutInitializing("end_solidification_z_red"), grid.nz);
        // Could potentially perform these reductions on the 2D view storing both start and end values, but depends on
        // 2D view data layout
        MPI_Reduce(start_solidification_z.data(), start_solidification_z_reduced.data(), grid.nz, MPI_FLOAT, MPI_MIN, 0,
                   MPI_COMM_WORLD);
        MPI_Reduce(end_solidification_z.data(), end_solidification_z_reduced.data(), grid.nz, MPI_FLOAT, MPI_MIN, 0,
                   MPI_COMM_WORLD);

        // Rank 0 - copy to host view
        view_type_float_2d_host start_end_solidification_z_host(
            Kokkos::ViewAllocateWithoutInitializing("start_end_solidification_z_host"), grid.nz, 2);
        if (id == 0) {
            view_type_float_host start_solidification_z_host =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), start_solidification_z_reduced);
            view_type_float_host end_solidification_z_host =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), end_solidification_z_reduced);
            for (int coord_z = 0; coord_z < grid.nz; coord_z++) {
                start_end_solidification_z_host(coord_z, 0) = start_solidification_z_host(coord_z);
                start_end_solidification_z_host(coord_z, 1) = end_solidification_z_host(coord_z);
            }
        }
        return start_end_solidification_z_host;
    }

    // Reset local cell undercooling (and if needed, the cell's starting undercooling) to 0
    KOKKOS_INLINE_FUNCTION
    void resetUndercooling(const int index) const {
        if (_store_solidification_start)
            undercooling_solidification_start(index) = 0.0;
        undercooling_current(index) = 0.0;
    }

    // Update local cell undercooling for the current melt-resolidification event
    KOKKOS_INLINE_FUNCTION
    void updateUndercooling(const int index) const {
        undercooling_current(index) += layer_time_temp_history(current_solidification_event(index), 2);
    }

    // (Optional based on inputs) Set the starting undercooling in the cell for the solidification event that just
    // started
    KOKKOS_INLINE_FUNCTION
    void setStartingUndercooling(const int index) const {
        if (_store_solidification_start)
            undercooling_solidification_start(index) = undercooling_current(index);
    }

    // Update the solidification event counter for a cell that has not finished the previous solidification event (i.e.,
    // solidification is not complete in this cell as it is not tempsolid or solid type)
    KOKKOS_INLINE_FUNCTION
    void updateSolidificationCounter(const int index) const { current_solidification_event(index)++; }

    // Update the solidification event counter for the cell (which is either tempsolid or solid type) and return whether
    // all solidification events have completed in the cell
    KOKKOS_INLINE_FUNCTION
    bool updateCheckSolidificationCounter(const int index) const {
        bool solidification_complete;
        current_solidification_event(index)++;
        if (current_solidification_event(index) == last_solidification_event(index))
            solidification_complete = true;
        else
            solidification_complete = false;
        return solidification_complete;
    }

    // Initialize the temperature data for the next layer of a multilayer simulation
    void initNextLayer(const int id, const std::string simulation_type, Grid &grid, const double deltat,
                       const int next_layer_number) {

        // If the next layer's temperature data isn't already stored, it should be read
        if ((simulation_type == "FromFile") && (_inputs.layerwise_temp_read))
            readTemperatureData(id, grid, next_layer_number);
        MPI_Barrier(MPI_COMM_WORLD);

        // Initialize next layer's temperature data
        initialize(next_layer_number, id, grid, deltat);

        // Get the subview for the next layer's undercooling
        getCurrentLayerUndercooling(grid.layer_range);
        if (_store_solidification_start)
            getCurrentLayerStartingUndercooling(grid.layer_range);
    }

    // Extract the next time that this point undergoes melting
    KOKKOS_INLINE_FUNCTION
    int getMeltTimeStep(const int cycle, const int index, const int event_num) const {
        int melt_time_step = static_cast<int>(layer_time_temp_history(event_num, 0));
        if (cycle > melt_time_step) {
            // If the cell has already exceeded the melt time step for the current melt-solidification event, get the
            // melt time step associated with the next solidification event - or, if there is no next
            // melt-solidification event, return the max possible int as the cell will not melt again during this layer
            // of the multilayer problem
            if (event_num < (last_solidification_event(index) - 1))
                melt_time_step = static_cast<int>(layer_time_temp_history(event_num + 1, 0));
            else
                melt_time_step = INT_MAX;
        }
        return melt_time_step;
    }

    // Extract the next time that this point cools below the liquidus
    // Uses the current value of the solidification event counter
    KOKKOS_INLINE_FUNCTION
    int getCritTimeStep(const int event_num) const {
        int crit_time_step = static_cast<int>(layer_time_temp_history(event_num, 1));
        return crit_time_step;
    }

    // Extract either the last time step that all points undergo melting in the layer, the last time they cools below
    // the liquidus, or the rate at which they cools from the liquidus from layer_time_temp_history (corresponds to
    // solidification event number `NumSolidificationEvents-1` for the cell) (can't just use subview here since
    // NumSolidificationEvents is different for each cell) If the cell does not undergo solidification, either print -1
    // or the specified default value
    template <typename extracted_view_data_type>
    extracted_view_data_type extractTmTlCrData(const int extracted_val, const int domain_size,
                                               const int default_val = -1) {
        extracted_view_data_type extracted_data(Kokkos::ViewAllocateWithoutInitializing("extracted_data"), domain_size);
        using extracted_value_type = typename extracted_view_data_type::value_type;

        // Get device copy of number_of_solidification_events
        auto number_of_solidification_events_device =
            Kokkos::create_mirror_view_and_copy(memory_space(), number_of_solidification_events);

        // Local copy for lambda capture.
        auto _layer_time_temp_history = layer_time_temp_history;
        auto _last_solidification_event = last_solidification_event;

        auto policy = Kokkos::RangePolicy<execution_space>(0, domain_size);
        Kokkos::parallel_for(
            "Extract_tm_tl_cr_data", policy, KOKKOS_LAMBDA(const int &index) {
                // If this cell doesn't undergo solidification at all, print -1
                if (number_of_solidification_events_device(index) == 0)
                    extracted_data(index) = static_cast<extracted_value_type>(default_val);
                else
                    extracted_data(index) = static_cast<extracted_value_type>(
                        _layer_time_temp_history(_last_solidification_event(index) - 1, extracted_val));
            });
        return extracted_data;
    }

    // Extract either the last time step that all points undergo melting in the layer, the last time they cools below
    // the liquidus, or the rate at which they cools from the liquidus from layer_time_temp_history (corresponds to
    // solidification event number `NumSolidificationEvents-1` for the cell) (can't just use subview here since
    // NumSolidificationEvents is different for each cell) If the cell does not undergo solidification, either print -1
    // or the specified default value
    template <typename extracted_view_data_type>
    extracted_view_data_type extractSolidificationEventCounter(const int domain_size) {
        extracted_view_data_type extracted_data(Kokkos::ViewAllocateWithoutInitializing("extracted_data"), domain_size);
        using extracted_value_type = typename extracted_view_data_type::value_type;

        // Local device copies for lambda capture.
        auto _current_solidification_event = current_solidification_event;
        auto _last_solidification_event = last_solidification_event;
        auto _number_of_solidification_events =
            Kokkos::create_mirror_view_and_copy(memory_space(), number_of_solidification_events);

        auto policy = Kokkos::RangePolicy<execution_space>(0, domain_size);
        Kokkos::parallel_for(
            "ExtractSEventCounter", policy, KOKKOS_LAMBDA(const int &index) {
                extracted_data(index) = static_cast<extracted_value_type>(
                    _current_solidification_event(index) -
                    (_last_solidification_event(index) - _number_of_solidification_events(index)));
            });
        return extracted_data;
    }
};

#endif
