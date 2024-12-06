// Copyright Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef EXACA_INTERFACE_HPP
#define EXACA_INTERFACE_HPP

#include "CAcelldata.hpp"
#include "CAconfig.hpp"
#include "CAorientation.hpp"
#include "CAparsefiles.hpp"
#include "CAtemperature.hpp"
#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Data representing the active cells at the solid-liquid interface, including MPI buffers
template <typename MemorySpace>
struct Interface {

    using memory_space = MemorySpace;
    using view_type_buffer = Kokkos::View<float **, memory_space>;
    using view_type_float = Kokkos::View<float *, memory_space>;
    using view_type_int = Kokkos::View<int *, memory_space>;
    using view_type_short = Kokkos::View<short *, memory_space>;
    using view_type_int_host = typename view_type_int::HostMirror;
    using neighbor_list_type = Kokkos::Array<int, 26>;
    using diagonal_list_type = Kokkos::Array<int, 6>;

    // Using the default exec space for this memory space.
    using execution_space = typename memory_space::execution_space;

    // Size of send/recv buffers
    int buf_size, buf_components;
    view_type_short nearest_diagonals;
    view_type_float diagonal_length, octahedron_center;
    view_type_buffer buffer_south_send, buffer_north_send, buffer_south_recv, buffer_north_recv;
    view_type_int send_size_south, send_size_north, steering_vector, num_steer;
    view_type_int_host send_size_south_host, send_size_north_host, num_steer_host;
    // Initial size of new octahedra
    float _init_oct_size;

    // Neighbor lists
    neighbor_list_type neighbor_x, neighbor_y, neighbor_z;

    // +/- 1 for positive and negative unit vector directions for each crystallographic <100>
    diagonal_list_type direction_negative;

    // Parallel dispatch tags.
    struct RefillBuffersTag {};

    // Constructor for views and view bounds for current layer
    // Use default initialization to 0 for num_steer_host and num_steer and buffer counts
    Interface(const int id, const int domain_size, const float init_oct_size, const int buf_size_initial_estimate = 25,
              const int buf_components_temp = 8)
        : nearest_diagonals(view_type_short(Kokkos::ViewAllocateWithoutInitializing("nearest_diagonals"), 78 * domain_size))
        , diagonal_length(view_type_float(Kokkos::ViewAllocateWithoutInitializing("diagonal_length"), 6 * domain_size))
        , octahedron_center(
              view_type_float(Kokkos::ViewAllocateWithoutInitializing("octahedron_center"), 3 * domain_size))
        , buffer_south_send(view_type_buffer(Kokkos::ViewAllocateWithoutInitializing("buffer_south_send"),
                                             buf_size_initial_estimate, buf_components_temp))
        , buffer_north_send(view_type_buffer(Kokkos::ViewAllocateWithoutInitializing("buffer_north_send"),
                                             buf_size_initial_estimate, buf_components_temp))
        , buffer_south_recv(view_type_buffer(Kokkos::ViewAllocateWithoutInitializing("buffer_south_recv"),
                                             buf_size_initial_estimate, buf_components_temp))
        , buffer_north_recv(view_type_buffer(Kokkos::ViewAllocateWithoutInitializing("buffer_north_recv"),
                                             buf_size_initial_estimate, buf_components_temp))
        , send_size_south(view_type_int("send_size_south", 1))
        , send_size_north(view_type_int("send_size_north", 1))
        , steering_vector(view_type_int(Kokkos::ViewAllocateWithoutInitializing("steering_vector"), domain_size))
        , num_steer(view_type_int("steering_vector_size", 1))
        , send_size_south_host(view_type_int_host("send_size_south_host", 1))
        , send_size_north_host(view_type_int_host("send_size_north_host", 1))
        , num_steer_host(view_type_int_host("steering_vector_size_host", 1))
        , _init_oct_size(init_oct_size) {

        // Set initial buffer size to the estimate
        buf_size = buf_size_initial_estimate;
        // Set number of components in the buffer
        buf_components = buf_components_temp;
        // Send/recv buffers for ghost node data should be initialized with -1s in the first index as placeholders for
        // empty positions in the buffer, and with send size counts of 0
        resetBuffers();
        // Initialize neighbor lists for iterating over active cells
        neighborListInit();
        // Initialize positive/negative numbers for unit vector directions
        directionNegativeInit();

        if (id == 0)
            std::cout << "Done with interface initialization" << std::endl;
    }

    // Set first index in send buffers to -1 (placeholder) for all cells in the buffer, and reset the counts of number
    // of cells contained in buffers to 0s
    void resetBuffers() {

        auto buffer_north_send_local = buffer_north_send;
        auto buffer_south_send_local = buffer_south_send;
        auto send_size_north_local = send_size_north;
        auto send_size_south_local = send_size_south;
        Kokkos::parallel_for(
            "BufferReset", buf_size, KOKKOS_LAMBDA(const int &i) {
                buffer_north_send_local(i, 0) = -1.0;
                buffer_south_send_local(i, 0) = -1.0;
            });
        Kokkos::parallel_for(
            "HaloCountReset", 1, KOKKOS_LAMBDA(const int) {
                send_size_north_local(0) = 0;
                send_size_south_local(0) = 0;
            });
    }

    // Initialize neighbor list structures (neighbor_x, neighbor_y, neighbor_z)
    void neighborListInit() {

        // Neighbors 0 through 5 are nearest neighbors, 6 through 17 are second nearest neighbors, and 18 through 25 are
        // third nearest neighbors
        neighbor_x = {1, 0, 0, -1, 0, 0, 1, 1, 0, -1, -1, 0, -1, -1, 0, 1, 1, 0, 1, -1, 1, 1, -1, -1, 1, -1};
        neighbor_y = {0, 1, 0, 0, -1, 0, 1, 0, 1, -1, 0, -1, 1, 0, -1, -1, 0, 1, 1, 1, -1, 1, -1, 1, -1, -1};
        neighbor_z = {0, 0, 1, 0, 0, -1, 0, 1, 1, 0, -1, -1, 0, 1, 1, 0, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1};
    }
    
    // Initialize directionNegativeInit to -1s and +1s for the positive and negative grain unit vector directions
    void directionNegativeInit() {
        direction_negative = {1, 1, 1, -1, -1, -1};
    }

    // Increase size of buffers if necessary, returning the new buffer size. Return true if the buffers were resized
    int resizeBuffers(const int id, const int cycle, const int num_cells_buffer_padding = 25) {

        bool resize_performed = false;
        int old_buf_size = buf_size;
        send_size_north_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_size_north);
        send_size_south_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_size_south);
        int max_count_local = Kokkos::max(send_size_north_host(0), send_size_south_host(0));
        int max_count_global;
        MPI_Allreduce(&max_count_local, &max_count_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (max_count_global > old_buf_size) {
            // Increase buffer size to fit all data
            // Add num_cells_buffer_padding (defaults to 25) cells as additional padding
            int new_buf_size = max_count_global + num_cells_buffer_padding;
            Kokkos::resize(buffer_north_send, new_buf_size, buf_components);
            Kokkos::resize(buffer_south_send, new_buf_size, buf_components);
            Kokkos::resize(buffer_north_recv, new_buf_size, buf_components);
            Kokkos::resize(buffer_south_recv, new_buf_size, buf_components);
            // Reset count variables on device to the old buffer size
            auto send_size_north_local = send_size_north;
            auto send_size_south_local = send_size_south;
            auto old_buf_size_local = old_buf_size;
            auto buffer_north_send_local = buffer_north_send;
            auto buffer_south_send_local = buffer_south_send;
            auto buf_components_local = buf_components;
            Kokkos::parallel_for(
                "ResetCounts", 1, KOKKOS_LAMBDA(const int &) {
                    send_size_north_local(0) = old_buf_size_local;
                    send_size_south_local(0) = old_buf_size_local;
                });
            // Set -1 values for the new (currently empty) positions in the resized buffer
            Kokkos::parallel_for(
                "InitNewBufCapacity", Kokkos::RangePolicy<>(old_buf_size_local, new_buf_size),
                KOKKOS_LAMBDA(const int &buf_position) {
                    for (int buf_comp = 0; buf_comp < buf_components_local; buf_comp++) {
                        buffer_north_send_local(buf_position, buf_comp) = -1.0;
                        buffer_south_send_local(buf_position, buf_comp) = -1.0;
                    }
                });
            buf_size = new_buf_size;
            resize_performed = true;
            if (id == 0)
                std::cout << "On time step " << cycle << ", resized the send/recv buffers to " << buf_size << std::endl;
        }
        return resize_performed;
    }

    // Resize and reinitialize structs governing the active cells before the next layer of a multilayer problem. Realloc
    // is used as the old values from the structs are not needed
    void initNextLayer(const int domain_size) {

        // Realloc steering vector as domain_size for the next layer may be different
        Kokkos::realloc(steering_vector, domain_size);

        // Realloc active cell data structure and halo regions
        Kokkos::realloc(diagonal_length, domain_size);
        Kokkos::realloc(octahedron_center, 3 * domain_size);
        Kokkos::realloc(nearest_diagonals, 78 * domain_size);

        // Reset active cell data structures to zeros
        Kokkos::deep_copy(diagonal_length, 0);
        Kokkos::deep_copy(octahedron_center, 0);
        Kokkos::deep_copy(nearest_diagonals, 0);
    }

    // Assign octahedron a small initial (equiaxed) size, and a center location
    // Note that the Y coordinate is relative to the domain origin to keep the coordinate system continuous across ranks
    template <typename ViewType>
    KOKKOS_INLINE_FUNCTION
    void createNewOctahedron(const int index, const int coord_x, const int coord_y, const int y_offset,
                             const int coord_z, const int my_orientation, const ViewType grain_unit_vector) const {
        for (int diagonal=0; diagonal<6; diagonal++)
            diagonal_length(6 * index + diagonal) = _init_oct_size;
        octahedron_center(3 * index) = coord_x + 0.5;
        octahedron_center(3 * index + 1) = coord_y + y_offset + 0.5;
        octahedron_center(3 * index + 2) = coord_z + 0.5;
        // Get the indices of the <100> closest aligned with the unit vector in the direction of each neighboring cell
        for (int n=0; n<26; n++) {
            // Unit vector in the direction of neighboring cell "n"
            const float mag = Kokkos::sqrt(neighbor_x[n] * neighbor_x[n] + neighbor_y[n] * neighbor_y[n] + neighbor_z[n] * neighbor_z[n]);
            const float neighbor_x_norm = neighbor_x[n] / mag;
            const float neighbor_y_norm = neighbor_y[n] / mag;
            const float neighbor_z_norm = neighbor_z[n] / mag;
            for (int diagonal=0; diagonal<3; diagonal++) {
                const float octahedron_growth_x = grain_unit_vector(9 * my_orientation + 3 * diagonal);
                const float octahedron_growth_y = grain_unit_vector(9 * my_orientation + 3 * diagonal + 1);
                const float octahedron_growth_z = grain_unit_vector(9 * my_orientation + 3 * diagonal + 2);
                // Will be between -1 and 1: use absolute value to get a positive number between 0 and 1 so magnitude can be compared
                const float cos_ang_neighbor_oct = neighbor_x_norm * octahedron_growth_x + neighbor_y_norm * octahedron_growth_y + neighbor_z_norm * octahedron_growth_z;
//                if (n == 25)
//                    printf("diagonal %d is %f\n",diagonal,cos_ang_neighbor_oct);
                // Either the index of the unit vector (0,1,2) or the index of the negative unit vector (3,4,5) is stored
                if (cos_ang_neighbor_oct > 0)
                    nearest_diagonals(78 * index + 3 * n + diagonal) = diagonal;
                else
                    nearest_diagonals(78 * index + 3 * n + diagonal) = diagonal + 3;
            }
//            printf("New cell diagonal indexes for direction %d are %d, %d, %d\n",n,nearest_diagonals(78 * index + 3 * n),nearest_diagonals(78 * index + 3 * n + 1),nearest_diagonals(78 * index + 3 * n + 2));
        }
    }

    // Load data (grain_id, octahedron_center, diagonal_length) into ghost nodes if the given coord_y is associated with
    // a 1D halo region Uses check to ensure that the buffer position does not reach the buffer size - if it does, keep
    // incrementing the send size counters for use resizing the buffers in the future
    KOKKOS_INLINE_FUNCTION
    bool loadGhostNodes(const int ghost_grain_id, const float ghost_octahedron_center_x,
                        const float ghost_octahedron_center_y, const float ghost_octahedron_center_z,
                        const float ghost_diagonal_length, const int ny_local, const int coord_x, const int coord_y,
                        const int coord_z, const bool at_north_boundary, const bool at_south_boundary,
                        const int n_grain_orientations) const {
        bool data_fits_in_buffer = true;
        if ((coord_y == 1) && (!(at_south_boundary))) {
            int ghost_position_south = Kokkos::atomic_fetch_add(&send_size_south(0), 1);
            if (ghost_position_south >= buf_size)
                data_fits_in_buffer = false;
            else {
                buffer_south_send(ghost_position_south, 0) = static_cast<float>(coord_x);
                buffer_south_send(ghost_position_south, 1) = static_cast<float>(coord_z);
                buffer_south_send(ghost_position_south, 2) =
                    static_cast<float>(getGrainOrientation(ghost_grain_id, n_grain_orientations, false));
                buffer_south_send(ghost_position_south, 3) =
                    static_cast<float>(getGrainNumber(ghost_grain_id, n_grain_orientations));
                buffer_south_send(ghost_position_south, 4) = ghost_octahedron_center_x;
                buffer_south_send(ghost_position_south, 5) = ghost_octahedron_center_y;
                buffer_south_send(ghost_position_south, 6) = ghost_octahedron_center_z;
                buffer_south_send(ghost_position_south, 7) = ghost_diagonal_length;
            }
        }
        else if ((coord_y == ny_local - 2) && (!(at_north_boundary))) {
            int ghost_position_north = Kokkos::atomic_fetch_add(&send_size_north(0), 1);
            if (ghost_position_north >= buf_size)
                data_fits_in_buffer = false;
            else {
                buffer_north_send(ghost_position_north, 0) = static_cast<float>(coord_x);
                buffer_north_send(ghost_position_north, 1) = static_cast<float>(coord_z);
                buffer_north_send(ghost_position_north, 2) =
                    static_cast<float>(getGrainOrientation(ghost_grain_id, n_grain_orientations, false));
                buffer_north_send(ghost_position_north, 3) =
                    static_cast<float>(getGrainNumber(ghost_grain_id, n_grain_orientations));
                buffer_north_send(ghost_position_north, 4) = ghost_octahedron_center_x;
                buffer_north_send(ghost_position_north, 5) = ghost_octahedron_center_y;
                buffer_north_send(ghost_position_north, 6) = ghost_octahedron_center_z;
                buffer_north_send(ghost_position_north, 7) = ghost_diagonal_length;
            }
        }
        return data_fits_in_buffer;
    }

    // If data doesn't fit in the buffer after the resize, warn that buffer data may have been lost
    KOKKOS_INLINE_FUNCTION
    void checkBufferSize([[maybe_unused]] const bool data_fits_in_buffer) const {
#if KOKKOS_VERSION >= 40200
        if (!data_fits_in_buffer)
            Kokkos::printf("Error: Send/recv buffer resize failed to include all necessary data, predicted "
                           "results at MPI processor boundaries may be inaccurate\n");
#endif
    }
};

#endif
