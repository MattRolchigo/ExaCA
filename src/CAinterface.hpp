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
    using view_type_int_host = typename view_type_int::HostMirror;
    using neighbor_list_type = Kokkos::Array<int, 26>;

    // Using the default exec space for this memory space.
    using execution_space = typename memory_space::execution_space;

    // Size of send/recv buffers
    int buf_size, buf_components;
    view_type_float diagonal_length, init_diagonal_length, octahedron_center_x, octahedron_center_y, octahedron_center_z, crit_diagonal_length;
    view_type_buffer buffer_south_send, buffer_north_send, buffer_south_recv, buffer_north_recv;
    view_type_int send_size_south, send_size_north, steering_vector, num_steer;
    view_type_int_host send_size_south_host, send_size_north_host, num_steer_host;
    // Initial size of new octahedra
    float _init_oct_size;

    // Neighbor lists
    neighbor_list_type neighbor_x, neighbor_y, neighbor_z;

    // Parallel dispatch tags.
    struct RefillBuffersTag {};

    // Constructor for views and view bounds for current layer
    // Use default initialization to 0 for num_steer_host and num_steer and buffer counts
    Interface(const int id, const int domain_size, const float init_oct_size, const int buf_size_initial_estimate = 25,
              const int buf_components_temp = 8)
        : diagonal_length(view_type_float(Kokkos::ViewAllocateWithoutInitializing("diagonal_length"), 26 * domain_size))
        , init_diagonal_length(view_type_float(Kokkos::ViewAllocateWithoutInitializing("init_diagonal_length"), 26 * domain_size))
        , octahedron_center_x(
              view_type_float(Kokkos::ViewAllocateWithoutInitializing("octahedron_center_x"), 26 * domain_size))
        , octahedron_center_y(
          view_type_float(Kokkos::ViewAllocateWithoutInitializing("octahedron_center_y"), 26 * domain_size))
        , octahedron_center_z(
          view_type_float(Kokkos::ViewAllocateWithoutInitializing("octahedron_center_z"), 26 * domain_size))
        , crit_diagonal_length(
              view_type_float(Kokkos::ViewAllocateWithoutInitializing("crit_diagonal_length"), 26 * domain_size))
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

        // Set up lists so that adjacent cells are sequential and x,y,z can be converted to 1D indices via getNeighborIndexInitiatingCell. Note that there is no (0,0,0)
        //             0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,13, 14,15,16, 17, 18, 19, 20,21,22, 23,24, 25
        neighbor_x = {-1, -1, -1,  0,  0,  0,  1,  1,  1, -1, -1, -1,  0, 0,  1, 1, 1, -1, -1, -1,  0, 0, 0,  1, 1, 1};
        neighbor_y = {-1,  0,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1, -1, 1, -1, 0, 1, -1,  0,  1, -1, 0, 1, -1, 0, 1};
        neighbor_z = {-1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0, 0,  0, 0, 0,  1,  1,  1,  1, 1, 1,  1, 1, 1};
    }

    // Cell at init_coord_x, init_coord_y, init_coord_z initiated the capture event
    // Cell at capt_coord_x, capt_coord_y, capt_coord_z was captured
    // The index neighbor_index_of_capt gives the coordinate of a cell adjacent to the captured cell
    // Return the index such that init_coord_x + neighbor_x[neighbor_index_of_init] = capt_coord_x + neighbor_coord_x[neighbor_index_of_capt], etc for y and z. If there is no value for neighbor_index_of_init to make these conditions true, return -1
    KOKKOS_INLINE_FUNCTION
    int getNeighborIndexInitiatingCell(const int init_coord_x, const int init_coord_y, const int init_coord_z, const int capt_coord_x, const int capt_coord_y, const int capt_coord_z, const int nx, const int ny_local, const int nz_layer, const int neighbor_index_of_capt) const {
        int neighbor_index_of_init;
        // Cell adjacent to to captured cell
        const int adjacent_capt_center_x = capt_coord_x + neighbor_x[neighbor_index_of_capt];
        const int adjacent_capt_center_y = capt_coord_y + neighbor_y[neighbor_index_of_capt];
        const int adjacent_capt_center_z = capt_coord_z + neighbor_z[neighbor_index_of_capt];
        // Distance between the cell adjacent to the captured cell center and the initiating cell
        int dist_adjacent_init_x = adjacent_capt_center_x - init_coord_x;
        int dist_adjacent_init_y = adjacent_capt_center_y - init_coord_y;
        int dist_adjacent_init_z = adjacent_capt_center_z - init_coord_z;
        const bool adjacent_capt_no_border_init = ((Kokkos::abs(dist_adjacent_init_x) > 1) || (Kokkos::abs(dist_adjacent_init_y) > 1) || (Kokkos::abs(dist_adjacent_init_z) > 1));
        const bool adjacent_capt_same_as_init = (Kokkos::abs(dist_adjacent_init_x) + Kokkos::abs(dist_adjacent_init_y) + Kokkos::abs(dist_adjacent_init_z) == 0);
        const bool adjacent_capt_not_in_bounds = ((adjacent_capt_center_x < 0) || (adjacent_capt_center_x >= nx) || (adjacent_capt_center_y < 0) || (adjacent_capt_center_x >= ny_local) || (adjacent_capt_center_z < 0) || (adjacent_capt_center_z >= nz_layer));
        // Return -1 if either of the following conditions are satisfied. 1) The cell adjacent to the captured cell is the initiating cell. 2) The distance between the cell adjacent to the captured cell center and the initiating cell is 2 or more in any cardinal direction. 3) The cell adjacent to the captured cell is out of bounds
        if ((adjacent_capt_no_border_init) || (adjacent_capt_same_as_init) || (adjacent_capt_not_in_bounds))
            neighbor_index_of_init = -1;
        else {
            // Add 1 so that capt_dir_x, capt_dir_y, capt_dir_z go from 0 to 2 for consistent indexing with the ordering in neighborListInit
            dist_adjacent_init_x++;
            dist_adjacent_init_y++;
            dist_adjacent_init_z++;
            // Convert the (i,j,k) to the relevant index in neighbor_x,neighbor_y,neighbor_z. Because (0,0,0) isn't an indexed direction in the neighbor lists, subtract one from the 1D index for cells past the midway index of the capture directions
            neighbor_index_of_init = 9 * dist_adjacent_init_z + 3 * dist_adjacent_init_x + dist_adjacent_init_y;
            if (neighbor_index_of_init >= 14)
                neighbor_index_of_init = neighbor_index_of_init - 1;
        }
        return neighbor_index_of_init;
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

//        // Realloc active cell data structure and halo regions
//        Kokkos::realloc(diagonal_length, 26 * domain_size);
//        Kokkos::realloc(octahedron_center, 3 * domain_size);
//        Kokkos::realloc(crit_diagonal_length, 26 * domain_size);
//
//        // Reset active cell data structures to zeros
//        Kokkos::deep_copy(diagonal_length, 0);
//        Kokkos::deep_copy(octahedron_center, 0);
//        Kokkos::deep_copy(crit_diagonal_length, 0);
    }

    // Assign octahedron a small initial size, and a center location
    // Note that the Y coordinate is relative to the domain origin to keep the coordinate system continuous across ranks
    KOKKOS_INLINE_FUNCTION
    void createNewOctahedron(const int index, const int coord_x, const int coord_y, const int y_offset,
                             const int coord_z) const {
        for (int capt_dir=0; capt_dir<26; capt_dir++) {
            diagonal_length(26 * index + capt_dir) = _init_oct_size;
            init_diagonal_length(26 * index + capt_dir) = _init_oct_size;
            octahedron_center_x(26 * index + capt_dir) = coord_x + 0.5;
            octahedron_center_y(26 * index + capt_dir) = coord_y + y_offset + 0.5;
            octahedron_center_z(26 * index + capt_dir) = coord_z + 0.5;
        }
    }

    // For the newly active cell located at 1D array position index (3D center coordinate of xp, yp, zp),
    // update crit_diagonal_length values for cell capture of neighboring cells. The octahedron has a center located at
    // (cx, cy, cz) Note that yp and cy are relative to the domain origin to keep the coordinate system continuous
    // across ranks
    template <typename ViewType>
    KOKKOS_INLINE_FUNCTION void calcCritDiagonalLength(const int index, const float xp, const float yp, const float zp,
                                                       const float cx, const float cy, const float cz,
                                                       const int my_orientation,
                                                       const ViewType grain_unit_vector, const int capt_dir) const {
        // Calculate critical octahedron diagonal length to activate nearest neighbor.
        // First, calculate the unique planes (4) associated with all octahedron faces (8)
        // Then just look at distance between face and the point of interest (cell center of
        // neighbor). The critical diagonal length will be the maximum of these (since all other
        // planes will have passed over the point by then
        // ... meaning it must be in the octahedron)
        float fx[4], fy[4], fz[4];

        fx[0] = grain_unit_vector(9 * my_orientation) + grain_unit_vector(9 * my_orientation + 3) +
                grain_unit_vector(9 * my_orientation + 6);
        fx[1] = grain_unit_vector(9 * my_orientation) - grain_unit_vector(9 * my_orientation + 3) +
                grain_unit_vector(9 * my_orientation + 6);
        fx[2] = grain_unit_vector(9 * my_orientation) + grain_unit_vector(9 * my_orientation + 3) -
                grain_unit_vector(9 * my_orientation + 6);
        fx[3] = grain_unit_vector(9 * my_orientation) - grain_unit_vector(9 * my_orientation + 3) -
                grain_unit_vector(9 * my_orientation + 6);

        fy[0] = grain_unit_vector(9 * my_orientation + 1) + grain_unit_vector(9 * my_orientation + 4) +
                grain_unit_vector(9 * my_orientation + 7);
        fy[1] = grain_unit_vector(9 * my_orientation + 1) - grain_unit_vector(9 * my_orientation + 4) +
                grain_unit_vector(9 * my_orientation + 7);
        fy[2] = grain_unit_vector(9 * my_orientation + 1) + grain_unit_vector(9 * my_orientation + 4) -
                grain_unit_vector(9 * my_orientation + 7);
        fy[3] = grain_unit_vector(9 * my_orientation + 1) - grain_unit_vector(9 * my_orientation + 4) -
                grain_unit_vector(9 * my_orientation + 7);

        fz[0] = grain_unit_vector(9 * my_orientation + 2) + grain_unit_vector(9 * my_orientation + 5) +
                grain_unit_vector(9 * my_orientation + 8);
        fz[1] = grain_unit_vector(9 * my_orientation + 2) - grain_unit_vector(9 * my_orientation + 5) +
                grain_unit_vector(9 * my_orientation + 8);
        fz[2] = grain_unit_vector(9 * my_orientation + 2) + grain_unit_vector(9 * my_orientation + 5) -
                grain_unit_vector(9 * my_orientation + 8);
        fz[3] = grain_unit_vector(9 * my_orientation + 2) - grain_unit_vector(9 * my_orientation + 5) -
                grain_unit_vector(9 * my_orientation + 8);

        float x0 = xp + neighbor_x[capt_dir] - cx;
        float y0 = yp + neighbor_y[capt_dir] - cy;
        float z0 = zp + neighbor_z[capt_dir] - cz;
        float d0 = x0 * fx[0] + y0 * fy[0] + z0 * fz[0];
        float d1 = x0 * fx[1] + y0 * fy[1] + z0 * fz[1];
        float d2 = x0 * fx[2] + y0 * fy[2] + z0 * fz[2];
        float d3 = x0 * fx[3] + y0 * fy[3] + z0 * fz[3];
        float dfabs = fmax(fmax(fabs(d0), fabs(d1)), fmax(fabs(d2), fabs(d3)));
        crit_diagonal_length(26 * index + capt_dir) = dfabs;
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
