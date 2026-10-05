#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

struct Kregion {
    int region_id;
    int region_size;
    std::vector<int> regions_above;
    Kregion(std::vector<std::vector<std::vector<int>>> &GrainID, std::vector<std::vector<std::vector<int>>> &RegionID,
            int region_size_, int region_id_, int k, const int n, const int nx, const int ny, const int nz) {
        region_id = region_id_;
        region_size = region_size_;
        regions_above.resize(region_size);
        int ct_above = 0;
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                if ((GrainID[k][j][i] == n) && (RegionID[k][j][i] == region_id)) {
                    if (k < nz - 1) {
                        if (GrainID[k + 1][j][i] == n) {
                            regions_above[ct_above] = RegionID[k + 1][j][i];
                            ct_above++;
                        }
                    }
                }
            }
        }
        regions_above.resize(ct_above);
        std::sort(regions_above.begin(), regions_above.end());
        regions_above.erase(std::unique(regions_above.begin(), regions_above.end()), regions_above.end());
    }
};

int parseASCIIData(std::istringstream &ss) {
    int read_value;
    ss >> read_value;
    return read_value;
}

std::vector<std::string> splitString(std::string input_string) {
    std::istringstream iss(input_string);
    std::string part;
    std::vector<std::string> parts;
    while (iss >> part) {
        parts.push_back(part);
    }
    return parts;
}

void readASCIIHeader(std::ifstream &input_data_stream, int &nx, int &ny, int &nz, float &deltax, float &xmin,
                     float &ymin, float &zmin) {
    // Ignore first 4 lines
    std::string line_read;
    for (int linenum = 0; linenum < 4; linenum++)
        getline(input_data_stream, line_read);
    // DIMENSIONS nx ny nz
    getline(input_data_stream, line_read);
    std::vector<std::string> dims = splitString(line_read);
    nx = std::stoi(dims[1]);
    ny = std::stoi(dims[2]);
    nz = std::stoi(dims[3]);
    // ORIGIN xmin ymin zmin
    getline(input_data_stream, line_read);
    std::vector<std::string> origin = splitString(line_read);
    xmin = std::stof(origin[1]);
    ymin = std::stof(origin[2]);
    zmin = std::stof(origin[3]);
    // SPACING deltax deltax deltax
    getline(input_data_stream, line_read);
    std::vector<std::string> dx = splitString(line_read);
    deltax = std::stof(dx[1]);
    // Ignore remaining 3 lines
    for (int linenum = 0; linenum < 3; linenum++)
        getline(input_data_stream, line_read);
}

std::vector<int> readASCIIField_ReturnUnique(std::ifstream &input_data_stream,
                                             std::vector<std::vector<std::vector<int>>> &GrainID, const int nx,
                                             const int ny, const int nz) {

    int counter = 0;
    std::vector<int> GrainID_1D(nx * ny * nz);

    for (int k = 0; k < nz; k++) {
        // Get line from file
        std::string line;
        getline(input_data_stream, line);
        // Parse string at spaces
        std::istringstream ss(line);
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int read_value = parseASCIIData(ss);
                GrainID[k][j][i] = read_value;
                GrainID_1D[counter] = read_value;
                counter++;
            }
        }
    }
    std::vector<int> unique_sorted = GrainID_1D;
    std::sort(unique_sorted.begin(), unique_sorted.end());
    unique_sorted.erase(std::unique(unique_sorted.begin(), unique_sorted.end()), unique_sorted.end());
    return unique_sorted;
}

// Does grain n appear at Z = k?
bool foundGrainAtZ(std::vector<std::vector<std::vector<int>>> &GrainID, const int k, const int n, const int nx,
                   const int ny) {
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            if (GrainID[k][j][i] == n)
                return true;
        }
    }
    return false;
}

// Largest Z coordinate at which each grain appears
std::vector<int> getMaxZGrain(std::vector<std::vector<std::vector<int>>> &GrainID, const int num_unique_grains,
                              const int nx, const int ny, const int nz) {

    std::vector<int> MaxZGrain(num_unique_grains, 0);
    for (int n = 0; n < num_unique_grains; n++) {
        for (int k = 0; k < nz; k++) {
            if (foundGrainAtZ(GrainID, k, n, nx, ny))
                MaxZGrain[n] = k;
            else
                k = nz;
        }
    }
    return MaxZGrain;
}

int findNewStartingXY(std::vector<std::vector<std::vector<int>>> &GrainID,
                      std::vector<std::vector<std::vector<int>>> &RegionID, const int n, const int k, int &start_x,
                      int &start_y, const int nx, const int ny) {

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            if ((GrainID[k][j][i] == n) && (RegionID[k][j][i] == -1)) {
                start_y = j;
                start_x = i;
                return true;
            }
        }
    }
    return false;
}

std::vector<int> getNeighboringGrainIDs(std::vector<std::vector<std::vector<int>>> &GrainID, const int k, const int j,
                                        const int i, const int dx[4], const int dy[4], const int nx, const int ny) {
    std::vector<int> NeighborGrainIDs(4);
    for (int dir = 0; dir < 4; dir++) {
        int xn = i + dx[dir];
        int yn = j + dy[dir];
        xn = (xn + nx) % nx;
        yn = (yn + ny) % ny;
        NeighborGrainIDs[dir] = GrainID[k][yn][xn];
    }
    return NeighborGrainIDs;
}

void assignRegions(std::vector<std::vector<std::vector<int>>> &RegionID,
                   std::vector<std::vector<std::vector<int>>> &GrainID, const int n, const int k, const int dx[4],
                   const int dy[4], const int nx, const int ny) {

    // Count how many cells have grain ID = n at Z = k, store an x,y value that has grain ID = n
    int start_x, start_y;
    int cells_to_assign = 0;
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            if (GrainID[k][j][i] == n) {
                cells_to_assign++;
                start_x = i;
                start_y = j;
            }
        }
    }
    // Region counter
    int region_count = 0;
    int cell_ct = 0;
    while (cells_to_assign > 0) {
        // Assign starting cell to the current region
        RegionID[k][start_y][start_x] = region_count;
        cells_to_assign--;
        cell_ct++;
        // Queue storage
        std::vector<int> qx(nx * ny), qy(nx * ny);
        // Queue counters
        int front = 0, back = 0;

        qx[back] = start_x;
        qy[back] = start_y;
        back++;

        while (front < back) {
            int x = qx[front];
            int y = qy[front];
            front++;
            // Check this cell's 4 neighbors to see if they should be added to the region
            for (int dir = 0; dir < 4; dir++) {
                int xn = x + dx[dir];
                int yn = y + dy[dir];
                xn = (xn + nx) % nx;
                yn = (yn + ny) % ny;
                // If neighbor has not been visited (RegionID still has default of -1), assign neighboring cell to this
                // region
                if ((RegionID[k][yn][xn] == -1) && (GrainID[k][yn][xn] == n)) {
                    RegionID[k][yn][xn] = region_count;
                    cells_to_assign--;
                    cell_ct++;
                    qx[back] = xn;
                    qy[back] = yn;
                    back++;
                }
            }
        }
        // Check to see if region size is sufficient: if less than 5 cells were assigned to it, remove those cells from
        // the grain (to be reassigned later to a larger grain). Exception to this is bottom surface at k = 0 where
        // exactly 1 region should exist regardless of its size
        if ((cell_ct < 5) && (k != 0)) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    if ((RegionID[k][j][i] == region_count) && (GrainID[k][j][i] == n)) {
                        GrainID[k][j][i] = -1;
                    }
                }
            }
        }
        // Check if cells are still unassigned
        if (cells_to_assign > 0) {
            // Cells are still unassigned - must be at least 1 more region, increment region counter and get a new
            // starting x and y location
            region_count++;
            cell_ct = 0;
            findNewStartingXY(GrainID, RegionID, n, k, start_x, start_y, nx, ny);
        }
    }
}

int getVectorMode(const std::vector<int> &v) {

    std::vector<int> v_new;
    for (int n = 0; n < 4; n++) {
        if (v[n] != -1)
            v_new.push_back(v[n]);
    }
    if (v_new.size() == 0)
        return -1;

    std::unordered_map<int, int> frequency;
    for (int num : v_new) {
        frequency[num]++;
    }

    int mode = v_new[0];
    int max_count = 0;

    for (const auto &pair : frequency) {
        if (pair.second > max_count) {
            max_count = pair.second;
            mode = pair.first;
        }
    }

    return mode;
}

// At the bottom surface of the domain, only one region ID value should exist per grain - keep only the largest
// connected region of cells with each grain ID, assigning others a placeholder of -1 to be reassigned later
void eliminateBottomSurfaceRegions(std::vector<std::vector<std::vector<int>>> &GrainID,
                                   std::vector<std::vector<std::vector<int>>> &RegionID, int num_unique_grains,
                                   const int nx, const int ny) {

    for (int n = 0; n < num_unique_grains; n++) {
        // Fill vector with region ID values
        std::vector<int> bottom_surface_ids(nx * ny);
        int ct = 0;
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                if (GrainID[0][j][i] == n) {
                    bottom_surface_ids[ct] = RegionID[0][j][i];
                    ct++;
                }
            }
        }
        bottom_surface_ids.resize(ct);
        // Get region ID of largest region
        int largest_region = getVectorMode(bottom_surface_ids);
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                // Remove cells from grain if they are not part of the largest region
                if ((GrainID[0][j][i] == n) && (RegionID[0][j][i] != largest_region))
                    GrainID[0][j][i] = -1;
            }
        }
    }
}

std::vector<int> getRegionSizeCounts(std::vector<std::vector<std::vector<int>>> &GrainID,
                                     std::vector<std::vector<std::vector<int>>> &RegionID, int n, int k, const int nx,
                                     const int ny) {
    std::vector<int> bottom_surface_ids(nx * ny);
    int ct = 0;
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            if (GrainID[k][j][i] == n) {
                bottom_surface_ids[ct] = RegionID[k][j][i];
                ct++;
            }
        }
    }
    bottom_surface_ids.resize(ct);
    if (ct == 0)
        return bottom_surface_ids;

    std::vector<int> unique_sorted = bottom_surface_ids;
    std::sort(unique_sorted.begin(), unique_sorted.end());
    unique_sorted.erase(std::unique(unique_sorted.begin(), unique_sorted.end()), unique_sorted.end());
    int num_regions = unique_sorted.size();
    std::vector<int> region_size_counts(num_regions);
    for (int reg_num = 0; reg_num < num_regions; reg_num++)
        region_size_counts[reg_num] = std::count(bottom_surface_ids.begin(), bottom_surface_ids.end(), reg_num);
    return region_size_counts;
}

void assignRemovedCellsToGrains(std::vector<std::vector<std::vector<int>>> &GrainID, const int k, const int dx[4],
                                const int dy[4], const int nx, const int ny) {

    int unassigned_count = 0;
    std::vector<int> unassigned_x;
    std::vector<int> unassigned_y;
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            if (GrainID[k][j][i] == -1) {
                unassigned_count++;
                unassigned_x.push_back(i);
                unassigned_y.push_back(j);
            }
        }
    }
    // std::cout << "Assigning grain IDs to " << unassigned_count << " cells at Z = " << k << std::endl;

    int unassigned_remaining = unassigned_count;
    while (unassigned_remaining > 0) {
        unassigned_remaining = 0;
        std::vector<int> new_grain_id(unassigned_count, -1);
        for (int cell_idx = 0; cell_idx < unassigned_count; cell_idx++) {
            const int i = unassigned_x[cell_idx];
            const int j = unassigned_y[cell_idx];
            if (GrainID[k][j][i] == -1) {
                std::vector<int> NeighborGrainIDs = getNeighboringGrainIDs(GrainID, k, j, i, dx, dy, nx, ny);
                int grainid_mode = getVectorMode(NeighborGrainIDs);
                if (grainid_mode != -1)
                    new_grain_id[cell_idx] = grainid_mode; // GrainID[k][j][i] = grainid_mode;
                else
                    unassigned_remaining++;
            }
        }
        for (int cell_idx = 0; cell_idx < unassigned_count; cell_idx++) {
            const int i = unassigned_x[cell_idx];
            const int j = unassigned_y[cell_idx];
            if (new_grain_id[cell_idx] != -1) {
                GrainID[k][j][i] = new_grain_id[cell_idx];
            }
        }
    }
}

int main(int argc, char *argv[]) {

    auto start_time = std::chrono::steady_clock::now();

    if (argc == 1)
        throw std::runtime_error("Error: Path/name of ExaCA vtk input file is needed on the command line");
    std::string filename_in = argv[1];
    std::string filename_base = filename_in.substr(0, filename_in.length() - 4);
    std::string filename_out = filename_base + "_Smooth.vtk";

    // Grid connectivity
    int dx[4] = {-1, 0, 1, 0};
    int dy[4] = {0, -1, 0, 1};

    // Reads portion of a paraview file and places data in the appropriate data structure
    // ASCII data at each Z value is separated by a newline
    std::ifstream grain_input_stream;
    grain_input_stream.open(filename_in);
    // Read header, get grid size and spacing
    int nx, ny, nz;
    float deltax, xmin, ymin, zmin;
    readASCIIHeader(grain_input_stream, nx, ny, nz, deltax, xmin, ymin, zmin);
    const int N = nx * ny * nz;

    // Empty structures for 3D grain ID, region ID data
    std::vector<std::vector<std::vector<int>>> GrainID, RegionID;
    for (int k = 0; k < nz; k++) {
        std::vector<std::vector<int>> dv2;
        for (int j = 0; j < ny; j++) {
            std::vector<int> dv1;
            for (int i = 0; i < nx; i++) {
                dv1.push_back(-1);
            }
            dv2.push_back(dv1);
        }
        GrainID.push_back(dv2);
        RegionID.push_back(dv2);
    }

    // Unique grain IDs list
    std::vector<int> UniqueIDs = readASCIIField_ReturnUnique(grain_input_stream, GrainID, nx, ny, nz);
    const int num_unique_grains = UniqueIDs.size();
    grain_input_stream.close();

    // Renumber grains - get unique IDs/map them
    std::cout << "Read file " << filename_in << " : number of unique grain IDs = " << num_unique_grains << std::endl;
    for (int n = 0; n < num_unique_grains; n++) {
        int unique_id = UniqueIDs[n];
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    if (GrainID[k][j][i] == unique_id) {
                        GrainID[k][j][i] = n;
                    }
                }
            }
        }
    }

    // Remove sharp edges from grains
    for (int smooth_iter = 0; smooth_iter < 8; smooth_iter++) {
        int num_changed = 0;
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    std::vector<int> NeighborGrainIDs = getNeighboringGrainIDs(GrainID, k, j, i, dx, dy, nx, ny);
                    bool neighbors01 = (NeighborGrainIDs[0] == NeighborGrainIDs[1]);
                    bool neighbors02 = (NeighborGrainIDs[0] == NeighborGrainIDs[2]);
                    bool neighbors03 = (NeighborGrainIDs[0] == NeighborGrainIDs[3]);
                    bool neighbors12 = (NeighborGrainIDs[1] == NeighborGrainIDs[2]);
                    bool neighbors13 = (NeighborGrainIDs[1] == NeighborGrainIDs[3]);
                    bool neighbors23 = (NeighborGrainIDs[2] == NeighborGrainIDs[3]);
                    int sum_neighbors_matching =
                        neighbors01 + neighbors02 + neighbors03 + neighbors12 + neighbors13 + neighbors23;
                    if (sum_neighbors_matching == 6) {
                        // All 4 neighbors match. Do I? If not, change my ID
                        if (GrainID[k][j][i] != NeighborGrainIDs[0]) {
                            GrainID[k][j][i] = NeighborGrainIDs[0];
                            num_changed++;
                        }
                    }
                    else if (sum_neighbors_matching == 3) {
                        // 3 of the 4 neighbors match. Which ones?
                        // Assume 0,2,3, 0,1,2, or 0,1,3 match
                        int majority_neighbor_id = NeighborGrainIDs[0];
                        if ((!neighbors01) && (!neighbors02)) {
                            // Neighbors 1,2,3 match
                            majority_neighbor_id = NeighborGrainIDs[1];
                        }
                        // Do I match these 3 neighbors? If not, change my ID
                        if (GrainID[k][j][i] != majority_neighbor_id) {
                            GrainID[k][j][i] = majority_neighbor_id;
                            num_changed++;
                        }
                    }
                    //                else if (sum_neighbors_matching == 2) {
                    //                    // Which 2 neighbors match?
                    //                    int majority_neighbor_id = NeighborGrainIDs[0];
                    //                    if ((neighbors12) || (neighbors13))
                    //                        majority_neighbor_id = NeighborGrainIDs[1];
                    //                    else if (neighbors23)
                    //                        majority_neighbor_id = NeighborGrainIDs[2];
                    //                    // Do I match these 2 neighbors? If not, change my ID
                    //                    if (GrainID[k][j][i] != majority_neighbor_id)
                    //                        GrainID[k][j][i] = majority_neighbor_id;
                    //                }
                }
            }
        }
        std::cout << "Number of cells reassigned grain IDs: " << num_changed << std::endl;
    }
    std::cout << "Preliminary smoothing of grains done" << std::endl;

    // Prelimiary max Z for each grain
    std::vector<int> MaxZGrain = getMaxZGrain(GrainID, num_unique_grains, nx, ny, nz);

    std::cout << "Assigning regions" << std::endl;
    // Create preliminary connectivity regions for each grain and each Z coordinate
    for (int n = 0; n < num_unique_grains; n++) {
        for (int k = 0; k <= MaxZGrain[n]; k++) {
            // Assign grains at this k value to a RegionID
            assignRegions(RegionID, GrainID, n, k, dx, dy, nx, ny);
        }
    }
    std::cout << "Done assigning regions" << std::endl;
    // Bottom surface: only the largest region should exist
    eliminateBottomSurfaceRegions(GrainID, RegionID, num_unique_grains, nx, ny);
    // Fill in -1 values for removed connectivity regions, first resetting RegionID to -1s
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                RegionID[k][j][i] = -1;
            }
        }
        assignRemovedCellsToGrains(GrainID, k, dx, dy, nx, ny);
    }
    std::cout << "Reassigning disconnected regions" << std::endl;

    // Reassign region IDs for each grain
    for (int n = 0; n < num_unique_grains; n++) {
        for (int k = 0; k <= MaxZGrain[n]; k++) {
            // Assign grains at this k value to a region, update vector of regions and counts
            assignRegions(RegionID, GrainID, n, k, dx, dy, nx, ny);
        }
    }
    // Updated max Z for each grain
    MaxZGrain = getMaxZGrain(GrainID, num_unique_grains, nx, ny, nz);
    std::cout << "Done reassigning regions" << std::endl;

    // Determine paths connecting the bottom surface of each grain to the top of each grain
    for (int n = 0; n < num_unique_grains; n++) {
        // Vector of "kregions" containing grain connectivity between Z slices
        std::vector<Kregion> path_data;
        // Stores the index of the last k region at this Z height
        std::vector<int> end_k(MaxZGrain[n] + 1);
        int kreg_count = 0;
        for (int k = 0; k <= MaxZGrain[n]; k++) {
            // Number of regions at this k value, and their sizes
            std::vector<int> region_size_counts = getRegionSizeCounts(GrainID, RegionID, n, k, nx, ny);
            int num_regions = region_size_counts.size();
            if (num_regions > 0) {
                for (int reg_num = 0; reg_num < num_regions; reg_num++) {
                    // Create new k region and add to vector of k regions: tracks region ID of regions above this one
                    // and the size and ID of this region
                    Kregion kregion(GrainID, RegionID, region_size_counts[reg_num], reg_num, k, n, nx, ny, nz);
                    path_data.push_back(kregion);
                    // Increment region count for this k value
                    kreg_count++;
                }
            }
            end_k[k] = kreg_count;
        }
        // Explore paths through the kregions for this grain
        // Number of cells along each path
        std::vector<int> path_sizes;
        // Whether each path was being tracked prior to the current iteration
        std::vector<bool> path_alive_old;
        // Whether each path is currently being tracked
        std::vector<bool> path_alive;
        // Index of each path
        std::vector<int> path_knum;
        // Region ID at each Z coordinate for each path
        std::vector<std::vector<int>> all_paths;
        for (int i = 0; i < nx * ny; i++) {
            std::vector<int> temp;
            for (int k = 0; k < nz; k++) {
                temp.push_back(-1);
            }
            all_paths.push_back(temp);
        }

        int path_count = 1;
        int path_count_old = 1;
        path_sizes.push_back(0);
        path_alive.push_back(true);
        path_alive_old.push_back(true);
        path_knum.push_back(0);
        int k = 0;
        bool checking_paths = true;
        while (checking_paths) {
            path_count_old = path_count;
            for (int i = 0; i < path_count_old; i++) {
                if (path_alive_old[i]) {
                    // If path was alive at the start of this time step, add the region ID and size
                    int my_path_knum = path_knum[i];
                    all_paths[i][k] = path_data[my_path_knum].region_id;
                    path_sizes[i] += path_data[my_path_knum].region_size;
                    std::vector<int> next_regions = path_data[my_path_knum].regions_above;

                    if (next_regions.size() == 1) {
                        // This region connects to one other region - path continues, update path_knum for Z = k + 1
                        for (int kk = end_k[k]; kk < end_k[k + 1]; kk++) {
                            if (path_data[kk].region_id == next_regions[0]) {
                                path_knum[i] = kk;
                            }
                        }
                    }
                    else if (next_regions.size() > 1) {
                        // This path splits into 2 or more regions - create new paths for the splits
                        int num_new_paths = next_regions.size() - 1;
                        for (int ii = 1; ii <= num_new_paths; ii++) {
                            // Copy current path size to new paths
                            path_count++;
                            path_alive.push_back(true);
                            path_sizes.push_back(path_sizes[i]);
                            for (int kold = 0; kold <= k; kold++)
                                all_paths[path_count - 1][kold] = all_paths[i][kold];
                            for (int kk = end_k[k]; kk < end_k[k + 1]; kk++) {
                                if (path_data[kk].region_id == next_regions[ii]) {
                                    path_knum.push_back(kk);
                                    kk = end_k[k + 1];
                                }
                            }
                        }
                        // Continuation of current path without splitting
                        for (int kk = end_k[k]; kk < end_k[k + 1]; kk++) {
                            if (path_data[kk].region_id == next_regions[0]) {
                                path_knum[i] = kk;
                            }
                        }
                    }
                    else if (next_regions.size() == 0) {
                        // Path ends
                        path_alive[i] = false;
                    }
                }
            }
            int act_paths = 0;
            // Copy path status for next iteration and count the number of active paths
            path_alive_old.resize(path_count);
            for (int i = 0; i < path_count; i++) {
                path_alive_old[i] = path_alive[i];
                if (path_alive_old[i])
                    act_paths++;
            }
            // Increment k to check next Z location
            k++;
            // Keep going until no active paths remain
            if (act_paths == 0) {
                checking_paths = false;
            }
        }

        // Index associated with the path that includes the largest number of cells for this grain ID
        auto max_it = std::max_element(path_sizes.begin(), path_sizes.end());
        auto max_idx = std::distance(path_sizes.begin(), max_it);
        bool printing_path = true;
        k = 0;
        // Iterate over all other paths and reset the grain IDs to the default of -1 for reassignment later - cannot
        // have grains that are disconnected at a given Z = k
        while (printing_path) {
            if (all_paths[max_idx][k] > -1) {
                for (int j = 0; j < ny; j++) {
                    for (int i = 0; i < nx; i++) {
                        if ((GrainID[k][j][i] == n) && (RegionID[k][j][i] != all_paths[max_idx][k]))
                            GrainID[k][j][i] = -1;
                    }
                }
                k++;
                if (k == nz)
                    printing_path = false;
            }
            else {
                // No cells should have the grain ID of n above this k value
                for (int kk = k; kk < nz; kk++) {
                    for (int j = 0; j < ny; j++) {
                        for (int i = 0; i < nx; i++) {
                            if (GrainID[kk][j][i] == n)
                                GrainID[kk][j][i] = -1;
                        }
                    }
                }
                printing_path = false;
            }
        }
    }

    // Fill in removed regions of grains
    for (int k = 0; k < nz; k++)
        assignRemovedCellsToGrains(GrainID, k, dx, dy, nx, ny);

    // Iteration of filling in removed regions of grains and removing grains from sections where they don't have at
    // least 2 neighbors
    bool reassigning_grains = true;
    while (reassigning_grains) {
        // For each grain in each cross-section, ensure that there are at least 3 neighboring grains - otherwise,
        // eliminate the grain
        std::cout << "Reassigning grain IDs to ensure triple junction connectivity" << std::endl;
        int removed_grains = 0;
        for (int k = 0; k < nz; k++) {
            for (int n = 0; n < num_unique_grains; n++) {
                // Max possible size for all_neighbor_ids - resize later
                std::vector<int> all_neighbor_ids(4 * nx * ny);
                // Counter for how many times grain n appears in the cross-section at Z = k
                int counter_this_grain = 0;
                for (int j = 0; j < ny; j++) {
                    for (int i = 0; i < nx; i++) {
                        if (GrainID[k][j][i] == n) {
                            // Store grain IDs of this cell's 4 neighbors (at least one of which should be the same as
                            // n)
                            std::vector<int> NeighborGrainIDs =
                                getNeighboringGrainIDs(GrainID, k, j, i, dx, dy, nx, ny);
                            for (int dir = 0; dir < 4; dir++)
                                all_neighbor_ids[4 * counter_this_grain + dir] = NeighborGrainIDs[dir];
                            // Increment the counter
                            counter_this_grain++;
                        }
                    }
                }
                // Get vector of unique grain IDs surrounding the cells associated with grain n
                all_neighbor_ids.resize(4 * counter_this_grain);
                std::sort(all_neighbor_ids.begin(), all_neighbor_ids.end());
                all_neighbor_ids.erase(std::unique(all_neighbor_ids.begin(), all_neighbor_ids.end()),
                                       all_neighbor_ids.end());
                //              // If grain n appears at this k location but borders only 2 other grains
                //              (all_neighbor_ids only has 3 of fewer unique members, including n), cells with this
                //              grain ID at this k location and all k locations above it should be reassigned
                if ((counter_this_grain > 0) && (all_neighbor_ids.size() < 4)) {
                    removed_grains++;
                    for (int kk = k; kk < nz; kk++) {
                        for (int j = 0; j < ny; j++) {
                            for (int i = 0; i < nx; i++) {
                                if (GrainID[kk][j][i] == n) {
                                    GrainID[kk][j][i] = -1;
                                }
                            }
                        }
                    }
                }
            }
        }
        // Fill in removed regions of grains
        for (int k = 0; k < nz; k++) {
            assignRemovedCellsToGrains(GrainID, k, dx, dy, nx, ny);
        }
        // If no change to the grain structure, triple junction insertion is complete
        if (removed_grains == 0)
            reassigning_grains = false;
    }

    std::cout << "Writing output file " << filename_out << std::endl;
    std::ofstream grain_output_stream;
    grain_output_stream.open(filename_out);
    grain_output_stream << "# vtk DataFile Version 3.0" << std::endl;
    grain_output_stream << "vtk output" << std::endl;
    grain_output_stream << "ASCII" << std::endl;
    grain_output_stream << "DATASET STRUCTURED_POINTS" << std::endl;
    grain_output_stream << "DIMENSIONS " << nx << " " << ny << " " << nz << std::endl;
    grain_output_stream << "ORIGIN " << xmin << " " << ymin << " " << zmin << std::endl;
    grain_output_stream << "SPACING " << deltax << " " << deltax << " " << deltax << std::endl;
    grain_output_stream << std::fixed << "POINT_DATA " << N << std::endl;
    grain_output_stream << "SCALARS grainID int 1" << std::endl;
    grain_output_stream << "LOOKUP_TABLE default" << std::endl;
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                // Map unique IDs back to original IDs
                int original_grain_id = UniqueIDs[GrainID[k][j][i]];
                grain_output_stream << original_grain_id << " ";
            }
        }
        grain_output_stream << std::endl;
    }
    grain_output_stream.close();

    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Execution time: " << elapsed_time.count() << " s" << std::endl;

    return 0;
}
