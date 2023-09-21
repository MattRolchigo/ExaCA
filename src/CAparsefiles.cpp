// Copyright 2021-2023 Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "CAparsefiles.hpp"

#include "CAconfig.hpp"
#include "CAfunctions.hpp"

#include "mpi.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <regex>

// Functions that are used to simplify the parsing of input files, either by ExaCA or related utilities

//*****************************************************************************/
// Remove whitespace from "line", optional argument to take only portion of the line after position "pos"
std::string removeWhitespace(std::string line, int pos) {

    std::string val = line.substr(pos + 1, std::string::npos);
    std::regex r("\\s+");
    val = std::regex_replace(val, r, "");
    return val;
}

// Check if a string is Y (true) or N (false)
bool getInputBool(std::string val_input) {
    std::string val = removeWhitespace(val_input);
    if (val == "N") {
        return false;
    }
    else if (val == "Y") {
        return true;
    }
    else {
        std::string error = "Input \"" + val + "\" must be \"Y\" or \"N\".";
        throw std::runtime_error(error);
    }
}

// Convert string "val_input" to base 10 integer
int getInputInt(std::string val_input) {
    int IntFromString = stoi(val_input, nullptr, 10);
    return IntFromString;
}

// Convert string "val_input" to float value multipled by 10^(factor)
float getInputFloat(std::string val_input, int factor) {
    float FloatFromString = atof(val_input.c_str()) * pow(10, factor);
    return FloatFromString;
}

// Convert string "val_input" to double value multipled by 10^(factor)
double getInputDouble(std::string val_input, int factor) {
    double DoubleFromString = std::stod(val_input.c_str()) * pow(10, factor);
    return DoubleFromString;
}

// Given a string ("line"), parse at "separator" (commas used by default)
// Modifies "parsed_line" to hold the separated values
// expected_num_values may be larger than parsed_line_size, if only a portion of the line is being parsed
void splitString(std::string line, std::vector<std::string> &parsed_line, int expected_num_values, char separator) {
    // Make sure the right number of values are present on the line - one more than the number of separators
    int actual_num_values = std::count(line.begin(), line.end(), separator) + 1;
    if (expected_num_values != actual_num_values) {
        std::string error = "Error: Expected " + std::to_string(expected_num_values) +
                            " values while reading file; but " + std::to_string(actual_num_values) + " were found";
        throw std::runtime_error(error);
    }
    // Separate the line into its components, now that the number of values has been checked
    std::size_t parsed_line_size = parsed_line.size();
    for (std::size_t n = 0; n < parsed_line_size - 1; n++) {
        std::size_t pos = line.find(separator);
        parsed_line[n] = line.substr(0, pos);
        line = line.substr(pos + 1, std::string::npos);
    }
    parsed_line[parsed_line_size - 1] = line;
}

// Check to make sure that all expected column names appear in the header for this temperature file
void checkForHeaderValues(std::string header_line) {

    // Header values from file
    std::size_t header_size = 6;
    std::vector<std::string> header_values(header_size, "");
    splitString(header_line, header_values, 6);

    std::vector<std::vector<std::string>> expected_values = {{"x"}, {"y"}, {"z"}, {"tm"}, {"tl", "ts"}, {"r", "cr"}};

    // Case insensitive comparison
    for (std::size_t n = 0; n < header_size; n++) {
        auto val = removeWhitespace(header_values[n]);
        std::transform(val.begin(), val.end(), val.begin(), ::tolower);

        // Check each header column label against the expected value(s) - throw error if no match
        std::size_t options_size = expected_values[n].size();
        for (std::size_t e = 0; e < options_size; e++) {
            auto ev = expected_values[n][e];
            if (val == ev)
                break;
            else if (e == options_size - 1)
                throw std::runtime_error(ev + " not found in temperature file header");
        }
    }
}

bool checkFileExists(const std::string path, const int id, const bool error) {
    std::ifstream stream;
    stream.open(path);
    if (!(stream.is_open())) {
        stream.close();
        if (error)
            throw std::runtime_error("Could not locate/open \"" + path + "\"");
        else
            return false;
    }
    stream.close();
    if (id == 0)
        std::cout << "Opened \"" << path << "\"" << std::endl;
    return true;
}

std::string checkFileInstalled(const std::string name, const int id) {
    // Path to file. Prefer installed location; if not installed use source location.
    std::string path = ExaCA_DATA_INSTALL;
    std::string file = path + "/" + name;
    bool files_installed = checkFileExists(file, id, false);
    if (!files_installed) {
        // If full file path, just use it.
        if (name.substr(0, 1) == "/") {
            file = name;
        }
        // If a relative path, it has to be with respect to the source path.
        else {
            path = ExaCA_DATA_SOURCE;
            file = path + "/" + name;
        }
        checkFileExists(file, id);
    }
    return file;
}

// Make sure file contains data
void checkFileNotEmpty(std::string testfilename) {
    std::ifstream testfilestream;
    testfilestream.open(testfilename);
    std::string testline;
    std::getline(testfilestream, testline);
    if (testline.empty())
        throw std::runtime_error("First line of file " + testfilename + " appears empty");
    testfilestream.close();
}

// Check the field names from the given input (Fieldtype = PrintFieldsInit or PrintFieldsFinal) against the possible
// fieldnames listed in Fieldnames_key. Fill the vector PrintFields_given with true or false values depending on whether
// the corresponding field name from Fieldnames_key appeared in the input or not
std::vector<bool> getPrintFieldValues(nlohmann::json inputdata, std::string Fieldtype,
                                      std::vector<std::string> Fieldnames_key) {
    int NumFields_key = Fieldnames_key.size();
    int NumFields_given = inputdata["Printing"][Fieldtype].size();
    std::vector<bool> PrintFields_given(NumFields_key, false);
    // Check each given field against each possible input field name
    for (int field_given = 0; field_given < NumFields_given; field_given++) {
        for (int field_key = 0; field_key < NumFields_key; field_key++) {
            if (inputdata["Printing"][Fieldtype][field_given] == Fieldnames_key[field_key])
                PrintFields_given[field_key] = true;
        }
    }
    return PrintFields_given;
}

// Read x, y, z coordinates in tempfile_thislayer (temperature file in either an ASCII or binary format) and return the
// min and max values. Also store the estimated starting/ending time steps for the layer
std::array<double, 6> parseTemperatureCoordinateMinMax(std::string tempfile_thislayer, bool BinaryInputData, double deltat, double FreezingRange, int* StartTimeStep, int* FinishTimeStep, int layernumber) {

    std::array<double, 6> XYZMinMax;
    std::ifstream TemperatureFilestream;
    TemperatureFilestream.open(tempfile_thislayer);

    if (!(BinaryInputData)) {
        // Read the header line data
        // Make sure the first line contains all required column names: x, y, z, tm, tl, cr
        std::string HeaderLine;
        getline(TemperatureFilestream, HeaderLine);
        checkForHeaderValues(HeaderLine);
    }

    // Units are assumed to be in meters, meters, seconds, seconds, and K/second
    int XYZPointCount_Estimate = 1000000;
    std::vector<double> XCoordinates(XYZPointCount_Estimate), YCoordinates(XYZPointCount_Estimate),
        ZCoordinates(XYZPointCount_Estimate);
    long unsigned int XYZPointCounter = 0;
    double tm_min = DBL_MAX;
    double tl_max = DBL_MIN;
    double stored_cr = 0.0;
    if (BinaryInputData) {
        while (!TemperatureFilestream.eof()) {
            // Get x from the binary string, or, if no data is left, exit the file read
            double XValue = ReadBinaryData<double>(TemperatureFilestream);
            if (!(TemperatureFilestream))
                break;
            // Store the x value that was read, and parse the y and z values
            XCoordinates[XYZPointCounter] = XValue;
            YCoordinates[XYZPointCounter] = ReadBinaryData<double>(TemperatureFilestream);
            ZCoordinates[XYZPointCounter] = ReadBinaryData<double>(TemperatureFilestream);
            // Get tm and tl values, ignored the cr value associated with this x, y, z
            double tmelt = ReadBinaryData<double>(TemperatureFilestream);
            double tl = ReadBinaryData<double>(TemperatureFilestream);
            double cr = ReadBinaryData<double>(TemperatureFilestream);
            // Store smallest tm and largest tl
            if (tmelt < tm_min)
                tm_min = tmelt;
            if (tl > tl_max) {
                tl_max = tl;
                stored_cr = cr;
            }
            XYZPointCounter++;
            if (XYZPointCounter == XCoordinates.size()) {
                XCoordinates.resize(XYZPointCounter + XYZPointCount_Estimate);
                YCoordinates.resize(XYZPointCounter + XYZPointCount_Estimate);
                ZCoordinates.resize(XYZPointCounter + XYZPointCount_Estimate);
            }
        }
    }
    else {
        while (!TemperatureFilestream.eof()) {
            std::vector<std::string> ParsedLine(6); // Get x, y, z - ignore tm, tl, cr
            std::string ReadLine;
            if (!getline(TemperatureFilestream, ReadLine))
                break;
            splitString(ReadLine, ParsedLine, 6);
            // Get values from ParsedLine
            XCoordinates[XYZPointCounter] = getInputDouble(ParsedLine[0]);
            YCoordinates[XYZPointCounter] = getInputDouble(ParsedLine[1]);
            ZCoordinates[XYZPointCounter] = getInputDouble(ParsedLine[2]);
            double tmelt = getInputDouble(ParsedLine[3]);
            double tl = getInputDouble(ParsedLine[4]);
            double cr = getInputDouble(ParsedLine[5]);
            // Store smallest tm and largest tl
            if (tmelt < tm_min)
                tm_min = tmelt;
            if (tl > tl_max) {
                tl_max = tl;
                stored_cr = cr;
            }
            XYZPointCounter++;
            if (XYZPointCounter == XCoordinates.size()) {
                XCoordinates.resize(XYZPointCounter + XYZPointCount_Estimate);
                YCoordinates.resize(XYZPointCounter + XYZPointCount_Estimate);
                ZCoordinates.resize(XYZPointCounter + XYZPointCount_Estimate);
            }
        }
    }

    XCoordinates.resize(XYZPointCounter);
    YCoordinates.resize(XYZPointCounter);
    ZCoordinates.resize(XYZPointCounter);
    StartTimeStep[layernumber] = round(tm_min / deltat);
    FinishTimeStep[layernumber] = round((tl_max + (FreezingRange / stored_cr)) / deltat);
    TemperatureFilestream.close();

    // Min/max x, y, and z coordinates from this layer's data
    XYZMinMax[0] = *min_element(XCoordinates.begin(), XCoordinates.end());
    XYZMinMax[1] = *max_element(XCoordinates.begin(), XCoordinates.end());
    XYZMinMax[2] = *min_element(YCoordinates.begin(), YCoordinates.end());
    XYZMinMax[3] = *max_element(YCoordinates.begin(), YCoordinates.end());
    XYZMinMax[4] = *min_element(ZCoordinates.begin(), ZCoordinates.end());
    XYZMinMax[5] = *max_element(ZCoordinates.begin(), ZCoordinates.end());
    return XYZMinMax;
}

// Determine the shift in the data in X or Y for LineToRaster problems
double shiftTemperatureCoordinateMinMax(std::array<double, 6> &XYZMinMax, double deltax, std::string scan_direction, std::string shift_direction) {
    // If calculating shift in X, shifting XYZMinMax[0] and XYZMinMax[1]
    // If calculating shift in Y, shifting XYZMinMax[2] and XYZMinMax[3]
    int shift_direction_upper_index, shift_direction_lower_index;
    if (shift_direction == "X") {
        shift_direction_lower_index = 0;
        shift_direction_upper_index = 1;
    }
    else if (shift_direction == "Y") {
        shift_direction_lower_index = 2;
        shift_direction_upper_index = 3;
    }
    else
        throw std::runtime_error("Error: Invalid shift direction encountered in shiftTemperatureCoordinateMinMax - should be X or Y");
    double raster_shift;
    if (scan_direction == shift_direction) {
        // Data starts at X or Y = 0, determine the offset to XYZMinMax associated with this shift
        double zero_loc = XYZMinMax[shift_direction_lower_index];
        // Ensure that the shift is an integer multiple of the cell size, to center the data as close to X or Y = 0 as possible
        int raster_shift_cells = - round(zero_loc / deltax);
        raster_shift = raster_shift_cells * deltax;
        XYZMinMax[shift_direction_upper_index] += raster_shift;
        XYZMinMax[shift_direction_lower_index] += raster_shift;
    }
    else {
        // The data should be centered at X or Y = 0, determine the offset to XYZMinMax associated with this shift
        double centerline = (XYZMinMax[shift_direction_upper_index] + XYZMinMax[shift_direction_lower_index]) / 2.0;
        // Ensure that the shift is an integer multiple of the cell size, to center the data as close to X or Y = centerline as possible
        int raster_shift_cells = - round(centerline / deltax);
        raster_shift = raster_shift_cells * deltax;
        XYZMinMax[shift_direction_upper_index] += raster_shift;
        XYZMinMax[shift_direction_lower_index] += raster_shift;
    }
    return raster_shift;
}

// Get the value for which the line melting/soldification times should be offset to ensure they solidify in succession
double shiftTmeltTliquidus(int StartTimeStep, int FinishTimeStep, double deltat) {
    double time_shift = deltat * (FinishTimeStep - StartTimeStep);
    return time_shift;
}
