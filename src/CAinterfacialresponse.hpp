// Copyright Lawrence Livermore National Security, LLC and other ExaCA Project Developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef EXACA_IRF_HPP
#define EXACA_IRF_HPP

#include "CAinputdata.hpp"

#include <Kokkos_Core.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Interfacial response function with various functional forms.
struct InterfacialResponseFunction {

    InterfacialResponseInputs _inputs;

    // Constructor
    InterfacialResponseFunction(const double deltat, const double deltax, const InterfacialResponseInputs inputs)
        : _inputs(inputs) {
        normalize(deltat, deltax);
    }

    void normalize(const double deltat, const double deltax) {
        if (_inputs.function_ferrite == _inputs.cubic) {
            // Normalize all 4 coefficients: V = A*x^3 + B*x^2 + C*x + D
            _inputs.A_ferrite *= static_cast<float>(deltat / deltax);
            _inputs.B_ferrite *= static_cast<float>(deltat / deltax);
            _inputs.C_ferrite *= static_cast<float>(deltat / deltax);
            _inputs.D_ferrite *= static_cast<float>(deltat / deltax);
        }
        else if (_inputs.function_ferrite == _inputs.quadratic) {
            // Normalize the 3 relevant coefficients: V = A*x^2 + B*x + C
            _inputs.A_ferrite *= static_cast<float>(deltat / deltax);
            _inputs.B_ferrite *= static_cast<float>(deltat / deltax);
            _inputs.C_ferrite *= static_cast<float>(deltat / deltax);
        }
        else if (_inputs.function_ferrite == _inputs.power) {
            // Normalize only the leading and last coefficient: V = A*x^B + C
            _inputs.A_ferrite *= static_cast<float>(deltat / deltax);
            _inputs.C_ferrite *= static_cast<float>(deltat / deltax);
        }
        
        if (_inputs.function_austenite == _inputs.cubic) {
            // Normalize all 4 coefficients: V = A*x^3 + B*x^2 + C*x + D
            _inputs.A_austenite *= static_cast<float>(deltat / deltax);
            _inputs.B_austenite *= static_cast<float>(deltat / deltax);
            _inputs.C_austenite *= static_cast<float>(deltat / deltax);
            _inputs.D_austenite *= static_cast<float>(deltat / deltax);
        }
        else if (_inputs.function_austenite == _inputs.quadratic) {
            // Normalize the 3 relevant coefficients: V = A*x^2 + B*x + C
            _inputs.A_austenite *= static_cast<float>(deltat / deltax);
            _inputs.B_austenite *= static_cast<float>(deltat / deltax);
            _inputs.C_austenite *= static_cast<float>(deltat / deltax);
        }
        else if (_inputs.function_austenite == _inputs.power) {
            // Normalize only the leading and last coefficient: V = A*x^B + C
            _inputs.A_austenite *= static_cast<float>(deltat / deltax);
            _inputs.C_austenite *= static_cast<float>(deltat / deltax);
        }
    }

    // Compute velocity from local undercooling.
    // functional form is assumed to be cubic if not explicitly given in input file
    KOKKOS_INLINE_FUNCTION
    float compute_austenite(const float loc_u) const {
        float V_austenite;
        if (_inputs.function_austenite == _inputs.quadratic)
            V_austenite = _inputs.A_austenite * Kokkos::pow(loc_u, 2.0) + _inputs.B_austenite * loc_u + _inputs.C_austenite;
        else if (_inputs.function_austenite == _inputs.power)
            V_austenite = _inputs.A_austenite * Kokkos::pow(loc_u, _inputs.B_austenite) + _inputs.C_austenite;
        else
            V_austenite = _inputs.A_austenite * Kokkos::pow(loc_u, 3.0) + _inputs.B_austenite * Kokkos::pow(loc_u, 2.0) + _inputs.C_austenite * loc_u +
                _inputs.D_austenite;
        return Kokkos::fmax(0.0, V_austenite);
    }
    KOKKOS_INLINE_FUNCTION
    float compute_ferrite(const float loc_u) const {
        float V_ferrite;
        if (_inputs.function_ferrite == _inputs.quadratic)
            V_ferrite = _inputs.A_ferrite * Kokkos::pow(loc_u, 2.0) + _inputs.B_ferrite * loc_u + _inputs.C_ferrite;
        else if (_inputs.function_ferrite == _inputs.power)
            V_ferrite = _inputs.A_ferrite * Kokkos::pow(loc_u, _inputs.B_ferrite) + _inputs.C_ferrite;
        else
            V_ferrite = _inputs.A_ferrite * Kokkos::pow(loc_u, 3.0) + _inputs.B_ferrite * Kokkos::pow(loc_u, 2.0) + _inputs.C_ferrite * loc_u +
                _inputs.D_ferrite;
        return Kokkos::fmax(0.0, V_ferrite);
    }

    std::string functionName() {
        // Not storing string due to Cuda warnings when constructing on device.
        if (_inputs.function_ferrite == _inputs.cubic)
            return "cubic";
        else if (_inputs.function_ferrite == _inputs.quadratic)
            return "quadratic";
        else if (_inputs.function_ferrite == _inputs.power)
            return "power";
        if (_inputs.function_austenite == _inputs.cubic)
            return "cubic";
        else if (_inputs.function_austenite == _inputs.quadratic)
            return "quadratic";
        else if (_inputs.function_austenite == _inputs.power)
            return "power";
        
        // Should never make it here
        return "none";
    }
    KOKKOS_INLINE_FUNCTION
    int getPreferredPhase(const float loc_u) const {
        const double V_ferrite = compute_ferrite(loc_u);
        const double V_austenite = compute_austenite(loc_u);
        if (V_ferrite > V_austenite)
            return 0;
        else
            return 1;
    }

    auto A_ferrite() { return _inputs.A_ferrite; }
    auto B_ferrite() { return _inputs.B_ferrite; }
    auto C_ferrite() { return _inputs.C_ferrite; }
    auto D_ferrite() { return _inputs.D_ferrite; }
    auto freezingRange_ferrite() { return _inputs.freezing_range_ferrite; }
    auto A_austenite() { return _inputs.A_austenite; }
    auto B_austenite() { return _inputs.B_austenite; }
    auto C_austenite() { return _inputs.C_austenite; }
    auto D_austenite() { return _inputs.D_austenite; }
    auto freezingRange_austenite() { return _inputs.freezing_range_austenite; }
};

#endif
