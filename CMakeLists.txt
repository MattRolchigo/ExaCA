cmake_minimum_required(VERSION 3.9)

project(ExaCA CXX)

include(GNUInstallDirs)

find_package(Kokkos 3 REQUIRED)
find_package(MPI REQUIRED)

add_subdirectory(src)
add_subdirectory(analysis)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ExaCAconfig.cmakein
  ${CMAKE_CURRENT_BINARY_DIR}/ExaCAconfig.cmake @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ExaCAconfig.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ExaCA)

# Install data files
file(GLOB MATERIALS examples/Materials/Inconel625)
file(GLOB SUBSTRATE examples/Substrate/GrainOrientationVectors*.csv)
install(FILES ${MATERIALS}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/ExaCA/Materials)
install(FILES ${SUBSTRATE}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/ExaCA/Substrate)

include(CTest)
#add_subdirectory(unit_test)
if(${BUILD_TESTING})
    add_test(NAME ExaCA-Test COMMAND ExaCA-Kokkos)
endif()

# clang-format
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package( CLANG_FORMAT 10)
if(CLANG_FORMAT_FOUND)
  file(GLOB_RECURSE FORMAT_SOURCES src/*.cpp src/*.hpp analysis/*.cpp analysis/*.hpp)
  add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXECUTABLE} -i -style=file ${FORMAT_SOURCES}
    DEPENDS ${FORMAT_SOURCES})
endif()
