cmake_minimum_required(VERSION 3.15)
project(MaxVolArmadilloHDF5)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
find_package(Armadillo REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS})
include_directories(${ARMADILLO_INCLUDE_DIRS} ${MPI_INCLUDE_PATH})
link_directories(${HDF5_LIBRARY_DIRS} ${ARMADILLO_LIBRARY_DIRS})
message(STATUS "Armadillo include dirs: ${ARMADILLO_INCLUDE_DIRS}")
message(STATUS "Armadillo libraries: ${ARMADILLO_LIBRARIES}")
add_executable(maxvol_experiment main1.cpp)
target_link_libraries(maxvol_experiment
    ${ARMADILLO_LIBRARIES}
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
)

set_target_properties(maxvol_experiment PROPERTIES
    BUILD_RPATH ${HDF5_LIBRARY_DIRS}:${ARMADILLO_LIBRARY_DIRS}
    INSTALL_RPATH ${HDF5_LIBRARY_DIRS}:${ARMADILLO_LIBRARY_DIRS}
)


