# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/lib/python3.5/dist-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /usr/local/lib/python3.5/dist-packages/cmake/data/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/nfs/zronaghi/cuhornet/hornetsnest

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nfs/zronaghi/cuhornet/hornetsnest/build

# Include any dependencies generated for this target.
include CMakeFiles/clus-coeff.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/clus-coeff.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/clus-coeff.dir/flags.make

CMakeFiles/clus-coeff.dir/test/ClusCoeffTest.cu.o: CMakeFiles/clus-coeff.dir/flags.make
CMakeFiles/clus-coeff.dir/test/ClusCoeffTest.cu.o: ../test/ClusCoeffTest.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nfs/zronaghi/cuhornet/hornetsnest/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/clus-coeff.dir/test/ClusCoeffTest.cu.o"
	/usr/local/cuda-10.0/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/nfs/zronaghi/cuhornet/hornetsnest/test/ClusCoeffTest.cu -o CMakeFiles/clus-coeff.dir/test/ClusCoeffTest.cu.o

CMakeFiles/clus-coeff.dir/test/ClusCoeffTest.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/clus-coeff.dir/test/ClusCoeffTest.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/clus-coeff.dir/test/ClusCoeffTest.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/clus-coeff.dir/test/ClusCoeffTest.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target clus-coeff
clus__coeff_OBJECTS = \
"CMakeFiles/clus-coeff.dir/test/ClusCoeffTest.cu.o"

# External object files for target clus-coeff
clus__coeff_EXTERNAL_OBJECTS =

CMakeFiles/clus-coeff.dir/cmake_device_link.o: CMakeFiles/clus-coeff.dir/test/ClusCoeffTest.cu.o
CMakeFiles/clus-coeff.dir/cmake_device_link.o: CMakeFiles/clus-coeff.dir/build.make
CMakeFiles/clus-coeff.dir/cmake_device_link.o: libhornetAlg.a
CMakeFiles/clus-coeff.dir/cmake_device_link.o: /home/nfs/zronaghi/miniconda3/envs/spgemm/lib/librmm.so
CMakeFiles/clus-coeff.dir/cmake_device_link.o: CMakeFiles/clus-coeff.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nfs/zronaghi/cuhornet/hornetsnest/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/clus-coeff.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/clus-coeff.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/clus-coeff.dir/build: CMakeFiles/clus-coeff.dir/cmake_device_link.o

.PHONY : CMakeFiles/clus-coeff.dir/build

# Object files for target clus-coeff
clus__coeff_OBJECTS = \
"CMakeFiles/clus-coeff.dir/test/ClusCoeffTest.cu.o"

# External object files for target clus-coeff
clus__coeff_EXTERNAL_OBJECTS =

clus-coeff: CMakeFiles/clus-coeff.dir/test/ClusCoeffTest.cu.o
clus-coeff: CMakeFiles/clus-coeff.dir/build.make
clus-coeff: libhornetAlg.a
clus-coeff: /home/nfs/zronaghi/miniconda3/envs/spgemm/lib/librmm.so
clus-coeff: CMakeFiles/clus-coeff.dir/cmake_device_link.o
clus-coeff: CMakeFiles/clus-coeff.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nfs/zronaghi/cuhornet/hornetsnest/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable clus-coeff"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/clus-coeff.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/clus-coeff.dir/build: clus-coeff

.PHONY : CMakeFiles/clus-coeff.dir/build

CMakeFiles/clus-coeff.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/clus-coeff.dir/cmake_clean.cmake
.PHONY : CMakeFiles/clus-coeff.dir/clean

CMakeFiles/clus-coeff.dir/depend:
	cd /home/nfs/zronaghi/cuhornet/hornetsnest/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nfs/zronaghi/cuhornet/hornetsnest /home/nfs/zronaghi/cuhornet/hornetsnest /home/nfs/zronaghi/cuhornet/hornetsnest/build /home/nfs/zronaghi/cuhornet/hornetsnest/build /home/nfs/zronaghi/cuhornet/hornetsnest/build/CMakeFiles/clus-coeff.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/clus-coeff.dir/depend

