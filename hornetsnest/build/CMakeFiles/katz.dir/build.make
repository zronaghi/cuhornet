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
include CMakeFiles/katz.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/katz.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/katz.dir/flags.make

CMakeFiles/katz.dir/test/KatzTest.cu.o: CMakeFiles/katz.dir/flags.make
CMakeFiles/katz.dir/test/KatzTest.cu.o: ../test/KatzTest.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nfs/zronaghi/cuhornet/hornetsnest/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/katz.dir/test/KatzTest.cu.o"
	/usr/local/cuda-10.0/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/nfs/zronaghi/cuhornet/hornetsnest/test/KatzTest.cu -o CMakeFiles/katz.dir/test/KatzTest.cu.o

CMakeFiles/katz.dir/test/KatzTest.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/katz.dir/test/KatzTest.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/katz.dir/test/KatzTest.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/katz.dir/test/KatzTest.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target katz
katz_OBJECTS = \
"CMakeFiles/katz.dir/test/KatzTest.cu.o"

# External object files for target katz
katz_EXTERNAL_OBJECTS =

CMakeFiles/katz.dir/cmake_device_link.o: CMakeFiles/katz.dir/test/KatzTest.cu.o
CMakeFiles/katz.dir/cmake_device_link.o: CMakeFiles/katz.dir/build.make
CMakeFiles/katz.dir/cmake_device_link.o: libhornetAlg.a
CMakeFiles/katz.dir/cmake_device_link.o: /home/nfs/zronaghi/miniconda3/envs/spgemm/lib/librmm.so
CMakeFiles/katz.dir/cmake_device_link.o: CMakeFiles/katz.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nfs/zronaghi/cuhornet/hornetsnest/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/katz.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/katz.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/katz.dir/build: CMakeFiles/katz.dir/cmake_device_link.o

.PHONY : CMakeFiles/katz.dir/build

# Object files for target katz
katz_OBJECTS = \
"CMakeFiles/katz.dir/test/KatzTest.cu.o"

# External object files for target katz
katz_EXTERNAL_OBJECTS =

katz: CMakeFiles/katz.dir/test/KatzTest.cu.o
katz: CMakeFiles/katz.dir/build.make
katz: libhornetAlg.a
katz: /home/nfs/zronaghi/miniconda3/envs/spgemm/lib/librmm.so
katz: CMakeFiles/katz.dir/cmake_device_link.o
katz: CMakeFiles/katz.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nfs/zronaghi/cuhornet/hornetsnest/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable katz"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/katz.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/katz.dir/build: katz

.PHONY : CMakeFiles/katz.dir/build

CMakeFiles/katz.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/katz.dir/cmake_clean.cmake
.PHONY : CMakeFiles/katz.dir/clean

CMakeFiles/katz.dir/depend:
	cd /home/nfs/zronaghi/cuhornet/hornetsnest/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nfs/zronaghi/cuhornet/hornetsnest /home/nfs/zronaghi/cuhornet/hornetsnest /home/nfs/zronaghi/cuhornet/hornetsnest/build /home/nfs/zronaghi/cuhornet/hornetsnest/build /home/nfs/zronaghi/cuhornet/hornetsnest/build/CMakeFiles/katz.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/katz.dir/depend

