# 18-645-Semester-Project

### Team Members
* Michael Chang
* Frank Lee
* Minh Truong

### Milestone Deadline
| Date | Task | Stage | Status |
|------|------|-------|--------|
|   Sep 20th   |   Layout Pipeline Modules   |   1    |        |
|   Sep 22nd   |   Develop Peripherals (makefiles, directories, ...)  |   1    |        |
|   Sep 23rd   |   Stage 1 Deadline | | |
|   Oct 5th    | Naive Implementation of Module | 2 |||
|   Oct 19th   | Parallelize Module | 2 |||
|   Nov 3rd    | Testing module | 3 |||
|   Nov 10th   | Benchmarking | 3 |||
|   Nov 17th  |  Assemble Pipeline | 4 |||
|   Dec 2nd   |   Final Deadline   |   5 6    |        |

* Stage 1: Top-level Development. Make sure that the entire pipeline of ORB is
compiled-able and modularized.
* Stage 2: Individual Module Development. Each team member will develop their
own module. Team member will meet up 1 - 2 times
a week for synchronization.
* Stage 3: Individual Module Testing. Each team member will develop testbench
that makes sure their module is air-tight.
* Stage 4: Pipeline assemble and testing. Develop top-level testbench to make
sure the pipeline works.
* Stage 5: Optimization.
* Stage 6: Profiling and Documentation.

### Directories

* `./app` stores any applications, presentations related program such
as ORB demo program and further documentation on ORB.
* `./data` stores data and data related program (cleaning, parsing, formatting)
used by other directories.
* `./lib` stores the shared library `.so` objects that can be used by any
applications.
* `./match` stores the Feature Matching module programs testbenches.
* `./oFAST` stores the oriented FAST module programs testbenches.
* `./rBRIEF` stores the rotated BRIEF module programs testbenches.
* `orb.h` links all modules together.
* `orb.c` implements neccessary pipeline peripherals.
* `tb_orb.c` test the pipeline.

Each Module directory have 4 starting files: `Makefile` `*.c` `*.h` `tb_*`.
These files are the minimum components needed to construct the pipeline.

### Make instruction
Each module has its own `Makefile` that can be tailored to fit the development of
that module. The `Makefile` in the root directory call on all sub-`Makefile`
to construct the pipeline. The root `Makefile` has 3 important recipe:
```
# this compiles all sub-directory and return .o files. It then link those  object to tb_orb.
make all
```
```
# this creates shared object that can be used for demonstration.
make lib
```
```
# this call on all sub-Makefile cleaning routine.
make clean
```
