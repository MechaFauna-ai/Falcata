# Falcata R-package

<img src="man/figures/logo.svg" align="right" alt="" width="175" />

### Contents

* [Installation](#installation)
    - [Installing the CRAN-style Tarball](#installing-the-cran-package)
    - [Installing from Source with CMake](#install)
    - [Installing a GPU-enabled Build](#installing-a-gpu-enabled-build)
    - [Installing Precompiled Binaries](#installing-precompiled-binaries)
    - [Installing from a Pre-compiled lib_falcata](#lib_falcata)
* [Examples](#examples)
* [Testing](#testing)
    - [Running the Tests](#running-the-tests)
    - [Code Coverage](#code-coverage)
* [Updating Documentation](#updating-documentation)
* [Preparing a CRAN Package](#preparing-a-cran-package)
* [Known Issues](#known-issues)

Installation
------------

Falcata is not published to CRAN; install by building the CRAN-style tarball locally (["Installing the CRAN-style tarball"](#installing-the-cran-package)) or ["Installing from Source with CMake"](#install).

If you experience any issues with that, try ["Installing from Source with CMake"](#install). This can produce a more efficient version of the library on Windows systems with Visual Studio.

To build a GPU-enabled version of the package, follow the steps in ["Installing a GPU-enabled Build"](#installing-a-gpu-enabled-build).

If any of the above options do not work for you or do not meet your needs, please let the maintainers know by [opening an issue](https://github.com/MechaFauna-ai/Falcata/issues).

When your package installation is done, you can check quickly if your Falcata R-package is working by running the following:

```r
library(falcata)
data(agaricus.train, package='falcata')
train <- agaricus.train
dtrain <- lgb.Dataset(train$data, label = train$label)
model <- lgb.cv(
    params = list(
        objective = "regression"
        , metric = "l2"
    )
    , data = dtrain
)
```

### Installing the CRAN-style tarball <a id="installing-the-cran-package"></a>

Falcata is not published to CRAN. Build the CRAN-style source tarball locally and install it:

```shell
sh build-cran-package.sh
R CMD INSTALL falcata_*.tar.gz
```

This does not require `CMake` or `Visual Studio`, and should work well on many different operating systems and compilers.

#### Custom Installation (Linux, Mac)

The steps above should work on most systems, but users with highly-customized environments might want to change how R builds packages from source.

To change the compiler used when installing the CRAN package, you can create a file `~/.R/Makevars` which overrides `CC` (`C` compiler) and `CXX` (`C++` compiler).

For example, to use `gcc-14` instead of `clang` on macOS, you could use something like the following:

```make
# ~/.R/Makevars
CC=gcc-14
CC17=gcc-14
CXX=g++-14
CXX17=g++-14
```

To check the values R is using, run the following:

```shell
R CMD config --all
```

### Installing from Source with CMake <a id="install"></a>

You need to install git and [CMake](https://cmake.org/) first.

Note: this method is only supported on 64-bit systems. If you need to run Falcata on 32-bit Windows (i386), follow the instructions in ["Installing the CRAN Package"](#installing-the-cran-package).

#### Windows Preparation

NOTE: Windows users may need to run with administrator rights (either R or the command prompt, depending on the way you are installing this package).

Installing a 64-bit version of [Rtools](https://cran.r-project.org/bin/windows/Rtools/) is mandatory.

After installing `Rtools` and `CMake`, be sure the following paths are added to the environment variable `PATH`. These may have been automatically added when installing other software.

* `Rtools`
    - If you have `Rtools` 4.0, example:
        - `C:\rtools40\mingw64\bin`
        - `C:\rtools40\usr\bin`
    - If you have `Rtools` 4.2+, example:
        - `C:\rtools42\x86_64-w64-mingw32.static.posix\bin`
        - `C:\rtools42\usr\bin`
        - **NOTE**: this is e.g. `rtools43\` for R 4.3
* `CMake`
    - example: `C:\Program Files\CMake\bin`
* `R`
    - example: `C:\Program Files\R\R-4.5.1\bin`

NOTE: Two `Rtools` paths are required from `Rtools` 4.0 onwards because paths and the list of included software was changed in `Rtools` 4.0.

NOTE: `Rtools42` and later take a very different approach to the compiler toolchain than previous releases, and how you install it changes what is required to build packages. See ["Howto: Building R 4.2 and packages on Windows"](https://cran.r-project.org/bin/windows/base/howto-R-4.2.html).

#### Windows Toolchain Options

A "toolchain" refers to the collection of software used to build the library. The R-package can be built with three different toolchains.

**Warning for Windows users**: it is recommended to use *Visual Studio* for its better multi-threading efficiency in Windows for many core systems. For very simple systems (dual core computers or worse), MinGW64 is recommended for maximum performance. If you do not know what to choose, it is recommended to use [Visual Studio](https://visualstudio.microsoft.com/downloads/), the default compiler. **Do not try using MinGW in Windows on many core systems. It may result in 10x slower results than Visual Studio.**

**Visual Studio (default)**

By default, the package will be built with [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/).

**MSYS2 (R 4.x)**

If you are using R 4.x and installation fails with Visual Studio, `Falcata` will fall back to using [MSYS2](https://www.msys2.org/). This should work with the tools already bundled in `Rtools` 4.0.

If you want to force `Falcata` to use MSYS2 (for any R version), pass `--use-msys2` to the installation script.

```shell
Rscript build_r.R --use-msys2
```

**MinGW**

If you want to force `Falcata` to use [MinGW](https://www.mingw-w64.org/) (for any R version), pass `--use-mingw` to the installation script.

```shell
Rscript build_r.R --use-mingw
```

#### Mac OS Preparation

You can perform installation either with **Apple Clang** or **gcc**. In case you prefer **Apple Clang**, you should install **OpenMP** (details for installation can be found in [Installation Guide](https://github.com/MechaFauna-ai/Falcata/blob/master/docs/Installation-Guide.rst#apple-clang)) first. In case you prefer **gcc**, you need to install it (details for installation can be found in [Installation Guide](https://github.com/MechaFauna-ai/Falcata/blob/master/docs/Installation-Guide.rst#gcc)) and set some environment variables to tell R to use `gcc` and `g++`. If you install these from Homebrew, your versions of `g++` and `gcc` are most likely in `/usr/local/bin`, as shown below.

```
# replace 8 with version of gcc installed on your machine
export CXX=/usr/local/bin/g++-8 CC=/usr/local/bin/gcc-8
```

#### Install with CMake

After following the "preparation" steps above for your operating system, build and install the R-package with the following commands:

```sh
git clone --recursive https://github.com/MechaFauna-ai/Falcata
cd Falcata
Rscript build_r.R
```

The `build_r.R` script builds the package in a temporary directory called `falcata_r`. It will destroy and recreate that directory each time you run the script. That script supports the following command-line options:

- `--no-build-vignettes`: Skip building vignettes.
- `-j[jobs]`: Number of threads to use when compiling Falcata. E.g., `-j4` will try to compile 4 objects at a time.
    - by default, this script uses single-thread compilation
    - for best results, set `-j` to the number of physical CPUs
- `--skip-install`: Build the package tarball, but do not install it.
- `--use-gpu`: Build a GPU-enabled version of the library.
- `--use-mingw`: Force the use of MinGW toolchain, regardless of R version.
- `--use-msys2`: Force the use of MSYS2 toolchain, regardless of R version.

Note: for the build with Visual Studio/VS Build Tools in Windows, you should use the Windows CMD or PowerShell.

### Installing a GPU-enabled Build

You will need to install Boost and OpenCL first: details for installation can be found in [Installation-Guide](https://github.com/MechaFauna-ai/Falcata/blob/master/docs/Installation-Guide.rst#build-gpu-version).

After installing these other libraries, follow the steps in ["Installing from Source with CMake"](#install). When you reach the step that mentions `build_r.R`, pass the flag `--use-gpu`.

```shell
Rscript build_r.R --use-gpu
```

You may also need or want to provide additional configuration, depending on your setup. For example, you may need to provide locations for Boost and OpenCL.

```shell
Rscript build_r.R \
    --use-gpu \
    --opencl-library=/usr/lib/x86_64-linux-gnu/libOpenCL.so \
    --boost-librarydir=/usr/lib/x86_64-linux-gnu
```

The following options correspond to the [CMake FindBoost options](https://cmake.org/cmake/help/latest/module/FindBoost.html) by the same names.

* `--boost-root`
* `--boost-dir`
* `--boost-include-dir`
* `--boost-librarydir`

The following options correspond to the [CMake FindOpenCL options](https://cmake.org/cmake/help/latest/module/FindOpenCL.html) by the same names.

* `--opencl-include-dir`
* `--opencl-library`

### Installing Precompiled Binaries

There are no precompiled binaries for the R package; install from the CRAN-style tarball or from source (see above).

### Installing from a Pre-compiled lib_falcata <a id="lib_falcata"></a>

Previous versions of Falcata offered the ability to first compile the C++ library (`lib_falcata.{dll,dylib,so}`) and then build an R-package that wraps it.

As of version 3.0.0, this is no longer supported. If building from source is difficult for you, reach out to the maintainers.

Examples
--------

Please visit [demo](https://github.com/MechaFauna-ai/Falcata/tree/master/R-package/demo):

* [Basic walkthrough of wrappers](https://github.com/MechaFauna-ai/Falcata/blob/master/R-package/demo/basic_walkthrough.R)
* [Boosting from existing prediction](https://github.com/MechaFauna-ai/Falcata/blob/master/R-package/demo/boost_from_prediction.R)
* [Early Stopping](https://github.com/MechaFauna-ai/Falcata/blob/master/R-package/demo/early_stopping.R)
* [Cross Validation](https://github.com/MechaFauna-ai/Falcata/blob/master/R-package/demo/cross_validation.R)
* [Multiclass Training/Prediction](https://github.com/MechaFauna-ai/Falcata/blob/master/R-package/demo/multiclass.R)
* [Leaf (in)Stability](https://github.com/MechaFauna-ai/Falcata/blob/master/R-package/demo/leaf_stability.R)
* [Weight-Parameter Adjustment Relationship](https://github.com/MechaFauna-ai/Falcata/blob/master/R-package/demo/weight_param.R)

Testing
-------

The R-package's unit tests are run automatically on every commit, via integrations like [GitHub Actions](https://github.com/MechaFauna-ai/Falcata/actions). Adding new tests in `R-package/tests/testthat` is a valuable way to improve the reliability of the R-package.

### Running the Tests

While developing the R-package, run the code below to run the unit tests.

```shell
sh build-cran-package.sh \
    --no-build-vignettes

R CMD INSTALL --with-keep.source falcata*.tar.gz
cd R-package/tests
Rscript testthat.R
```

To run the tests with more verbose logs, set environment variable `FALCATA_TEST_VERBOSITY` to a valid value for parameter [`verbosity`](https://github.com/MechaFauna-ai/Falcata/blob/master/docs/Parameters.rst#verbosity).

```shell
export FALCATA_TEST_VERBOSITY=1
cd R-package/tests
Rscript testthat.R
```

### Code Coverage

When adding tests, you may want to use test coverage to identify untested areas and to check if the tests you've added are covering all branches of the intended code.

The example below shows how to generate code coverage for the R-package on a macOS or Linux setup. To adjust for your environment, refer to [the customization step described above](#custom-installation-linux-mac).

```shell
# Install
sh build-cran-package.sh \
    --no-build-vignettes

# Get coverage
Rscript -e " \
    library(covr);
    coverage <- covr::package_coverage('./falcata_r', type = 'tests', quiet = FALSE);
    print(coverage);
    covr::report(coverage, file = file.path(getwd(), 'coverage.html'), browse = TRUE);
    "
```

Updating Documentation
----------------------

The R-package uses [`{roxygen2}`](https://CRAN.R-project.org/package=roxygen2) to generate its documentation.
The generated `DESCRIPTION`, `NAMESPACE`, and `man/` files are checked into source control.
To regenerate those files, run the following.

```shell
Rscript \
    --vanilla \
    -e "install.packages('roxygen2', repos = 'https://cran.rstudio.com')"

sh build-cran-package.sh --no-build-vignettes
R CMD INSTALL \
  --with-keep.source \
  ./falcata_*.tar.gz

cd R-package
Rscript \
    --vanilla \
    -e "roxygen2::roxygenize(load = 'installed')"
```

Preparing a CRAN Package
------------------------

This section is primarily for maintainers, but may help users and contributors to understand the structure of the R-package.

Most of `Falcata` uses `CMake` to handle tasks like setting compiler and linker flags, including header file locations, and linking to other libraries. Because CRAN packages typically do not assume the presence of `CMake`, the R-package uses an alternative method that is in the CRAN-supported toolchain for building R packages with C++ code: `Autoconf`.

For more information on this approach, see ["Writing R Extensions"](https://cran.r-project.org/doc/manuals/r-release/R-exts.html#Configure-and-cleanup).

### Build a CRAN Package

From the root of the repository, run the following.

```shell
git submodule update --init --recursive
sh build-cran-package.sh
```

This will create a file `falcata_${VERSION}.tar.gz`, where `VERSION` is the version of `Falcata`.

That script supports the following command-line options:

- `--no-build-vignettes`: Skip building vignettes.
- `--r-executable=[path-to-executable]`: Use an alternative build of R.

### Standard Installation from CRAN Package

After building the package, install it with a command like the following:

```shell
R CMD install falcata_*.tar.gz
```

### Changing the CRAN Package

A lot of details are handled automatically by `R CMD build` and `R CMD install`, so it can be difficult to understand how the files in the R-package are related to each other. An extensive treatment of those details is available in ["Writing R Extensions"](https://cran.r-project.org/doc/manuals/r-release/R-exts.html).

This section briefly explains the key files for building a CRAN package. To update the package, edit the files relevant to your change and re-run the steps in [Build a CRAN Package](#build-a-cran-package).

**Linux or Mac**

At build time, `configure` will be run and used to create a file `Makevars`, using `Makevars.in` as a template.

1. Edit `configure.ac`.
2. Create `configure` with `autoconf`. Do not edit it by hand. This file must be generated on Ubuntu 22.04.

    If you have an Ubuntu 22.04 environment available, run the provided script from the root of the `Falcata` repository.

    ```shell
    ./R-package/recreate-configure.sh
    ```

    If you do not have easy access to an Ubuntu 22.04 environment, the `configure` script can be generated using Docker by running the code below from the root of this repo.

    ```shell
    docker run \
        --rm \
        -v $(pwd):/opt/Falcata \
        -w /opt/Falcata \
        ubuntu:22.04 \
        ./R-package/recreate-configure.sh
    ```

    The version of `autoconf` used by this project is stored in `R-package/AUTOCONF_UBUNTU_VERSION`. To update that version, update that file and run the commands above. To see available versions, see https://packages.ubuntu.com/search?keywords=autoconf.

3. Edit `src/Makevars.in`.

Alternatively, GitHub Actions can re-generate this file for you.

1. navigate to https://github.com/MechaFauna-ai/Falcata/actions/workflows/r_configure.yml
2. click "Run workflow" (drop-down)
3. enter the branch from the pull request for the `pr-branch` input
4. click "Run workflow" (button)

**Configuring for Windows**

At build time, `configure.win` will be run and used to create a file `Makevars.win`, using `Makevars.win.in` as a template.

1. Edit `configure.win` directly.
2. Edit `src/Makevars.win.in`.

### Testing the CRAN Package

`{falcata}` is tested automatically on every commit, across many combinations of operating system, R version, and compiler. This section describes how to test the package locally while you are developing.

#### Windows, Mac, and Linux

```shell
sh build-cran-package.sh
R CMD check --as-cran falcata_*.tar.gz
```

#### <a id="UBSAN"></a>ASAN and UBSAN

All packages uploaded to CRAN must pass builds using `gcc` and `clang`, instrumented with two sanitizers: the Address Sanitizer (ASAN) and the Undefined Behavior Sanitizer (UBSAN).

For more background, see

* [this blog post](https://dirk.eddelbuettel.com/code/sanitizers.html)
* [top-level CRAN documentation on these checks](https://cran.r-project.org/web/checks/check_issue_kinds.html)
* [CRAN's configuration of these checks](https://www.stats.ox.ac.uk/pub/bdr/memtests/README.txt)

You can replicate these checks locally using Docker.
For more information on the image used for testing, see https://github.com/wch/r-debug.

In the code below, environment variable `R_CUSTOMIZATION` should be set to one of two values.

* `"san"` = replicates CRAN's `gcc-ASAN` and `gcc-UBSAN` checks
* `"csan"` = replicates CRAN's `clang-ASAN` and `clang-UBSAN` checks

```shell
docker run \
  --rm \
  -it \
  -v $(pwd):/opt/Falcata \
  -w /opt/Falcata \
  --env R_CUSTOMIZATION=san \
  wch1/r-debug:latest \
  /bin/bash

# install dependencies
RDscript${R_CUSTOMIZATION} \
  -e "install.packages(c('R6', 'data.table', 'jsonlite', 'knitr', 'markdown', 'Matrix', 'RhpcBLASctl', 'testthat'), repos = 'https://cran.r-project.org', Ncpus = parallel::detectCores())"

# install falcata
sh build-cran-package.sh --r-executable=RD${R_CUSTOMIZATION}
RD${R_CUSTOMIZATION} \
  CMD INSTALL falcata_*.tar.gz

# run tests
cd R-package/tests
rm -f ./tests.log
RDscript${R_CUSTOMIZATION} testthat.R >> tests.log 2>&1

# check that tests passed
echo "test exit code: $?"
tail -300 ./tests.log
```

#### Valgrind

All packages uploaded to CRAN must be built and tested without raising any issues from `valgrind`. `valgrind` is a profiler that can catch serious issues like memory leaks and illegal writes. For more information, see [this blog post](https://reside-ic.github.io/blog/debugging-and-fixing-crans-additional-checks-errors/).

You can replicate these checks locally using Docker. Note that instrumented versions of R built to use `valgrind` run much slower, and these tests may take as long as 20 minutes to run.

```shell
docker run \
    --rm \
    -v $(pwd):/opt/Falcata \
    -w /opt/Falcata \
    -it \
        wch1/r-debug

RDscriptvalgrind -e "install.packages(c('R6', 'data.table', 'jsonlite', 'knitr', 'markdown', 'Matrix', 'RhpcBLASctl', 'testthat'), repos = 'https://cran.rstudio.com', Ncpus = parallel::detectCores())"

sh build-cran-package.sh \
    --r-executable=RDvalgrind

RDvalgrind CMD INSTALL \
    --preclean \
    --install-tests \
        falcata_*.tar.gz

cd R-package/tests

RDvalgrind \
    --no-readline \
    --vanilla \
    -d "valgrind --tool=memcheck --leak-check=full --track-origins=yes" \
        -f testthat.R \
2>&1 \
| tee out.log \
| cat
```

These tests can also be triggered on a pull request branch, using GitHub Actions.

1. navigate to https://github.com/MechaFauna-ai/Falcata/actions/workflows/r_valgrind.yml
2. click "Run workflow" (drop-down)
3. enter the branch from the pull request for the `pr-branch` input
4. enter the pull request ID for the `pr-number` input
5. click "Run workflow" (button)

Or by using the GitHub CLI, using a command similar to this:

```shell
gh workflow run \
    --repo MechaFauna-ai/Falcata \
    r_valgrind.yml \
    -f pr-branch=ci/fix-rerun-workflow \
    -f pr-number=7072
```

Known Issues
------------

