# Using Falcata via Docker

This directory contains `Dockerfile`s to make it easy to build and run Falcata via [Docker](https://www.docker.com/).

These builds of Falcata all train on the CPU. For GPU-enabled builds, see [the gpu/ directory](./gpu).

## Installing Docker

Follow the general installation instructions [on the Docker site](https://docs.docker.com/install/):

* [macOS](https://docs.docker.com/docker-for-mac/install/)
* [Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
* [Windows](https://docs.docker.com/docker-for-windows/install/)

## Using CLI Version of Falcata via Docker

Build an image with the Falcata CLI.

```shell
# run from the root of a Falcata checkout: the image is built from the
# working tree (COPY), not from a clone
docker build \
    -t falcata-cli \
    -f docker/dockerfile-cli \
    .
```

Once that completes, the built image can be used to run the CLI in a container.
To try it out, run the following.

```shell
# configure the CLI
cat << EOF > train.conf
task = train
objective = binary
data = binary.train
num_trees = 10
output_model = Falcata-CLI-model.txt
EOF

# training data ships in the checkout
cp examples/binary_classification/binary.train .

# train, and save model to a text file
docker run \
  --rm \
  --volume "${PWD}":/opt/training \
  --workdir /opt/training \
  falcata-cli \
  config=train.conf
```

After this runs, a Falcata model can be found at `Falcata-CLI-model.txt`.

For more details on how to configure and use the Falcata CLI, see https://github.com/BelixRogner/Falcata/blob/master/docs/Quick-Start.rst.

## Running the Python-package Container

Build an image with the Falcata Python-package installed.

```shell
# run from the root of a Falcata checkout: the image is built from the
# working tree (COPY), not from a clone
docker build \
    -t falcata-python \
    -f docker/dockerfile-python \
    .
```

Once that completes, the built image can be used to run Falcata's Python-package in a container.
Run the following to produce a model using the Python-package.

```shell
# training data ships in the checkout
cp examples/binary_classification/binary.train .

# create training script
cat << EOF > train.py
import falcata as lgb
import numpy as np
params = {
    "objective": "binary",
    "num_trees": 10
}

bst = lgb.train(
    train_set=lgb.Dataset("binary.train"),
    params=params
)
bst.save_model("Falcata-python-model.txt")
EOF

# run training in a container
docker run \
    --rm \
    --volume "${PWD}":/opt/training \
    --workdir /opt/training \
    falcata-python \
    python train.py
```

After this runs, a Falcata model can be found at `Falcata-python-model.txt`.

Or run an interactive Python session in a container.

```shell
docker run \
    --rm \
    --volume "${PWD}":/opt/training \
    --workdir /opt/training \
    -it falcata-python \
    python
```

## Running the R-package Container

Build an image with the Falcata R-package installed.

```shell
# run from the root of a Falcata checkout: the image is built from the
# working tree (COPY), not from a clone
docker build \
    -t falcata-r \
    -f docker/dockerfile-r \
    .
```

Once that completes, the built image can be used to run Falcata's R-package in a container.
Run the following to produce a model using the R-package.

```shell
# training data ships in the checkout
cp examples/binary_classification/binary.train .

# create training script
cat << EOF > train.R
library(falcata)
params <- list(
    objective = "binary"
    , num_trees = 10L
)

bst <- lgb.train(
    data = lgb.Dataset("binary.train"),
    params = params
)
lgb.save(bst, "Falcata-R-model.txt")
EOF

# run training in a container
docker run \
    --rm \
    --volume "${PWD}":/opt/training \
    --workdir /opt/training \
    falcata-r \
    Rscript train.R
```

After this runs, a Falcata model can be found at `Falcata-R-model.txt`.

Run the following to get an interactive R session in a container.

```shell
docker run \
    --rm \
    -it falcata-r \
    R
```

To use [RStudio](https://www.rstudio.com/products/rstudio/), an interactive development environment, run the following.

```shell
docker run \
    --rm \
    --env PASSWORD="falcata" \
    -p 8787:8787 \
    falcata-r
```

Then navigate to `localhost:8787` in your local web browser, and log in with username `rstudio` and password `falcata`.

To target a different R version, pass any [valid rocker/verse tag](https://hub.docker.com/r/rocker/verse/tags) to `docker build`.

For example, to test Falcata with R 4.5:

```shell
docker build \
    -t falcata-r-45 \
    -f dockerfile-r \
    --build-arg R_VERSION=4.5 \
    .
```
