Regression Example
==================

Here is an example for Falcata to run regression task.

***You must follow the [installation instructions](https://falcata.readthedocs.io/en/latest/Installation-Guide.html)
for the following commands to work. The `falcata` binary must be built and available at the root of this project.***

Training
--------

Run the following command in this folder:

```bash
"../../falcata" config=train.conf
```

Prediction
----------

You should finish training first.

Run the following command in this folder:

```bash
"../../falcata" config=predict.conf
```
