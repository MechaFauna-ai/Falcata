LambdaRank Example
==================

Here is an example for Falcata to run LambdaRank task.

***You must follow the [installation instructions](https://github.com/BelixRogner/Falcata/blob/master/docs/Installation-Guide.rst)
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

Data Format
-----------

To learn more about the query format used in this example, check out the
[query data format](https://github.com/BelixRogner/Falcata/blob/master/docs/Parameters.rst#query-data).
