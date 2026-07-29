Distributed Learning Example
============================
<a name="parallel-learning-example"></a>

Here is an example for Falcata to perform distributed learning for 2 machines.

1. Edit [mlist.txt](./mlist.txt): write the ip of these 2 machines that you want to run application on.

   ```
   machine1_ip 12400
   machine2_ip 12400
   ```

2. Copy this folder and executable file to these 2 machines that you want to run application on.

3. Run command in this folder on both 2 machines:

   ```"./falcata" config=train.conf```

This distributed learning example is based on socket. Falcata also supports distributed learning based on MPI.

For more details about the usage of distributed learning, please refer to [this](https://github.com/BelixRogner/Falcata/blob/master/docs/Parallel-Learning-Guide.rst).
