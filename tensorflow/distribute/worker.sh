#!/usr/bin/env bash

bazel-bin/tensorflow/core/distributed_runtime/rpc/grpc_tensorflow_server \
    --cluster_spec='local|localhost:2222;localhost:2223' --job_name=local --task_id=0

bazel-bin/tensorflow/core/distributed_runtime/rpc/grpc_tensorflow_server \
    --cluster_spec='local|localhost:2222;localhost:2223' --job_name=local --task_id=1

bazel-bin/tensorflow/core/distributed_runtime/rpc/grpc_tensorflow_server \
    --cluster_spec='worker|localhost:2500;localhost:2501' --job_name=worker --task_id=0