import tensorflow as tf

# worker1 = "cq01-dt-dev03.cq01:2222"
# worker2 = "cq01-dt-dev03.cq01:2223"
# worker3 = "cp01-sys-hic-gpu-26.cp01:2222"
# worker4 = "cp01-sys-hic-gpu-26.cp01:2223"
# worker_hosts = [worker1, worker2, worker3, worker4]
# cluster_spec = tf.train.ClusterSpec({"worker": worker_hosts})
cluster_spec = tf.train.ClusterSpec({"worker": ["localhost:2223"]})
server = tf.train.Server(cluster_spec, job_name="worker", task_index=0)
server.join()
