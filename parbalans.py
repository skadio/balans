from balans.solver import ParBalans

instance_path = "tests/data/noswot.mps"
n_machines = 2

balans = ParBalans(instance_path=instance_path, n_machines=n_machines)
balans.solve()