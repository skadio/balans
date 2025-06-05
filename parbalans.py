from balans.solver import ParBalans

instance_path = "tests/data/noswot.mps"
n_machines = 2

balans = ParBalans(n_jobs=n_machines)
balans.run()