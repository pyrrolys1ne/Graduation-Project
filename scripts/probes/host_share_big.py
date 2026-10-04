import os, sys
os.chdir("/home/lyon/projects/bishe")
sys.path.insert(0, "/home/lyon/projects/bishe/scripts/probes")
import diag_host_bound_sweep as m
m.SIZES = (224, 672, 1024, 1344)
m.REPS = 60
m.main()
