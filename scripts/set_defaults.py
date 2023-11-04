import multiprocessing
import os
import pprint

from baynes.model_utils import get_models_path, set_models_path, update_config

pp = pprint.PrettyPrinter(indent=1, compact=True)
curr = os.getcwd()
models_path = os.path.join(curr, "stan")
print("Setting default Stan models path:\n", models_path)
set_models_path(models_path)

num_cores = multiprocessing.cpu_count()
try:
    max_threads = os.sysconf("SC_NPROCESSORS_ONLN")
except ValueError:
    max_threads = num_cores

compiler_kwargs = {
    "cpp_options": {"STAN_THREADS": True, "jN": max_threads},
    "stanc_options": {
        "include-paths": os.path.join(get_models_path(), "include"),
        "O1": None,
    },
}
print("\nSetting default Stan compiler options:")
pp.pprint(compiler_kwargs)
update_config({"STAN_COMPILER_KWARGS": compiler_kwargs})
