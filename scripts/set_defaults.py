from baynes.model_utils import set_models_path, set_compiler_kwargs, get_models_path
import os
import pprint

pp = pprint.PrettyPrinter(indent=1, compact=True)
curr = os.getcwd()
models_path = os.path.join(curr, "stan")
print('Setting default Stan models path:\n', models_path)
set_models_path(models_path)
compiler_kwargs = {"cpp_options": {"STAN_THREADS": True, "jN": 16},
                    "stanc_options": {"include-paths": os.path.join(get_models_path(), 'include')}}
print('\nSetting default Stan compiler options:')
pp.pprint(compiler_kwargs)
set_compiler_kwargs(compiler_kwargs)