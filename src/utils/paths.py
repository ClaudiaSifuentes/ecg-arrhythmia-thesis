import os

def get_project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))

def get_data_path() -> str:
    return os.path.join(get_project_root(), '..', 'data')

def get_logs_path() -> str:
    return os.path.join(get_project_root(), '..', 'logs')

def get_configs_path() -> str:
    return os.path.join(get_project_root(), '..', 'configs')

def get_scripts_path() -> str:
    return os.path.join(get_project_root(), '..', 'scripts')

def get_notebooks_path() -> str:
    return os.path.join(get_project_root(), '..', 'notebooks')