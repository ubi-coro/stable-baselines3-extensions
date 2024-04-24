import os, json
import platform, re
import torch
from importlib import metadata

def save_system_info(save_path: str) -> None:
    """Save all versions to a file (system_info.txt).

    :param save_path: The output path
    """
    system_info_dict = {}

    os_version = re.sub(r"#(\d)", r"# \1", f"{platform.platform()} {platform.version()}")
    system_info_dict.update({"OS": os_version})
    system_info_dict.update({"Python": platform.python_version()})
    system_info_dict.update({"GPU Enabled": torch.cuda.is_available()})
    for dist in metadata.distributions():
        system_info_dict.update({dist.name: dist.version})
    with open(os.path.join(save_path, "system_info.txt"), "w") as f:
        json.dump(system_info_dict, f, indent=2)