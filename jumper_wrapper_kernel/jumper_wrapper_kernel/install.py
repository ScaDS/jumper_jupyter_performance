"""
Installation script for the Jumper Wrapper Kernel.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from jupyter_client.kernelspec import KernelSpecManager

from .icon_utils import create_wrapped_kernel_icons, create_base_kernel_icons


KERNEL_JSON = {
    "argv": [sys.executable, "-m", "jumper_wrapper_kernel", "-f", "{connection_file}"],
    "display_name": "Jumper Wrapper Kernel",
    "language": "python",
    "metadata": {
        "debugger": False
    }
}


def install_kernel(user=True, prefix=None):
    """Install the Jumper Wrapper Kernel specification.

    Args:
        user: If True, install for current user only. Ignored if prefix is set.
        prefix: Install to specific prefix path (e.g., sys.prefix for virtualenv).
    """
    import tempfile
    import shutil
    
    kernel_spec_manager = KernelSpecManager()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        kernel_json_path = os.path.join(temp_dir, 'kernel.json')
        
        with open(kernel_json_path, 'w') as f:
            json.dump(KERNEL_JSON, f, indent=2)

        # Add branded icons for the base launcher entry (full kangaroo icon)
        create_base_kernel_icons(Path(temp_dir))
        
        # Install the kernel spec
        if prefix:
            kernel_spec_manager.install_kernel_spec(
                temp_dir,
                kernel_name='jumper_wrapper',
                user=False,
                prefix=prefix
            )
        else:
            kernel_spec_manager.install_kernel_spec(
                temp_dir,
                kernel_name='jumper_wrapper',
                user=user
            )
    
    print(f"Jumper Wrapper Kernel installed successfully!")
    print(f"You can now use it by selecting 'Jumper Wrapper Kernel' in Jupyter.")


def uninstall_kernel():
    """Remove the Jumper Wrapper Kernel specification."""
    kernel_spec_manager = KernelSpecManager()
    
    try:
        kernel_spec_manager.remove_kernel_spec('jumper_wrapper')
        print("Jumper Wrapper Kernel uninstalled successfully!")
    except Exception as e:
        print(f"Failed to uninstall kernel: {e}")
        sys.exit(1)


def main():
    """CLI entry point for kernel installation and removal."""
    parser = argparse.ArgumentParser(description='Install/Uninstall Jumper Wrapper Kernel')
    parser.add_argument('action', choices=['install', 'uninstall'], help='Action to perform')
    parser.add_argument('--user', action='store_true', default=True,
                        help='Install for the current user (default)')
    parser.add_argument('--sys-prefix', action='store_true',
                        help='Install to sys.prefix (e.g., for virtualenv/conda)')
    parser.add_argument('--prefix', type=str,
                        help='Install to a specific prefix')
    
    args = parser.parse_args()
    
    if args.action == 'install':
        if args.sys_prefix:
            install_kernel(user=False, prefix=sys.prefix)
        elif args.prefix:
            install_kernel(user=False, prefix=args.prefix)
        else:
            install_kernel(user=args.user)
    elif args.action == 'uninstall':
        uninstall_kernel()


if __name__ == '__main__':
    main()
