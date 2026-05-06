"""
Jumper Wrapper Kernel - Main kernel implementation.

This kernel wraps other Jupyter kernels and forwards execution to them,
while keeping jumper-extension magic commands local.
"""

import sys
import os
from pathlib import Path
from ipykernel.ipkernel import IPythonKernel
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.core.interactiveshell import ExecutionInfo, ExecutionResult
from traitlets import Unicode

from .utilities import is_local_magic_cell
from .icon_utils import create_wrapped_kernel_icons


# Check for jumper-extension dependency
import importlib.util
JUMPER_EXTENSION_AVAILABLE = importlib.util.find_spec("jumper_extension") is not None

if not JUMPER_EXTENSION_AVAILABLE:
    raise ImportError(
        "jumper-extension is required but not installed. "
        "Please install it with: pip install jumper-extension"
    )


@magics_class
class JumperWrapperMagics(Magics):
    """Magic commands for the Jumper Wrapper Kernel."""
    
    def __init__(self, shell, kernel):
        """Initialize magic commands with shell and kernel references.

        Args:
            shell: IPython shell instance.
            kernel: JumperWrapperKernel instance for delegating operations.
        """
        super().__init__(shell)
        self._kernel = kernel
    
    @line_magic
    def list_kernels(self, line):
        """List all available Jupyter kernels."""
        self._kernel._list_kernels()
    
    @line_magic
    def wrap_kernel(self, line):
        """Wrap an existing Jupyter kernel.
        
        Usage:
            %wrap_kernel <kernel_name>
            %wrap_kernel <kernel_name> --save <new_kernel_name>
        
        The --save option creates a new permanent kernel spec that auto-wraps
        the specified kernel on startup.
        """
        self._kernel._wrap_kernel(line.strip())


class JumperWrapperKernel(IPythonKernel):
    """A Jupyter kernel that wraps other kernels."""
    
    implementation = 'jumper_wrapper_kernel'
    implementation_version = '0.1.0'
    language = 'python'
    language_version = sys.version.split()[0]
    language_info = {
        'name': 'python',
        'mimetype': 'text/x-python',
        'file_extension': '.py',
        'codemirror_mode': {
            'name': 'ipython',
            'version': 3,
        },
        'pygments_lexer': 'ipython3',
    }
    banner = "Jumper Wrapper Kernel - Wrap any Jupyter kernel with jumper-extension support"
    
    # Configuration trait for auto-wrapping a kernel on startup
    auto_wrap_kernel = Unicode('', config=True,
        help="Kernel name to automatically wrap on startup. If empty, no auto-wrap.")
    
    def __init__(self, **kwargs):
        """Initialize the wrapper kernel.

        Loads jumper-extension, registers wrapper magics, and prepares
        for optional auto-wrapping based on configuration.
        """
        super().__init__(**kwargs)

        # Wrapped kernel state
        self._wrapped_kernel_name = None
        self._kernel_manager = None
        self._kernel_client = None
        self._kernel_spec_manager = KernelSpecManager()

        # Set of magic commands registered (populated after loading extensions)
        self._jumper_magic_commands = set()
        self._wrapper_magic_commands = set()
        self._load_jumper_extension()
        self._register_wrapper_magics()

        # Deferred auto-wrap flag (don't block __init__)
        self._auto_wrap_pending = bool(self.auto_wrap_kernel)
    
    def _auto_wrap_on_startup(self):
        """Automatically wrap the configured kernel on startup."""
        kernel_name = self.auto_wrap_kernel
        
        # Check if kernel exists
        available_kernels = self._get_available_kernels()
        if kernel_name not in available_kernels:
            self.log.error(f"Auto-wrap kernel '{kernel_name}' not found")
            return
        
        try:
            self._kernel_manager = KernelManager(kernel_name=kernel_name)
            self._kernel_manager.start_kernel()
            self._kernel_client = self._kernel_manager.client()
            self._kernel_client.start_channels()
            self._kernel_client.wait_for_ready(timeout=60)
            self._wrapped_kernel_name = kernel_name
            
            # Get language info from wrapped kernel
            self._update_language_info_from_wrapped_kernel()
            
            self.log.info(f"Auto-wrapped kernel: {kernel_name}")
        except Exception as e:
            self.log.error(f"Failed to auto-wrap kernel '{kernel_name}': {e}")
    
    def _load_jumper_extension(self):
        """Load jumper-extension and capture its registered magic commands."""
        # Ensure matplotlib picks an inline-friendly backend (before
        # IPKernelApp.init_gui_pylab sets MPLBACKEND).
        os.environ.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")

        # Get magics before loading extension
        magics_before = self._get_all_magics()
        
        # Load jumper extension using the kernel's shell
        self.shell.run_line_magic('load_ext', 'jumper_extension')

        # Get magics after loading extension
        magics_after = self._get_all_magics()
        
        # The difference is the magics registered by jumper-extension
        self._jumper_magic_commands = magics_after - magics_before

    def _register_wrapper_magics(self):
        """Register wrapper magic commands and capture their names."""
        # Get magics before registering
        magics_before = self._get_all_magics()
        
        # Register our magic class
        wrapper_magics = JumperWrapperMagics(self.shell, self)
        self.shell.register_magics(wrapper_magics)
        
        # Get magics after registering
        magics_after = self._get_all_magics()
        
        # The difference is our wrapper magics
        self._wrapper_magic_commands = magics_after - magics_before
    
    def _get_all_magics(self):
        """Get all currently registered magic command names."""
        magics = set()
        magic_manager = self.shell.magics_manager
        
        # Get line magics
        for name in magic_manager.magics.get('line', {}).keys():
            magics.add(name)
        
        # Get cell magics
        for name in magic_manager.magics.get('cell', {}).keys():
            magics.add(name)
        
        return magics
    
    def _get_available_kernels(self):
        """Get a list of available kernel specs."""
        return self._kernel_spec_manager.get_all_specs()
    
    def _get_local_magics(self):
        """Get combined set of local magic commands."""
        return self._jumper_magic_commands | self._wrapper_magic_commands
    
    def _list_kernels(self):
        """List all available Jupyter kernels."""
        kernels = self._get_available_kernels()
        
        output = "Available Jupyter Kernels:\n"
        output += "-" * 50 + "\n"
        
        for name, spec in kernels.items():
            display_name = spec.get('spec', {}).get('display_name', name)
            language = spec.get('spec', {}).get('language', 'unknown')
            output += f"  {name}: {display_name} ({language})\n"
        
        if self._wrapped_kernel_name:
            output += "\n" + "-" * 50 + "\n"
            output += f"Currently wrapped kernel: {self._wrapped_kernel_name}\n"
        
        self.send_response(self.iopub_socket, 'stream', {'name': 'stdout', 'text': output})
        
        return {
            'status': 'ok',
            'execution_count': self.execution_count,
            'payload': [],
            'user_expressions': {},
        }
    
    def _wrap_kernel(self, args):
        """Wrap the specified kernel."""
        # Parse arguments
        parts = args.split()
        kernel_name = None
        save_name = None
        
        i = 0
        while i < len(parts):
            if parts[i] == '--save' and i + 1 < len(parts):
                save_name = parts[i + 1]
                i += 2
            elif kernel_name is None:
                kernel_name = parts[i]
                i += 1
            else:
                i += 1
        
        if not kernel_name:
            error_msg = "Usage: %wrap_kernel <kernel_name> [--save <new_kernel_name>]\nUse %list_kernels to see available kernels."
            self.send_response(self.iopub_socket, 'stream', {'name': 'stderr', 'text': error_msg})
            return {
                'status': 'error',
                'ename': 'ValueError',
                'evalue': 'No kernel name specified',
                'traceback': [error_msg],
                'execution_count': self.execution_count,
            }
        
        # Check if kernel exists
        available_kernels = self._get_available_kernels()
        if kernel_name not in available_kernels:
            error_msg = f"Kernel '{kernel_name}' not found. Use %list_kernels to see available kernels."
            self.send_response(self.iopub_socket, 'stream', {'name': 'stderr', 'text': error_msg})
            return {
                'status': 'error',
                'ename': 'ValueError',
                'evalue': f'Kernel not found: {kernel_name}',
                'traceback': [error_msg],
                'execution_count': self.execution_count,
            }
        
        # Shutdown existing wrapped kernel if any
        if self._kernel_manager is not None:
            self._shutdown_wrapped_kernel()
        
        # Start the new kernel
        try:
            self._kernel_manager = KernelManager(kernel_name=kernel_name)
            self._kernel_manager.start_kernel()
            self._kernel_client = self._kernel_manager.client()
            self._kernel_client.start_channels()
            self._kernel_client.wait_for_ready(timeout=60)
            self._wrapped_kernel_name = kernel_name
            
            # Get language info from wrapped kernel via kernel_info request
            self._update_language_info_from_wrapped_kernel()
            
            # Notify frontend to refresh kernel info for updated syntax highlighting
            self._notify_frontend_language_change()
            
            success_msg = f"Successfully wrapped kernel: {kernel_name}\n"
            success_msg += "Hint: Refresh the page (without restarting the kernel) to enable syntax highlighting for the wrapped language.\n"
            
            # Handle --save option to create permanent kernel spec
            if save_name:
                save_result = self._save_wrapped_kernel_spec(kernel_name, save_name)
                if save_result:
                    success_msg += f"Created permanent kernel '{save_name}' that auto-wraps '{kernel_name}'.\n"
                else:
                    success_msg += f"Warning: Failed to create permanent kernel spec.\n"
            else:
                success_msg += f"Tip: Use '%wrap_kernel {kernel_name} --save <name>' to create a permanent kernel that auto-wraps on startup.\n"
            
            self.send_response(self.iopub_socket, 'stream', {'name': 'stdout', 'text': success_msg})
            
            return {
                'status': 'ok',
                'execution_count': self.execution_count,
                'payload': [],
                'user_expressions': {},
            }
        except Exception as e:
            error_msg = f"Failed to start kernel '{kernel_name}': {str(e)}"
            self.send_response(self.iopub_socket, 'stream', {'name': 'stderr', 'text': error_msg})
            return {
                'status': 'error',
                'ename': type(e).__name__,
                'evalue': str(e),
                'traceback': [error_msg],
                'execution_count': self.execution_count,
            }
    
    def _save_wrapped_kernel_spec(self, wrapped_kernel_name, new_kernel_name):
        """Create a new kernel spec that auto-wraps the specified kernel on startup."""
        import json
        import os
        
        try:
            # Get the wrapped kernel's display name
            available_kernels = self._get_available_kernels()
            wrapped_spec = available_kernels.get(wrapped_kernel_name, {}).get('spec', {})
            wrapped_display_name = wrapped_spec.get('display_name', wrapped_kernel_name)
            base_kernel_spec = self._kernel_spec_manager.get_kernel_spec(wrapped_kernel_name)
            
            # Create kernel spec directory
            kernel_dir = os.path.join(
                os.path.expanduser('~'),
                '.local', 'share', 'jupyter', 'kernels',
                new_kernel_name
            )
            os.makedirs(kernel_dir, exist_ok=True)
            
            # Create kernel.json with auto-wrap configuration
            kernel_spec = {
                'argv': [
                    sys.executable,
                    '-m', 'jumper_wrapper_kernel',
                    '-f', '{connection_file}',
                    '--JumperWrapperKernel.auto_wrap_kernel=' + wrapped_kernel_name,
                ],
                'display_name': f'{new_kernel_name} (Wrapped)',
                'language': wrapped_spec.get('language', 'python'),
                'metadata': {
                    'debugger': False,
                    'jumper_wrapper': {
                        'wrapped_kernel': wrapped_kernel_name,
                    }
                }
            }
            
            kernel_json_path = os.path.join(kernel_dir, 'kernel.json')
            with open(kernel_json_path, 'w') as f:
                json.dump(kernel_spec, f, indent=2)

            # Generate launcher icons with kangaroo badge
            create_wrapped_kernel_icons(
                Path(kernel_dir),
                wrapped_spec=base_kernel_spec,
                logger=self.log
            )
            
            return True
        except Exception as e:
            return False
    
    def _update_language_info_from_wrapped_kernel(self):
        """Get language info from the wrapped kernel and update our language_info."""
        if self._kernel_client is None:
            return
        
        try:
            # Request kernel_info from wrapped kernel
            msg_id = self._kernel_client.kernel_info()
            reply = self._kernel_client.get_shell_msg(timeout=10)
            
            if reply['content'].get('status') == 'ok':
                wrapped_language_info = reply['content'].get('language_info', {})
                
                # Update our language info to match the wrapped kernel
                if wrapped_language_info:
                    self.language = wrapped_language_info.get('name', 'python')
                    self.language_info = wrapped_language_info.copy()
        except Exception:
            pass
    
    def _notify_frontend_language_change(self):
        """Send JavaScript to frontend to trigger kernel info refresh for syntax highlighting."""
        js_code = """
        if (typeof Jupyter !== 'undefined' && Jupyter.notebook) {
            // JupyterLab classic notebook
            Jupyter.notebook.kernel.kernel_info(function(reply) {
                if (reply.content && reply.content.language_info) {
                    Jupyter.notebook.metadata.language_info = reply.content.language_info;
                    // Trigger CodeMirror mode change for all cells
                    var mode = reply.content.language_info.codemirror_mode || reply.content.language_info.name;
                    Jupyter.notebook.get_cells().forEach(function(cell) {
                        if (cell.cell_type === 'code') {
                            cell.code_mirror.setOption('mode', mode);
                        }
                    });
                }
            });
        }
        """
        self.send_response(
            self.iopub_socket,
            'display_data',
            {
                'data': {'application/javascript': js_code},
                'metadata': {},
            }
        )
    
    def _shutdown_wrapped_kernel(self):
        """Shutdown the currently wrapped kernel."""
        if self._kernel_client is not None:
            self._kernel_client.stop_channels()
            self._kernel_client = None
        
        if self._kernel_manager is not None:
            self._kernel_manager.shutdown_kernel(now=True)
            self._kernel_manager = None
        
        self._wrapped_kernel_name = None
    
    def _trigger_pre_run_cell(self, code, silent, store_history):
        """Trigger pre_run_cell event for jumper-extension hooks."""
        info = ExecutionInfo(
            raw_cell=code,
            store_history=store_history,
            silent=silent,
            shell_futures=True,
            cell_id=None
        )
        self.shell.events.trigger('pre_run_cell', info)
        return info
    
    def _trigger_post_run_cell(self, info, success=True, result_value=None, error=None):
        """Trigger post_run_cell event for jumper-extension hooks."""
        exec_result = ExecutionResult(info)
        exec_result.result = result_value
        if error:
            exec_result.error_in_exec = error
        self.shell.events.trigger('post_run_cell', exec_result)
        return exec_result
    
    def _forward_to_wrapped_kernel(self, code, silent, store_history, user_expressions, allow_stdin):
        """Forward code execution to the wrapped kernel."""
        if self._kernel_client is None:
            error_msg = "No kernel is currently wrapped. Use %wrap_kernel <kernel_name> to wrap a kernel."
            self.send_response(self.iopub_socket, 'stream', {'name': 'stderr', 'text': error_msg})
            return {
                'status': 'error',
                'ename': 'RuntimeError',
                'evalue': 'No wrapped kernel',
                'traceback': [error_msg],
                'execution_count': self.execution_count,
            }
        
        # Trigger pre_run_cell event for jumper-extension
        exec_info = self._trigger_pre_run_cell(code, silent, store_history)
        
        # Execute on wrapped kernel
        msg_id = self._kernel_client.execute(
            code,
            silent=silent,
            store_history=store_history,
            user_expressions=user_expressions,
            allow_stdin=allow_stdin,
        )
        
        # Process messages from the wrapped kernel until we get the shell reply
        # We need to process iopub messages while waiting for the reply
        execution_error = None
        got_idle = False
        result = None
        
        # Process iopub messages until we see idle status
        while not got_idle:
            try:
                msg = self._kernel_client.get_iopub_msg(timeout=0.1)
                msg_type = msg['header']['msg_type']
                content = msg['content']
                
                # Only process messages for our execution request
                parent_msg_id = msg.get('parent_header', {}).get('msg_id')
                if parent_msg_id != msg_id:
                    continue
                
                if msg_type == 'status':
                    if content.get('execution_state') == 'idle':
                        got_idle = True
                elif msg_type == 'error':
                    execution_error = Exception(content.get('evalue', 'Unknown error'))
                    self.send_response(self.iopub_socket, msg_type, content)
                elif msg_type in ('stream', 'display_data', 'execute_result'):
                    self.send_response(self.iopub_socket, msg_type, content)
                    
            except Exception:
                # Timeout - check if shell reply is available
                try:
                    reply = self._kernel_client.get_shell_msg(timeout=0.1)
                    result = reply['content']
                except Exception:
                    pass
        
        # If we didn't get the shell reply yet, get it now
        if result is None:
            try:
                reply = self._kernel_client.get_shell_msg(timeout=30)
                result = reply['content']
            except Exception as e:
                execution_error = e
                result = {
                    'status': 'error',
                    'ename': type(e).__name__,
                    'evalue': str(e),
                    'traceback': [str(e)],
                    'execution_count': self.execution_count,
                }
        
        # Trigger post_run_cell event for jumper-extension
        self._trigger_post_run_cell(
            exec_info,
            success=(result.get('status') == 'ok'),
            error=execution_error
        )

        # KernelBase.execute_request increments self.execution_count before
        # calling do_execute, but IPythonKernel's setter is a no-op (it defers
        # to self.shell.execution_count).  For local magics shell.run_cell
        # handles the increment internally; for forwarded cells we must do it
        # ourselves.  We also override the execution_count in the result so the
        # frontend shows the wrapper's sequential In[N], not the wrapped
        # kernel's independent counter.
        result['execution_count'] = self.shell.execution_count
        self.shell.execution_count += 1

        return result
    
    def _execute_local_magic(self, code):
        """Execute a magic command locally using IPython."""
        try:
            result = self.shell.run_cell(code, store_history=True)
            execution_count = self.execution_count - 1

            if result.success:
                if result.result is not None:
                    self.send_response(self.iopub_socket, 'stream',
                                      {'name': 'stdout', 'text': str(result.result) + '\n'})
                return {
                    'status': 'ok',
                    'execution_count': execution_count,
                    'payload': [],
                    'user_expressions': {},
                }
            else:
                error_info = result.error_in_exec or result.error_before_exec
                if error_info:
                    return {
                        'status': 'error',
                        'ename': type(error_info).__name__,
                        'evalue': str(error_info),
                        'traceback': [str(error_info)],
                        'execution_count': execution_count,
                    }
                return {
                    'status': 'error',
                    'ename': 'Error',
                    'evalue': 'Unknown error',
                    'traceback': [],
                    'execution_count': execution_count,
                }
        except Exception as e:
            return {
                'status': 'error',
                'ename': type(e).__name__,
                'evalue': str(e),
                'traceback': [str(e)],
                'execution_count': self.execution_count - 1,
            }
    
    def _is_local_magic(self, code):
        """Check if code is a magic command that should be executed locally."""
        return is_local_magic_cell(code, self._get_local_magics())
    
    def do_execute(self, code, silent, store_history=True, user_expressions=None, allow_stdin=False, **kwargs):
        """Execute code - either locally or forwarded to wrapped kernel."""
        user_expressions = user_expressions or {}

        # Perform deferred auto-wrap on first execution (not in __init__ to avoid blocking)
        if self._auto_wrap_pending:
            self._auto_wrap_pending = False
            self._auto_wrap_on_startup()

        # Check for local magic commands (wrapper + jumper) - execute locally
        if self._is_local_magic(code):
            return self._execute_local_magic(code)

        # Forward everything else to the wrapped kernel
        return self._forward_to_wrapped_kernel(code, silent, store_history, user_expressions, allow_stdin)
    
    def do_shutdown(self, restart):
        """Shutdown the kernel."""
        self._shutdown_wrapped_kernel()
        return {'status': 'ok', 'restart': restart}
    
    def do_complete(self, code, cursor_pos):
        """Handle code completion - forward to wrapped kernel if available."""
        # Perform deferred auto-wrap if pending
        if self._auto_wrap_pending:
            self._auto_wrap_pending = False
            self._auto_wrap_on_startup()

        if self._kernel_client is not None:
            msg_id = self._kernel_client.complete(code, cursor_pos)
            try:
                reply = self._kernel_client.get_shell_msg(timeout=10)
                return reply['content']
            except Exception:
                pass
        
        return {
            'status': 'ok',
            'matches': [],
            'cursor_start': cursor_pos,
            'cursor_end': cursor_pos,
            'metadata': {},
        }
    
    def do_inspect(self, code, cursor_pos, detail_level=0):
        """Handle object inspection - forward to wrapped kernel if available."""
        # Perform deferred auto-wrap if pending
        if self._auto_wrap_pending:
            self._auto_wrap_pending = False
            self._auto_wrap_on_startup()

        if self._kernel_client is not None:
            msg_id = self._kernel_client.inspect(code, cursor_pos, detail_level)
            try:
                reply = self._kernel_client.get_shell_msg(timeout=10)
                return reply['content']
            except Exception:
                pass
        
        return {
            'status': 'ok',
            'found': False,
            'data': {},
            'metadata': {},
        }
    
    @property
    def kernel_info(self):
        """Return kernel info with current language_info (may be from wrapped kernel)."""
        return {
            'protocol_version': '5.3',
            'implementation': self.implementation,
            'implementation_version': self.implementation_version,
            'language_info': self.language_info,
            'banner': self.banner,
            'help_links': [],
        }


if __name__ == '__main__':
    from ipykernel.kernelapp import IPKernelApp
    IPKernelApp.launch_instance(kernel_class=JumperWrapperKernel)
