import os, sys
import warnings
import traceback

def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    """Show detailed warning messages with stack trace."""
    # Format the basic warning message
    warning_msg = warnings.formatwarning(message, category, filename, lineno, line)
    # Get the current stack trace, excluding this custom function call
    stack = ''.join(traceback.format_stack(limit=5)[:-1])
    # Print the warning message with the stack trace
    print(warning_msg + '\nStack trace (most recent call last):\n' + stack)

class HiddenPrints:
    """Hides print statements in a context manager."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout