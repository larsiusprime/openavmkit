import sys
import os
import warnings


def setup_environment():
    # Add the repository root to PYTHONPATH
    repo_root = os.path.abspath("..")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
        oldformatwarning = warnings.formatwarning

        # Customize warning format
        def custom_formatwarning(msg, category, filename, lineno, line):
            # if it's a user warning:
            if issubclass(category, UserWarning):
                return f"UserWarning: {msg}\n"
            else:
                return oldformatwarning(msg, category, filename, lineno, line)

        warnings.formatwarning = custom_formatwarning

    print("Environment setup completed.")

