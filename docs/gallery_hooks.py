import os

import matplotlib as mpl

_RC_FILE = os.path.join(os.path.dirname(__file__), "..", "examples", "matplotlibrc")
_POSTER_RCPARAMS = dict(mpl.rc_params_from_file(_RC_FILE, use_default_template=False))


def apply_poster_rcparams(gallery_conf, fname):
    """Apply poster-style rcParams before each gallery example.

    Parameters are required by sphinx-gallery's reset_modules callback API.
    """
    mpl.rcParams.update(_POSTER_RCPARAMS)
