from typing import Any

# MoviePy integration for video support.
# Supports both MoviePy 1.x and 2.x versions.
try:
    try:
        # Standard import for MoviePy 1.x and some 2.x setups
        from moviepy.editor import VideoClip
    except ImportError:
        # Alternative import for some MoviePy 2.x distributions
        from moviepy import VideoClip
    HAS_MOVIEPY = True
except ImportError:
    # Fallback if MoviePy is not installed
    class VideoClip: pass
    HAS_MOVIEPY = False
