"""
DataProbe: Advanced Data Engineering Tools for Python

A comprehensive toolkit for data pipeline debugging, profiling, and optimization.
"""

__version__ = "0.1.5"
__author__ = "SANTHOSH KRISHNAN R"
__email__ = "santhoshkrishnan3006@gmail.com"

from dataprobe.debugger.pipeline_debugger import PipelineDebugger
from dataprobe.utils.logger import setup_logger

__all__ = [
    "PipelineDebugger",
    "setup_logger"
]


