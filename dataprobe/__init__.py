"""
DataProbe: Enterprise-grade Data Engineering Tools for Python

A comprehensive toolkit for data pipeline debugging, profiling, and optimization
with advanced visualization capabilities that rival commercial ETL monitoring tools.
"""

__version__ = "2.1.0"  # Updated from 0.1.5
__author__ = "SANTHOSH KRISHNAN R"
__email__ = "santhoshkrishnan3006@gmail.com"

# Import main classes with enhanced features
from dataprobe.debugger.pipeline_debugger import PipelineDebugger, OperationMetrics, DataLineage
from dataprobe.utils.logger import setup_logger

__all__ = [
    "PipelineDebugger",
    "OperationMetrics", 
    "DataLineage",
    "setup_logger",
    "__version__",
    "__author__",
    "__email__"
]

# Package metadata
DESCRIPTION = "Enterprise-grade debugging and profiling toolkit for data pipelines"
FEATURES = [
    "🔍 Advanced Operation Tracking with detailed metrics",
    "📊 Enterprise-grade Dashboard Visualizations", 
    "💾 Comprehensive Memory Profiling and leak detection",
    "🔗 Complete Data Lineage tracking and visualization",
    "⚠️ Intelligent Bottleneck Detection and optimization suggestions",
    "📈 Executive-level Performance Reports",
    "🎯 Smart Error Tracking with full traceback analysis",
    "🌳 Support for Nested Operations and complex workflows",
    "🎨 3D Pipeline Network visualizations",
    "📋 Professional Executive Reports for stakeholders"
]

def get_version():
    """Return the current version of DataProbe."""
    return __version__

def get_features():
    """Return a list of key features."""
    return FEATURES

def print_info():
    """Print package information."""
    print(f"DataProbe v{__version__}")
    print(f"Enterprise-grade pipeline debugging toolkit")
    print(f"Author: {__author__}")
    print(f"\nKey Features:")
    for feature in FEATURES:
        print(f"  {feature}")
    print(f"\nDocumentation: https://dataprobe.readthedocs.io/")
    print(f"GitHub: https://github.com/santhoshkrishnan30/dataprobe")
    print(f"Issues: https://github.com/santhoshkrishnan30/dataprobe/issues")

# Version compatibility check
import sys
if sys.version_info < (3, 8):
    raise RuntimeError("DataProbe requires Python 3.8 or higher")

# Optional rich console setup for better output
try:
    from rich.console import Console
    console = Console()
    
    def welcome_message():
        """Display a welcome message with rich formatting."""
        console.print(f"[bold green]DataProbe v{__version__} loaded successfully![/bold green]")
        console.print("[cyan]Enterprise-grade pipeline debugging toolkit ready to use.[/cyan]")
        console.print("[dim]Use PipelineDebugger() to get started.[/dim]")
    
except ImportError:
    def welcome_message():
        """Display a simple welcome message."""
        print(f"DataProbe v{__version__} loaded successfully!")
        print("Enterprise-grade pipeline debugging toolkit ready to use.")

# Display welcome message on import (optional - can be disabled)
# welcome_message()