# dataalchemy/debugger/pipeline_debugger.py

import time
import traceback
import psutil
import pandas as pd
import polars as pl
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from datetime import datetime
from functools import wraps
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.tree import Tree
from rich.panel import Panel
from rich.syntax import Syntax
import warnings
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import sys
import gc
import numpy as np

# Initialize Rich console for beautiful output
console = Console()

@dataclass
class OperationMetrics:
    """Store metrics for a single operation."""
    operation_id: str
    operation_name: str
    start_time: float
    end_time: float = 0.0
    duration: float = 0.0
    memory_before: float = 0.0
    memory_after: float = 0.0
    memory_delta: float = 0.0
    input_shape: Optional[Tuple] = None
    output_shape: Optional[Tuple] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataLineage:
    """Track data lineage information."""
    data_id: str
    source: str
    transformations: List[Dict[str, Any]] = field(default_factory=list)
    current_shape: Optional[Tuple] = None
    data_type: str = "unknown"
    column_changes: List[Dict[str, Any]] = field(default_factory=list)
    
class PipelineDebugger:
    """
    A comprehensive debugging tool for data pipelines that tracks operations,
    memory usage, data lineage, and provides visual debugging capabilities.
    """
    
    def __init__(self, 
                 name: str = "Pipeline",
                 track_memory: bool = True,
                 track_lineage: bool = True,
                 auto_save: bool = True,
                 save_path: Optional[Path] = None,
                 memory_threshold_mb: float = 100.0):
        """
        Initialize the PipelineDebugger.
        
        Args:
            name: Name of the pipeline
            track_memory: Whether to track memory usage
            track_lineage: Whether to track data lineage
            auto_save: Whether to auto-save debugging information
            save_path: Path to save debugging information
            memory_threshold_mb: Memory threshold for warnings (in MB)
        """
        self.name = name
        self.track_memory = track_memory
        self.track_lineage = track_lineage
        self.auto_save = auto_save
        self.save_path = save_path or Path(f"./debug_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.memory_threshold_mb = memory_threshold_mb
        
        # Operation tracking
        self.operations: Dict[str, OperationMetrics] = {}
        self.operation_order: List[str] = []
        self.current_operation_stack: List[str] = []
        
        # Data lineage tracking
        self.data_lineages: Dict[str, DataLineage] = {}
        
        # Performance metrics
        self.bottlenecks: List[str] = []
        self.memory_peaks: List[Tuple[str, float]] = []
        
        # Create save directory if needed
        if self.auto_save:
            self.save_path.mkdir(parents=True, exist_ok=True)
            
        console.print(f"[green]Pipeline Debugger initialized: {self.name}[/green]")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.track_memory:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        return 0.0
    
    def _generate_operation_id(self, operation_name: str) -> str:
        """Generate unique operation ID."""
        timestamp = str(time.time())
        return hashlib.md5(f"{operation_name}_{timestamp}".encode()).hexdigest()[:8]
    
    def _detect_data_type(self, data: Any) -> str:
        """Detect the type of data being processed."""
        if isinstance(data, pd.DataFrame):
            return "pandas.DataFrame"
        elif isinstance(data, pl.DataFrame):
            return "polars.DataFrame"
        elif isinstance(data, np.ndarray):
            return "numpy.ndarray"
        elif isinstance(data, list):
            return "list"
        elif isinstance(data, dict):
            return "dict"
        else:
            return type(data).__name__
    
    def _get_data_shape(self, data: Any) -> Optional[Tuple]:
        """Get shape of data if applicable."""
        if hasattr(data, 'shape'):
            return data.shape
        elif isinstance(data, (list, dict)):
            return (len(data),)
        return None
    
    def _track_column_changes(self, data_before: Any, data_after: Any) -> List[Dict[str, Any]]:
        """Track column changes in DataFrames."""
        changes = []
        
        if isinstance(data_before, (pd.DataFrame, pl.DataFrame)) and isinstance(data_after, (pd.DataFrame, pl.DataFrame)):
            cols_before = set(data_before.columns)
            cols_after = set(data_after.columns)
            
            added = cols_after - cols_before
            removed = cols_before - cols_after
            
            if added:
                changes.append({"type": "columns_added", "columns": list(added)})
            if removed:
                changes.append({"type": "columns_removed", "columns": list(removed)})
                
        return changes
    
    def track_operation(self, operation_name: str, **metadata):
        """
        Decorator to track an operation in the pipeline.
        
        Args:
            operation_name: Name of the operation
            **metadata: Additional metadata to store
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                operation_id = self._generate_operation_id(operation_name)
                parent_id = self.current_operation_stack[-1] if self.current_operation_stack else None
                
                # Initialize operation metrics
                metrics = OperationMetrics(
                    operation_id=operation_id,
                    operation_name=operation_name,
                    start_time=time.time(),
                    memory_before=self._get_memory_usage(),
                    parent_id=parent_id,
                    metadata=metadata
                )
                
                # Add to parent's children if exists
                if parent_id and parent_id in self.operations:
                    self.operations[parent_id].children_ids.append(operation_id)
                
                # Track operation
                self.operations[operation_id] = metrics
                self.operation_order.append(operation_id)
                self.current_operation_stack.append(operation_id)
                
                # Show progress
                console.print(f"\n[blue]▶ Starting operation: {operation_name}[/blue]")
                
                try:
                    # Execute the function
                    result = func(*args, **kwargs)
                    
                    # Update metrics
                    metrics.end_time = time.time()
                    metrics.duration = metrics.end_time - metrics.start_time
                    metrics.memory_after = self._get_memory_usage()
                    metrics.memory_delta = metrics.memory_after - metrics.memory_before
                    
                    # Get data shapes
                    if args:
                        metrics.input_shape = self._get_data_shape(args[0])
                    metrics.output_shape = self._get_data_shape(result)
                    
                    # Track lineage
                    if self.track_lineage and result is not None:
                        self._update_lineage(operation_id, operation_name, args, result)
                    
                    # Check for bottlenecks
                    if metrics.duration > 1.0:  # Operations taking more than 1 second
                        self.bottlenecks.append(operation_id)
                        console.print(f"[yellow]⚠ Bottleneck detected: {operation_name} took {metrics.duration:.2f}s[/yellow]")
                    
                    # Check memory usage
                    if metrics.memory_delta > self.memory_threshold_mb:
                        self.memory_peaks.append((operation_id, metrics.memory_delta))
                        console.print(f"[yellow]⚠ High memory usage: {metrics.memory_delta:.2f}MB[/yellow]")
                    
                    console.print(f"[green]✓ Completed: {operation_name} ({metrics.duration:.3f}s, {metrics.memory_delta:+.1f}MB)[/green]")
                    
                    return result
                    
                except Exception as e:
                    # Record error
                    metrics.end_time = time.time()
                    metrics.duration = metrics.end_time - metrics.start_time
                    metrics.error = str(e)
                    metrics.traceback = traceback.format_exc()
                    
                    console.print(f"[red]✗ Error in {operation_name}: {str(e)}[/red]")
                    raise
                    
                finally:
                    self.current_operation_stack.pop()
                    
                    # Auto-save if enabled
                    if self.auto_save:
                        self.save_checkpoint()
                        
            return wrapper
        return decorator
    
    def _update_lineage(self, operation_id: str, operation_name: str, inputs: tuple, output: Any):
        """Update data lineage information."""
        # Generate data ID for output
        data_id = hashlib.md5(str(id(output)).encode()).hexdigest()[:8]
        
        # Create or update lineage
        if data_id not in self.data_lineages:
            self.data_lineages[data_id] = DataLineage(
                data_id=data_id,
                source=operation_name,
                data_type=self._detect_data_type(output),
                current_shape=self._get_data_shape(output)
            )
        
        # Add transformation
        transformation = {
            "operation_id": operation_id,
            "operation_name": operation_name,
            "timestamp": datetime.now().isoformat(),
            "input_shapes": [self._get_data_shape(inp) for inp in inputs if inp is not None],
            "output_shape": self._get_data_shape(output)
        }
        
        # Track column changes if applicable
        if inputs and hasattr(inputs[0], 'columns') and hasattr(output, 'columns'):
            column_changes = self._track_column_changes(inputs[0], output)
            if column_changes:
                transformation["column_changes"] = column_changes
                self.data_lineages[data_id].column_changes.extend(column_changes)
        
        self.data_lineages[data_id].transformations.append(transformation)
    
    def profile_memory(self, func: Callable) -> Callable:
        """
        Decorator for detailed memory profiling of a function.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Force garbage collection
            gc.collect()
            
            # Get initial memory
            mem_before = self._get_memory_usage()
            
            # Track memory during execution
            memory_samples = []
            start_time = time.time()
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Get final memory
            gc.collect()
            mem_after = self._get_memory_usage()
            
            # Report
            console.print(Panel(
                f"Memory Profile: {func.__name__}\n"
                f"Before: {mem_before:.2f}MB\n"
                f"After: {mem_after:.2f}MB\n"
                f"Delta: {mem_after - mem_before:+.2f}MB",
                title="Memory Usage",
                border_style="cyan"
            ))
            
            return result
        return wrapper
    
    def analyze_dataframe(self, df: Union[pd.DataFrame, pl.DataFrame], name: str = "DataFrame"):
        """
        Analyze a DataFrame and provide detailed statistics.
        """
        console.print(f"\n[cyan]Analyzing {name}...[/cyan]")
        
        # Create analysis table
        table = Table(title=f"DataFrame Analysis: {name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Basic info
        table.add_row("Shape", str(df.shape))
        table.add_row("Memory Usage", f"{df.memory_usage().sum() / 1024 / 1024:.2f} MB" if isinstance(df, pd.DataFrame) else "N/A")
        table.add_row("Columns", str(len(df.columns)))
        table.add_row("Data Types", str(df.dtypes.value_counts().to_dict()) if isinstance(df, pd.DataFrame) else str(df.dtypes))
        
        # Missing values
        if isinstance(df, pd.DataFrame):
            missing = df.isnull().sum()
            table.add_row("Missing Values", str(missing[missing > 0].to_dict()) if any(missing > 0) else "None")
        
        # Duplicates
        table.add_row("Duplicate Rows", str(df.duplicated().sum()) if isinstance(df, pd.DataFrame) else "N/A")
        
        console.print(table)
        
        # Column details
        col_table = Table(title="Column Details")
        col_table.add_column("Column", style="cyan")
        col_table.add_column("Type", style="green")
        col_table.add_column("Non-Null", style="yellow")
        col_table.add_column("Unique", style="magenta")
        
        for col in df.columns[:10]:  # Show first 10 columns
            if isinstance(df, pd.DataFrame):
                col_table.add_row(
                    col,
                    str(df[col].dtype),
                    str(df[col].count()),
                    str(df[col].nunique())
                )
        
        console.print(col_table)
    
    def visualize_pipeline(self, save_path: Optional[Path] = None):
        """
        Create a visual representation of the pipeline execution.
        """
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes
        for op_id, metrics in self.operations.items():
            label = f"{metrics.operation_name}\n{metrics.duration:.3f}s"
            color = 'red' if metrics.error else ('yellow' if op_id in self.bottlenecks else 'lightblue')
            G.add_node(op_id, label=label, color=color)
        
        # Add edges
        for op_id, metrics in self.operations.items():
            for child_id in metrics.children_ids:
                G.add_edge(op_id, child_id)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        
        # Draw nodes
        node_colors = [G.nodes[node].get('color', 'lightblue') for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, arrowsize=20)
        
        # Draw labels
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title(f"Pipeline Execution Flow: {self.name}")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.save_path / "pipeline_flow.png", dpi=300, bbox_inches='tight')
        
        plt.close()
        console.print("[green]✓ Pipeline visualization saved[/green]")
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive debugging report.
        """
        report = {
            "pipeline_name": self.name,
            "total_operations": len(self.operations),
            "total_duration": sum(op.duration for op in self.operations.values()),
            "total_memory_used": sum(op.memory_delta for op in self.operations.values()),
            "errors": sum(1 for op in self.operations.values() if op.error),
            "bottlenecks": len(self.bottlenecks),
            "timestamp": datetime.now().isoformat()
        }
        
        # Operation details
        report["operations"] = []
        for op_id in self.operation_order:
            metrics = self.operations[op_id]
            report["operations"].append({
                "id": op_id,
                "name": metrics.operation_name,
                "duration": metrics.duration,
                "memory_delta": metrics.memory_delta,
                "input_shape": metrics.input_shape,
                "output_shape": metrics.output_shape,
                "error": metrics.error,
                "is_bottleneck": op_id in self.bottlenecks
            })
        
        # Bottleneck analysis
        if self.bottlenecks:
            report["bottleneck_operations"] = [
                {
                    "id": op_id,
                    "name": self.operations[op_id].operation_name,
                    "duration": self.operations[op_id].duration
                }
                for op_id in self.bottlenecks
            ]
        
        # Memory peaks
        if self.memory_peaks:
            report["memory_peaks"] = [
                {
                    "operation": self.operations[op_id].operation_name,
                    "memory_increase_mb": mem_delta
                }
                for op_id, mem_delta in self.memory_peaks
            ]
        
        return report
    
    def print_summary(self):
        """
        Print a summary of the pipeline execution.
        """
        # Create summary tree
        tree = Tree(f"[bold cyan]Pipeline Summary: {self.name}[/bold cyan]")
        
        # Execution stats
        stats_branch = tree.add("[yellow]Execution Statistics[/yellow]")
        stats_branch.add(f"Total Operations: {len(self.operations)}")
        stats_branch.add(f"Total Duration: {sum(op.duration for op in self.operations.values()):.3f}s")
        stats_branch.add(f"Total Memory Used: {sum(op.memory_delta for op in self.operations.values()):.2f}MB")
        
        # Errors
        errors = [op for op in self.operations.values() if op.error]
        if errors:
            error_branch = tree.add(f"[red]Errors ({len(errors)})[/red]")
            for op in errors:
                error_branch.add(f"{op.operation_name}: {op.error}")
        
        # Bottlenecks
        if self.bottlenecks:
            bottleneck_branch = tree.add(f"[yellow]Bottlenecks ({len(self.bottlenecks)})[/yellow]")
            for op_id in self.bottlenecks:
                op = self.operations[op_id]
                bottleneck_branch.add(f"{op.operation_name}: {op.duration:.3f}s")
        
        # Memory peaks
        if self.memory_peaks:
            memory_branch = tree.add(f"[magenta]Memory Peaks ({len(self.memory_peaks)})[/magenta]")
            for op_id, mem_delta in self.memory_peaks:
                op = self.operations[op_id]
                memory_branch.add(f"{op.operation_name}: +{mem_delta:.2f}MB")
        
        console.print(tree)
    
    def save_checkpoint(self):
        """
        Save current debugging state to disk.
        """
        checkpoint = {
            "pipeline_name": self.name,
            "operations": self.operations,
            "operation_order": self.operation_order,
            "data_lineages": self.data_lineages,
            "bottlenecks": self.bottlenecks,
            "memory_peaks": self.memory_peaks,
            "timestamp": datetime.now().isoformat()
        }
        
        checkpoint_path = self.save_path / f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def export_lineage(self, format: str = "json") -> Union[str, Dict]:
        """
        Export data lineage information.
        
        Args:
            format: Export format ('json' or 'dict')
        """
        lineage_data = {
            "pipeline": self.name,
            "lineages": {}
        }
        
        for data_id, lineage in self.data_lineages.items():
            lineage_data["lineages"][data_id] = {
                "source": lineage.source,
                "data_type": lineage.data_type,
                "current_shape": lineage.current_shape,
                "transformations": lineage.transformations,
                "column_changes": lineage.column_changes
            }
        
        if format == "json":
            return json.dumps(lineage_data, indent=2)
        return lineage_data
    
    def suggest_optimizations(self) -> List[Dict[str, Any]]:
        """
        Analyze the pipeline and suggest optimizations.
        """
        suggestions = []
        
        # Check for slow operations
        for op_id in self.bottlenecks:
            op = self.operations[op_id]
            suggestions.append({
                "type": "performance",
                "operation": op.operation_name,
                "issue": f"Operation took {op.duration:.2f}s",
                "suggestion": "Consider optimizing this operation or parallelizing if possible"
            })
        
        # Check for high memory usage
        for op_id, mem_delta in self.memory_peaks:
            op = self.operations[op_id]
            suggestions.append({
                "type": "memory",
                "operation": op.operation_name,
                "issue": f"High memory usage: +{mem_delta:.2f}MB",
                "suggestion": "Consider processing data in chunks or optimizing memory usage"
            })
        
        # Check for inefficient operations
        for op_id, op in self.operations.items():
            if op.input_shape and op.output_shape:
                if isinstance(op.input_shape, tuple) and isinstance(op.output_shape, tuple):
                    if len(op.input_shape) > 0 and len(op.output_shape) > 0:
                        if op.output_shape[0] > op.input_shape[0] * 10:
                            suggestions.append({
                                "type": "data_explosion",
                                "operation": op.operation_name,
                                "issue": f"Output size ({op.output_shape[0]}) is much larger than input ({op.input_shape[0]})",
                                "suggestion": "Review if this data expansion is necessary"
                            })
        
        return suggestions
