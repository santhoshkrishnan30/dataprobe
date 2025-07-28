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
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Arrow, Ellipse
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
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation

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
        
        # Enhanced color schemes for professional visualization
        self.colors = {
            'primary': '#1E3A8A',      # Deep blue
            'secondary': '#10B981',    # Emerald green
            'accent': '#F59E0B',       # Amber
            'danger': '#EF4444',       # Red
            'warning': '#F97316',      # Orange
            'info': '#3B82F6',         # Blue
            'success': '#22C55E',      # Green
            'background': '#F8FAFC',   # Light gray
            'dark_bg': '#0F172A',      # Dark blue
            'text_primary': '#1F2937', # Dark gray
            'text_secondary': '#6B7280', # Medium gray
            'border': '#E5E7EB'        # Light border
        }
        
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
        Create an enterprise-grade, professional dashboard visualization of the pipeline execution.
        This creates a comprehensive visual report that rivals commercial ETL monitoring tools.
        """
        # Set professional style
        plt.style.use('default')
        
        # Create the main figure with professional layout
        fig = plt.figure(figsize=(24, 16))
        fig.patch.set_facecolor('#FAFBFC')
        
        # Create sophisticated grid layout
        gs = GridSpec(4, 6, height_ratios=[0.8, 2.5, 1.5, 1.2], width_ratios=[1, 1, 1, 1, 1, 1],
                     hspace=0.25, wspace=0.15, left=0.03, right=0.97, top=0.93, bottom=0.05)
        
        # ====================== HEADER SECTION ======================
        self._create_header(fig, gs)
        
        # ====================== MAIN DASHBOARD PANELS ======================
        self._create_kpi_dashboard(fig, gs)
        self._create_pipeline_flowchart(fig, gs)
        self._create_performance_analytics(fig, gs)
        self._create_data_insights_panel(fig, gs)
        
        # ====================== SAVE AND DISPLAY ======================
        save_file = save_path or (self.save_path / "enterprise_pipeline_dashboard.png")
        plt.savefig(save_file, dpi=300, bbox_inches='tight', facecolor='#FAFBFC', 
                   edgecolor='none', pad_inches=0.2)
        
        plt.show()
        console.print(f"[green]✓ Enterprise pipeline dashboard saved to: {save_file}[/green]")
        return str(save_file)

    def _create_header(self, fig, gs):
        """Create professional header with branding and key metrics"""
        ax_header = fig.add_subplot(gs[0, :])
        ax_header.axis('off')
        
        # Main title with professional styling
        title_text = f"DataProbe Enterprise Analytics Dashboard"
        subtitle_text = f"Pipeline: {self.name} | Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}"
        
        ax_header.text(0.02, 0.7, title_text, fontsize=26, fontweight='bold', 
                      color=self.colors['primary'], transform=ax_header.transAxes)
        ax_header.text(0.02, 0.3, subtitle_text, fontsize=14, color=self.colors['text_secondary'], 
                      transform=ax_header.transAxes)
        
        # Status indicator
        status_color = self.colors['danger'] if any(op.error for op in self.operations.values()) else self.colors['success']
        status_text = "ISSUES DETECTED" if any(op.error for op in self.operations.values()) else "HEALTHY"
        
        # Status badge
        bbox = dict(boxstyle="round,pad=0.3", facecolor=status_color, alpha=0.2, edgecolor=status_color, linewidth=2)
        ax_header.text(0.85, 0.5, f"STATUS: {status_text}", fontsize=12, fontweight='bold',
                      color=status_color, transform=ax_header.transAxes, 
                      bbox=bbox, ha='center', va='center')

    def _create_kpi_dashboard(self, fig, gs):
        """Create KPI dashboard with key performance indicators"""
        # Calculate KPIs
        total_ops = len(self.operations)
        successful_ops = sum(1 for op in self.operations.values() if not op.error)
        failed_ops = total_ops - successful_ops
        total_duration = sum(op.duration for op in self.operations.values())
        total_memory = sum(op.memory_delta for op in self.operations.values())
        avg_duration = total_duration / max(total_ops, 1)
        
        success_rate = (successful_ops / max(total_ops, 1)) * 100
        
        # KPI data
        kpis = [
            ("Total Operations", f"{total_ops}", self.colors['info'], "📊"),
            ("Success Rate", f"{success_rate:.1f}%", self.colors['success'] if success_rate >= 95 else self.colors['warning'], "✅"),
            ("Total Duration", f"{total_duration:.2f}s", self.colors['primary'], "⏱️"),
            ("Memory Impact", f"{total_memory:+.1f}MB", self.colors['accent'], "💾"),
            ("Avg. Op. Time", f"{avg_duration:.3f}s", self.colors['secondary'], "📈"),
            ("Bottlenecks", f"{len(self.bottlenecks)}", self.colors['danger'] if self.bottlenecks else self.colors['success'], "🚨")
        ]
        
        # Create KPI panels
        for i, (label, value, color, icon) in enumerate(kpis):
            ax_kpi = fig.add_subplot(gs[1, i])
            ax_kpi.axis('off')
            
            # KPI box with modern design
            box = FancyBboxPatch((0.05, 0.1), 0.9, 0.8, boxstyle="round,pad=0.02",
                               facecolor=color, alpha=0.1, edgecolor=color, linewidth=2)
            ax_kpi.add_patch(box)
            
            # Icon and value
            ax_kpi.text(0.5, 0.75, icon, ha='center', va='center', fontsize=24, 
                       transform=ax_kpi.transAxes)
            ax_kpi.text(0.5, 0.45, value, ha='center', va='center', fontsize=20, 
                       fontweight='bold', color=color, transform=ax_kpi.transAxes)
            ax_kpi.text(0.5, 0.2, label, ha='center', va='center', fontsize=10, 
                       color=self.colors['text_primary'], transform=ax_kpi.transAxes, wrap=True)

    def _create_pipeline_flowchart(self, fig, gs):
        """Create sophisticated pipeline flowchart"""
        ax_flow = fig.add_subplot(gs[2, :4])
        ax_flow.set_xlim(0, 100)
        ax_flow.set_ylim(0, 100)
        ax_flow.axis('off')
        
        # Title
        ax_flow.text(50, 95, 'Pipeline Execution Flow', ha='center', va='center', 
                    fontsize=16, fontweight='bold', color=self.colors['primary'])
        
        if not self.operations:
            ax_flow.text(50, 50, 'No operations recorded', ha='center', va='center', 
                        fontsize=14, color=self.colors['text_secondary'])
            return
        
        # Calculate positions for operations
        sorted_ops = sorted(self.operations.items(), key=lambda x: x[1].start_time)
        n_ops = len(sorted_ops)
        
        # Create network graph for better positioning
        G = nx.DiGraph()
        positions = {}
        
        # Add nodes and calculate positions
        for i, (op_id, op) in enumerate(sorted_ops):
            x = 10 + (i * 80 / max(n_ops - 1, 1))
            
            # Vary Y position based on performance characteristics
            if op.error:
                y = 25  # Errors at bottom
            elif op_id in self.bottlenecks:
                y = 45  # Bottlenecks in middle
            else:
                y = 65  # Normal operations on top
                
            # Add some randomness to avoid overlap
            y += np.sin(i * 0.7) * 8
            positions[op_id] = (x, y)
            
            G.add_node(op_id, pos=(x, y))
            
            # Add edges
            if i > 0:
                prev_op_id = sorted_ops[i-1][0]
                G.add_edge(prev_op_id, op_id)
        
        # Draw connections first
        for edge in G.edges():
            start_pos = positions[edge[0]]
            end_pos = positions[edge[1]]
            
            # Create curved arrow
            mid_x = (start_pos[0] + end_pos[0]) / 2
            mid_y = max(start_pos[1], end_pos[1]) + 5
            
            # Draw curved line
            x_vals = [start_pos[0] + 8, mid_x, end_pos[0] - 8]
            y_vals = [start_pos[1], mid_y, end_pos[1]]
            
            ax_flow.plot(x_vals, y_vals, color=self.colors['info'], linewidth=2, alpha=0.7)
            
            # Arrow head
            ax_flow.annotate('', xy=(end_pos[0] - 8, end_pos[1]), 
                           xytext=(end_pos[0] - 12, end_pos[1]),
                           arrowprops=dict(arrowstyle='->', lw=2, color=self.colors['info']))
        
        # Draw operation nodes
        for op_id, (x, y) in positions.items():
            op = self.operations[op_id]
            
            # Determine node style
            if op.error:
                node_color = self.colors['danger']
                edge_color = '#B91C1C'
                icon = '❌'
            elif op_id in self.bottlenecks:
                node_color = self.colors['warning']
                edge_color = '#D97706'
                icon = '⚠️'
            else:
                node_color = self.colors['success']
                edge_color = '#059669'
                icon = '✅'
            
            # Main node circle
            circle = Circle((x, y), 7, facecolor=node_color, edgecolor=edge_color, 
                          linewidth=2, alpha=0.9, zorder=3)
            ax_flow.add_patch(circle)
            
            # Operation name (shortened)
            op_name = op.operation_name
            if len(op_name) > 12:
                op_name = op_name[:12] + "..."
            
            ax_flow.text(x, y + 12, op_name, ha='center', va='center', fontsize=8, 
                        fontweight='bold', color=self.colors['text_primary'])
            
            # Performance metrics below
            ax_flow.text(x, y - 12, f"{op.duration:.3f}s", ha='center', va='center', 
                        fontsize=7, color=self.colors['text_secondary'])
            
            # Status icon
            ax_flow.text(x, y, icon, ha='center', va='center', fontsize=10, zorder=4)

    def _create_performance_analytics(self, fig, gs):
        """Create performance analytics section"""
        # Memory usage timeline
        ax_memory = fig.add_subplot(gs[2, 4:])
        
        if self.operations:
            sorted_ops = sorted(self.operations.items(), key=lambda x: x[1].start_time)
            
            # Calculate cumulative memory
            memory_timeline = []
            cumulative_memory = 0
            labels = []
            
            for i, (op_id, op) in enumerate(sorted_ops):
                cumulative_memory += op.memory_delta
                memory_timeline.append(cumulative_memory)
                labels.append(op.operation_name[:8] + "..." if len(op.operation_name) > 8 else op.operation_name)
            
            x_pos = range(len(memory_timeline))
            
            # Create gradient fill
            ax_memory.fill_between(x_pos, 0, memory_timeline, alpha=0.3, color=self.colors['accent'])
            ax_memory.plot(x_pos, memory_timeline, color=self.colors['accent'], linewidth=3, 
                          marker='o', markersize=6, markerfacecolor='white', markeredgecolor=self.colors['accent'], 
                          markeredgewidth=2)
            
            # Highlight memory peaks
            for i, (op_id, memory_delta) in enumerate(self.memory_peaks):
                if op_id in dict(sorted_ops):
                    idx = [id for id, _ in sorted_ops].index(op_id)
                    ax_memory.scatter(idx, memory_timeline[idx], color=self.colors['danger'], 
                                    s=100, zorder=5, edgecolor='white', linewidth=2)
                    ax_memory.annotate(f'+{memory_delta:.1f}MB', 
                                     xy=(idx, memory_timeline[idx]),
                                     xytext=(5, 10), textcoords='offset points',
                                     fontsize=8, color=self.colors['danger'],
                                     bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                                             edgecolor=self.colors['danger'], alpha=0.8))
            
            ax_memory.set_title('Memory Usage Timeline', fontsize=12, fontweight='bold', 
                              color=self.colors['primary'], pad=10)
            ax_memory.set_xlabel('Operations', fontsize=10, color=self.colors['text_primary'])
            ax_memory.set_ylabel('Cumulative Memory (MB)', fontsize=10, color=self.colors['text_primary'])
            ax_memory.grid(True, alpha=0.3, linestyle='--')
            ax_memory.set_facecolor('#FAFBFC')
            
            # Style the plot
            ax_memory.spines['top'].set_visible(False)
            ax_memory.spines['right'].set_visible(False)
            ax_memory.spines['left'].set_color(self.colors['border'])
            ax_memory.spines['bottom'].set_color(self.colors['border'])

    def _create_data_insights_panel(self, fig, gs):
        """Create data insights and lineage panel"""
        # Data lineage summary
        ax_lineage = fig.add_subplot(gs[3, :3])
        ax_lineage.axis('off')
        
        # Background panel
        panel_bg = FancyBboxPatch((0.02, 0.1), 0.96, 0.8, boxstyle="round,pad=0.02",
                                facecolor=self.colors['info'], alpha=0.05, 
                                edgecolor=self.colors['info'], linewidth=1)
        ax_lineage.add_patch(panel_bg)
        
        ax_lineage.text(0.05, 0.8, 'Data Lineage & Insights', fontsize=14, fontweight='bold',
                       color=self.colors['primary'], transform=ax_lineage.transAxes)
        
        # Lineage statistics
        lineage_count = len(self.data_lineages)
        transform_count = sum(len(l.transformations) for l in self.data_lineages.values())
        
        insights = [
            f"📊 {lineage_count} data objects tracked",
            f"🔄 {transform_count} transformations recorded",
            f"⚡ {len(self.operation_order)} operations executed",
            f"🎯 {len([op for op in self.operations.values() if not op.error])} successful operations"
        ]
        
        for i, insight in enumerate(insights):
            ax_lineage.text(0.05, 0.6 - i*0.12, insight, fontsize=11, 
                           color=self.colors['text_primary'], transform=ax_lineage.transAxes)
        
        # Performance insights panel
        ax_insights = fig.add_subplot(gs[3, 3:])
        ax_insights.axis('off')
        
        # Background panel
        panel_bg2 = FancyBboxPatch((0.02, 0.1), 0.96, 0.8, boxstyle="round,pad=0.02",
                                 facecolor=self.colors['success'], alpha=0.05, 
                                 edgecolor=self.colors['success'], linewidth=1)
        ax_insights.add_patch(panel_bg2)
        
        ax_insights.text(0.05, 0.8, 'Performance Insights', fontsize=14, fontweight='bold',
                        color=self.colors['primary'], transform=ax_insights.transAxes)
        
        # Generate insights
        total_duration = sum(op.duration for op in self.operations.values())
        avg_duration = total_duration / max(len(self.operations), 1)
        
        perf_insights = []
        
        if self.bottlenecks:
            slowest_op = max(self.operations.values(), key=lambda x: x.duration)
            perf_insights.append(f"🐌 Slowest: {slowest_op.operation_name} ({slowest_op.duration:.2f}s)")
        else:
            perf_insights.append("🚀 No bottlenecks detected")
        
        if self.memory_peaks:
            highest_mem = max(self.memory_peaks, key=lambda x: x[1])
            op_name = self.operations[highest_mem[0]].operation_name
            perf_insights.append(f"💾 Peak memory: {op_name} (+{highest_mem[1]:.1f}MB)")
        else:
            perf_insights.append("✅ Memory usage within limits")
        
        perf_insights.append(f"⏱️ Average operation time: {avg_duration:.3f}s")
        
        error_count = sum(1 for op in self.operations.values() if op.error)
        if error_count > 0:
            perf_insights.append(f"❌ {error_count} operations failed")
        else:
            perf_insights.append("✅ All operations completed successfully")
        
        for i, insight in enumerate(perf_insights):
            ax_insights.text(0.05, 0.6 - i*0.12, insight, fontsize=11, 
                           color=self.colors['text_primary'], transform=ax_insights.transAxes)

    def create_3d_pipeline_visualization(self, save_path: Optional[Path] = None):
        """Create an advanced 3D visualization of the pipeline network"""
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set background
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(False)
        
        if not self.operations:
            ax.text(0.5, 0.5, 0.5, 'No operations to visualize', 
                   transform=ax.transAxes, fontsize=16, ha='center')
            return
        
        # Create network graph
        G = nx.DiGraph()
        sorted_ops = sorted(self.operations.items(), key=lambda x: x[1].start_time)
        
        # Add nodes and edges
        for i, (op_id, op) in enumerate(sorted_ops):
            G.add_node(op_id, operation=op)
            if i > 0:
                prev_op_id = sorted_ops[i-1][0]
                G.add_edge(prev_op_id, op_id)
        
        # Calculate 3D positions
        pos_2d = nx.spring_layout(G, k=3, iterations=50)
        pos_3d = {}
        
        for i, (op_id, op) in enumerate(sorted_ops):
            if op_id in pos_2d:
                x, y = pos_2d[op_id]
                # Z-axis represents performance metrics
                z = op.duration * 10  # Scale duration for visibility
                pos_3d[op_id] = (x * 10, y * 10, z)
        
        # Draw nodes
        for op_id, (x, y, z) in pos_3d.items():
            op = self.operations[op_id]
            
            # Determine color and size based on status
            if op.error:
                color = self.colors['danger']
                size = 200
            elif op_id in self.bottlenecks:
                color = self.colors['warning']
                size = 150
            else:
                color = self.colors['success']
                size = 100
            
            # Memory usage affects alpha
            alpha = min(1.0, 0.3 + abs(op.memory_delta) / 50)
            
            ax.scatter(x, y, z, c=color, s=size, alpha=alpha, edgecolors='white', linewidth=2)
            
            # Add labels
            ax.text(x, y, z + 0.5, op.operation_name[:10], fontsize=8, ha='center')
        
        # Draw edges
        for edge in G.edges():
            if edge[0] in pos_3d and edge[1] in pos_3d:
                x1, y1, z1 = pos_3d[edge[0]]
                x2, y2, z2 = pos_3d[edge[1]]
                ax.plot([x1, x2], [y1, y2], [z1, z2], 
                       color=self.colors['info'], linewidth=2, alpha=0.6)
        
        # Styling
        ax.set_xlabel('Network Flow →', fontsize=12, color=self.colors['primary'])
        ax.set_ylabel('Complexity →', fontsize=12, color=self.colors['primary'])
        ax.set_zlabel('Performance (Duration) →', fontsize=12, color=self.colors['primary'])
        ax.set_title(f'3D Pipeline Network: {self.name}', fontsize=16, 
                    color=self.colors['primary'], fontweight='bold', pad=20)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['success'], 
                      markersize=10, label='Successful Operation'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['warning'], 
                      markersize=10, label='Performance Bottleneck'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['danger'], 
                      markersize=10, label='Failed Operation')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
        
        # Save
        save_file = save_path or (self.save_path / "pipeline_3d_network.png")
        plt.savefig(save_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        console.print(f"[green]✓ 3D pipeline network saved to: {save_file}[/green]")
        return str(save_file)

    def generate_executive_report(self, save_path: Optional[Path] = None):
        """Generate an executive-level visual report"""
        # Create multi-page report
        fig = plt.figure(figsize=(16, 20))
        fig.patch.set_facecolor('white')
        
        # Create sections
        gs = GridSpec(6, 2, height_ratios=[0.5, 1.5, 1.5, 1.5, 1.5, 0.5], 
                     hspace=0.3, wspace=0.2)
        
        # Header
        ax_header = fig.add_subplot(gs[0, :])
        ax_header.axis('off')
        
        # Executive header
        header_bg = FancyBboxPatch((0.02, 0.1), 0.96, 0.8, boxstyle="round,pad=0.02",
                                 facecolor=self.colors['primary'], alpha=0.1, 
                                 edgecolor=self.colors['primary'], linewidth=2)
        ax_header.add_patch(header_bg)
        
        ax_header.text(0.5, 0.6, f'Executive Pipeline Report: {self.name}', 
                      ha='center', va='center', fontsize=24, fontweight='bold', 
                      color=self.colors['primary'], transform=ax_header.transAxes)
        ax_header.text(0.5, 0.3, f'Generated on {datetime.now().strftime("%B %d, %Y")}', 
                      ha='center', va='center', fontsize=12, 
                      color=self.colors['text_secondary'], transform=ax_header.transAxes)
        
        # Executive Summary Section
        ax_summary = fig.add_subplot(gs[1, :])
        ax_summary.axis('off')
        
        # Calculate executive metrics
        total_ops = len(self.operations)
        success_rate = (sum(1 for op in self.operations.values() if not op.error) / max(total_ops, 1)) * 100
        total_duration = sum(op.duration for op in self.operations.values())
        total_memory = sum(op.memory_delta for op in self.operations.values())
        
        # Executive summary text
        summary_text = f"""
EXECUTIVE SUMMARY

Pipeline Execution Status: {'SUCCESSFUL' if success_rate == 100 else 'ISSUES DETECTED'}
Total Operations Processed: {total_ops}
Success Rate: {success_rate:.1f}%
Total Processing Time: {total_duration:.2f} seconds
Memory Impact: {total_memory:+.1f} MB

KEY FINDINGS:
{'• All operations completed successfully without errors' if success_rate == 100 else f'• {total_ops - int(total_ops * success_rate / 100)} operations encountered errors'}
{'• No performance bottlenecks detected' if not self.bottlenecks else f'• {len(self.bottlenecks)} performance bottlenecks identified'}
{'• Memory usage within acceptable limits' if not self.memory_peaks else f'• {len(self.memory_peaks)} memory usage spikes detected'}
• Average operation duration: {total_duration/max(total_ops, 1):.3f} seconds
        """
        
        ax_summary.text(0.05, 0.95, summary_text, fontsize=12, color=self.colors['text_primary'],
                       transform=ax_summary.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['background'], 
                               alpha=0.8, edgecolor=self.colors['border']))
        
        # Performance Trends
        ax_trends = fig.add_subplot(gs[2, :])
        
        if self.operations:
            sorted_ops = sorted(self.operations.items(), key=lambda x: x[1].start_time)
            
            # Create performance trend chart
            durations = [op.duration for _, op in sorted_ops]
            memory_usage = [op.memory_delta for _, op in sorted_ops]
            x_pos = range(len(sorted_ops))
            
            # Dual-axis chart
            ax_trends2 = ax_trends.twinx()
            
            # Duration trend
            line1 = ax_trends.plot(x_pos, durations, color=self.colors['primary'], 
                                 linewidth=3, marker='o', markersize=6, label='Duration (s)')
            ax_trends.fill_between(x_pos, durations, alpha=0.3, color=self.colors['primary'])
            
            # Memory trend
            line2 = ax_trends2.plot(x_pos, memory_usage, color=self.colors['accent'], 
                                  linewidth=3, marker='s', markersize=6, label='Memory Delta (MB)')
            ax_trends2.fill_between(x_pos, memory_usage, alpha=0.3, color=self.colors['accent'])
            
            # Styling
            ax_trends.set_xlabel('Operation Sequence', fontsize=12, color=self.colors['text_primary'])
            ax_trends.set_ylabel('Duration (seconds)', fontsize=12, color=self.colors['primary'])
            ax_trends2.set_ylabel('Memory Delta (MB)', fontsize=12, color=self.colors['accent'])
            ax_trends.set_title('Performance Trends Analysis', fontsize=14, fontweight='bold', 
                              color=self.colors['primary'], pad=15)
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax_trends.legend(lines, labels, loc='upper left')
            
            ax_trends.grid(True, alpha=0.3)
            ax_trends.spines['top'].set_visible(False)
            ax_trends2.spines['top'].set_visible(False)
        
        # Error Analysis and Recommendations
        ax_errors = fig.add_subplot(gs[3, 0])
        ax_recommendations = fig.add_subplot(gs[3, 1])
        
        # Error Analysis
        ax_errors.axis('off')
        ax_errors.text(0.05, 0.9, 'Error Analysis', fontsize=14, fontweight='bold',
                      color=self.colors['danger'], transform=ax_errors.transAxes)
        
        errors = [op for op in self.operations.values() if op.error]
        if errors:
            error_text = f"Total Errors: {len(errors)}\n\n"
            for i, op in enumerate(errors[:3]):  # Show top 3 errors
                error_text += f"{i+1}. {op.operation_name}\n   Error: {op.error[:50]}...\n\n"
        else:
            error_text = "✅ No errors detected\nAll operations completed successfully"
        
        ax_errors.text(0.05, 0.75, error_text, fontsize=10, color=self.colors['text_primary'],
                      transform=ax_errors.transAxes, verticalalignment='top',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor='#FEF2F2', 
                              alpha=0.8, edgecolor=self.colors['danger']))
        
        # Recommendations
        ax_recommendations.axis('off')
        ax_recommendations.text(0.05, 0.9, 'Recommendations', fontsize=14, fontweight='bold',
                               color=self.colors['success'], transform=ax_recommendations.transAxes)
        
        recommendations = self.suggest_optimizations()
        if recommendations:
            rec_text = "Priority Actions:\n\n"
            for i, rec in enumerate(recommendations[:3]):
                rec_text += f"{i+1}. {rec['type'].upper()}: {rec['operation']}\n"
                rec_text += f"   {rec['suggestion']}\n\n"
        else:
            rec_text = "✅ Pipeline performing optimally\nNo immediate actions required"
        
        ax_recommendations.text(0.05, 0.75, rec_text, fontsize=10, color=self.colors['text_primary'],
                               transform=ax_recommendations.transAxes, verticalalignment='top',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='#F0FDF4', 
                                       alpha=0.8, edgecolor=self.colors['success']))
        
        # Data Quality Metrics
        ax_quality = fig.add_subplot(gs[4, :])
        ax_quality.axis('off')
        
        ax_quality.text(0.05, 0.9, 'Data Quality & Lineage Summary', fontsize=14, fontweight='bold',
                       color=self.colors['info'], transform=ax_quality.transAxes)
        
        # Data quality metrics
        lineage_count = len(self.data_lineages)
        transform_count = sum(len(l.transformations) for l in self.data_lineages.values())
        
        quality_metrics = [
            f"📊 Data Objects Tracked: {lineage_count}",
            f"🔄 Transformations Applied: {transform_count}",
            f"📈 Operations with Shape Changes: {sum(1 for l in self.data_lineages.values() if l.column_changes)}",
            f"⚡ Pipeline Complexity Score: {len(self.operation_order) * (1 + len(self.bottlenecks))}",
            f"🎯 Data Processing Efficiency: {(success_rate/100) * (1/(max(total_duration, 1)/10)):.2f}"
        ]
        
        quality_text = "\n".join(quality_metrics)
        ax_quality.text(0.05, 0.7, quality_text, fontsize=12, color=self.colors['text_primary'],
                       transform=ax_quality.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='#F0F9FF', 
                               alpha=0.8, edgecolor=self.colors['info']))
        
        # Footer
        ax_footer = fig.add_subplot(gs[5, :])
        ax_footer.axis('off')
        
        footer_text = f"Generated by DataProbe v2.0 | Pipeline Analytics & Debugging Tool | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ax_footer.text(0.5, 0.5, footer_text, ha='center', va='center', fontsize=10,
                      color=self.colors['text_secondary'], transform=ax_footer.transAxes,
                      style='italic')
        
        # Save
        save_file = save_path or (self.save_path / "executive_pipeline_report.png")
        plt.savefig(save_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        console.print(f"[green]✓ Executive report saved to: {save_file}[/green]")
        return str(save_file)

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

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive pipeline report"""
        total_duration = sum(op.duration for op in self.operations.values())
        total_memory = sum(op.memory_delta for op in self.operations.values())
        
        # Find bottlenecks
        bottlenecks = [
            op_name for op_name, op_data in self.operations.items()
            if op_data.duration > total_duration * 0.3
        ]
        
        return {
            'pipeline_name': self.name,
            'total_operations': len(self.operations),
            'total_duration': total_duration,
            'total_memory_used': total_memory,
            'bottlenecks': len(bottlenecks),
            'errors': len([op for op in self.operations.values() if op.error]),
            'success_rate': (len(self.operations) - len([op for op in self.operations.values() if op.error])) / max(len(self.operations), 1),
            'operations_detail': {op_id: {
                'name': op.operation_name,
                'duration': op.duration,
                'memory_delta': op.memory_delta,
                'status': 'error' if op.error else 'success'
            } for op_id, op in self.operations.items()},
            'error_detail': {op.operation_name: op.error for op in self.operations.values() if op.error}
        }