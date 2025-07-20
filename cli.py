# dataalchemy/cli.py

"""
Command Line Interface for DataAlchemy
"""

import click
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import print as rprint
import pickle

console = Console()

@click.group()
@click.version_option(version="0.1.0", prog_name="dataalchemy")
def main():
    """DataAlchemy - Advanced Data Pipeline Debugging Tools"""
    pass

@main.command()
@click.argument('checkpoint_file', type=click.Path(exists=True))
@click.option('--format', '-f', type=click.Choice(['summary', 'detailed', 'json']), 
              default='summary', help='Output format')
def analyze(checkpoint_file, format):
    """Analyze a saved pipeline checkpoint."""
    try:
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
        
        if format == 'json':
            click.echo(json.dumps({
                "pipeline_name": checkpoint['pipeline_name'],
                "total_operations": len(checkpoint['operations']),
                "timestamp": checkpoint['timestamp']
            }, indent=2))
        elif format == 'summary':
            console.print(f"\n[bold cyan]Pipeline: {checkpoint['pipeline_name']}[/bold cyan]")
            console.print(f"Timestamp: {checkpoint['timestamp']}")
            console.print(f"Total Operations: {len(checkpoint['operations'])}")
            console.print(f"Bottlenecks: {len(checkpoint['bottlenecks'])}")
        else:  # detailed
            table = Table(title=f"Pipeline Analysis: {checkpoint['pipeline_name']}")
            table.add_column("Operation", style="cyan")
            table.add_column("Duration (s)", style="green")
            table.add_column("Memory (MB)", style="yellow")
            table.add_column("Status", style="red")
            
            for op_id in checkpoint['operation_order']:
                op = checkpoint['operations'][op_id]
                status = "❌ Error" if op.error else "✅ Success"
                table.add_row(
                    op.operation_name,
                    f"{op.duration:.3f}",
                    f"{op.memory_delta:+.1f}",
                    status
                )
            
            console.print(table)
    
    except Exception as e:
        console.print(f"[red]Error reading checkpoint: {e}[/red]")
        raise click.Abort()

@main.command()
@click.argument('checkpoint_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file for lineage data')
def export_lineage(checkpoint_file, output):
    """Export data lineage from a checkpoint."""
    try:
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
        
        lineage_data = {
            "pipeline": checkpoint['pipeline_name'],
            "lineages": {}
        }
        
        for data_id, lineage in checkpoint.get('data_lineages', {}).items():
            lineage_data["lineages"][data_id] = {
                "source": lineage.source,
                "data_type": lineage.data_type,
                "transformations": lineage.transformations
            }
        
        if output:
            with open(output, 'w') as f:
                json.dump(lineage_data, f, indent=2)
            console.print(f"[green]Lineage data exported to {output}[/green]")
        else:
            console.print(json.dumps(lineage_data, indent=2))
    
    except Exception as e:
        console.print(f"[red]Error exporting lineage: {e}[/red]")
        raise click.Abort()

@main.command()
def info():
    """Display information about DataAlchemy."""
    console.print("\n[bold cyan]DataAlchemy[/bold cyan] - Advanced Data Pipeline Debugging Tools\n")
    console.print("Version: 0.1.0")
    console.print("Author: Your Name")
    console.print("License: MIT\n")
    
    console.print("[yellow]Features:[/yellow]")
    features = [
        "🔍 Operation tracking and profiling",
        "📊 Visual pipeline flow generation",
        "💾 Memory usage monitoring",
        "🔗 Data lineage tracking",
        "⚠️ Bottleneck detection",
        "📈 Performance optimization suggestions"
    ]
    for feature in features:
        console.print(f"  • {feature}")
    
    console.print("\n[green]Get started:[/green]")
    console.print("  from dataalchemy import PipelineDebugger")
    console.print("  debugger = PipelineDebugger(name='My_Pipeline')")
    console.print("\n[blue]Documentation:[/blue] https://dataalchemy.readthedocs.io")

if __name__ == "__main__":
    main()