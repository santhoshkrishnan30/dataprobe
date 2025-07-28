## Project description

# DataProbe

**DataProbe** is a comprehensive Python toolkit for debugging, profiling, and optimizing data pipelines. It provides powerful tools to track data lineage, identify bottlenecks, monitor memory usage, and visualize pipeline execution flow with **enterprise-grade visualizations**.

## 🎨 **NEW: Enterprise-Grade Visualizations**

DataProbe v1.0.0 introduces professional-quality visualizations that rival commercial ETL monitoring tools like Airflow, Prefect, and Dagster.

### **Dashboard Features**

#### 🏢 **Enterprise Dashboard**

- **KPI Panels**: Real-time success rates, duration, memory usage
- **Pipeline Flowchart**: Interactive operation flow with status indicators
- **Performance Analytics**: Memory usage timelines with peak detection
- **Data Insights**: Comprehensive lineage and transformation tracking

```python
# Generate enterprise dashboard
debugger.visualize_pipeline()
```

#### 🌐 **3D Pipeline Network**

- **3D Visualization**: Interactive network showing operation relationships
- **Performance Mapping**: Z-axis represents operation duration
- **Status Color-coding**: Visual error and bottleneck identification

```python
# Create 3D network visualization
debugger.create_3d_pipeline_visualization()
```

#### 📊 **Executive Reports**

- **Multi-page Reports**: Professional stakeholder-ready documentation
- **Performance Trends**: Dual-axis charts showing duration and memory patterns
- **Optimization Recommendations**: AI-powered suggestions for improvements
- **Data Quality Metrics**: Comprehensive pipeline health scoring

```python
# Generate executive report
debugger.generate_executive_report()
```

### **Color-Coded Status System**

- 🟢 **Success**: Operations completed without issues
- 🟡 **Warning**: Performance bottlenecks detected
- 🔴 **Error**: Failed operations requiring attention
- 🟦 **Info**: Data flow and transformation indicators

## 🚀 Features

### PipelineDebugger

* **🔍 Operation Tracking** : Automatically track execution time, memory usage, and data shapes for each operation
* **📊 Enterprise-Grade Visualizations** : Professional dashboards, 3D networks, and executive reports
* **💾 Memory Profiling** : Monitor memory usage and identify memory-intensive operations
* **🔗 Data Lineage** : Track data transformations and column changes throughout the pipeline
* **⚠️ Bottleneck Detection** : Automatically identify slow operations and memory peaks
* **📈 Performance Reports** : Generate comprehensive debugging reports with optimization suggestions
* **🎯 Error Tracking** : Capture and track errors with full traceback information
* **🌳 Nested Operations** : Support for tracking nested function calls and their relationships

## 📦 Installation

```bash
pip install dataprobe
```

For development installation:

```bash
git clone https://github.com/santhoshkrishnan30/dataprobe.git
cd dataprobe
pip install -e ".[dev]"
```

## 🎯 Quick Start

### Basic Usage with Enhanced Visualizations

```python
from dataprobe import PipelineDebugger
import pandas as pd

# Initialize the debugger with enhanced features
debugger = PipelineDebugger(
    name="My_ETL_Pipeline",
    track_memory=True,
    track_lineage=True
)

# Use decorators to track operations
@debugger.track_operation("Load Data")
def load_data(file_path):
    return pd.read_csv(file_path)

@debugger.track_operation("Transform Data")
def transform_data(df):
    df['new_column'] = df['value'] * 2
    return df

# Run your pipeline
df = load_data("data.csv")
df = transform_data(df)

# Generate enterprise-grade visualizations
debugger.visualize_pipeline()              # Enterprise dashboard
debugger.create_3d_pipeline_visualization() # 3D network view  
debugger.generate_executive_report()       # Executive report

# Get AI-powered optimization suggestions
suggestions = debugger.suggest_optimizations()
for suggestion in suggestions:
    print(f"💡 {suggestion['suggestion']}")

# Print summary and reports
debugger.print_summary()
report = debugger.generate_report()
```

### Memory Profiling

```python
@debugger.profile_memory
def memory_intensive_operation():
    large_df = pd.DataFrame(np.random.randn(1000000, 50))
    result = large_df.groupby(large_df.index % 1000).mean()
    return result
```

### DataFrame Analysis

```python
# Analyze DataFrames for potential issues
debugger.analyze_dataframe(df, name="Sales Data")
```

## 📊 Example Output

### Enterprise Dashboard

Professional KPI dashboard with real-time metrics, pipeline flowchart, memory analytics, and performance insights.

### Pipeline Summary

```
Pipeline Summary: My_ETL_Pipeline
├── Execution Statistics
│   ├── Total Operations: 5
│   ├── Total Duration: 2.34s
│   └── Total Memory Used: 125.6MB
├── Bottlenecks (1)
│   └── Transform Data: 1.52s
└── Memory Peaks (1)
    └── Load Large Dataset: +85.3MB
```

### Optimization Suggestions

```
💡 OPTIMIZATION RECOMMENDATIONS:

1. [PERFORMANCE] Transform Data
   Issue: Operation took 1.52s
   💡 Consider optimizing this operation or parallelizing if possible

2. [MEMORY] Load Large Dataset  
   Issue: High memory usage: +85.3MB
   💡 Consider processing data in chunks or optimizing memory usage
```

## 🔧 Advanced Features

### Multiple Visualization Options

```python
# Enterprise dashboard - Professional KPI dashboard
debugger.visualize_pipeline()

# 3D network visualization - Interactive operation relationships  
debugger.create_3d_pipeline_visualization()

# Executive report - Multi-page stakeholder documentation
debugger.generate_executive_report()
```

### Data Lineage Tracking

```python
# Export data lineage information
lineage_json = debugger.export_lineage(format="json")

# Track column changes automatically
@debugger.track_operation("Add Features")
def add_features(df):
    df['feature_1'] = df['value'].rolling(7).mean()
    df['feature_2'] = df['value'].shift(1)
    return df
```

### Custom Metadata

```python
@debugger.track_operation("Process Batch", batch_id=123, source="api")
def process_batch(data):
    # Operation metadata is stored and included in reports
    return processed_data
```

### Checkpoint Saving

```python
# Auto-save is enabled by default
debugger = PipelineDebugger(name="Pipeline", auto_save=True)

# Manual checkpoint
debugger.save_checkpoint()
```

## 📈 Performance Tips

1. **Use with Context** : The debugger adds minimal overhead, but for production pipelines, you can disable tracking:

```python
   debugger = PipelineDebugger(name="Pipeline", track_memory=False, track_lineage=False)
```

2. **Batch Operations** : Group small operations together to reduce tracking overhead
3. **Memory Monitoring** : Set appropriate memory thresholds to catch issues early:

```python
   debugger = PipelineDebugger(name="Pipeline", memory_threshold_mb=500)
```

## 💼 **Enterprise Features**

✅ **Professional Styling**: Modern design matching enterprise standards
✅ **Executive Ready**: Suitable for stakeholder presentations
✅ **Performance Insights**: AI-powered optimization recommendations
✅ **Export Options**: High-resolution PNG outputs
✅ **Responsive Design**: Scales from detailed debugging to executive overview
✅ **Real-time Metrics**: Live performance and memory tracking

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/santhoshkrishnan30/dataprobe/blob/main/LICENSE) file for details.

## 🙏 Acknowledgments

* Built with [Rich](https://github.com/Textualize/rich) for beautiful terminal output
* Uses [NetworkX](https://networkx.org/) for pipeline visualization
* Enhanced with [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for enterprise-grade visualizations
* Inspired by the need for better data pipeline debugging tools

## 📞 Support

* 📧 Email: [santhoshkrishnan3006@gmail.com](mailto:santhoshkrishnan3006@gmail.com)
* 🐛 Issues: [GitHub Issues](https://github.com/santhoshkrishnan30/dataprobe/issues)
* 📖 Documentation: [Read the Docs](https://dataprobe.readthedocs.io/)



---

⭐ Star this repository if DataProbe helped you!

Made with ❤️ by Santhosh Krishnan R
