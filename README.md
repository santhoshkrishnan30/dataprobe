## Project description

# DataProbe

**DataProbe** is a comprehensive Python toolkit for debugging, profiling, and optimizing data pipelines. It provides powerful tools to track data lineage, identify bottlenecks, monitor memory usage, and visualize pipeline execution flow.

## 🚀 Features

### PipelineDebugger

* **🔍 Operation Tracking** : Automatically track execution time, memory usage, and data shapes for each operation
* **📊 Visual Pipeline Flow** : Generate interactive visualizations of your pipeline execution
* **💾 Memory Profiling** : Monitor memory usage and identify memory-intensive operations
* **🔗 Data Lineage** : Track data transformations and column changes throughout the pipeline
* **⚠️ Bottleneck Detection** : Automatically identify slow operations and memory peaks
* **📈 Performance Reports** : Generate comprehensive debugging reports with optimization suggestions
* **🎯 Error Tracking** : Capture and track errors with full traceback information
* **🌳 Nested Operations** : Support for tracking nested function calls and their relationships

## 📦 Installation

```
pip install dataprobe
```

For development installation:

```
git clone https://github.com/santhoshkrishnan30/dataprobe.git
cd dataprobe
pip install -e ".[dev]"
```

## 🎯 Quick Start

### Basic Usage

```
from dataprobe import PipelineDebugger
import pandas as pd

# Initialize the debugger
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

# Generate reports and visualizations
debugger.print_summary()
debugger.visualize_pipeline()
report = debugger.generate_report()
```

### Memory Profiling

```
@debugger.profile_memory
def memory_intensive_operation():
    large_df = pd.DataFrame(np.random.randn(1000000, 50))
    result = large_df.groupby(large_df.index % 1000).mean()
    return result
```

### DataFrame Analysis

```
# Analyze DataFrames for potential issues
debugger.analyze_dataframe(df, name="Sales Data")
```

## 📊 Example Output

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
- [PERFORMANCE] Transform Data: Operation took 1.52s
  Suggestion: Consider optimizing this operation or parallelizing if possible

- [MEMORY] Load Large Dataset: High memory usage: +85.3MB
  Suggestion: Consider processing data in chunks or optimizing memory usage
```

## 🔧 Advanced Features

### Data Lineage Tracking

```
# Export data lineage information
lineage_json = debugger.export_lineage(format="json")

# Track column changes automatically
@debugger.track_operation("Add Features")
defadd_features(df):
    df['feature_1'] = df['value'].rolling(7).mean()
    df['feature_2'] = df['value'].shift(1)
    return df
```

### Custom Metadata

```
@debugger.track_operation("Process Batch", batch_id=123, source="api")
defprocess_batch(data):
    # Operation metadata is stored and included in reports
    return processed_data
```

### Checkpoint Saving

```
# Auto-save is enabled by default
debugger = PipelineDebugger(name="Pipeline", auto_save=True)

# Manual checkpoint
debugger.save_checkpoint()
```

## 📈 Performance Tips

1. **Use with Context** : The debugger adds minimal overhead, but for production pipelines, you can disable tracking:

```
   debugger = PipelineDebugger(name="Pipeline", track_memory=False, track_lineage=False)
```

1. **Batch Operations** : Group small operations together to reduce tracking overhead
2. **Memory Monitoring** : Set appropriate memory thresholds to catch issues early:

```
   debugger = PipelineDebugger(name="Pipeline", memory_threshold_mb=500)
```

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
* Inspired by the need for better data pipeline debugging tools

## 📞 Support

* 📧 Email: [santhoshkrishnan3006@gmail.com](mailto:santhoshkrishnan3006@gmail.com)
* 🐛 Issues: [GitHub Issues](https://github.com/santhoshkrishnan30/dataprobe/issues)
* 📖 Documentation: [Read the Docs](https://dataprobe.readthedocs.io/)

## 🗺️ Roadmap

* [ ] Support for distributed pipeline debugging
* [ ] Integration with popular orchestration tools (Airflow, Prefect, Dagster)
* [ ] Real-time pipeline monitoring dashboard
* [ ] Advanced anomaly detection in data flow
* [ ] Support for streaming data pipelines

---

Made with ❤️ by Santhosh Krishnan R
