"""
Tests for PipelineDebugger
"""

import pytest
import pandas as pd
import numpy as np
import time
from pathlib import Path
import tempfile
import shutil
from dataprobe.debugger import PipelineDebugger, OperationMetrics, DataLineage
from dataprobe.utils import get_dataframe_info, detect_dataframe_changes

class TestPipelineDebugger:
    """Test cases for PipelineDebugger."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'id': range(100),
            'value': np.random.randn(100),
            'category': np.random.choice(['A', 'B', 'C'], 100)
        })
    
    @pytest.fixture
    def debugger(self, temp_dir):
        """Create a PipelineDebugger instance for testing."""
        return PipelineDebugger(
            name="TestPipeline",
            save_path=temp_dir,
            auto_save=False
        )
    
    def test_initialization(self, debugger):
        """Test debugger initialization."""
        assert debugger.name == "TestPipeline"
        assert debugger.track_memory == True
        assert debugger.track_lineage == True
        assert len(debugger.operations) == 0
    
    def test_track_operation_decorator(self, debugger, sample_df):
        """Test operation tracking decorator."""
        
        @debugger.track_operation("Test Operation")
        def process_data(df):
            time.sleep(0.1)  # Simulate processing
            return df.copy()
        
        result = process_data(sample_df)
        
        # Check operation was tracked
        assert len(debugger.operations) == 1
        op_id = debugger.operation_order[0]
        op = debugger.operations[op_id]
        
        assert op.operation_name == "Test Operation"
        assert op.duration > 0.1
        assert op.input_shape == (100, 3)
        assert op.output_shape == (100, 3)
        assert op.error is None
    
    def test_error_tracking(self, debugger):
        """Test error tracking in operations."""
        
        @debugger.track_operation("Failing Operation")
        def failing_operation():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_operation()
        
        # Check error was tracked
        assert len(debugger.operations) == 1
        op = list(debugger.operations.values())[0]
        assert op.error == "Test error"
        assert op.traceback is not None
    
    def test_memory_tracking(self, debugger):
        """Test memory usage tracking."""
        
        @debugger.track_operation("Memory Test")
        def memory_operation():
            # Allocate some memory
            data = np.random.randn(1000000)
            return data
        
        result = memory_operation()
        
        op = list(debugger.operations.values())[0]
        assert op.memory_before >= 0
        assert op.memory_after >= 0
        # Memory delta might be positive or negative due to garbage collection
        assert hasattr(op, 'memory_delta')
    
    def test_nested_operations(self, debugger):
        """Test nested operation tracking."""
        
        @debugger.track_operation("Parent")
        def parent_operation():
            @debugger.track_operation("Child")
            def child_operation():
                return "child_result"
            
            return child_operation()
        
        result = parent_operation()
        
        assert len(debugger.operations) == 2
        
        # Find parent and child
        parent_op = next(op for op in debugger.operations.values() if op.operation_name == "Parent")
        child_op = next(op for op in debugger.operations.values() if op.operation_name == "Child")
        
        assert child_op.parent_id == parent_op.operation_id
        assert child_op.operation_id in parent_op.children_ids
    
    def test_bottleneck_detection(self, debugger):
        """Test bottleneck detection for slow operations."""
        
        @debugger.track_operation("Slow Operation")
        def slow_operation():
            time.sleep(1.1)  # Trigger bottleneck threshold
            return "done"
        
        result = slow_operation()
        
        assert len(debugger.bottlenecks) == 1
        assert debugger.bottlenecks[0] in debugger.operations
    
    def test_data_lineage_tracking(self, debugger, sample_df):
        """Test data lineage tracking."""
        
        @debugger.track_operation("Transform 1")
        def transform1(df):
            df['new_col'] = df['value'] * 2
            return df
        
        @debugger.track_operation("Transform 2")
        def transform2(df):
            return df.drop(columns=['category'])
        
        df1 = transform1(sample_df)
        df2 = transform2(df1)
        
        # Check lineage was tracked
        assert len(debugger.data_lineages) > 0
        
        # Find lineage for final DataFrame
        lineage = list(debugger.data_lineages.values())[-1]
        assert len(lineage.transformations) >= 1
        assert lineage.data_type == "pandas.DataFrame"
    
    def test_profile_memory_decorator(self, debugger):
        """Test memory profiling decorator."""
        
        @debugger.profile_memory
        def memory_intensive():
            data = np.random.randn(1000000)
            return data.sum()
        
        result = memory_intensive()
        assert isinstance(result, (int, float))
    
    def test_analyze_dataframe(self, debugger, sample_df, capsys):
        """Test DataFrame analysis functionality."""
        debugger.analyze_dataframe(sample_df, name="Test DF")
        
        captured = capsys.readouterr()
        assert "Test DF" in captured.out
        assert "Shape" in captured.out
        assert "(100, 3)" in captured.out
    
    def test_generate_report(self, debugger, sample_df):
        """Test report generation."""
        
        @debugger.track_operation("Report Test")
        def operation(df):
            return df
        
        operation(sample_df)
        
        report = debugger.generate_report()
        
        assert report['pipeline_name'] == "TestPipeline"
        assert report['total_operations'] == 1
        assert 'operations' in report
        assert len(report['operations']) == 1
    
    def test_export_lineage(self, debugger, sample_df):
        """Test lineage export functionality."""
        
        @debugger.track_operation("Lineage Test")
        def operation(df):
            return df
        
        operation(sample_df)
        
        # Test JSON export
        lineage_json = debugger.export_lineage(format="json")
        assert isinstance(lineage_json, str)
        assert "TestPipeline" in lineage_json
        
        # Test dict export
        lineage_dict = debugger.export_lineage(format="dict")
        assert isinstance(lineage_dict, dict)
        assert lineage_dict['pipeline'] == "TestPipeline"
    
    def test_suggest_optimizations(self, debugger):
        """Test optimization suggestions."""
        
        @debugger.track_operation("Slow Op")
        def slow_op():
            time.sleep(1.5)
            return "done"
        
        @debugger.track_operation("Memory Heavy Op")
        def memory_op():
            # Allocate large array to trigger memory warning
            debugger.memory_threshold_mb = 0.1  # Set low threshold
            data = np.random.randn(1000000)
            return data
        
        slow_op()
        memory_op()
        
        suggestions = debugger.suggest_optimizations()
        assert len(suggestions) >= 2
        
        # Check for performance suggestion
        perf_suggestions = [s for s in suggestions if s['type'] == 'performance']
        assert len(perf_suggestions) >= 1
        
        # Check for memory suggestion
        mem_suggestions = [s for s in suggestions if s['type'] == 'memory']
        assert len(mem_suggestions) >= 1
    
    def test_save_checkpoint(self, debugger, temp_dir, sample_df):
        """Test checkpoint saving."""
        
        @debugger.track_operation("Checkpoint Test")
        def operation(df):
            return df
        
        operation(sample_df)
        
        debugger.save_checkpoint()
        
        # Check checkpoint file was created
        checkpoint_files = list(temp_dir.glob("checkpoint_*.pkl"))
        assert len(checkpoint_files) >= 1

class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_dataframe_info(self):
        """Test DataFrame info extraction."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': ['x', 'y', 'z'],
            'c': [1.1, 2.2, np.nan]
        })
        
        info = get_dataframe_info(df)
        
        assert info['type'] == 'DataFrame'
        assert info['shape'] == (3, 3)
        assert 'memory_usage_mb' in info
        assert info['null_counts']['c'] == 1
    
    def test_detect_dataframe_changes(self):
        """Test DataFrame change detection."""
        df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2 = pd.DataFrame({'a': [1, 2], 'c': [5, 6]})
        
        changes = detect_dataframe_changes(df1, df2)
        
        assert changes['columns_added'] == ['c']
        assert changes['columns_removed'] == ['b']
        assert changes['shape_change']['before'] == (2, 2)
        assert changes['shape_change']['after'] == (2, 2)
