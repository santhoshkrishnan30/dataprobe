#!/usr/bin/env python3
"""
DataProbe v2.1.0 - Enhanced Enterprise Visualization Demo
Showcasing professional-grade pipeline debugging and visualization capabilities
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_wine, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from dataprobe import PipelineDebugger

def main():
    print("=" * 80)
    print("DataProbe v2.1.0 - Enterprise Visualization Demo".center(80))
    print("Professional Pipeline Debugging & Analytics".center(80))
    print("=" * 80)
    print()

    # Initialize enhanced debugger
    debugger = PipelineDebugger(
        name="Enterprise_Demo_Pipeline_v2",
        track_memory=True,
        track_lineage=True,
        auto_save=True
    )

    # ===================== DATA LOADING OPERATIONS =====================
    print("🔄 EXECUTING ENHANCED PIPELINE OPERATIONS")
    print("=" * 80)

    @debugger.track_operation("Load Wine Dataset", source="sklearn", samples=178)
    def load_wine_data():
        print("   Loading wine quality dataset...")
        wine = load_wine()
        wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
        wine_df['quality_class'] = wine.target
        wine_df['quality_name'] = wine_df['quality_class'].map({
            0: 'Class_0', 1: 'Class_1', 2: 'Class_2'
        })
        return wine_df, wine

    @debugger.track_operation("Load Housing Dataset", source="sklearn", samples=20640)
    def load_housing_data():
        print("   Loading California housing dataset...")
        housing = fetch_california_housing()
        housing_df = pd.DataFrame(housing.data, columns=housing.feature_names)
        housing_df['MedHouseValue'] = housing.target
        
        # Add geographic clustering
        housing_df['location_cluster'] = pd.qcut(
            housing_df['Latitude'] + housing_df['Longitude'], 
            q=5, 
            labels=['Region_1', 'Region_2', 'Region_3', 'Region_4', 'Region_5']
        )
        return housing_df, housing

    @debugger.track_operation("Advanced Feature Engineering", algorithms=8)
    def engineer_features(wine_df, housing_df):
        print("   Creating advanced features...")
        
        # Wine feature engineering with error handling
        wine_df = wine_df.copy()
        
        # Create ratios and interactions
        wine_df['alcohol_acid_ratio'] = wine_df['alcohol'] / (wine_df['malic_acid'] + 0.001)
        if 'total_phenols' in wine_df.columns and 'flavanoids' in wine_df.columns:
            wine_df['phenol_flavanoid_interaction'] = wine_df['total_phenols'] * wine_df['flavanoids']
        
        # Statistical features
        numeric_cols = wine_df.select_dtypes(include=[np.number]).columns
        wine_df['feature_mean'] = wine_df[numeric_cols].mean(axis=1)
        wine_df['feature_std'] = wine_df[numeric_cols].std(axis=1)
        
        # Housing feature engineering
        housing_df = housing_df.copy()
        housing_df['rooms_per_household'] = housing_df['AveRooms'] / housing_df['AveOccup']
        housing_df['bedrooms_per_room'] = housing_df['AveBedrms'] / housing_df['AveRooms']
        housing_df['population_per_household'] = housing_df['Population'] / housing_df['AveOccup']
        
        return wine_df, housing_df

    @debugger.track_operation("Statistical Analysis", methods=4)
    def analyze_statistics(wine_df, housing_df):
        print("   Performing comprehensive statistical analysis...")
        
        # Wine analysis
        wine_numeric = wine_df.select_dtypes(include=[np.number])
        wine_corr = wine_numeric.corr()
        wine_stats = wine_df.describe()
        
        # Housing analysis
        housing_numeric = housing_df.select_dtypes(include=[np.number])
        housing_corr = housing_numeric.corr()
        housing_stats = housing_df.describe()
        
        # Advanced analysis
        wine_quality_analysis = wine_df.groupby('quality_name')[wine_numeric.columns[:5]].agg(['mean', 'std']).round(3)
        
        return {
            'wine_correlation': wine_corr,
            'housing_correlation': housing_corr,
            'wine_stats': wine_stats,
            'housing_stats': housing_stats,
            'wine_quality_analysis': wine_quality_analysis
        }

    @debugger.track_operation("ML Pipeline Preparation", train_test_split=True)
    def prepare_ml_pipeline(wine_df, housing_df):
        print("   Preparing comprehensive ML pipeline...")
        
        # Wine classification setup
        wine_numeric = wine_df.select_dtypes(include=[np.number]).columns
        wine_features = [col for col in wine_numeric if col not in ['quality_class']]
        X_wine = wine_df[wine_features]
        y_wine = wine_df['quality_class']
        
        X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(
            X_wine, y_wine, test_size=0.3, random_state=42, stratify=y_wine
        )
        
        # Housing regression setup
        housing_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                           'Population', 'AveOccup', 'Latitude', 'Longitude',
                           'rooms_per_household', 'bedrooms_per_room', 'population_per_household']
        X_housing = housing_df[housing_features]
        y_housing = housing_df['MedHouseValue']
        
        X_train_housing, X_test_housing, y_train_housing, y_test_housing = train_test_split(
            X_housing, y_housing, test_size=0.2, random_state=42
        )
        
        return {
            'wine': (X_train_wine, X_test_wine, y_train_wine, y_test_wine),
            'housing': (X_train_housing, X_test_housing, y_train_housing, y_test_housing)
        }

    @debugger.profile_memory
    def simulate_large_scale_processing():
        print("   Simulating enterprise-scale data processing...")
        
        # Create large dataset for comprehensive memory testing
        large_df = pd.DataFrame(
            np.random.randn(100000, 30),
            columns=[f'feature_{i}' for i in range(30)]
        )
        
        # Add categorical features
        large_df['category'] = np.random.choice(['A', 'B', 'C', 'D', 'E'], 100000)
        large_df['region'] = np.random.choice(['North', 'South', 'East', 'West'], 100000)
        
        # Complex aggregations and transformations
        result = large_df.groupby(['category', 'region']).agg({
            'feature_0': ['mean', 'std', 'min', 'max'],
            'feature_1': ['sum', 'count'],
            'feature_2': ['median', 'var']
        })
        
        # Correlation matrix
        correlation_matrix = large_df.select_dtypes(include=[np.number]).corr()
        
        return result, correlation_matrix

    @debugger.track_operation("Data Quality Validation", rules=6)
    def validate_data_quality(wine_df, housing_df):
        print("   Performing comprehensive data quality validation...")
        
        # Wine data validation
        assert not wine_df.isnull().any().any(), "Wine data contains null values"
        assert len(wine_df) > 0, "Wine dataset is empty"
        assert len(wine_df.columns) > 10, "Wine dataset has insufficient features"
        
        # Housing data validation  
        assert not housing_df.isnull().any().any(), "Housing data contains null values"
        assert len(housing_df) > 0, "Housing dataset is empty"
        assert housing_df['MedHouseValue'].min() > 0, "Invalid housing values detected"
        
        return True

    # ===================== EXECUTE ENHANCED PIPELINE =====================
    try:
        # Load datasets
        wine_df, wine_data = load_wine_data()
        housing_df, housing_data = load_housing_data()
        
        # Feature engineering
        wine_df, housing_df = engineer_features(wine_df, housing_df)
        
        # Statistical analysis
        stats_results = analyze_statistics(wine_df, housing_df)
        
        # ML preparation
        ml_data = prepare_ml_pipeline(wine_df, housing_df)
        
        # Memory profiling demo
        large_result, correlation_matrix = simulate_large_scale_processing()
        
        # Data validation
        validation_result = validate_data_quality(wine_df, housing_df)
        
        print("\n✅ All pipeline operations completed successfully!")
        
    except Exception as e:
        print(f"   ✗ Pipeline error: {e}")

    # ===================== ENHANCED ANALYSIS & VISUALIZATION =====================
    print("\n" + "=" * 80)
    print("📊 GENERATING ENTERPRISE-GRADE VISUALIZATIONS")
    print("=" * 80)

    # Analyze datasets with enhanced features
    debugger.analyze_dataframe(wine_df, name="Enhanced Wine Quality Dataset")
    debugger.analyze_dataframe(housing_df[:1000], name="California Housing with Features")

    print("\n🎨 Creating Professional Dashboards...")
    
    # Generate all visualization types
    try:
        # Enterprise dashboard
        enterprise_path = debugger.visualize_pipeline()
        print(f"✅ Enterprise dashboard: {enterprise_path}")
        
        # 3D visualization
        threed_path = debugger.create_3d_pipeline_visualization()
        print(f"✅ 3D network visualization: {threed_path}")
        
        # Executive report
        exec_path = debugger.generate_executive_report()
        print(f"✅ Executive report: {exec_path}")
        
    except Exception as e:
        print(f"⚠️  Visualization error: {e}")
        print("Note: Ensure matplotlib backend supports display")

    # ===================== ENHANCED PERFORMANCE ANALYSIS =====================
    print("\n" + "=" * 80)
    print("📈 ENHANCED PERFORMANCE ANALYSIS & AI INSIGHTS")
    print("=" * 80)
    
    # Print comprehensive pipeline summary
    debugger.print_summary()
    
    # Get AI-powered optimization suggestions
    suggestions = debugger.suggest_optimizations()
    if suggestions:
        print("\n💡 AI-POWERED OPTIMIZATION RECOMMENDATIONS:")
        for i, suggestion in enumerate(suggestions[:5], 1):
            print(f"\n{i}. [{suggestion['type'].upper()}] {suggestion['operation']}")
            print(f"   Issue: {suggestion['issue']}")
            print(f"   💡 {suggestion['suggestion']}")
    else:
        print("\n✅ Pipeline performing optimally - no issues detected!")
    
    # Generate comprehensive performance report
    report = debugger.generate_report()
    print(f"\n📋 COMPREHENSIVE PIPELINE METRICS:")
    print(f"   • Success Rate: {report['success_rate']*100:.1f}%")
    print(f"   • Total Duration: {report['total_duration']:.3f}s")
    print(f"   • Memory Efficiency: {report['total_memory_used']:.2f}MB")
    print(f"   • Operations Tracked: {report['total_operations']}")
    print(f"   • Bottlenecks Detected: {report['bottlenecks']}")
    print(f"   • Errors Encountered: {report['errors']}")
    
    # Export comprehensive lineage data
    lineage_json = debugger.export_lineage(format="json")
    print(f"   • Data Lineage: {len(debugger.data_lineages)} transformations tracked")
    
    # Display dataset statistics
    print(f"\n📊 FINAL DATASET STATISTICS:")
    print(f"   • Wine: {len(wine_df)} samples, {len(wine_df.columns)} features")
    print(f"   • Housing: {len(housing_df):,} samples, {len(housing_df.columns)} features")
    print(f"   • Feature Engineering: Added {len(wine_df.columns) - len(wine_data.feature_names)} wine features")
    
    print("\n" + "=" * 80)
    print("🎉 DataProbe v2.1.0 Enterprise Demo Complete!")
    print("=" * 80)
    print("\n🚀 Key Enterprise Features Demonstrated:")
    print("   ✅ Professional KPI dashboards with real-time metrics")
    print("   ✅ Interactive 3D pipeline network visualizations")
    print("   ✅ Executive-level multi-page reports")
    print("   ✅ Advanced memory profiling and leak detection")
    print("   ✅ AI-powered optimization recommendations")
    print("   ✅ Comprehensive data lineage tracking")
    print("   ✅ Enterprise-grade error handling and reporting")
    print("   ✅ Professional color-coded status indicators")
    
    print(f"\n🎯 DataProbe v2.1.0 Benefits:")
    print(f"   • Rivals commercial ETL monitoring tools")
    print(f"   • Suitable for executive presentations")
    print(f"   • Production-ready performance insights")
    print(f"   • Professional visualization exports")
    
    print(f"\n📚 Learn More:")
    print(f"   • Documentation: https://dataprobe.readthedocs.io")
    print(f"   • GitHub: https://github.com/santhoshkrishnan30/dataprobe")
    print(f"   • PyPI: https://pypi.org/project/dataprobe/")
    
    print(f"\n💡 Get Started:")
    print(f"   pip install --upgrade dataprobe")

if __name__ == "__main__":
    main()