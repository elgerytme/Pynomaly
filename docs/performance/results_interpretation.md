# Interpreting Results

### Common Result Artifacts

#### CSV (Comma-Separated Values)
- **Description**: A simple format for storing tabular data.
- **Use**: Easy sharing and integration with tools like spreadsheets and databases.
- **Analysis**: Look for trends, averages, and outliers.

#### HTML Flame Graph
- **Description**: A visual representation of hierarchical data, typically used for profiling.
- **Use**: Identifies where CPU time is being spent.
- **Analysis**: Look for wide segments and deep stacks which may indicate performance bottlenecks.

#### Trend Charts
- **Description**: Visual representations showing changes over time.
- **Use**: Identify trends and anomalies in your data.
- **Analysis**: Look for patterns, sudden spikes, or drops which could signal a regression or improvement.

### Heuristics for Analysis

#### Spotting Regressions
- **Indicators**: Degradation in key performance metrics compared to previous results.
- **Techniques**: Compare baseline data against current results using trend charts and summaries.

#### Variance Analysis
- **Description**: Assess how data points differ from the mean.
- **Use**: Helps in understanding data consistency.
- **Tools**: Plot standard deviations or use box plots.

#### Statistical Significance (Mann-Whitney U Test)
- **Purpose**: Determine if two independent samples come from the same distribution.
- **Use Case**: Advocate changes in software or configurations when benchmark results vary.

### Steps for Using Mann-Whitney U Test
1. **Collect Data**: Gather performance metrics before and after changes.
2. **Hypothesis Testing**:
   - Null: No difference between sample distributions.
   - Alternative: A significant difference exists.
3. **Calculate the U Statistic**.
4. **Interpret Results**:
   - A low p-value (< 0.05) typically indicates significant differences.

By leveraging these tools and techniques, one can efficiently interpret performance data, spot issues, and drive continuous improvement.
