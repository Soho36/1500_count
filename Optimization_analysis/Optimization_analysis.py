import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ============================================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================================

# File paths
INPUT_FILE = "0130 0200.csv"  # Change to your file name
OUTPUT_FILE = "optimization_analysis.xlsx"

# CSV settings (based on your data structure)
CSV_SEPARATOR = '\t'  # Tab-separated (change to ',' if comma-separated)
CSV_DECIMAL = '.'  # Decimal separator

# Filter settings
MIN_TRADES = 30  # Minimum number of trades required
MAX_DRAWDOWN = 20.0  # Maximum drawdown percentage allowed
MIN_PROFIT_FACTOR = 1.1  # Minimum Profit Factor required

# Display settings
TOP_N_DISPLAY = 20  # Number of top runs to show in report
TOP_N_SAVE = 100  # Number of top runs to save in Excel
TOP_N_SEASONALITY = 10  # Number of runs to analyze for seasonality
N_RECOMMENDATIONS = 5  # Number of recommendations to generate

# Composite score weights
SCORE_WEIGHTS = {
    'Profit': 0,  # Note: Changed from 'profit' to 'Profit'
    'Profit Factor': 0.25,  # Changed from 'profit_factor'
    'Recovery Factor': 0.20,  # Changed from 'recovery_factor'
    'Sharpe Ratio': 0.20,  # Changed from 'sharpe_ratio'
    'Expected Payoff': 0.15,  # Changed from 'expected_payoff'
    'Custom Equity DD %': -0.10,  # Changed from 'max_dd'
}

# Pareto analysis metrics
PARETO_X_METRIC = 'Trades'
PARETO_Y_METRIC = 'Profit Factor'


# ============================================================================
# MAIN SCRIPT - NO NEED TO EDIT BELOW
# ============================================================================

class OptimizationAnalyzer:
    def __init__(self, weights=None):
        """Initialize analyzer with customizable weights"""
        self.default_weights = SCORE_WEIGHTS
        self.weights = weights or self.default_weights

    def load_data(self, filepath):
        """Load optimization data from CSV/Excel"""
        print(f"Loading data from {filepath}...")
        print(
            f"Using separator: '{CSV_SEPARATOR}' (tab)" if CSV_SEPARATOR == '\t' else f"Using separator: '{CSV_SEPARATOR}'")

        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath, sep=CSV_SEPARATOR, decimal=CSV_DECIMAL)
        elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")

        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()

        # Debug: Print column names
        print(f"Found {len(df.columns)} columns:")
        for i, col in enumerate(df.columns):
            print(f"  {i + 1}. '{col}'")

        print(f"\nLoaded {len(df)} rows")
        return df

    def filter_runs(self, df):
        """Apply initial filters to remove poor runs"""
        print("\nApplying filters...")
        print(f"Available columns: {list(df.columns)}")

        filtered = df.copy()
        initial_count = len(filtered)

        # Check which columns we have
        profit_col = None
        trades_col = None
        pf_col = None
        dd_col = None

        # Find the actual column names
        for col in df.columns:
            if 'Profit' in col and col != 'Profit Factor' and col != 'Expected Payoff':
                profit_col = col
            elif 'Trades' in col:
                trades_col = col
            elif 'Profit Factor' in col:
                pf_col = col
            elif 'DD %' in col or 'Drawdown' in col:
                dd_col = col

        print(f"Detected columns:")
        print(f"  Profit column: {profit_col}")
        print(f"  Trades column: {trades_col}")
        print(f"  Profit Factor column: {pf_col}")
        print(f"  Drawdown column: {dd_col}")

        # Apply filters based on detected columns
        if profit_col:
            filtered = filtered[filtered[profit_col] > 0]
            print(f"  Positive {profit_col}: {len(filtered)}/{initial_count}")

        if trades_col:
            filtered = filtered[filtered[trades_col] >= MIN_TRADES]
            print(f"  Min {trades_col} {MIN_TRADES}: {len(filtered)}/{initial_count}")

        if pf_col:
            filtered = filtered[filtered[pf_col] >= MIN_PROFIT_FACTOR]
            print(f"  Min {pf_col} {MIN_PROFIT_FACTOR}: {len(filtered)}/{initial_count}")

        # Drawdown filter - check both possible column names
        dd_filtered = False
        if 'Custom Equity DD %' in filtered.columns:
            filtered = filtered[filtered['Custom Equity DD %'] <= MAX_DRAWDOWN]
            dd_filtered = True
            print(f"  Max Custom Equity DD % {MAX_DRAWDOWN}%: {len(filtered)}/{initial_count}")
        elif 'DD %' in filtered.columns:
            filtered = filtered[filtered['DD %'] <= MAX_DRAWDOWN]
            dd_filtered = True
            print(f"  Max DD % {MAX_DRAWDOWN}%: {len(filtered)}/{initial_count}")
        elif dd_col:
            filtered = filtered[filtered[dd_col] <= MAX_DRAWDOWN]
            dd_filtered = True
            print(f"  Max {dd_col} {MAX_DRAWDOWN}%: {len(filtered)}/{initial_count}")

        if not dd_filtered:
            print(f"  Note: No drawdown column found for filtering")

        print(f"\nFiltered from {initial_count} to {len(filtered)} runs "
              f"({(len(filtered) / initial_count * 100):.1f}% remaining)")

        return filtered

    def calculate_composite_score(self, df):
        """Calculate weighted composite score for each run"""
        print("\nCalculating composite scores with weights:")
        for metric, weight in self.weights.items():
            print(f"  {metric}: {weight:.2f}")

        df_scored = df.copy()

        # Normalize each metric (0-1 range)
        scaler = MinMaxScaler()

        for metric, weight in self.weights.items():
            # Check if this metric exists in the dataframe
            if metric in df.columns:
                if 'DD %' in metric or 'Drawdown' in metric:
                    # For drawdown, invert so lower values get higher scores
                    normalized = 1 - scaler.fit_transform(df[[metric]].values).flatten()
                else:
                    normalized = scaler.fit_transform(df[[metric]].values).flatten()
                df_scored[f'{metric}_norm'] = normalized
            else:
                print(f"  Warning: Metric '{metric}' not found in dataframe")

        # Calculate weighted composite score
        df_scored['composite_score'] = 0

        for metric, weight in self.weights.items():
            col_name = f'{metric}_norm'
            if col_name in df_scored.columns:
                df_scored['composite_score'] += df_scored[col_name] * weight

        # Sort by composite score
        df_scored = df_scored.sort_values('composite_score', ascending=False)

        print(f"Composite scores calculated for {len(df_scored)} runs")

        return df_scored

    def find_pareto_front(self, df):
        """Identify Pareto-optimal runs (non-dominated solutions)"""
        try:
            x_metric = PARETO_X_METRIC
            y_metric = PARETO_Y_METRIC

            if x_metric not in df.columns or y_metric not in df.columns:
                print(f"  Warning: Pareto metrics not found ({x_metric}, {y_metric})")
                return pd.DataFrame()

            pareto_points = []

            for idx, row in df.iterrows():
                # Check if this point is dominated by any other point
                dominated = False
                for _, other_row in df.iterrows():
                    if (other_row[x_metric] >= row[x_metric] and
                            other_row[y_metric] >= row[y_metric] and
                            not (other_row[x_metric] == row[x_metric] and
                                 other_row[y_metric] == row[y_metric])):
                        dominated = True
                        break

                if not dominated:
                    pareto_points.append(idx)

            return df.loc[pareto_points]
        except Exception as e:
            print(f"  Warning: Pareto analysis failed: {e}")
            return pd.DataFrame()

    def analyze_seasonality(self, df):
        """Analyze monthly performance patterns in top runs"""
        month_columns = ['TradeJanuary', 'TradeFebruary', 'TradeMarch',
                         'TradeApril', 'TradeMay', 'TradeJune',
                         'TradeJuly', 'TradeAugust', 'TradeSeptember',
                         'TradeOctober', 'TradeNovember', 'TradeDecember']

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # Get top runs
        top_runs = df.head(TOP_N_SEASONALITY)

        # Calculate monthly success rate
        seasonal_summary = {}
        month_columns_found = []

        for month_col, month_name in zip(month_columns, month_names):
            if month_col in top_runs.columns:
                success_rate = top_runs[month_col].mean() * 100
                seasonal_summary[month_name] = success_rate
                month_columns_found.append(month_col)

        if not month_columns_found:
            print("  No month columns found for seasonality analysis")

        return seasonal_summary

    def recommend_runs(self, df):
        """Get top recommendations with diversity consideration"""
        recommendations = []

        # Get top by composite score
        top_by_score = df.head(N_RECOMMENDATIONS * 3)  # Get more for diversity

        # Group by similar characteristics (simplified example)
        recommendations.append({
            'type': 'Best Overall',
            'runs': df.head(N_RECOMMENDATIONS).to_dict('records')
        })

        # Get best by specific metrics for diversity
        metrics_to_consider = ['Profit Factor', 'Recovery Factor', 'Sharpe Ratio']

        for metric in metrics_to_consider:
            if metric in df.columns:
                best_by_metric = df.nlargest(2, metric)
                recommendations.append({
                    'type': f'Best by {metric}',
                    'runs': best_by_metric.head(2).to_dict('records')
                })

        return recommendations

    def generate_report(self, df):
        """Generate comprehensive analysis report"""
        report = []

        report.append("=" * 60)
        report.append("OPTIMIZATION ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Configuration:")
        report.append(f"  Input file: {INPUT_FILE}")
        report.append(f"  Min trades: {MIN_TRADES}")
        report.append(f"  Max drawdown: {MAX_DRAWDOWN}%")
        report.append(f"  Min Profit Factor: {MIN_PROFIT_FACTOR}")
        report.append(f"Total runs analyzed: {len(df)}")
        report.append(f"Top {TOP_N_DISPLAY} runs by composite score:")
        report.append("-" * 60)

        # Top runs summary
        top_runs = df.head(TOP_N_DISPLAY)

        # Find the actual profit column name
        profit_col = None
        for col in df.columns:
            if 'Profit' in col and col != 'Profit Factor' and col != 'Expected Payoff':
                profit_col = col
                break

        for i, (idx, row) in enumerate(top_runs.iterrows(), 1):
            pass_num = row.get('Pass', idx)
            report.append(f"\n{i}. Pass {pass_num}")
            report.append(f"   Composite Score: {row.get('composite_score', 0):.3f}")

            if profit_col:
                report.append(f"   Profit: ${row.get(profit_col, 0):.2f}")

            if 'Profit Factor' in row:
                report.append(f"   Profit Factor: {row.get('Profit Factor', 0):.3f}")

            if 'Sharpe Ratio' in row:
                report.append(f"   Sharpe Ratio: {row.get('Sharpe Ratio', 0):.3f}")

            if 'Trades' in row:
                report.append(f"   Trades: {row.get('Trades', 0)}")

        # Statistical summary
        report.append("\n" + "=" * 60)
        report.append("STATISTICAL SUMMARY")
        report.append("-" * 60)

        metrics_to_check = ['Profit', 'Profit Factor', 'Sharpe Ratio', 'Recovery Factor', 'Trades']

        for metric_name in metrics_to_check:
            # Find the actual column name
            actual_col = None
            for col in df.columns:
                if metric_name in col:
                    actual_col = col
                    break

            if actual_col and actual_col in df.columns:
                report.append(f"\n{metric_name}:")
                report.append(f"  Mean: {df[actual_col].mean():.3f}")
                report.append(f"  Std: {df[actual_col].std():.3f}")
                report.append(f"  Max: {df[actual_col].max():.3f}")
                report.append(f"  Min: {df[actual_col].min():.3f}")

        return "\n".join(report)


def main():
    print("=" * 60)
    print("MT5 OPTIMIZATION ANALYZER")
    print("=" * 60)

    # Initialize analyzer
    analyzer = OptimizationAnalyzer()

    try:
        # Load data
        df = analyzer.load_data(INPUT_FILE)

        if len(df) == 0:
            print("ERROR: No data loaded. Check file path and format.")
            return

        # Filter runs
        df_filtered = analyzer.filter_runs(df)

        if len(df_filtered) == 0:
            print("\nERROR: No runs passed the filters. Adjust filter parameters in the config section.")
            print("Consider reducing MIN_TRADES, increasing MAX_DRAWDOWN, or lowering MIN_PROFIT_FACTOR")
            return

        # Calculate composite scores
        df_scored = analyzer.calculate_composite_score(df_filtered)

        # Generate report
        report = analyzer.generate_report(df_scored)
        print("\n" + report)

        # Optional: Analyze seasonality
        try:
            seasonality = analyzer.analyze_seasonality(df_scored)
            if seasonality:
                print("\n" + "=" * 60)
                print(f"SEASONALITY ANALYSIS (Top {TOP_N_SEASONALITY} runs)")
                print("-" * 60)
                for month, success_rate in seasonality.items():
                    print(f"{month}: {success_rate:.1f}% profitable runs")
        except Exception as e:
            print(f"\nNote: Seasonality analysis skipped: {e}")

        # Optional: Find Pareto front
        try:
            pareto_runs = analyzer.find_pareto_front(df_scored)
            if len(pareto_runs) > 0:
                print(f"\nFound {len(pareto_runs)} Pareto-optimal runs")
        except Exception as e:
            print(f"\nNote: Pareto analysis skipped: {e}")

        # Save results
        print(f"\nSaving results to {OUTPUT_FILE}...")
        with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
            df_scored.to_excel(writer, sheet_name='All_Runs', index=False)
            df_scored.head(TOP_N_SAVE).to_excel(writer, sheet_name=f'Top_{TOP_N_SAVE}', index=False)

            # Add summary sheet
            summary_data = {
                'Metric': ['Total Runs', 'Filtered Runs', 'Filter %'],
                'Value': [
                    len(df),
                    len(df_filtered),
                    f"{(len(df_filtered) / len(df) * 100):.1f}%"
                ]
            }

            # Add metrics to summary
            metrics_to_add = ['Profit', 'Profit Factor', 'Sharpe Ratio', 'Recovery Factor']
            for metric_name in metrics_to_add:
                # Find actual column
                actual_col = None
                for col in df_scored.columns:
                    if metric_name in col and '_norm' not in col and 'composite' not in col:
                        actual_col = col
                        break

                if actual_col and actual_col in df_scored.columns:
                    summary_data['Metric'].append(f'Best {metric_name}')
                    summary_data['Value'].append(df_scored[actual_col].max())

            summary_data['Metric'].append('Best Composite Score')
            summary_data['Value'].append(df_scored['composite_score'].max())

            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

            # Add configuration sheet
            config_data = {
                'Parameter': ['INPUT_FILE', 'OUTPUT_FILE', 'CSV_SEPARATOR', 'MIN_TRADES',
                              'MAX_DRAWDOWN', 'MIN_PROFIT_FACTOR', 'TOP_N_DISPLAY',
                              'TOP_N_SAVE', 'TOP_N_SEASONALITY', 'N_RECOMMENDATIONS'],
                'Value': [INPUT_FILE, OUTPUT_FILE, repr(CSV_SEPARATOR), MIN_TRADES, MAX_DRAWDOWN,
                          MIN_PROFIT_FACTOR, TOP_N_DISPLAY, TOP_N_SAVE,
                          TOP_N_SEASONALITY, N_RECOMMENDATIONS]
            }
            pd.DataFrame(config_data).to_excel(writer, sheet_name='Configuration', index=False)

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)

        # Display top 3 recommendations
        print(f"\nTop 3 recommended passes:")
        print("-" * 40)

        top_3 = df_scored.head(3)

        # Find the actual profit column name
        profit_col = None
        for col in df_scored.columns:
            if 'Profit' in col and col != 'Profit Factor' and col != 'Expected Payoff' and '_norm' not in col:
                profit_col = col
                break

        for i, (idx, row) in enumerate(top_3.iterrows(), 1):
            pass_num = row.get('Pass', idx)
            print(f"\n{i}. Pass {pass_num}")
            print(f"   Composite Score: {row['composite_score']:.3f}")

            if profit_col:
                print(f"   Profit: ${row[profit_col]:.2f}")

            if 'Profit Factor' in row:
                print(f"   Profit Factor: {row['Profit Factor']:.2f}")

            if 'Sharpe Ratio' in row:
                print(f"   Sharpe Ratio: {row['Sharpe Ratio']:.2f}")

            if 'Trades' in row:
                print(f"   Trades: {row['Trades']}")

            if 'Custom Equity DD %' in row:
                print(f"   DD %: {row['Custom Equity DD %']:.2f}%")

        print(f"\nFull results saved to: {OUTPUT_FILE}")

    except FileNotFoundError:
        print(f"\nERROR: File '{INPUT_FILE}' not found.")
        print(f"Please ensure the file exists in the current directory.")
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()