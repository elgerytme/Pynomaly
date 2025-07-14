import pandas as pd
from ...domain.entities.profiles import StatisticalProfile

class StatisticalProfilingService:
    """Service to perform statistical profiling on numeric data."""
    def analyze(self, df: pd.DataFrame) -> StatisticalProfile:
        """Compute descriptive stats for each numeric column."""
        stats: dict[str, dict[str, float]] = {}
        for col in df.select_dtypes(include='number').columns:
            series = df[col].dropna()
            if series.empty:
                continue
            mode_val = series.mode()
            stats[col] = {
                'mean': float(series.mean()),
                'median': float(series.median()),
                'mode': float(mode_val.iloc[0]) if not mode_val.empty else None,
                'std': float(series.std())
            }
        return StatisticalProfile(numeric_stats=stats)