import re
from typing import Dict, List
import pandas as pd
from ...domain.entities.profiles import Pattern

class PatternDiscoveryService:
    """Service to discover common patterns in text data."""
    EMAIL_REGEX = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
    PHONE_REGEX = r'\+?\d[\d \-]{7,}\d'
    URL_REGEX = r'https?://[^\s]+'

    def discover(self, df: pd.DataFrame) -> Dict[str, List[Pattern]]:
        """Return patterns detected per column."""
        patterns: Dict[str, List[Pattern]] = {}
        for col in df.select_dtypes(include='object').columns:
            series = df[col].dropna().astype(str)
            examples = series.unique().tolist()[:100]
            col_patterns: List[Pattern] = []
            # email patterns
            emails = [v for v in examples if re.fullmatch(self.EMAIL_REGEX, v)]
            if emails:
                col_patterns.append(Pattern(
                    pattern_type='email',
                    regex=self.EMAIL_REGEX,
                    frequency=len(emails),
                    percentage=len(emails) / len(examples) if examples else 0.0,
                    examples=emails[:5],
                    confidence=1.0
                ))
            # phone patterns
            phones = [v for v in examples if re.fullmatch(self.PHONE_REGEX, v)]
            if phones:
                col_patterns.append(Pattern(
                    pattern_type='phone',
                    regex=self.PHONE_REGEX,
                    frequency=len(phones),
                    percentage=len(phones) / len(examples) if examples else 0.0,
                    examples=phones[:5],
                    confidence=1.0
                ))
            # URL patterns
            urls = [v for v in examples if re.fullmatch(self.URL_REGEX, v)]
            if urls:
                col_patterns.append(Pattern(
                    pattern_type='url',
                    regex=self.URL_REGEX,
                    frequency=len(urls),
                    percentage=len(urls) / len(examples) if examples else 0.0,
                    examples=urls[:5],
                    confidence=1.0
                ))
            if col_patterns:
                patterns[col] = col_patterns
        return patterns