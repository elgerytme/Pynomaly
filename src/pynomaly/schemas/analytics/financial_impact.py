"""
Financial impact assessment schemas for analyzing costs, savings, and revenue.

This module provides Pydantic schemas for analyzing the financial impact of
monitored systems, including cost metrics, savings, revenue, and return on
investment (ROI) calculations.

Schemas:
    FinancialImpactFrame: Main financial impact frame
    ROICalculation: ROI calculation model
    CostMetrics: Cost-related metrics
    SavingsMetrics: Savings-related metrics
    RevenueMetrics: Revenue-related metrics
"""

from __future__ import annotations

from typing import Optional, Dict, Any
from datetime import date

from pydantic import BaseModel, Field, root_validator, validator

from .base import RealTimeMetricFrame


class CostMetrics(BaseModel):
    """Financial cost metrics."""
    
    total_cost: float = Field(ge=0.0, description="Total cost incurred")
    cost_per_unit: float = Field(ge=0.0, description="Cost per unit")
    budget: Optional[float] = Field(None, ge=0.0, description="Allocated budget")
    
    @validator('total_cost', 'cost_per_unit')
    def validate_non_negative(cls, v: float) -> float:
        """Ensure cost metrics are non-negative."""
        if v < 0:
            raise ValueError('Cost values must be non-negative')
        return v
    
    def is_within_budget(self) -> bool:
        """Check if total cost is within budget."""
        return self.budget is not None and self.total_cost <= self.budget


class SavingsMetrics(BaseModel):
    """Metrics related to financial savings achieved through optimization."""
    
    total_savings: float = Field(ge=0.0, description="Total savings achieved")
    savings_rate: float = Field(ge=0.0, le=1.0, description="Rate of savings")
    
    @validator('savings_rate')
    def validate_savings_rate(cls, v: float) -> float:
        """Ensure savings rate is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError('Savings rate must be between 0 and 1')
        return v


class RevenueMetrics(BaseModel):
    """Revenue metrics for financial analysis."""
    
    total_revenue: float = Field(ge=0.0, description="Total revenue generated")
    revenue_per_unit: float = Field(ge=0.0, description="Revenue per unit")
    revenue_growth_rate: Optional[float] = Field(None, description="Revenue growth rate")
    
    @validator('total_revenue', 'revenue_per_unit')
    def validate_non_negative(cls, v: float) -> float:
        """Ensure revenue metrics are non-negative."""
        if v < 0:
            raise ValueError('Revenue values must be non-negative')
        return v


class ROICalculation(BaseModel):
    """Calculation of return on investment (ROI)."""
    
    investment: float = Field(ge=0.0, description="Total investment amount")
    returns: float = Field(ge=0.0, description="Returns from investment")
    roi: Optional[float] = Field(None, description="Calculated ROI")
    period_start_date: Optional[date] = Field(None, description="Start date of the investment period")
    period_end_date: Optional[date] = Field(None, description="End date of the investment period")
    
    @root_validator(pre=True)
    def calculate_roi(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate ROI based on investment and returns."""
        investment = values.get('investment', 0)
        returns = values.get('returns', 0)
        if investment > 0:
            values['roi'] = (returns - investment) / investment
        return values
    
    def is_profitable(self) -> bool:
        """Check if the investment is profitable based on ROI."""
        return self.roi is not None and self.roi > 0


class FinancialImpactFrame(RealTimeMetricFrame):
    """Main financial impact frame integrating cost, savings, and revenue metrics."""
    
    cost_metrics: CostMetrics
    savings_metrics: SavingsMetrics
    revenue_metrics: RevenueMetrics
    roi_calculation: ROICalculation
    
    @property
    def total_benefits(self) -> float:
        """Calculate total financial benefits."""
        return self.savings_metrics.total_savings + self.revenue_metrics.total_revenue
    
    def get_financial_summary(self) -> Dict[str, Any]:
        """Get a summary of financial metrics."""
        return {
            "total_cost": self.cost_metrics.total_cost,
            "total_revenue": self.revenue_metrics.total_revenue,
            "total_savings": self.savings_metrics.total_savings,
            "roi": self.roi_calculation.roi,
            "is_profitable": self.roi_calculation.is_profitable(),
        }
    
    def meets_financial_goals(self) -> bool:
        """Check if financial goals are met based on cost, savings, and ROI."""
        return all([
            self.cost_metrics.is_within_budget(),
            self.roi_calculation.is_profitable(),
            self.savings_metrics.savings_rate >= 0.2,  # Example savings goal
        ])
    
    def __str__(self) -> str:
        """String representation of the financial impact frame."""
        return (
            f"FinancialImpact(cost={self.cost_metrics.total_cost:.2f}, "
            f"revenue={self.revenue_metrics.total_revenue:.2f}, "
            f"savings={self.savings_metrics.total_savings:.2f}, ROI={self.roi_calculation.roi:.2%})"
        )

