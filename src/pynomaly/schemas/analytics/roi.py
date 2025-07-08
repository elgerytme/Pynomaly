"""
ROI analysis schemas for assessing return on investment.

This module provides Pydantic schemas for return on investment (ROI) analysis
including cost-benefit analysis and investment metrics.

Schemas:
    ROIFrame: Main ROI analysis frame
    CostBenefitAnalysis: Cost-benefit analysis information
    InvestmentMetrics: Investment-related metrics
"""

from __future__ import annotations

from typing import Optional, Dict, Any
from datetime import date

from pydantic import BaseModel, Field, validator

from .base import RealTimeMetricFrame


class CostBenefitAnalysis(BaseModel):
    """Cost-benefit analysis information."""
    
    total_benefits: float = Field(ge=0.0, description="Total benefits calculated")
    total_costs: float = Field(ge=0.0, description="Total costs calculated")
    net_benefits: Optional[float] = Field(None, description="Net benefits (benefits - costs)")
    benefit_cost_ratio: Optional[float] = Field(None, description="Benefit-cost ratio")
    internal_rate_of_return: Optional[float] = Field(None, description="Internal rate of return")
    
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.net_benefits = self.total_benefits - self.total_costs
        self.benefit_cost_ratio = self.total_benefits / self.total_costs if self.total_costs else None
    

class InvestmentMetrics(BaseModel):
    """Investment metrics for ROI calculation."""
    
    initial_investment: float = Field(ge=0.0, description="Initial investment amount")
    investment_period_years: int = Field(ge=1, description="Investment period in years")
    annual_return_rate: float = Field(ge=0.0, le=1.0, description="Annual return rate")
    future_value: Optional[float] = Field(None, description="Future value of the investment")
    
    def calculate_future_value(self) -> float:
        """Calculate the future value of the investment."""
        self.future_value = self.initial_investment * ((1 + self.annual_return_rate) ** self.investment_period_years)
        return self.future_value


class ROIFrame(RealTimeMetricFrame):
    """Main ROI analysis frame integrating cost-benefit and investment metrics."""
    
    cost_benefit_analysis: CostBenefitAnalysis
    investment_metrics: InvestmentMetrics
    analysis_date: date = Field(description="Date of the ROI analysis")
    
    def get_roi_summary(self) -> Dict[str, Any]:
        """Get a summary of the ROI analysis."""
        self.investment_metrics.calculate_future_value()
        return {
            "net_benefits": self.cost_benefit_analysis.net_benefits,
            "benefit_cost_ratio": self.cost_benefit_analysis.benefit_cost_ratio,
            "internal_rate_of_return": self.cost_benefit_analysis.internal_rate_of_return,
            "future_value": self.investment_metrics.future_value,
            "investment_period_years": self.investment_metrics.investment_period_years,
            "annual_return_rate": self.investment_metrics.annual_return_rate,
        }
    
    def is_viable_investment(self) -> bool:
        """Check if the investment is viable based on ROI metrics."""
        return all([
            self.cost_benefit_analysis.net_benefits > 0,
            self.cost_benefit_analysis.benefit_cost_ratio and self.cost_benefit_analysis.benefit_cost_ratio > 1,
            self.investment_metrics.annual_return_rate >= 0.05,  # Example investment goal
        ])
    
    def meets_investment_goals(self) -> bool:
        """Check if investment goals are met based on future value and IRR."""
        return all([
            self.investment_metrics.future_value and self.investment_metrics.future_value >= self.investment_metrics.initial_investment * 1.5,  # Example future value goal
            self.cost_benefit_analysis.internal_rate_of_return and self.cost_benefit_analysis.internal_rate_of_return >= 0.07,  # Example IRR goal
        ])
    
    def __str__(self) -> str:
        """String representation of the ROI analysis frame."""
        return (
            f"ROIAnalysis(net_benefits={self.cost_benefit_analysis.net_benefits:.2f}, "
            f"BCR={self.cost_benefit_analysis.benefit_cost_ratio:.2f}, "
            f"IRR={self.cost_benefit_analysis.internal_rate_of_return:.2%})"
        )

