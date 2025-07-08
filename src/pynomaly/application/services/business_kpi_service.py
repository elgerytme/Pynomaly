class BusinessKPIService:
    def calculate_revenue_at_risk(self, total_revenue, risk_factor):
        """Calculate the revenue at risk based on a given risk factor."""
        return total_revenue * risk_factor

    def calculate_prevented_loss(self, loss_prevented, total_potential_loss):
        """Calculate the percentage of prevented loss."""
        return (loss_prevented / total_potential_loss) * 100

    def calculate_roi(self, gains_from_investment, cost_of_investment):
        """Calculate Return on Investment (ROI)."""
        return ((gains_from_investment - cost_of_investment) / cost_of_investment) * 100

    def calculate_cost_savings_trends(self, monthly_savings):
        """Calculate the trend in cost savings over time."""
        # For simplicity, let's return a static trend value for now
        return sum(monthly_savings) / len(monthly_savings) if monthly_savings else 0.0
