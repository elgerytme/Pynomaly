import pytest
from pynomaly.application.services.business_kpi_service import BusinessKPIService


@pytest.fixture
def kpi_service():
    return BusinessKPIService()


def test_calculate_revenue_at_risk(kpi_service):
    total_revenue = 1000000
    risk_factor = 0.05
    expected = 50000
    result = kpi_service.calculate_revenue_at_risk(total_revenue, risk_factor)
    assert result == expected


def test_calculate_prevented_loss(kpi_service):
    loss_prevented = 50000
    total_potential_loss = 200000
    expected = 25.0
    result = kpi_service.calculate_prevented_loss(loss_prevented, total_potential_loss)
    assert result == expected


def test_calculate_roi(kpi_service):
    gains_from_investment = 150000
    cost_of_investment = 50000
    expected = 200.0
    result = kpi_service.calculate_roi(gains_from_investment, cost_of_investment)
    assert result == expected


def test_calculate_cost_savings_trends(kpi_service):
    monthly_savings = [10000, 15000, 12000, 13000, 11000]
    expected = 12200.0
    result = kpi_service.calculate_cost_savings_trends(monthly_savings)
    assert result == expected


def test_calculate_cost_savings_trends_empty(kpi_service):
    monthly_savings = []
    expected = 0.0
    result = kpi_service.calculate_cost_savings_trends(monthly_savings)
    assert result == expected

