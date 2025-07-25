"""
Pricing value objects for the MLOps Marketplace.

Defines pricing models, billing cycles, and discount structures
for marketplace solutions and subscriptions.
"""

from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator


class Currency(str, Enum):
    """Supported currencies."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CAD = "CAD"
    AUD = "AUD"
    CHF = "CHF"
    CNY = "CNY"
    INR = "INR"


class BillingCycle(str, Enum):
    """Billing cycle options."""
    ONE_TIME = "one_time"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class PricingType(str, Enum):
    """Types of pricing models."""
    FREE = "free"
    FIXED = "fixed"
    TIERED = "tiered"
    USAGE_BASED = "usage_based"
    FREEMIUM = "freemium"
    SUBSCRIPTION = "subscription"


class UsageUnit(str, Enum):
    """Units for usage-based pricing."""
    API_CALLS = "api_calls"
    PREDICTIONS = "predictions"
    PROCESSING_TIME = "processing_time"
    DATA_VOLUME = "data_volume"
    STORAGE = "storage"
    COMPUTE_HOURS = "compute_hours"
    TRANSACTIONS = "transactions"


class Price(BaseModel):
    """Price value object with currency and amount."""
    
    amount: Decimal = Field(..., ge=0)
    currency: Currency = Field(default=Currency.USD)
    
    def __init__(self, amount: any = None, currency: Currency = Currency.USD, **kwargs):
        """Initialize price with proper decimal conversion."""
        if amount is not None:
            if isinstance(amount, (int, float, str)):
                amount = Decimal(str(amount))
            elif not isinstance(amount, Decimal):
                raise ValueError(f"Invalid amount type: {type(amount)}")
        super().__init__(amount=amount, currency=currency, **kwargs)
    
    def __str__(self) -> str:
        """String representation of price."""
        return f"{self.amount} {self.currency.value}"
    
    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if not isinstance(other, Price):
            return False
        return self.amount == other.amount and self.currency == other.currency
    
    def __add__(self, other: 'Price') -> 'Price':
        """Add two prices (must be same currency)."""
        if self.currency != other.currency:
            raise ValueError("Cannot add prices with different currencies")
        return Price(amount=self.amount + other.amount, currency=self.currency)
    
    def __sub__(self, other: 'Price') -> 'Price':
        """Subtract two prices (must be same currency)."""
        if self.currency != other.currency:
            raise ValueError("Cannot subtract prices with different currencies")
        result_amount = self.amount - other.amount
        if result_amount < 0:
            raise ValueError("Price cannot be negative")
        return Price(amount=result_amount, currency=self.currency)
    
    def __mul__(self, multiplier: Decimal) -> 'Price':
        """Multiply price by a factor."""
        if not isinstance(multiplier, (Decimal, int, float)):
            raise ValueError("Multiplier must be a numeric value")
        return Price(amount=self.amount * Decimal(str(multiplier)), currency=self.currency)
    
    def is_free(self) -> bool:
        """Check if price is zero."""
        return self.amount == Decimal('0')
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary."""
        return {
            "amount": str(self.amount),
            "currency": self.currency.value
        }
    
    @classmethod
    def free(cls, currency: Currency = Currency.USD) -> 'Price':
        """Create a free price."""
        return cls(amount=Decimal('0'), currency=currency)
    
    @classmethod
    def from_cents(cls, cents: int, currency: Currency = Currency.USD) -> 'Price':
        """Create price from cents/minor units."""
        return cls(amount=Decimal(cents) / 100, currency=currency)
    
    @validator('amount')
    def validate_amount(cls, v: Decimal) -> Decimal:
        """Validate amount is non-negative and has appropriate precision."""
        if v < 0:
            raise ValueError("Price amount cannot be negative")
        
        # Limit to 2 decimal places for most currencies
        if v.as_tuple().exponent < -2:
            raise ValueError("Price cannot have more than 2 decimal places")
        
        return v
    
    class Config:
        """Pydantic configuration."""
        frozen = True
        json_encoders = {
            Decimal: str,
        }


class PricingTier(BaseModel):
    """A single tier in tiered pricing."""
    
    name: str = Field(..., min_length=1, max_length=50)
    min_quantity: int = Field(..., ge=0)
    max_quantity: Optional[int] = Field(None, ge=1)
    price_per_unit: Price
    
    @validator('max_quantity')
    def validate_max_quantity(cls, v: Optional[int], values: Dict) -> Optional[int]:
        """Validate max_quantity is greater than min_quantity."""
        if v is not None and 'min_quantity' in values:
            if v <= values['min_quantity']:
                raise ValueError("max_quantity must be greater than min_quantity")
        return v
    
    def applies_to_quantity(self, quantity: int) -> bool:
        """Check if this tier applies to the given quantity."""
        if quantity < self.min_quantity:
            return False
        if self.max_quantity is not None and quantity > self.max_quantity:
            return False
        return True
    
    class Config:
        """Pydantic configuration."""
        frozen = True


class Discount(BaseModel):
    """Discount value object."""
    
    name: str = Field(..., min_length=1, max_length=100)
    discount_type: str = Field(..., regex="^(percentage|fixed_amount)$")
    value: Decimal = Field(..., gt=0)
    currency: Optional[Currency] = None
    max_discount_amount: Optional[Price] = None
    
    @validator('currency')
    def validate_currency_for_fixed_amount(cls, v: Optional[Currency], values: Dict) -> Optional[Currency]:
        """Validate currency is provided for fixed amount discounts."""
        if values.get('discount_type') == 'fixed_amount' and v is None:
            raise ValueError("Currency is required for fixed amount discounts")
        return v
    
    @validator('value')
    def validate_discount_value(cls, v: Decimal, values: Dict) -> Decimal:
        """Validate discount value based on type."""
        discount_type = values.get('discount_type')
        if discount_type == 'percentage' and v > 100:
            raise ValueError("Percentage discount cannot exceed 100%")
        return v
    
    def apply_to_price(self, price: Price) -> Price:
        """Apply discount to a price."""
        if self.discount_type == 'percentage':
            discount_amount = price.amount * (self.value / 100)
        else:  # fixed_amount
            if self.currency != price.currency:
                raise ValueError("Discount and price currencies must match")
            discount_amount = self.value
        
        # Apply max discount limit if specified
        if self.max_discount_amount and discount_amount > self.max_discount_amount.amount:
            discount_amount = self.max_discount_amount.amount
        
        # Ensure final amount is not negative
        final_amount = max(Decimal('0'), price.amount - discount_amount)
        return Price(amount=final_amount, currency=price.currency)
    
    class Config:
        """Pydantic configuration."""
        frozen = True


class UsagePricing(BaseModel):
    """Usage-based pricing configuration."""
    
    unit: UsageUnit
    price_per_unit: Price
    minimum_charge: Optional[Price] = None
    included_units: int = Field(default=0, ge=0)
    
    def calculate_cost(self, usage: int) -> Price:
        """Calculate cost for given usage."""
        billable_usage = max(0, usage - self.included_units)
        cost = self.price_per_unit * Decimal(billable_usage)
        
        if self.minimum_charge and cost.amount < self.minimum_charge.amount:
            return self.minimum_charge
        
        return cost
    
    class Config:
        """Pydantic configuration."""
        frozen = True


class PricingModel(BaseModel):
    """Complete pricing model for a solution."""
    
    pricing_type: PricingType
    base_price: Optional[Price] = None
    billing_cycle: BillingCycle = Field(default=BillingCycle.ONE_TIME)
    
    # Tiered pricing
    tiers: List[PricingTier] = Field(default_factory=list)
    
    # Usage-based pricing
    usage_pricing: List[UsagePricing] = Field(default_factory=list)
    
    # Freemium limits
    free_tier_limits: Dict[str, int] = Field(default_factory=dict)
    
    # Trial period
    trial_period_days: int = Field(default=0, ge=0)
    
    def is_free(self) -> bool:
        """Check if pricing model is completely free."""
        return (
            self.pricing_type == PricingType.FREE or
            (self.base_price is not None and self.base_price.is_free())
        )
    
    def has_trial(self) -> bool:
        """Check if pricing model includes a trial period."""
        return self.trial_period_days > 0
    
    def calculate_price_for_quantity(self, quantity: int) -> Price:
        """Calculate price for a given quantity using tiered pricing."""
        if not self.tiers:
            if self.base_price:
                return self.base_price * Decimal(quantity)
            return Price.free()
        
        total_cost = Decimal('0')
        remaining_quantity = quantity
        
        for tier in sorted(self.tiers, key=lambda t: t.min_quantity):
            if remaining_quantity <= 0:
                break
            
            if tier.applies_to_quantity(quantity):
                tier_quantity = min(
                    remaining_quantity,
                    (tier.max_quantity or quantity) - tier.min_quantity + 1
                )
                total_cost += tier.price_per_unit.amount * tier_quantity
                remaining_quantity -= tier_quantity
        
        currency = self.tiers[0].price_per_unit.currency if self.tiers else Currency.USD
        return Price(amount=total_cost, currency=currency)
    
    def calculate_usage_cost(self, usage_data: Dict[UsageUnit, int]) -> Price:
        """Calculate cost based on usage data."""
        if not self.usage_pricing:
            return Price.free()
        
        total_cost = Decimal('0')
        currency = self.usage_pricing[0].price_per_unit.currency
        
        for usage_config in self.usage_pricing:
            usage_amount = usage_data.get(usage_config.unit, 0)
            cost = usage_config.calculate_cost(usage_amount)
            total_cost += cost.amount
        
        return Price(amount=total_cost, currency=currency)
    
    @validator('tiers')
    def validate_tiers(cls, v: List[PricingTier], values: Dict) -> List[PricingTier]:
        """Validate pricing tiers."""
        if values.get('pricing_type') == PricingType.TIERED and not v:
            raise ValueError("Tiered pricing requires at least one tier")
        
        # Check for overlapping tiers
        for i, tier1 in enumerate(v):
            for j, tier2 in enumerate(v[i+1:], i+1):
                if (tier1.min_quantity <= tier2.min_quantity <= (tier1.max_quantity or float('inf')) or
                    tier2.min_quantity <= tier1.min_quantity <= (tier2.max_quantity or float('inf'))):
                    raise ValueError(f"Overlapping tiers: {tier1.name} and {tier2.name}")
        
        return v
    
    @validator('base_price')
    def validate_base_price(cls, v: Optional[Price], values: Dict) -> Optional[Price]:
        """Validate base price based on pricing type."""
        pricing_type = values.get('pricing_type')
        
        if pricing_type in [PricingType.FIXED, PricingType.SUBSCRIPTION] and v is None:
            raise ValueError(f"{pricing_type} pricing requires a base price")
        
        if pricing_type == PricingType.FREE and v is not None and not v.is_free():
            raise ValueError("Free pricing cannot have a non-zero base price")
        
        return v
    
    class Config:
        """Pydantic configuration."""
        frozen = True