"""Cost optimization service for cloud resource management."""

import asyncio
import logging
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from pynomaly.domain.entities.cost_optimization import (
    CloudResource, CostBudget, CostOptimizationPlan, OptimizationRecommendation,
    ResourceType, CloudProvider, OptimizationStrategy, RecommendationType,
    RecommendationPriority, ResourceStatus, ResourceUsageMetrics, ResourceCost
)


logger = logging.getLogger(__name__)


class CostAnalysisEngine:
    """Engine for analyzing cost patterns and trends."""
    
    def __init__(self):
        self.cost_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def analyze_cost_trends(self, resources: List[CloudResource], days: int = 30) -> Dict[str, Any]:
        """Analyze cost trends across resources."""
        analysis = {
            "total_monthly_cost": 0.0,
            "projected_annual_cost": 0.0,
            "cost_by_resource_type": defaultdict(float),
            "cost_by_provider": defaultdict(float),
            "cost_by_environment": defaultdict(float),
            "top_cost_drivers": [],
            "cost_trends": {
                "7d_change": 0.0,
                "30d_change": 0.0,
                "growth_rate": 0.0
            },
            "inefficiency_indicators": {
                "idle_resources": 0,
                "underutilized_resources": 0,
                "total_waste": 0.0
            }
        }
        
        total_cost = 0.0
        cost_trends_7d = []
        cost_trends_30d = []
        idle_count = 0
        underutilized_count = 0
        total_waste = 0.0
        
        for resource in resources:
            monthly_cost = resource.cost_info.monthly_cost
            total_cost += monthly_cost
            
            # Aggregate by dimensions
            analysis["cost_by_resource_type"][resource.resource_type.value] += monthly_cost
            analysis["cost_by_provider"][resource.provider.value] += monthly_cost
            analysis["cost_by_environment"][resource.environment] += monthly_cost
            
            # Track trends
            cost_trends_7d.append(resource.cost_info.cost_trend_7d)
            cost_trends_30d.append(resource.cost_info.cost_trend_30d)
            
            # Identify waste
            if resource.is_idle():
                idle_count += 1
                total_waste += monthly_cost
            elif resource.usage_metrics.is_underutilized():
                underutilized_count += 1
                efficiency = resource.usage_metrics.get_efficiency_score()
                total_waste += monthly_cost * (1 - efficiency)
        
        analysis["total_monthly_cost"] = total_cost
        analysis["projected_annual_cost"] = total_cost * 12
        
        # Calculate trends (handle empty lists)
        if cost_trends_7d:
            analysis["cost_trends"]["7d_change"] = statistics.mean(cost_trends_7d) if cost_trends_7d else 0.0
        if cost_trends_30d:
            analysis["cost_trends"]["30d_change"] = statistics.mean(cost_trends_30d) if cost_trends_30d else 0.0
            analysis["cost_trends"]["growth_rate"] = analysis["cost_trends"]["30d_change"] / 30  # Daily growth rate
        
        # Top cost drivers
        resource_costs = [(r, r.cost_info.monthly_cost) for r in resources]
        resource_costs.sort(key=lambda x: x[1], reverse=True)
        analysis["top_cost_drivers"] = [
            {
                "resource_id": str(r.resource_id),
                "name": r.name,
                "resource_type": r.resource_type.value,
                "monthly_cost": cost,
                "efficiency_score": r.usage_metrics.get_efficiency_score()
            }
            for r, cost in resource_costs[:10]
        ]
        
        # Inefficiency indicators
        analysis["inefficiency_indicators"]["idle_resources"] = idle_count
        analysis["inefficiency_indicators"]["underutilized_resources"] = underutilized_count
        analysis["inefficiency_indicators"]["total_waste"] = total_waste
        
        return analysis
    
    def predict_future_costs(self, resources: List[CloudResource], months_ahead: int = 12) -> Dict[str, Any]:
        """Predict future costs based on usage patterns."""
        predictions = {
            "monthly_predictions": [],
            "total_predicted_cost": 0.0,
            "confidence_interval": (0.0, 0.0),
            "key_assumptions": []
        }
        
        if not resources:
            return predictions
        
        # Extract features for prediction
        features = []
        targets = []
        
        for resource in resources:
            if resource.cost_info.monthly_cost > 0:
                feature_vector = [
                    resource.usage_metrics.cpu_utilization_avg,
                    resource.usage_metrics.memory_utilization_avg,
                    resource.usage_metrics.requests_per_second,
                    resource.cost_info.cost_trend_7d,
                    resource.cost_info.cost_trend_30d,
                    float(resource.resource_type.value == "compute"),
                    float(resource.resource_type.value == "storage"),
                    float(resource.environment == "production"),
                    resource.cpu_cores,
                    resource.memory_gb
                ]
                features.append(feature_vector)
                targets.append(resource.cost_info.monthly_cost)
        
        if len(features) < 10:  # Not enough data for ML prediction
            # Use simple trend extrapolation
            current_total = sum(r.cost_info.monthly_cost for r in resources)
            trend_values = [r.cost_info.cost_trend_30d for r in resources if r.cost_info.cost_trend_30d != 0]
            avg_growth_rate = statistics.mean(trend_values) if trend_values else 0.0
            
            for month in range(1, months_ahead + 1):
                predicted_cost = current_total * (1 + avg_growth_rate) ** month
                predictions["monthly_predictions"].append({
                    "month": month,
                    "predicted_cost": predicted_cost,
                    "confidence": 0.6
                })
            
            predictions["total_predicted_cost"] = sum(p["predicted_cost"] for p in predictions["monthly_predictions"])
            predictions["key_assumptions"].append("Used trend extrapolation due to limited data")
        
        else:
            # Use ML prediction
            try:
                X = np.array(features)
                y = np.array(targets)
                
                # Train the model
                X_scaled = self.scaler.fit_transform(X)
                self.cost_predictor.fit(X_scaled, y)
                self.is_trained = True
                
                # Make predictions
                total_predicted = 0.0
                for month in range(1, months_ahead + 1):
                    # Adjust features for future month (simplified)
                    future_features = X.copy()
                    growth_factor = 1 + (avg_growth_rate or 0.02) * month  # Default 2% monthly growth
                    future_features[:, 2] *= growth_factor  # Adjust requests per second
                    
                    future_features_scaled = self.scaler.transform(future_features)
                    month_predictions = self.cost_predictor.predict(future_features_scaled)
                    month_total = np.sum(month_predictions)
                    total_predicted += month_total
                    
                    predictions["monthly_predictions"].append({
                        "month": month,
                        "predicted_cost": month_total,
                        "confidence": 0.8
                    })
                
                predictions["total_predicted_cost"] = total_predicted
                predictions["confidence_interval"] = (total_predicted * 0.85, total_predicted * 1.15)
                predictions["key_assumptions"].append("Used machine learning model with historical usage patterns")
                
            except Exception as e:
                logger.error(f"Error in ML cost prediction: {e}")
                # Fallback to trend extrapolation
                return self.predict_future_costs(resources[:9], months_ahead)  # Reduce data to trigger fallback
        
        return predictions
    
    def identify_cost_anomalies(self, resources: List[CloudResource]) -> List[Dict[str, Any]]:
        """Identify cost anomalies and unusual spending patterns."""
        anomalies = []
        
        if not resources:
            return anomalies
        
        # Calculate cost statistics
        costs = [r.cost_info.monthly_cost for r in resources if r.cost_info.monthly_cost > 0]
        if not costs:
            return anomalies
        
        mean_cost = statistics.mean(costs)
        std_cost = statistics.stdev(costs) if len(costs) > 1 else 0
        
        # Identify outliers using IQR method
        costs_sorted = sorted(costs)
        q1 = costs_sorted[len(costs_sorted) // 4]
        q3 = costs_sorted[3 * len(costs_sorted) // 4]
        iqr = q3 - q1
        outlier_threshold = q3 + 1.5 * iqr
        
        for resource in resources:
            cost = resource.cost_info.monthly_cost
            
            # High cost outlier
            if cost > outlier_threshold:
                anomalies.append({
                    "resource_id": str(resource.resource_id),
                    "anomaly_type": "high_cost_outlier",
                    "severity": "high",
                    "description": f"Resource cost ${cost:.2f} is {cost/mean_cost:.1f}x the average",
                    "monthly_cost": cost,
                    "average_cost": mean_cost,
                    "suggested_action": "Review resource sizing and usage patterns"
                })
            
            # Cost trend anomalies
            if resource.cost_info.cost_trend_7d > 0.5:  # 50% increase in 7 days
                anomalies.append({
                    "resource_id": str(resource.resource_id),
                    "anomaly_type": "cost_spike",
                    "severity": "medium",
                    "description": f"Cost increased by {resource.cost_info.cost_trend_7d*100:.1f}% in the last 7 days",
                    "trend_7d": resource.cost_info.cost_trend_7d,
                    "suggested_action": "Investigate usage changes or scaling events"
                })
            
            # Idle but expensive resources
            if resource.is_idle() and cost > mean_cost:
                anomalies.append({
                    "resource_id": str(resource.resource_id),
                    "anomaly_type": "expensive_idle_resource",
                    "severity": "high",
                    "description": f"Idle resource still costing ${cost:.2f}/month",
                    "idle_hours": resource.calculate_idle_time().total_seconds() / 3600,
                    "suggested_action": "Consider terminating or downsizing the resource"
                })
            
            # Poor cost efficiency
            efficiency = resource.usage_metrics.get_efficiency_score()
            cost_efficiency = resource.cost_info.get_cost_per_efficiency_unit(efficiency)
            if efficiency < 0.3 and cost > mean_cost * 0.5:
                anomalies.append({
                    "resource_id": str(resource.resource_id),
                    "anomaly_type": "poor_cost_efficiency",
                    "severity": "medium",
                    "description": f"Low efficiency ({efficiency:.2f}) for significant cost (${cost:.2f})",
                    "efficiency_score": efficiency,
                    "cost_per_efficiency": cost_efficiency,
                    "suggested_action": "Optimize resource configuration or workload"
                })
        
        return sorted(anomalies, key=lambda x: {"high": 3, "medium": 2, "low": 1}[x["severity"]], reverse=True)


class RecommendationEngine:
    """Engine for generating cost optimization recommendations."""
    
    def __init__(self):
        self.rightsizing_models = {}
        self.recommendation_cache = {}
        
    async def generate_recommendations(
        self, 
        resources: List[CloudResource], 
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    ) -> List[OptimizationRecommendation]:
        """Generate cost optimization recommendations."""
        recommendations = []
        
        for resource in resources:
            if not resource.can_be_optimized():
                continue
            
            # Generate different types of recommendations
            recommendations.extend(await self._generate_rightsizing_recommendations(resource, strategy))
            recommendations.extend(await self._generate_scheduling_recommendations(resource, strategy))
            recommendations.extend(await self._generate_instance_type_recommendations(resource, strategy))
            recommendations.extend(await self._generate_storage_optimization_recommendations(resource, strategy))
            recommendations.extend(await self._generate_idle_cleanup_recommendations(resource, strategy))
        
        # Filter and rank recommendations
        filtered_recs = self._filter_recommendations(recommendations, strategy)
        return sorted(filtered_recs, key=lambda r: r.get_priority_score(), reverse=True)
    
    async def _generate_rightsizing_recommendations(
        self, 
        resource: CloudResource, 
        strategy: OptimizationStrategy
    ) -> List[OptimizationRecommendation]:
        """Generate rightsizing recommendations."""
        recommendations = []
        
        metrics = resource.usage_metrics
        current_cost = resource.cost_info.monthly_cost
        
        # Downsize if underutilized
        if metrics.is_underutilized():
            # Calculate suggested specs
            suggested_cpu = max(1, int(resource.cpu_cores * metrics.cpu_utilization_p95 * 1.2))  # 20% buffer
            suggested_memory = max(1, resource.memory_gb * metrics.memory_utilization_p95 * 1.2)
            
            # Estimate cost savings (simplified calculation)
            cpu_reduction = (resource.cpu_cores - suggested_cpu) / resource.cpu_cores
            memory_reduction = (resource.memory_gb - suggested_memory) / resource.memory_gb
            avg_reduction = (cpu_reduction + memory_reduction) / 2
            
            projected_cost = current_cost * (1 - avg_reduction * 0.8)  # 80% of reduction translates to cost savings
            monthly_savings = current_cost - projected_cost
            
            if monthly_savings > 10:  # Only recommend if savings > $10/month
                priority = RecommendationPriority.HIGH if monthly_savings > 100 else RecommendationPriority.MEDIUM
                
                recommendation = OptimizationRecommendation(
                    resource_id=resource.resource_id,
                    recommendation_type=RecommendationType.RIGHTSIZING,
                    priority=priority,
                    title=f"Downsize {resource.name}",
                    description=f"Reduce CPU from {resource.cpu_cores} to {suggested_cpu} cores and memory from {resource.memory_gb:.1f} to {suggested_memory:.1f} GB",
                    rationale=f"Resource is underutilized: CPU {metrics.cpu_utilization_avg*100:.1f}% avg, Memory {metrics.memory_utilization_avg*100:.1f}% avg",
                    current_monthly_cost=current_cost,
                    projected_monthly_cost=projected_cost,
                    monthly_savings=monthly_savings,
                    annual_savings=monthly_savings * 12,
                    implementation_cost=0.0,  # Assuming no cost for rightsizing
                    performance_impact="minimal",
                    risk_level="low",
                    confidence_score=0.85,
                    action_required=f"Resize instance to {suggested_cpu} vCPU, {suggested_memory:.1f} GB RAM",
                    automation_possible=True,
                    estimated_implementation_time="5-10 minutes"
                )
                recommendations.append(recommendation)
        
        # Upsize if overutilized
        elif metrics.is_overutilized():
            # Calculate suggested specs
            suggested_cpu = int(resource.cpu_cores * 1.5)  # 50% increase
            suggested_memory = resource.memory_gb * 1.3  # 30% increase
            
            # Estimate cost increase
            cpu_increase = (suggested_cpu - resource.cpu_cores) / resource.cpu_cores
            memory_increase = (suggested_memory - resource.memory_gb) / resource.memory_gb
            avg_increase = (cpu_increase + memory_increase) / 2
            
            projected_cost = current_cost * (1 + avg_increase * 0.8)
            monthly_increase = projected_cost - current_cost
            
            recommendation = OptimizationRecommendation(
                resource_id=resource.resource_id,
                recommendation_type=RecommendationType.RIGHTSIZING,
                priority=RecommendationPriority.HIGH,
                title=f"Upsize {resource.name}",
                description=f"Increase CPU from {resource.cpu_cores} to {suggested_cpu} cores and memory from {resource.memory_gb:.1f} to {suggested_memory:.1f} GB",
                rationale=f"Resource is overutilized: CPU P95 {metrics.cpu_utilization_p95*100:.1f}%, Memory P95 {metrics.memory_utilization_p95*100:.1f}%",
                current_monthly_cost=current_cost,
                projected_monthly_cost=projected_cost,
                monthly_savings=-monthly_increase,  # Negative savings (cost increase)
                annual_savings=-monthly_increase * 12,
                implementation_cost=0.0,
                performance_impact="positive",
                risk_level="low",
                confidence_score=0.9,
                action_required=f"Resize instance to {suggested_cpu} vCPU, {suggested_memory:.1f} GB RAM",
                automation_possible=True,
                estimated_implementation_time="5-10 minutes"
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    async def _generate_scheduling_recommendations(
        self, 
        resource: CloudResource, 
        strategy: OptimizationStrategy
    ) -> List[OptimizationRecommendation]:
        """Generate scheduling recommendations."""
        recommendations = []
        
        # Only recommend for non-production environments
        if resource.environment == "production":
            return recommendations
        
        current_cost = resource.cost_info.monthly_cost
        
        # Stop during off-hours
        if resource.environment in ["development", "test", "staging"]:
            # Assume 12 hours/day, 5 days/week usage for dev/test
            # Assume 16 hours/day, 7 days/week for staging
            if resource.environment in ["development", "test"]:
                hours_per_week = 12 * 5  # 60 hours
                potential_savings_percent = (168 - hours_per_week) / 168  # ~64% savings
            else:  # staging
                hours_per_week = 16 * 7  # 112 hours
                potential_savings_percent = (168 - hours_per_week) / 168  # ~33% savings
            
            monthly_savings = current_cost * potential_savings_percent
            
            if monthly_savings > 5:  # Only recommend if savings > $5/month
                priority = RecommendationPriority.HIGH if monthly_savings > 50 else RecommendationPriority.MEDIUM
                
                recommendation = OptimizationRecommendation(
                    resource_id=resource.resource_id,
                    recommendation_type=RecommendationType.SCHEDULING,
                    priority=priority,
                    title=f"Schedule {resource.name} for off-hours shutdown",
                    description=f"Automatically stop resource during off-hours in {resource.environment} environment",
                    rationale=f"Non-production resource can be stopped when not in use",
                    current_monthly_cost=current_cost,
                    projected_monthly_cost=current_cost - monthly_savings,
                    monthly_savings=monthly_savings,
                    annual_savings=monthly_savings * 12,
                    implementation_cost=0.0,
                    performance_impact="none",
                    risk_level="low",
                    confidence_score=0.9,
                    action_required="Configure automated start/stop schedule",
                    automation_possible=True,
                    estimated_implementation_time="15-30 minutes",
                    prerequisites=["Verify no 24/7 dependencies", "Configure monitoring"]
                )
                recommendations.append(recommendation)
        
        return recommendations
    
    async def _generate_instance_type_recommendations(
        self, 
        resource: CloudResource, 
        strategy: OptimizationStrategy
    ) -> List[OptimizationRecommendation]:
        """Generate instance type optimization recommendations."""
        recommendations = []
        
        if resource.resource_type != ResourceType.COMPUTE:
            return recommendations
        
        current_cost = resource.cost_info.monthly_cost
        metrics = resource.usage_metrics
        
        # Recommend spot instances for fault-tolerant workloads
        if (resource.environment in ["development", "test", "staging"] and 
            "spot" not in resource.instance_type.lower()):
            
            # Spot instances typically 60-90% cheaper
            spot_savings_percent = 0.75  # 75% savings
            monthly_savings = current_cost * spot_savings_percent
            
            if monthly_savings > 10:
                recommendation = OptimizationRecommendation(
                    resource_id=resource.resource_id,
                    recommendation_type=RecommendationType.SPOT_INSTANCES,
                    priority=RecommendationPriority.MEDIUM,
                    title=f"Migrate {resource.name} to spot instances",
                    description="Use spot instances for significant cost savings in non-critical environment",
                    rationale="Non-production workload suitable for spot instances",
                    current_monthly_cost=current_cost,
                    projected_monthly_cost=current_cost - monthly_savings,
                    monthly_savings=monthly_savings,
                    annual_savings=monthly_savings * 12,
                    implementation_cost=0.0,
                    performance_impact="minimal",
                    risk_level="medium",
                    confidence_score=0.8,
                    action_required="Migrate to spot instance type",
                    automation_possible=False,
                    estimated_implementation_time="30-60 minutes",
                    prerequisites=["Ensure workload is fault-tolerant", "Test spot instance behavior"]
                )
                recommendations.append(recommendation)
        
        # Recommend reserved instances for stable workloads
        if (resource.environment == "production" and 
            resource.cost_info.billing_model == "on_demand" and
            current_cost > 100):  # Only for significant costs
            
            # Reserved instances typically 30-60% cheaper
            reserved_savings_percent = 0.4  # 40% savings
            monthly_savings = current_cost * reserved_savings_percent
            implementation_cost = current_cost * 12 * 0.6  # Upfront payment
            
            recommendation = OptimizationRecommendation(
                resource_id=resource.resource_id,
                recommendation_type=RecommendationType.RESERVED_INSTANCES,
                priority=RecommendationPriority.MEDIUM,
                title=f"Purchase reserved instance for {resource.name}",
                description="Convert to reserved instance for long-term cost savings",
                rationale="Stable production workload suitable for 1-year commitment",
                current_monthly_cost=current_cost,
                projected_monthly_cost=current_cost - monthly_savings,
                monthly_savings=monthly_savings,
                annual_savings=monthly_savings * 12,
                implementation_cost=implementation_cost,
                payback_period_days=int(implementation_cost / monthly_savings * 30),
                performance_impact="none",
                risk_level="low",
                confidence_score=0.9,
                action_required="Purchase 1-year reserved instance",
                automation_possible=False,
                estimated_implementation_time="5-10 minutes",
                prerequisites=["Confirm 1-year usage commitment", "Budget approval for upfront cost"]
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    async def _generate_storage_optimization_recommendations(
        self, 
        resource: CloudResource, 
        strategy: OptimizationStrategy
    ) -> List[OptimizationRecommendation]:
        """Generate storage optimization recommendations."""
        recommendations = []
        
        if resource.resource_type != ResourceType.STORAGE:
            return recommendations
        
        current_cost = resource.cost_info.monthly_cost
        
        # Storage tier optimization
        if resource.storage_gb > 100:  # Only for significant storage
            # Recommend moving to cheaper storage tier
            tier_savings_percent = 0.3  # 30% savings by moving to infrequent access
            monthly_savings = current_cost * tier_savings_percent
            
            if monthly_savings > 5:
                recommendation = OptimizationRecommendation(
                    resource_id=resource.resource_id,
                    recommendation_type=RecommendationType.STORAGE_OPTIMIZATION,
                    priority=RecommendationPriority.MEDIUM,
                    title=f"Optimize storage tier for {resource.name}",
                    description="Move infrequently accessed data to cheaper storage tier",
                    rationale="Large storage volume suitable for tiering optimization",
                    current_monthly_cost=current_cost,
                    projected_monthly_cost=current_cost - monthly_savings,
                    monthly_savings=monthly_savings,
                    annual_savings=monthly_savings * 12,
                    implementation_cost=0.0,
                    performance_impact="minimal",
                    risk_level="low",
                    confidence_score=0.8,
                    action_required="Configure lifecycle policies for storage tiering",
                    automation_possible=True,
                    estimated_implementation_time="30-45 minutes"
                )
                recommendations.append(recommendation)
        
        return recommendations
    
    async def _generate_idle_cleanup_recommendations(
        self, 
        resource: CloudResource, 
        strategy: OptimizationStrategy
    ) -> List[OptimizationRecommendation]:
        """Generate idle resource cleanup recommendations."""
        recommendations = []
        
        if not resource.is_idle():
            return recommendations
        
        current_cost = resource.cost_info.monthly_cost
        idle_hours = resource.calculate_idle_time().total_seconds() / 3600
        
        if idle_hours > 72:  # Idle for more than 3 days
            priority = RecommendationPriority.CRITICAL if current_cost > 100 else RecommendationPriority.HIGH
            
            recommendation = OptimizationRecommendation(
                resource_id=resource.resource_id,
                recommendation_type=RecommendationType.IDLE_RESOURCE_CLEANUP,
                priority=priority,
                title=f"Terminate idle resource {resource.name}",
                description=f"Resource has been idle for {idle_hours:.1f} hours, costing ${current_cost:.2f}/month",
                rationale=f"Resource last accessed {resource.last_accessed} and is consuming unnecessary costs",
                current_monthly_cost=current_cost,
                projected_monthly_cost=0.0,
                monthly_savings=current_cost,
                annual_savings=current_cost * 12,
                implementation_cost=0.0,
                performance_impact="none",
                risk_level="low",
                confidence_score=0.95,
                action_required="Verify no dependencies and terminate resource",
                automation_possible=False,  # Manual verification required
                estimated_implementation_time="10-15 minutes",
                prerequisites=["Verify no hidden dependencies", "Backup any important data"]
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def _filter_recommendations(
        self, 
        recommendations: List[OptimizationRecommendation], 
        strategy: OptimizationStrategy
    ) -> List[OptimizationRecommendation]:
        """Filter recommendations based on strategy."""
        filtered = []
        
        strategy_filters = {
            OptimizationStrategy.AGGRESSIVE: {
                "min_savings": 5,
                "max_risk": "high",
                "min_confidence": 0.6
            },
            OptimizationStrategy.BALANCED: {
                "min_savings": 10,
                "max_risk": "medium",
                "min_confidence": 0.7
            },
            OptimizationStrategy.CONSERVATIVE: {
                "min_savings": 25,
                "max_risk": "low",
                "min_confidence": 0.8
            },
            OptimizationStrategy.COST_FIRST: {
                "min_savings": 1,
                "max_risk": "high",
                "min_confidence": 0.5
            },
            OptimizationStrategy.PERFORMANCE_FIRST: {
                "min_savings": 50,
                "max_risk": "low",
                "min_confidence": 0.9
            }
        }
        
        filters = strategy_filters.get(strategy, strategy_filters[OptimizationStrategy.BALANCED])
        risk_levels = {"low": 1, "medium": 2, "high": 3}
        max_risk_level = risk_levels[filters["max_risk"]]
        
        for rec in recommendations:
            if (rec.monthly_savings >= filters["min_savings"] and
                risk_levels.get(rec.risk_level, 2) <= max_risk_level and
                rec.confidence_score >= filters["min_confidence"]):
                filtered.append(rec)
        
        return filtered


class CostOptimizationService:
    """Main service for cost optimization and resource management."""
    
    def __init__(self):
        self.resources: Dict[UUID, CloudResource] = {}
        self.budgets: Dict[UUID, CostBudget] = {}
        self.optimization_plans: Dict[UUID, CostOptimizationPlan] = {}
        
        self.analysis_engine = CostAnalysisEngine()
        self.recommendation_engine = RecommendationEngine()
        
        # Metrics and caching
        self.metrics = {
            "total_resources": 0,
            "total_monthly_cost": 0.0,
            "total_savings_identified": 0.0,
            "recommendations_generated": 0,
            "recommendations_implemented": 0
        }
        
    async def register_resource(self, resource: CloudResource) -> bool:
        """Register a resource for cost optimization."""
        try:
            self.resources[resource.resource_id] = resource
            self.metrics["total_resources"] = len(self.resources)
            self.metrics["total_monthly_cost"] = sum(r.cost_info.monthly_cost for r in self.resources.values())
            
            logger.info(f"Registered resource {resource.name} ({resource.resource_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error registering resource: {e}")
            return False
    
    async def update_resource_metrics(self, resource_id: UUID, metrics: ResourceUsageMetrics) -> bool:
        """Update usage metrics for a resource."""
        try:
            resource = self.resources.get(resource_id)
            if not resource:
                return False
            
            resource.usage_metrics = metrics
            resource.last_accessed = datetime.utcnow()
            
            # Update resource status based on metrics
            if metrics.is_underutilized():
                resource.status = ResourceStatus.UNDERUTILIZED
            elif metrics.is_overutilized():
                resource.status = ResourceStatus.OVERUTILIZED
            elif resource.is_idle():
                resource.status = ResourceStatus.IDLE
            else:
                resource.status = ResourceStatus.ACTIVE
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating resource metrics: {e}")
            return False
    
    async def analyze_costs(self, tenant_id: Optional[UUID] = None) -> Dict[str, Any]:
        """Perform comprehensive cost analysis."""
        try:
            # Filter resources by tenant if specified
            resources = list(self.resources.values())
            if tenant_id:
                resources = [r for r in resources if r.tenant_id == tenant_id]
            
            analysis = self.analysis_engine.analyze_cost_trends(resources)
            
            # Add anomaly detection
            anomalies = self.analysis_engine.identify_cost_anomalies(resources)
            analysis["cost_anomalies"] = anomalies
            
            # Add future cost predictions
            predictions = self.analysis_engine.predict_future_costs(resources)
            analysis["cost_predictions"] = predictions
            
            # Add optimization opportunities
            optimization_potential = sum(r.get_optimization_potential() for r in resources)
            analysis["optimization_potential"] = {
                "total_resources": len(resources),
                "optimizable_resources": len([r for r in resources if r.can_be_optimized()]),
                "total_potential_score": optimization_potential,
                "avg_potential_score": optimization_potential / len(resources) if resources else 0
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in cost analysis: {e}")
            return {}
    
    async def generate_optimization_plan(
        self, 
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        tenant_id: Optional[UUID] = None,
        target_savings_percent: float = 0.2
    ) -> CostOptimizationPlan:
        """Generate a comprehensive optimization plan."""
        try:
            # Filter resources
            resources = list(self.resources.values())
            if tenant_id:
                resources = [r for r in resources if r.tenant_id == tenant_id]
            
            # Generate recommendations
            recommendations = await self.recommendation_engine.generate_recommendations(resources, strategy)
            self.metrics["recommendations_generated"] += len(recommendations)
            
            # Create optimization plan
            plan = CostOptimizationPlan(
                name=f"Cost Optimization Plan - {strategy.value.title()}",
                description=f"Automated cost optimization plan using {strategy.value} strategy",
                strategy=strategy,
                tenant_id=tenant_id,
                target_cost_reduction_percent=target_savings_percent,
                target_monthly_savings=sum(r.cost_info.monthly_cost for r in resources) * target_savings_percent
            )
            
            # Add recommendations to plan
            for rec in recommendations:
                plan.add_recommendation(rec)
            
            # Calculate implementation timeline
            quick_wins = plan.get_quick_wins()
            plan.estimated_implementation_days = len(quick_wins) + (len(recommendations) - len(quick_wins)) * 2
            
            # Store plan
            self.optimization_plans[plan.plan_id] = plan
            self.metrics["total_savings_identified"] += plan.total_potential_savings
            
            logger.info(f"Generated optimization plan with {len(recommendations)} recommendations, "
                       f"potential savings: ${plan.total_potential_savings:.2f}/year")
            
            return plan
            
        except Exception as e:
            logger.error(f"Error generating optimization plan: {e}")
            return CostOptimizationPlan()
    
    async def implement_recommendation(self, recommendation_id: UUID) -> Dict[str, Any]:
        """Implement a specific recommendation."""
        # This would integrate with cloud provider APIs
        # For now, return a simulation
        result = {
            "recommendation_id": str(recommendation_id),
            "status": "simulated",
            "implementation_time": "5 minutes",
            "actual_savings": 0.0,
            "success": True,
            "message": "Recommendation implementation simulated successfully"
        }
        
        self.metrics["recommendations_implemented"] += 1
        logger.info(f"Simulated implementation of recommendation {recommendation_id}")
        
        return result
    
    async def create_budget(self, budget: CostBudget) -> bool:
        """Create a cost budget."""
        try:
            self.budgets[budget.budget_id] = budget
            logger.info(f"Created budget {budget.name} with ${budget.monthly_limit:.2f}/month limit")
            return True
            
        except Exception as e:
            logger.error(f"Error creating budget: {e}")
            return False
    
    async def check_budget_alerts(self) -> List[Dict[str, Any]]:
        """Check for budget alert conditions."""
        alerts = []
        
        for budget in self.budgets.values():
            triggered_thresholds = budget.get_triggered_alerts()
            
            for threshold in triggered_thresholds:
                alerts.append({
                    "budget_id": str(budget.budget_id),
                    "budget_name": budget.name,
                    "alert_type": "budget_threshold",
                    "threshold": threshold,
                    "current_utilization": budget.get_monthly_utilization(),
                    "current_spend": budget.current_monthly_spend,
                    "budget_limit": budget.monthly_limit,
                    "days_until_exhausted": budget.days_until_budget_exhausted(),
                    "severity": "critical" if threshold >= 1.0 else "high" if threshold >= 0.9 else "medium"
                })
        
        return alerts
    
    async def get_resource_summary(self, tenant_id: Optional[UUID] = None) -> Dict[str, Any]:
        """Get summary of resources and their optimization status."""
        resources = list(self.resources.values())
        if tenant_id:
            resources = [r for r in resources if r.tenant_id == tenant_id]
        
        summary = {
            "total_resources": len(resources),
            "total_monthly_cost": sum(r.cost_info.monthly_cost for r in resources),
            "resource_breakdown": {
                "by_type": defaultdict(int),
                "by_status": defaultdict(int),
                "by_environment": defaultdict(int),
                "by_provider": defaultdict(int)
            },
            "optimization_summary": {
                "optimizable_resources": 0,
                "idle_resources": 0,
                "underutilized_resources": 0,
                "overutilized_resources": 0,
                "total_potential_savings": 0.0
            }
        }
        
        for resource in resources:
            # Resource breakdown
            summary["resource_breakdown"]["by_type"][resource.resource_type.value] += 1
            summary["resource_breakdown"]["by_status"][resource.status.value] += 1
            summary["resource_breakdown"]["by_environment"][resource.environment] += 1
            summary["resource_breakdown"]["by_provider"][resource.provider.value] += 1
            
            # Optimization summary
            if resource.can_be_optimized():
                summary["optimization_summary"]["optimizable_resources"] += 1
            
            if resource.is_idle():
                summary["optimization_summary"]["idle_resources"] += 1
            elif resource.usage_metrics.is_underutilized():
                summary["optimization_summary"]["underutilized_resources"] += 1
            elif resource.usage_metrics.is_overutilized():
                summary["optimization_summary"]["overutilized_resources"] += 1
            
            # Estimate potential savings (simplified)
            if resource.is_idle():
                summary["optimization_summary"]["total_potential_savings"] += resource.cost_info.monthly_cost
            elif resource.usage_metrics.is_underutilized():
                efficiency = resource.usage_metrics.get_efficiency_score()
                summary["optimization_summary"]["total_potential_savings"] += resource.cost_info.monthly_cost * (1 - efficiency) * 0.5
        
        return summary
    
    async def get_service_metrics(self) -> Dict[str, Any]:
        """Get service performance metrics."""
        return {
            **self.metrics,
            "optimization_plans": len(self.optimization_plans),
            "budgets": len(self.budgets),
            "avg_cost_per_resource": self.metrics["total_monthly_cost"] / max(1, self.metrics["total_resources"]),
            "savings_rate": (self.metrics["total_savings_identified"] / max(1, self.metrics["total_monthly_cost"] * 12)) * 100
        }