#!/usr/bin/env python3
"""
Beta User Onboarding Email Templates for Pynomaly

Comprehensive email template system for beta user onboarding workflows,
including welcome emails, feature announcements, and engagement campaigns.
"""

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

class EmailType(Enum):
    WELCOME = "welcome"
    INVITATION = "invitation"
    ONBOARDING_STEP = "onboarding_step"
    FEATURE_ANNOUNCEMENT = "feature_announcement"
    FEEDBACK_REQUEST = "feedback_request"
    TIPS_AND_TRICKS = "tips_and_tricks"
    REMINDER = "reminder"
    THANK_YOU = "thank_you"

@dataclass
class EmailTemplate:
    template_id: str
    email_type: EmailType
    subject: str
    html_body: str
    text_body: str
    variables: List[str]
    tags: List[str]
    priority: str = "normal"

class BetaEmailTemplateSystem:
    """Beta user onboarding email template system."""
    
    def __init__(self):
        self.templates = {}
        self.create_email_templates()
    
    def create_email_templates(self):
        """Create all beta onboarding email templates."""
        
        # Beta Invitation Email
        self.templates["beta_invitation"] = EmailTemplate(
            template_id="beta_invitation",
            email_type=EmailType.INVITATION,
            subject="🧪 You're Invited to the Pynomaly Beta Program!",
            html_body="""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Welcome to Pynomaly Beta</title>
                <style>
                    body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                    .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                    .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-align: center; padding: 30px; border-radius: 8px 8px 0 0; }
                    .content { background: #f9f9f9; padding: 30px; border-radius: 0 0 8px 8px; }
                    .cta-button { display: inline-block; background: #4CAF50; color: white; padding: 15px 30px; text-decoration: none; border-radius: 5px; font-weight: bold; margin: 20px 0; }
                    .feature-list { background: white; padding: 20px; border-radius: 5px; margin: 20px 0; }
                    .footer { text-align: center; color: #666; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🧪 Welcome to Pynomaly Beta!</h1>
                        <p>Exclusive Early Access to Advanced Anomaly Detection</p>
                    </div>
                    
                    <div class="content">
                        <p>Hi {{user_name}},</p>
                        
                        <p>Congratulations! You've been selected to join the exclusive Pynomaly Beta Program. As a beta user, you'll get early access to our cutting-edge anomaly detection platform and help shape its future.</p>
                        
                        <div class="feature-list">
                            <h3>🎯 What You Get as a Beta User:</h3>
                            <ul>
                                <li><strong>Early Access:</strong> Be the first to try new features</li>
                                <li><strong>Direct Impact:</strong> Your feedback shapes product development</li>
                                <li><strong>Premium Support:</strong> Direct line to our engineering team</li>
                                <li><strong>Extended Trial:</strong> Full access throughout the beta period</li>
                                <li><strong>Special Pricing:</strong> Exclusive discounts when we launch</li>
                            </ul>
                        </div>
                        
                        <div style="text-align: center;">
                            <a href="{{registration_url}}" class="cta-button">🚀 Get Started Now</a>
                        </div>
                        
                        <h3>🏢 Perfect for {{industry}} Use Cases:</h3>
                        <p>{{use_case_description}}</p>
                        
                        <div style="background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0;">
                            <h4>📚 Getting Started Resources:</h4>
                            <ul>
                                <li><a href="{{docs_url}}">📖 Beta Documentation</a></li>
                                <li><a href="{{api_docs_url}}">🔧 API Reference</a></li>
                                <li><a href="{{examples_url}}">💡 Industry Examples</a></li>
                                <li><a href="{{support_url}}">💬 Beta Support Channel</a></li>
                            </ul>
                        </div>
                        
                        <p>Ready to detect anomalies like never before? Click the button above to create your account and start exploring!</p>
                        
                        <p>Best regards,<br>
                        The Pynomaly Team</p>
                    </div>
                    
                    <div class="footer">
                        <p>Questions? Reply to this email or contact us at <a href="mailto:{{support_email}}">{{support_email}}</a></p>
                        <p>Pynomaly Beta Program | <a href="{{unsubscribe_url}}">Unsubscribe</a></p>
                    </div>
                </div>
            </body>
            </html>
            """,
            text_body="""
            🧪 Welcome to Pynomaly Beta!

            Hi {{user_name}},

            Congratulations! You've been selected to join the exclusive Pynomaly Beta Program.

            As a beta user, you'll get:
            • Early access to new features
            • Direct impact on product development
            • Premium support from our engineering team
            • Extended trial period
            • Special pricing when we launch

            Get started: {{registration_url}}

            Perfect for {{industry}} use cases:
            {{use_case_description}}

            Resources:
            • Documentation: {{docs_url}}
            • API Reference: {{api_docs_url}}
            • Examples: {{examples_url}}
            • Support: {{support_url}}

            Questions? Contact us at {{support_email}}

            Best regards,
            The Pynomaly Team
            """,
            variables=["user_name", "industry", "use_case_description", "registration_url", "docs_url", "api_docs_url", "examples_url", "support_url", "support_email", "unsubscribe_url"],
            tags=["beta", "invitation", "onboarding"]
        )
        
        # Welcome Email (after registration)
        self.templates["welcome_post_registration"] = EmailTemplate(
            template_id="welcome_post_registration",
            email_type=EmailType.WELCOME,
            subject="🎉 Welcome to Pynomaly! Let's Get You Started",
            html_body="""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Welcome to Pynomaly</title>
                <style>
                    body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                    .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                    .header { background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); color: white; text-align: center; padding: 30px; border-radius: 8px 8px 0 0; }
                    .content { background: #f9f9f9; padding: 30px; border-radius: 0 0 8px 8px; }
                    .step-card { background: white; padding: 20px; margin: 15px 0; border-radius: 5px; border-left: 4px solid #4CAF50; }
                    .cta-button { display: inline-block; background: #2196F3; color: white; padding: 12px 25px; text-decoration: none; border-radius: 5px; font-weight: bold; margin: 10px 5px; }
                    .footer { text-align: center; color: #666; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🎉 Welcome to Pynomaly, {{user_name}}!</h1>
                        <p>Your anomaly detection journey starts now</p>
                    </div>
                    
                    <div class="content">
                        <p>Hi {{user_name}},</p>
                        
                        <p>Thank you for joining Pynomaly Beta! Your account is now active and ready to use. Let's get you up and running with a quick 3-step process:</p>
                        
                        <div class="step-card">
                            <h3>📊 Step 1: Upload Your First Dataset</h3>
                            <p>Start by uploading a sample dataset to see Pynomaly in action. We support CSV, JSON, and real-time data streams.</p>
                            <a href="{{dashboard_url}}/upload" class="cta-button">Upload Dataset</a>
                        </div>
                        
                        <div class="step-card">
                            <h3>🔍 Step 2: Run Anomaly Detection</h3>
                            <p>Configure your detection parameters and let our ML models identify anomalies in your data.</p>
                            <a href="{{dashboard_url}}/detection" class="cta-button">Run Detection</a>
                        </div>
                        
                        <div class="step-card">
                            <h3>📈 Step 3: Explore Results</h3>
                            <p>View detailed analytics, visualizations, and insights about the detected anomalies.</p>
                            <a href="{{dashboard_url}}/results" class="cta-button">View Results</a>
                        </div>
                        
                        <div style="background: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; border: 1px solid #ffeeba;">
                            <h4>💡 Pro Tips for {{industry}} Users:</h4>
                            <ul>
                                <li>{{tip_1}}</li>
                                <li>{{tip_2}}</li>
                                <li>{{tip_3}}</li>
                            </ul>
                        </div>
                        
                        <p><strong>Need Help?</strong> Our beta support team is here for you:</p>
                        <ul>
                            <li>📖 <a href="{{docs_url}}">Beta Documentation</a></li>
                            <li>💬 <a href="{{support_url}}">Beta Support Channel</a></li>
                            <li>📧 Email us at <a href="mailto:{{support_email}}">{{support_email}}</a></li>
                            <li>📞 Schedule a 1-on-1 demo: <a href="{{demo_url}}">{{demo_url}}</a></li>
                        </ul>
                        
                        <p>We're excited to see what insights you'll discover!</p>
                        
                        <p>Best regards,<br>
                        The Pynomaly Team</p>
                    </div>
                    
                    <div class="footer">
                        <p>Pynomaly Beta Program | <a href="{{dashboard_url}}">Login to Dashboard</a> | <a href="{{unsubscribe_url}}">Unsubscribe</a></p>
                    </div>
                </div>
            </body>
            </html>
            """,
            text_body="""
            🎉 Welcome to Pynomaly, {{user_name}}!

            Thank you for joining Pynomaly Beta! Let's get you started:

            📊 Step 1: Upload Your First Dataset
            {{dashboard_url}}/upload

            🔍 Step 2: Run Anomaly Detection  
            {{dashboard_url}}/detection

            📈 Step 3: Explore Results
            {{dashboard_url}}/results

            💡 Pro Tips for {{industry}} Users:
            • {{tip_1}}
            • {{tip_2}}
            • {{tip_3}}

            Need Help?
            • Documentation: {{docs_url}}
            • Support: {{support_url}}
            • Email: {{support_email}}
            • Demo: {{demo_url}}

            Best regards,
            The Pynomaly Team
            """,
            variables=["user_name", "industry", "dashboard_url", "docs_url", "support_url", "support_email", "demo_url", "tip_1", "tip_2", "tip_3", "unsubscribe_url"],
            tags=["beta", "welcome", "onboarding", "getting-started"]
        )
        
        # Feature Announcement Email
        self.templates["feature_announcement"] = EmailTemplate(
            template_id="feature_announcement",
            email_type=EmailType.FEATURE_ANNOUNCEMENT,
            subject="🚀 New Feature: {{feature_name}} is Now Available!",
            html_body="""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>New Feature Available</title>
                <style>
                    body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                    .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                    .header { background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%); color: white; text-align: center; padding: 30px; border-radius: 8px 8px 0 0; }
                    .content { background: #f9f9f9; padding: 30px; border-radius: 0 0 8px 8px; }
                    .feature-highlight { background: white; padding: 25px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                    .cta-button { display: inline-block; background: #FF6B6B; color: white; padding: 15px 30px; text-decoration: none; border-radius: 5px; font-weight: bold; margin: 15px 0; }
                    .benefit-list { background: #e8f5e8; padding: 20px; border-radius: 5px; margin: 15px 0; }
                    .footer { text-align: center; color: #666; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🚀 {{feature_name}} is Here!</h1>
                        <p>Exclusive beta feature now available</p>
                    </div>
                    
                    <div class="content">
                        <p>Hi {{user_name}},</p>
                        
                        <p>We're excited to announce that <strong>{{feature_name}}</strong> is now available in your Pynomaly beta account!</p>
                        
                        <div class="feature-highlight">
                            <h3>✨ What's New:</h3>
                            <p>{{feature_description}}</p>
                            
                            <div class="benefit-list">
                                <h4>🎯 Key Benefits:</h4>
                                <ul>
                                    <li>{{benefit_1}}</li>
                                    <li>{{benefit_2}}</li>
                                    <li>{{benefit_3}}</li>
                                </ul>
                            </div>
                        </div>
                        
                        <div style="text-align: center;">
                            <a href="{{feature_url}}" class="cta-button">🔍 Try {{feature_name}} Now</a>
                        </div>
                        
                        <h3>📝 How to Get Started:</h3>
                        <ol>
                            <li>{{step_1}}</li>
                            <li>{{step_2}}</li>
                            <li>{{step_3}}</li>
                        </ol>
                        
                        <div style="background: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; border: 1px solid #ffeeba;">
                            <h4>💡 Beta Feedback Wanted!</h4>
                            <p>As a beta user, your feedback is invaluable. Please try out {{feature_name}} and let us know:</p>
                            <ul>
                                <li>How does it work for your use case?</li>
                                <li>What would make it even better?</li>
                                <li>Any bugs or issues you encounter?</li>
                            </ul>
                            <p><a href="{{feedback_url}}">Share your feedback here</a></p>
                        </div>
                        
                        <p>Questions about this feature? Check out our <a href="{{docs_url}}">updated documentation</a> or reach out to our beta support team.</p>
                        
                        <p>Happy anomaly hunting!</p>
                        
                        <p>Best regards,<br>
                        The Pynomaly Team</p>
                    </div>
                    
                    <div class="footer">
                        <p>Pynomaly Beta Program | <a href="{{dashboard_url}}">Dashboard</a> | <a href="{{unsubscribe_url}}">Unsubscribe</a></p>
                    </div>
                </div>
            </body>
            </html>
            """,
            text_body="""
            🚀 {{feature_name}} is Here!

            Hi {{user_name}},

            {{feature_name}} is now available in your Pynomaly beta account!

            ✨ What's New:
            {{feature_description}}

            🎯 Key Benefits:
            • {{benefit_1}}
            • {{benefit_2}}
            • {{benefit_3}}

            📝 How to Get Started:
            1. {{step_1}}
            2. {{step_2}}
            3. {{step_3}}

            Try it now: {{feature_url}}

            💡 We want your feedback!
            {{feedback_url}}

            Questions? Check our docs: {{docs_url}}

            Best regards,
            The Pynomaly Team
            """,
            variables=["user_name", "feature_name", "feature_description", "benefit_1", "benefit_2", "benefit_3", "step_1", "step_2", "step_3", "feature_url", "feedback_url", "docs_url", "dashboard_url", "unsubscribe_url"],
            tags=["beta", "feature-announcement", "product-update"]
        )
        
        # Feedback Request Email
        self.templates["feedback_request"] = EmailTemplate(
            template_id="feedback_request",
            email_type=EmailType.FEEDBACK_REQUEST,
            subject="📝 Help Us Improve Pynomaly - 2 Minute Survey",
            html_body="""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Feedback Request</title>
                <style>
                    body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                    .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                    .header { background: linear-gradient(135deg, #9C27B0 0%, #673AB7 100%); color: white; text-align: center; padding: 30px; border-radius: 8px 8px 0 0; }
                    .content { background: #f9f9f9; padding: 30px; border-radius: 0 0 8px 8px; }
                    .survey-highlight { background: white; padding: 25px; border-radius: 8px; margin: 20px 0; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                    .cta-button { display: inline-block; background: #9C27B0; color: white; padding: 15px 30px; text-decoration: none; border-radius: 5px; font-weight: bold; margin: 15px 0; }
                    .footer { text-align: center; color: #666; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>📝 Your Feedback Matters</h1>
                        <p>Help us build the perfect anomaly detection platform</p>
                    </div>
                    
                    <div class="content">
                        <p>Hi {{user_name}},</p>
                        
                        <p>You've been using Pynomaly for {{usage_days}} days now, and we'd love to hear about your experience!</p>
                        
                        <div class="survey-highlight">
                            <h3>🎯 Quick 2-Minute Survey</h3>
                            <p>Your insights help us prioritize features and improvements that matter most to {{industry}} professionals like you.</p>
                            
                            <div style="margin: 20px 0;">
                                <p><strong>Survey Topics:</strong></p>
                                <ul style="text-align: left; display: inline-block;">
                                    <li>Overall satisfaction with Pynomaly</li>
                                    <li>Most valuable features for your use case</li>
                                    <li>Areas for improvement</li>
                                    <li>Feature requests and suggestions</li>
                                </ul>
                            </div>
                            
                            <a href="{{survey_url}}" class="cta-button">📝 Take 2-Minute Survey</a>
                        </div>
                        
                        <div style="background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0;">
                            <h4>🎁 As a Thank You:</h4>
                            <p>Complete the survey and receive:</p>
                            <ul>
                                <li>🎯 Personalized best practices guide for {{industry}}</li>
                                <li>📊 Early access to our new dashboard features</li>
                                <li>💰 Exclusive 25% discount for GA launch</li>
                            </ul>
                        </div>
                        
                        <p><strong>Your Impact:</strong> Previous feedback has led to:</p>
                        <ul>
                            <li>✅ Improved API response times (requested by 78% of users)</li>
                            <li>✅ Enhanced visualization options (top feature request)</li>
                            <li>✅ Better documentation and examples</li>
                        </ul>
                        
                        <p>Can't complete the survey? Feel free to reply directly to this email with any thoughts or suggestions.</p>
                        
                        <p>Thank you for being an amazing beta user!</p>
                        
                        <p>Best regards,<br>
                        The Pynomaly Team</p>
                    </div>
                    
                    <div class="footer">
                        <p>Pynomaly Beta Program | <a href="{{dashboard_url}}">Dashboard</a> | <a href="{{unsubscribe_url}}">Unsubscribe</a></p>
                    </div>
                </div>
            </body>
            </html>
            """,
            text_body="""
            📝 Your Feedback Matters

            Hi {{user_name}},

            You've been using Pynomaly for {{usage_days}} days - we'd love your feedback!

            🎯 Quick 2-Minute Survey
            {{survey_url}}

            Survey covers:
            • Overall satisfaction
            • Most valuable features  
            • Areas for improvement
            • Feature requests

            🎁 Thank you rewards:
            • Personalized best practices guide
            • Early dashboard access
            • 25% GA launch discount

            Your previous feedback helped us:
            ✅ Improve API response times
            ✅ Enhance visualizations  
            ✅ Better documentation

            Questions? Reply to this email.

            Best regards,
            The Pynomaly Team
            """,
            variables=["user_name", "usage_days", "industry", "survey_url", "dashboard_url", "unsubscribe_url"],
            tags=["beta", "feedback", "survey", "engagement"]
        )
        
        # Tips and Tricks Email
        self.templates["tips_and_tricks"] = EmailTemplate(
            template_id="tips_and_tricks",
            email_type=EmailType.TIPS_AND_TRICKS,
            subject="💡 Pro Tips: Get More from Pynomaly ({{industry}} Edition)",
            html_body="""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Pynomaly Pro Tips</title>
                <style>
                    body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                    .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                    .header { background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%); color: white; text-align: center; padding: 30px; border-radius: 8px 8px 0 0; }
                    .content { background: #f9f9f9; padding: 30px; border-radius: 0 0 8px 8px; }
                    .tip-card { background: white; padding: 20px; margin: 15px 0; border-radius: 8px; border-left: 4px solid #FF9800; }
                    .pro-tip { background: #fff3e0; padding: 15px; border-radius: 5px; margin: 10px 0; }
                    .cta-button { display: inline-block; background: #FF9800; color: white; padding: 12px 25px; text-decoration: none; border-radius: 5px; font-weight: bold; margin: 10px 0; }
                    .footer { text-align: center; color: #666; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>💡 Pynomaly Pro Tips</h1>
                        <p>{{industry}} Edition - Level up your anomaly detection</p>
                    </div>
                    
                    <div class="content">
                        <p>Hi {{user_name}},</p>
                        
                        <p>Want to get even more value from Pynomaly? Here are some pro tips specifically for {{industry}} professionals:</p>
                        
                        <div class="tip-card">
                            <h3>🎯 Tip #1: {{tip_1_title}}</h3>
                            <p>{{tip_1_description}}</p>
                            <div class="pro-tip">
                                <strong>💡 Pro Insight:</strong> {{tip_1_insight}}
                            </div>
                        </div>
                        
                        <div class="tip-card">
                            <h3>🔧 Tip #2: {{tip_2_title}}</h3>
                            <p>{{tip_2_description}}</p>
                            <div class="pro-tip">
                                <strong>💡 Pro Insight:</strong> {{tip_2_insight}}
                            </div>
                        </div>
                        
                        <div class="tip-card">
                            <h3>📊 Tip #3: {{tip_3_title}}</h3>
                            <p>{{tip_3_description}}</p>
                            <div class="pro-tip">
                                <strong>💡 Pro Insight:</strong> {{tip_3_insight}}
                            </div>
                        </div>
                        
                        <div style="text-align: center; margin: 30px 0;">
                            <a href="{{dashboard_url}}" class="cta-button">🚀 Try These Tips Now</a>
                        </div>
                        
                        <div style="background: #e3f2fd; padding: 20px; border-radius: 5px; margin: 20px 0;">
                            <h4>📚 Want to Learn More?</h4>
                            <ul>
                                <li><a href="{{advanced_guide_url}}">Advanced {{industry}} Guide</a></li>
                                <li><a href="{{api_examples_url}}">API Integration Examples</a></li>
                                <li><a href="{{webinar_url}}">Upcoming {{industry}} Webinar</a></li>
                                <li><a href="{{community_url}}">Join the Beta Community</a></li>
                            </ul>
                        </div>
                        
                        <p><strong>Have your own tips to share?</strong> We'd love to feature successful beta users in future newsletters. <a href="{{contact_url}}">Get in touch!</a></p>
                        
                        <p>Happy detecting!</p>
                        
                        <p>Best regards,<br>
                        The Pynomaly Team</p>
                    </div>
                    
                    <div class="footer">
                        <p>Pynomaly Beta Program | <a href="{{dashboard_url}}">Dashboard</a> | <a href="{{unsubscribe_url}}">Unsubscribe</a></p>
                    </div>
                </div>
            </body>
            </html>
            """,
            text_body="""
            💡 Pynomaly Pro Tips - {{industry}} Edition

            Hi {{user_name}},

            Pro tips for {{industry}} professionals:

            🎯 Tip #1: {{tip_1_title}}
            {{tip_1_description}}
            💡 {{tip_1_insight}}

            🔧 Tip #2: {{tip_2_title}}
            {{tip_2_description}}
            💡 {{tip_2_insight}}

            📊 Tip #3: {{tip_3_title}}
            {{tip_3_description}}
            💡 {{tip_3_insight}}

            Try these tips: {{dashboard_url}}

            Learn More:
            • Advanced Guide: {{advanced_guide_url}}
            • API Examples: {{api_examples_url}}
            • Webinar: {{webinar_url}}
            • Community: {{community_url}}

            Share your tips: {{contact_url}}

            Best regards,
            The Pynomaly Team
            """,
            variables=["user_name", "industry", "tip_1_title", "tip_1_description", "tip_1_insight", "tip_2_title", "tip_2_description", "tip_2_insight", "tip_3_title", "tip_3_description", "tip_3_insight", "dashboard_url", "advanced_guide_url", "api_examples_url", "webinar_url", "community_url", "contact_url", "unsubscribe_url"],
            tags=["beta", "tips", "education", "engagement"]
        )
    
    def get_template(self, template_id: str) -> EmailTemplate:
        """Get email template by ID."""
        return self.templates.get(template_id)
    
    def render_template(self, template_id: str, variables: Dict[str, str]) -> Dict[str, str]:
        """Render email template with variables."""
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")
        
        # Replace variables in subject, html_body, and text_body
        rendered_subject = template.subject
        rendered_html = template.html_body
        rendered_text = template.text_body
        
        for var_name, var_value in variables.items():
            placeholder = f"{{{{{var_name}}}}}"
            rendered_subject = rendered_subject.replace(placeholder, str(var_value))
            rendered_html = rendered_html.replace(placeholder, str(var_value))
            rendered_text = rendered_text.replace(placeholder, str(var_value))
        
        return {
            "subject": rendered_subject,
            "html_body": rendered_html,
            "text_body": rendered_text,
            "template_id": template_id,
            "email_type": template.email_type.value
        }
    
    def get_industry_specific_content(self, industry: str) -> Dict[str, str]:
        """Get industry-specific content for templates."""
        industry_content = {
            "fintech": {
                "use_case_description": "Perfect for fraud detection, transaction monitoring, algorithmic trading analysis, and regulatory compliance anomaly detection.",
                "tip_1": "Set up real-time alerts for transaction anomalies above your risk threshold",
                "tip_2": "Use our time-series models for detecting unusual trading patterns",
                "tip_3": "Integrate with your existing fraud prevention systems via our REST API",
                "tip_1_title": "Real-Time Transaction Monitoring",
                "tip_1_description": "Configure automated alerts for suspicious transaction patterns that deviate from normal customer behavior.",
                "tip_1_insight": "FinTech customers see 40% faster fraud detection with real-time monitoring.",
                "tip_2_title": "Optimize Model Sensitivity",
                "tip_2_description": "Fine-tune detection thresholds based on your risk tolerance and false positive costs.",
                "tip_2_insight": "Start with 95% confidence for critical transactions, 85% for general monitoring.",
                "tip_3_title": "Historical Pattern Analysis",
                "tip_3_description": "Use historical data to establish baseline patterns and improve detection accuracy.",
                "tip_3_insight": "6-month historical baselines reduce false positives by up to 60%."
            },
            "manufacturing": {
                "use_case_description": "Ideal for predictive maintenance, quality control, supply chain monitoring, and production line optimization.",
                "tip_1": "Monitor sensor data in real-time to predict equipment failures before they happen",
                "tip_2": "Set up quality control alerts for production line anomalies",
                "tip_3": "Use batch processing for analyzing large volumes of historical production data",
                "tip_1_title": "Predictive Maintenance Setup",
                "tip_1_description": "Monitor vibration, temperature, and pressure sensors to predict equipment failures 2-4 weeks in advance.",
                "tip_1_insight": "Early detection can reduce unplanned downtime by up to 75%.",
                "tip_2_title": "Quality Control Automation",
                "tip_2_description": "Automatically flag products that deviate from quality specifications during production.",
                "tip_2_insight": "Automated QC can detect defects 10x faster than manual inspection.",
                "tip_3_title": "Supply Chain Visibility",
                "tip_3_description": "Track supplier performance and delivery patterns to identify potential disruptions.",
                "tip_3_insight": "Supply chain anomaly detection reduces material shortage risks by 45%."
            },
            "healthcare": {
                "use_case_description": "Essential for patient monitoring, clinical trial analysis, medical device performance, and healthcare operations optimization.",
                "tip_1": "Configure patient vital sign monitoring with personalized thresholds",
                "tip_2": "Analyze clinical trial data to identify unexpected drug reactions or effects",
                "tip_3": "Monitor medical device performance and alert for potential malfunctions",
                "tip_1_title": "Patient Monitoring Optimization",
                "tip_1_description": "Set personalized alert thresholds based on individual patient baselines rather than population averages.",
                "tip_1_insight": "Personalized thresholds reduce false alarms by 50% while improving early detection.",
                "tip_2_title": "Clinical Data Analysis",
                "tip_2_description": "Use anomaly detection to identify unexpected patterns in clinical trial data and adverse events.",
                "tip_2_insight": "Early anomaly detection in trials can save 6-12 months in drug development.",
                "tip_3_title": "Device Performance Monitoring",
                "tip_3_description": "Track medical device metrics to predict maintenance needs and prevent failures during critical procedures.",
                "tip_3_insight": "Predictive device maintenance improves patient safety scores by 35%."
            }
        }
        
        return industry_content.get(industry.lower(), industry_content["fintech"])
    
    def create_email_campaign(self, user_data: List[Dict[str, Any]], template_id: str) -> List[Dict[str, Any]]:
        """Create email campaign for multiple users."""
        campaign_emails = []
        
        for user in user_data:
            # Get industry-specific content
            industry_content = self.get_industry_specific_content(user.get("industry", "fintech"))
            
            # Prepare template variables
            variables = {
                "user_name": user.get("name", user.get("email", "User")),
                "user_email": user.get("email"),
                "industry": user.get("industry", "fintech").title(),
                "registration_url": f"https://beta.pynomaly.io/register?token={hash(user.get('email', '')) % 100000}",
                "dashboard_url": "https://app.pynomaly.io",
                "docs_url": "https://docs.pynomaly.io/beta",
                "api_docs_url": "https://docs.pynomaly.io/api",
                "examples_url": "https://docs.pynomaly.io/examples",
                "support_url": "https://support.pynomaly.io/beta",
                "support_email": "beta-support@pynomaly.io",
                "demo_url": "https://calendly.com/pynomaly/beta-demo",
                "feedback_url": "https://feedback.pynomaly.io",
                "survey_url": "https://survey.pynomaly.io/beta-satisfaction",
                "unsubscribe_url": f"https://beta.pynomaly.io/unsubscribe?email={user.get('email')}",
                "contact_url": "mailto:beta-support@pynomaly.io",
                "feature_url": "https://app.pynomaly.io/features/new",
                "advanced_guide_url": f"https://docs.pynomaly.io/guides/{user.get('industry', 'fintech')}",
                "api_examples_url": "https://docs.pynomaly.io/api/examples",
                "webinar_url": "https://pynomaly.io/webinars",
                "community_url": "https://community.pynomaly.io",
                "usage_days": "7"  # Default to 7 days
            }
            
            # Add industry-specific content
            variables.update(industry_content)
            
            # Template-specific variables
            if template_id == "feature_announcement":
                variables.update({
                    "feature_name": "Real-time Streaming",
                    "feature_description": "Process and analyze data streams in real-time with sub-second latency.",
                    "benefit_1": "Instant anomaly detection as data arrives",
                    "benefit_2": "Scalable to millions of events per second",
                    "benefit_3": "Seamless integration with your existing data pipeline",
                    "step_1": "Navigate to the Streaming section in your dashboard",
                    "step_2": "Connect your data source using our connector or API",
                    "step_3": "Configure detection parameters and start monitoring"
                })
            
            # Render the template
            rendered_email = self.render_template(template_id, variables)
            rendered_email["recipient"] = user.get("email")
            rendered_email["user_data"] = user
            
            campaign_emails.append(rendered_email)
        
        return campaign_emails
    
    def generate_sample_campaigns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate sample email campaigns for beta users."""
        sample_users = [
            {
                "name": "Sarah Chen",
                "email": "sarah.chen@fintechcorp.com",
                "industry": "fintech",
                "company": "FinTech Corp",
                "role": "Risk Manager"
            },
            {
                "name": "David Park",
                "email": "david.park@automaker.com",
                "industry": "manufacturing",
                "company": "AutoMaker Inc",
                "role": "Quality Engineer"
            },
            {
                "name": "Dr. Smith",
                "email": "dr.smith@hospital.org",
                "industry": "healthcare",
                "company": "City General Hospital",
                "role": "Chief Medical Officer"
            }
        ]
        
        campaigns = {}
        
        # Generate campaigns for each template
        for template_id in self.templates.keys():
            campaigns[template_id] = self.create_email_campaign(sample_users, template_id)
        
        return campaigns


def main():
    """Demo of email template system."""
    email_system = BetaEmailTemplateSystem()
    
    print("🧪 Beta Email Template System Demo")
    print("=" * 50)
    
    # Generate sample campaigns
    campaigns = email_system.generate_sample_campaigns()
    
    print(f"📧 Generated {len(campaigns)} email campaigns:")
    for template_id, emails in campaigns.items():
        print(f"  • {template_id}: {len(emails)} emails")
    
    # Demo: Render welcome email for a specific user
    print("\n📝 Sample Welcome Email:")
    print("-" * 30)
    
    user_data = {
        "name": "Sarah Chen",
        "email": "sarah.chen@fintechcorp.com",
        "industry": "fintech"
    }
    
    industry_content = email_system.get_industry_specific_content("fintech")
    variables = {
        "user_name": "Sarah Chen",
        "industry": "FinTech",
        "dashboard_url": "https://app.pynomaly.io",
        "docs_url": "https://docs.pynomaly.io/beta",
        "support_url": "https://support.pynomaly.io/beta",
        "support_email": "beta-support@pynomaly.io",
        "demo_url": "https://calendly.com/pynomaly/beta-demo",
        "unsubscribe_url": "https://beta.pynomaly.io/unsubscribe?email=sarah.chen@fintechcorp.com",
        **industry_content
    }
    
    rendered_email = email_system.render_template("welcome_post_registration", variables)
    
    print(f"Subject: {rendered_email['subject']}")
    print(f"Type: {rendered_email['email_type']}")
    print("\nText Body Preview:")
    print(rendered_email['text_body'][:500] + "...")
    
    # Save sample campaign
    project_root = Path(__file__).parent.parent.parent
    output_file = project_root / "beta_email_campaigns_sample.json"
    
    with open(output_file, "w") as f:
        json.dump(campaigns, f, indent=2, default=str)
    
    print(f"\n📄 Sample campaigns saved to: {output_file}")
    
    return campaigns


if __name__ == "__main__":
    main()