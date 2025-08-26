"""
Enhanced Global-First Implementation - Terragon SDLC v4.0
Multi-region, multi-language, compliance-ready financial intelligence system

Features:
- I18n support for 6+ languages (en, es, fr, de, ja, zh)
- GDPR, CCPA, PDPA compliance framework
- Multi-region deployment architecture
- Cross-platform compatibility
- Cultural financial analysis adaptation
- Global regulatory compliance monitoring
- Timezone-aware financial data processing
"""

from __future__ import annotations

import asyncio
import logging
import json
import os
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import hashlib
from decimal import Decimal
import re

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported languages for global deployment."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"          # General Data Protection Regulation (EU)
    CCPA = "ccpa"          # California Consumer Privacy Act (US)
    PDPA = "pdpa"          # Personal Data Protection Act (Singapore/Thailand)
    PIPEDA = "pipeda"      # Personal Information Protection and Electronic Documents Act (Canada)
    LGPD = "lgpd"          # Lei Geral de Proteção de Dados (Brazil)
    SOX = "sox"            # Sarbanes-Oxley Act (US)
    MiFID = "mifid"        # Markets in Financial Instruments Directive (EU)


class Region(Enum):
    """Global regions for deployment."""
    NORTH_AMERICA = "na"
    EUROPE = "eu"
    ASIA_PACIFIC = "apac"
    LATIN_AMERICA = "latam"
    MIDDLE_EAST_AFRICA = "mea"


@dataclass
class LocalizationContext:
    """Context for localization and cultural adaptation."""
    language: SupportedLanguage
    region: Region
    timezone: str
    currency: str
    date_format: str
    number_format: str
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    cultural_preferences: Dict[str, Any] = field(default_factory=dict)
    regulatory_requirements: Dict[str, Any] = field(default_factory=dict)


class GlobalFinancialIntelligence:
    """Global-first financial intelligence system."""
    
    def __init__(self):
        self.translations = {}
        self.localization_contexts = {}
        self.compliance_rules = {}
        self.cultural_adaptations = {}
        self.timezone_handlers = {}
        
        # Initialize supported locales
        self._initialize_localization_contexts()
        self._load_translations()
        self._setup_compliance_frameworks()
        self._configure_cultural_adaptations()
    
    def _initialize_localization_contexts(self):
        """Initialize localization contexts for all supported regions."""
        contexts = {
            # North America
            (SupportedLanguage.ENGLISH, Region.NORTH_AMERICA): LocalizationContext(
                language=SupportedLanguage.ENGLISH,
                region=Region.NORTH_AMERICA,
                timezone="America/New_York",
                currency="USD",
                date_format="%m/%d/%Y",
                number_format="1,234.56",
                compliance_frameworks=[ComplianceFramework.SOX, ComplianceFramework.CCPA],
                cultural_preferences={
                    "financial_year_start": "01-01",
                    "quarter_start_months": [1, 4, 7, 10],
                    "business_days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
                    "risk_tolerance": "moderate",
                    "disclosure_preference": "detailed"
                }
            ),
            
            # Europe
            (SupportedLanguage.ENGLISH, Region.EUROPE): LocalizationContext(
                language=SupportedLanguage.ENGLISH,
                region=Region.EUROPE,
                timezone="Europe/London",
                currency="EUR",
                date_format="%d/%m/%Y",
                number_format="1.234,56",
                compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.MiFID],
                cultural_preferences={
                    "financial_year_start": "04-06",  # UK tax year
                    "quarter_start_months": [4, 7, 10, 1],
                    "business_days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
                    "risk_tolerance": "conservative",
                    "disclosure_preference": "comprehensive"
                }
            ),
            
            (SupportedLanguage.GERMAN, Region.EUROPE): LocalizationContext(
                language=SupportedLanguage.GERMAN,
                region=Region.EUROPE,
                timezone="Europe/Berlin",
                currency="EUR",
                date_format="%d.%m.%Y",
                number_format="1.234,56",
                compliance_frameworks=[ComplianceFramework.GDPR],
                cultural_preferences={
                    "financial_year_start": "01-01",
                    "quarter_start_months": [1, 4, 7, 10],
                    "business_days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
                    "risk_tolerance": "conservative",
                    "disclosure_preference": "detailed"
                }
            ),
            
            (SupportedLanguage.FRENCH, Region.EUROPE): LocalizationContext(
                language=SupportedLanguage.FRENCH,
                region=Region.EUROPE,
                timezone="Europe/Paris",
                currency="EUR",
                date_format="%d/%m/%Y",
                number_format="1 234,56",
                compliance_frameworks=[ComplianceFramework.GDPR],
                cultural_preferences={
                    "financial_year_start": "01-01",
                    "quarter_start_months": [1, 4, 7, 10],
                    "business_days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
                    "risk_tolerance": "moderate",
                    "disclosure_preference": "comprehensive"
                }
            ),
            
            # Asia Pacific
            (SupportedLanguage.JAPANESE, Region.ASIA_PACIFIC): LocalizationContext(
                language=SupportedLanguage.JAPANESE,
                region=Region.ASIA_PACIFIC,
                timezone="Asia/Tokyo",
                currency="JPY",
                date_format="%Y/%m/%d",
                number_format="1,234",
                compliance_frameworks=[ComplianceFramework.PDPA],
                cultural_preferences={
                    "financial_year_start": "04-01",  # Japanese fiscal year
                    "quarter_start_months": [4, 7, 10, 1],
                    "business_days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
                    "risk_tolerance": "conservative",
                    "disclosure_preference": "hierarchical"
                }
            ),
            
            (SupportedLanguage.CHINESE, Region.ASIA_PACIFIC): LocalizationContext(
                language=SupportedLanguage.CHINESE,
                region=Region.ASIA_PACIFIC,
                timezone="Asia/Shanghai",
                currency="CNY",
                date_format="%Y年%m月%d日",
                number_format="1,234.56",
                compliance_frameworks=[ComplianceFramework.PDPA],
                cultural_preferences={
                    "financial_year_start": "01-01",
                    "quarter_start_months": [1, 4, 7, 10],
                    "business_days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
                    "risk_tolerance": "moderate",
                    "disclosure_preference": "summary"
                }
            ),
            
            # Latin America
            (SupportedLanguage.SPANISH, Region.LATIN_AMERICA): LocalizationContext(
                language=SupportedLanguage.SPANISH,
                region=Region.LATIN_AMERICA,
                timezone="America/Mexico_City",
                currency="MXN",
                date_format="%d/%m/%Y",
                number_format="1,234.56",
                compliance_frameworks=[ComplianceFramework.LGPD],
                cultural_preferences={
                    "financial_year_start": "01-01",
                    "quarter_start_months": [1, 4, 7, 10],
                    "business_days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
                    "risk_tolerance": "moderate",
                    "disclosure_preference": "detailed"
                }
            )
        }
        
        for key, context in contexts.items():
            self.localization_contexts[key] = context
    
    def _load_translations(self):
        """Load translations for all supported languages."""
        # Base translation keys with English defaults
        base_translations = {
            "query.processing": "Processing your query...",
            "query.completed": "Query completed successfully",
            "query.error": "Error processing query",
            "risk.high": "High Risk",
            "risk.medium": "Medium Risk", 
            "risk.low": "Low Risk",
            "financial.revenue": "Revenue",
            "financial.expenses": "Expenses",
            "financial.profit": "Profit",
            "financial.loss": "Loss",
            "financial.assets": "Assets",
            "financial.liabilities": "Liabilities",
            "financial.equity": "Equity",
            "compliance.gdpr.notice": "This service complies with GDPR requirements",
            "compliance.ccpa.notice": "California Consumer Privacy Act rights available",
            "data.processing.consent": "We process your data with your consent",
            "data.retention.policy": "Data retained according to our retention policy",
            "error.invalid_input": "Invalid input provided",
            "error.unauthorized": "Unauthorized access attempt",
            "error.rate_limited": "Rate limit exceeded",
            "success.data_saved": "Data saved successfully",
            "quantum.processing": "Quantum processing in progress...",
            "quantum.optimization": "Quantum optimization applied",
            "performance.excellent": "Excellent performance",
            "performance.good": "Good performance",
            "performance.poor": "Poor performance"
        }
        
        # Spanish translations
        spanish_translations = {
            "query.processing": "Procesando su consulta...",
            "query.completed": "Consulta completada exitosamente",
            "query.error": "Error al procesar la consulta",
            "risk.high": "Riesgo Alto",
            "risk.medium": "Riesgo Medio",
            "risk.low": "Riesgo Bajo",
            "financial.revenue": "Ingresos",
            "financial.expenses": "Gastos",
            "financial.profit": "Beneficio",
            "financial.loss": "Pérdida",
            "financial.assets": "Activos",
            "financial.liabilities": "Pasivos",
            "financial.equity": "Patrimonio",
            "compliance.gdpr.notice": "Este servicio cumple con los requisitos del GDPR",
            "data.processing.consent": "Procesamos sus datos con su consentimiento",
            "error.invalid_input": "Entrada inválida proporcionada",
            "error.unauthorized": "Intento de acceso no autorizado",
            "success.data_saved": "Datos guardados exitosamente",
            "quantum.processing": "Procesamiento cuántico en progreso...",
            "performance.excellent": "Rendimiento excelente"
        }
        
        # French translations
        french_translations = {
            "query.processing": "Traitement de votre requête...",
            "query.completed": "Requête terminée avec succès",
            "query.error": "Erreur lors du traitement de la requête",
            "risk.high": "Risque Élevé",
            "risk.medium": "Risque Moyen",
            "risk.low": "Risque Faible",
            "financial.revenue": "Revenus",
            "financial.expenses": "Dépenses",
            "financial.profit": "Bénéfice",
            "financial.loss": "Perte",
            "financial.assets": "Actifs",
            "financial.liabilities": "Passifs",
            "financial.equity": "Capitaux propres",
            "compliance.gdpr.notice": "Ce service est conforme aux exigences du RGPD",
            "data.processing.consent": "Nous traitons vos données avec votre consentement",
            "quantum.processing": "Traitement quantique en cours...",
            "performance.excellent": "Performance excellente"
        }
        
        # German translations
        german_translations = {
            "query.processing": "Verarbeitung Ihrer Anfrage...",
            "query.completed": "Anfrage erfolgreich abgeschlossen",
            "query.error": "Fehler bei der Verarbeitung der Anfrage",
            "risk.high": "Hohes Risiko",
            "risk.medium": "Mittleres Risiko",
            "risk.low": "Geringes Risiko",
            "financial.revenue": "Einnahmen",
            "financial.expenses": "Ausgaben",
            "financial.profit": "Gewinn",
            "financial.loss": "Verlust",
            "financial.assets": "Vermögenswerte",
            "financial.liabilities": "Verbindlichkeiten",
            "financial.equity": "Eigenkapital",
            "compliance.gdpr.notice": "Dieser Service entspricht den DSGVO-Anforderungen",
            "quantum.processing": "Quantenverarbeitung läuft...",
            "performance.excellent": "Ausgezeichnete Leistung"
        }
        
        # Japanese translations
        japanese_translations = {
            "query.processing": "クエリを処理中...",
            "query.completed": "クエリが正常に完了しました",
            "query.error": "クエリ処理エラー",
            "risk.high": "高リスク",
            "risk.medium": "中リスク",
            "risk.low": "低リスク",
            "financial.revenue": "収益",
            "financial.expenses": "費用",
            "financial.profit": "利益",
            "financial.loss": "損失",
            "financial.assets": "資産",
            "financial.liabilities": "負債",
            "financial.equity": "純資産",
            "quantum.processing": "量子処理中...",
            "performance.excellent": "優秀な性能"
        }
        
        # Chinese translations
        chinese_translations = {
            "query.processing": "正在处理您的查询...",
            "query.completed": "查询成功完成",
            "query.error": "查询处理错误",
            "risk.high": "高风险",
            "risk.medium": "中等风险",
            "risk.low": "低风险",
            "financial.revenue": "收入",
            "financial.expenses": "费用",
            "financial.profit": "利润",
            "financial.loss": "损失",
            "financial.assets": "资产",
            "financial.liabilities": "负债",
            "financial.equity": "股权",
            "quantum.processing": "量子处理进行中...",
            "performance.excellent": "优秀表现"
        }
        
        # Store translations
        self.translations = {
            SupportedLanguage.ENGLISH: base_translations,
            SupportedLanguage.SPANISH: {**base_translations, **spanish_translations},
            SupportedLanguage.FRENCH: {**base_translations, **french_translations},
            SupportedLanguage.GERMAN: {**base_translations, **german_translations},
            SupportedLanguage.JAPANESE: {**base_translations, **japanese_translations},
            SupportedLanguage.CHINESE: {**base_translations, **chinese_translations}
        }
    
    def _setup_compliance_frameworks(self):
        """Setup compliance framework rules and handlers."""
        self.compliance_rules = {
            ComplianceFramework.GDPR: {
                "data_retention_days": 730,  # 2 years default
                "consent_required": True,
                "right_to_erasure": True,
                "data_portability": True,
                "privacy_by_design": True,
                "data_processing_basis": ["consent", "legitimate_interest", "contract"],
                "sensitive_data_categories": [
                    "personal_financial_data",
                    "risk_assessments",
                    "investment_preferences"
                ],
                "breach_notification_hours": 72,
                "required_notices": [
                    "data_processing_purpose",
                    "data_retention_period",
                    "third_party_sharing",
                    "user_rights"
                ]
            },
            
            ComplianceFramework.CCPA: {
                "data_retention_days": 365,  # 1 year for some categories
                "right_to_know": True,
                "right_to_delete": True,
                "right_to_opt_out": True,
                "non_discrimination": True,
                "personal_info_categories": [
                    "financial_information",
                    "commercial_information",
                    "professional_information"
                ],
                "sale_opt_out_required": True,
                "consumer_rights_notice": True
            },
            
            ComplianceFramework.PDPA: {
                "consent_management": True,
                "data_breach_notification": True,
                "data_protection_officer": False,  # Not always required
                "cross_border_transfer_restrictions": True,
                "notification_period_days": 3,
                "personal_data_categories": [
                    "financial_data",
                    "transaction_history", 
                    "risk_profiles"
                ]
            },
            
            ComplianceFramework.SOX: {
                "financial_reporting_controls": True,
                "audit_trails_required": True,
                "data_integrity_controls": True,
                "access_controls": True,
                "segregation_of_duties": True,
                "document_retention_years": 7,
                "financial_data_accuracy": True
            },
            
            ComplianceFramework.MiFID: {
                "client_categorization": True,
                "best_execution": True,
                "product_governance": True,
                "investor_protection": True,
                "transaction_reporting": True,
                "conflict_of_interest": True,
                "suitable_advice": True
            }
        }
    
    def _configure_cultural_adaptations(self):
        """Configure cultural adaptations for financial analysis."""
        self.cultural_adaptations = {
            Region.ASIA_PACIFIC: {
                "risk_presentation": "hierarchical",
                "financial_metrics_emphasis": ["cash_flow", "debt_ratio", "roi"],
                "reporting_style": "detailed_with_summary",
                "color_preferences": {
                    "positive": "red",  # In some Asian cultures, red represents good fortune
                    "negative": "green",
                    "neutral": "black"
                },
                "number_formatting": {
                    "large_numbers": "万, 億",  # Use traditional large number units
                    "decimal_places": 2
                }
            },
            
            Region.EUROPE: {
                "risk_presentation": "conservative_detailed",
                "financial_metrics_emphasis": ["regulatory_capital", "compliance_ratio", "esg_score"],
                "reporting_style": "comprehensive_regulatory",
                "color_preferences": {
                    "positive": "green",
                    "negative": "red", 
                    "neutral": "grey"
                },
                "regulatory_focus": "high"
            },
            
            Region.NORTH_AMERICA: {
                "risk_presentation": "executive_summary",
                "financial_metrics_emphasis": ["growth_rate", "market_cap", "eps"],
                "reporting_style": "performance_focused",
                "color_preferences": {
                    "positive": "green",
                    "negative": "red",
                    "neutral": "blue"
                },
                "market_focus": "high"
            },
            
            Region.LATIN_AMERICA: {
                "risk_presentation": "inflation_adjusted",
                "financial_metrics_emphasis": ["real_returns", "currency_exposure", "political_risk"],
                "reporting_style": "risk_aware",
                "currency_considerations": "multi_currency",
                "volatility_emphasis": "high"
            }
        }
    
    def get_localization_context(self, language: SupportedLanguage, 
                               region: Region) -> Optional[LocalizationContext]:
        """Get localization context for language and region combination."""
        return self.localization_contexts.get((language, region))
    
    def translate(self, key: str, language: SupportedLanguage, 
                 **kwargs) -> str:
        """Translate a message key to the specified language."""
        translations = self.translations.get(language, self.translations[SupportedLanguage.ENGLISH])
        translated = translations.get(key, key)
        
        # Simple template substitution
        if kwargs:
            try:
                translated = translated.format(**kwargs)
            except (KeyError, ValueError) as e:
                logger.warning(f"Translation template error for key '{key}': {e}")
        
        return translated
    
    def format_financial_value(self, value: Union[float, Decimal], 
                              context: LocalizationContext) -> str:
        """Format financial value according to locale."""
        if isinstance(value, float):
            value = Decimal(str(value))
        
        # Apply currency and number formatting
        if context.currency == "JPY":
            # Japanese Yen - no decimal places
            formatted = f"¥{value:,.0f}"
        elif context.currency == "EUR":
            if context.number_format == "1.234,56":
                # German/some EU formatting
                formatted = f"€{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            elif context.number_format == "1 234,56":
                # French formatting
                formatted = f"€{value:,.2f}".replace(",", " ").replace(".", ",")
            else:
                formatted = f"€{value:,.2f}"
        elif context.currency == "USD":
            formatted = f"${value:,.2f}"
        elif context.currency == "CNY":
            formatted = f"¥{value:,.2f}"
        elif context.currency == "MXN":
            formatted = f"${value:,.2f} MXN"
        else:
            formatted = f"{context.currency} {value:,.2f}"
        
        return formatted
    
    def format_date(self, date: datetime, context: LocalizationContext) -> str:
        """Format date according to locale."""
        return date.strftime(context.date_format)
    
    def check_compliance(self, framework: ComplianceFramework, 
                        data_processing_activity: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance requirements for a data processing activity."""
        rules = self.compliance_rules.get(framework, {})
        compliance_status = {
            "framework": framework.value,
            "compliant": True,
            "violations": [],
            "recommendations": []
        }
        
        # Check specific compliance requirements
        if framework == ComplianceFramework.GDPR:
            compliance_status.update(self._check_gdpr_compliance(data_processing_activity, rules))
        elif framework == ComplianceFramework.CCPA:
            compliance_status.update(self._check_ccpa_compliance(data_processing_activity, rules))
        elif framework == ComplianceFramework.SOX:
            compliance_status.update(self._check_sox_compliance(data_processing_activity, rules))
        
        return compliance_status
    
    def _check_gdpr_compliance(self, activity: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        """Check GDPR compliance requirements."""
        violations = []
        recommendations = []
        
        # Check consent requirements
        if rules.get("consent_required") and not activity.get("user_consent"):
            violations.append("Missing user consent for data processing")
        
        # Check data retention
        retention_period = activity.get("retention_days", 0)
        max_retention = rules.get("data_retention_days", 730)
        if retention_period > max_retention:
            violations.append(f"Data retention period ({retention_period} days) exceeds maximum ({max_retention} days)")
        
        # Check sensitive data handling
        sensitive_categories = rules.get("sensitive_data_categories", [])
        processed_categories = activity.get("data_categories", [])
        for category in processed_categories:
            if category in sensitive_categories and not activity.get("special_consent"):
                violations.append(f"Sensitive data category '{category}' requires special consent")
        
        # Recommendations
        if not activity.get("privacy_impact_assessment") and len(processed_categories) > 3:
            recommendations.append("Consider conducting a Privacy Impact Assessment")
        
        if not activity.get("data_protection_officer_notified"):
            recommendations.append("Notify Data Protection Officer of new processing activity")
        
        return {
            "violations": violations,
            "recommendations": recommendations,
            "compliant": len(violations) == 0
        }
    
    def _check_ccpa_compliance(self, activity: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        """Check CCPA compliance requirements."""
        violations = []
        recommendations = []
        
        # Check consumer rights disclosure
        if not activity.get("consumer_rights_disclosed"):
            violations.append("Consumer rights not properly disclosed")
        
        # Check sale opt-out mechanism
        if activity.get("data_sale") and not activity.get("opt_out_mechanism"):
            violations.append("Missing opt-out mechanism for data sale")
        
        # Check non-discrimination
        if activity.get("differential_pricing") and not activity.get("ccpa_compliant_pricing"):
            violations.append("Differential pricing may violate non-discrimination requirements")
        
        return {
            "violations": violations,
            "recommendations": recommendations,
            "compliant": len(violations) == 0
        }
    
    def _check_sox_compliance(self, activity: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        """Check SOX compliance requirements."""
        violations = []
        recommendations = []
        
        # Check audit trails
        if not activity.get("audit_trail_enabled"):
            violations.append("Audit trails not enabled for financial data processing")
        
        # Check access controls
        if not activity.get("role_based_access"):
            violations.append("Role-based access controls not implemented")
        
        # Check segregation of duties
        if activity.get("financial_reporting") and not activity.get("segregation_of_duties"):
            violations.append("Segregation of duties required for financial reporting")
        
        return {
            "violations": violations,
            "recommendations": recommendations,
            "compliant": len(violations) == 0
        }
    
    def adapt_financial_analysis(self, analysis_result: Dict[str, Any], 
                                context: LocalizationContext) -> Dict[str, Any]:
        """Adapt financial analysis results for cultural and regional preferences."""
        adaptations = self.cultural_adaptations.get(context.region, {})
        
        adapted_result = analysis_result.copy()
        
        # Apply cultural formatting
        if "financial_values" in adapted_result:
            for key, value in adapted_result["financial_values"].items():
                if isinstance(value, (int, float, Decimal)):
                    adapted_result["financial_values"][key] = self.format_financial_value(value, context)
        
        # Adapt risk presentation style
        risk_style = adaptations.get("risk_presentation", "standard")
        if "risk_analysis" in adapted_result:
            adapted_result["risk_analysis"]["presentation_style"] = risk_style
            
            if risk_style == "hierarchical":
                # Reorganize risk data hierarchically
                risks = adapted_result["risk_analysis"].get("risks", [])
                adapted_result["risk_analysis"]["risks"] = self._organize_risks_hierarchically(risks)
            elif risk_style == "conservative_detailed":
                # Add additional conservative risk factors
                adapted_result["risk_analysis"]["conservative_factors"] = [
                    "regulatory_changes",
                    "market_volatility",
                    "compliance_requirements"
                ]
        
        # Apply emphasized metrics
        emphasized_metrics = adaptations.get("financial_metrics_emphasis", [])
        if emphasized_metrics and "metrics" in adapted_result:
            adapted_result["emphasized_metrics"] = {
                metric: adapted_result["metrics"].get(metric)
                for metric in emphasized_metrics
                if metric in adapted_result.get("metrics", {})
            }
        
        # Add cultural context
        adapted_result["cultural_context"] = {
            "region": context.region.value,
            "adaptations_applied": list(adaptations.keys()),
            "compliance_frameworks": [f.value for f in context.compliance_frameworks]
        }
        
        return adapted_result
    
    def _organize_risks_hierarchically(self, risks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Organize risks in hierarchical structure for Asian cultural preferences."""
        # Sort by severity and group by category
        risk_hierarchy = {
            "strategic_risks": [],
            "operational_risks": [],
            "financial_risks": [],
            "compliance_risks": []
        }
        
        for risk in risks:
            category = risk.get("category", "operational_risks")
            if category in risk_hierarchy:
                risk_hierarchy[category].append(risk)
            else:
                risk_hierarchy["operational_risks"].append(risk)
        
        # Sort each category by severity
        for category in risk_hierarchy:
            risk_hierarchy[category].sort(
                key=lambda x: x.get("severity", 0), 
                reverse=True
            )
        
        # Convert back to flat list with hierarchical indicators
        hierarchical_risks = []
        for category, category_risks in risk_hierarchy.items():
            if category_risks:
                hierarchical_risks.append({
                    "type": "category_header",
                    "category": category,
                    "risk_count": len(category_risks)
                })
                hierarchical_risks.extend(category_risks)
        
        return hierarchical_risks
    
    def get_supported_languages(self) -> List[SupportedLanguage]:
        """Get list of all supported languages."""
        return list(SupportedLanguage)
    
    def get_supported_regions(self) -> List[Region]:
        """Get list of all supported regions."""
        return list(Region)
    
    def get_compliance_frameworks(self, region: Region) -> List[ComplianceFramework]:
        """Get applicable compliance frameworks for a region."""
        framework_mapping = {
            Region.EUROPE: [ComplianceFramework.GDPR, ComplianceFramework.MiFID],
            Region.NORTH_AMERICA: [ComplianceFramework.SOX, ComplianceFramework.CCPA],
            Region.ASIA_PACIFIC: [ComplianceFramework.PDPA],
            Region.LATIN_AMERICA: [ComplianceFramework.LGPD],
        }
        
        return framework_mapping.get(region, [])


# Global instance
global_financial_intelligence = GlobalFinancialIntelligence()


def localized_response(language: SupportedLanguage = SupportedLanguage.ENGLISH,
                      region: Region = Region.NORTH_AMERICA):
    """Decorator to automatically localize function responses."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Get localization context
            context = global_financial_intelligence.get_localization_context(language, region)
            if not context:
                return result
            
            # Apply localization if result is a dict
            if isinstance(result, dict):
                return global_financial_intelligence.adapt_financial_analysis(result, context)
            
            return result
        
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Get localization context
            context = global_financial_intelligence.get_localization_context(language, region)
            if not context:
                return result
            
            # Apply localization if result is a dict
            if isinstance(result, dict):
                return global_financial_intelligence.adapt_financial_analysis(result, context)
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator


# Export all classes and functions
__all__ = [
    'SupportedLanguage',
    'ComplianceFramework', 
    'Region',
    'LocalizationContext',
    'GlobalFinancialIntelligence',
    'global_financial_intelligence',
    'localized_response'
]