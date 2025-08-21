"""
Global Compliance System v4.0
Comprehensive internationalization, compliance, and global-first implementation.
"""

import asyncio
import json
import time
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import logging
import functools
import re
from datetime import datetime, timezone
import hashlib

logger = logging.getLogger(__name__)


class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    GDPR = "gdpr"          # General Data Protection Regulation (EU)
    CCPA = "ccpa"          # California Consumer Privacy Act (US)
    PDPA = "pdpa"          # Personal Data Protection Act (Singapore)
    PIPEDA = "pipeda"      # Personal Information Protection and Electronic Documents Act (Canada)
    LGPD = "lgpd"          # Lei Geral de Proteção de Dados (Brazil)
    POPIA = "popia"        # Protection of Personal Information Act (South Africa)
    SOX = "sox"            # Sarbanes-Oxley Act (US)
    PCI_DSS = "pci_dss"    # Payment Card Industry Data Security Standard
    ISO27001 = "iso27001"  # ISO/IEC 27001
    NIST = "nist"          # NIST Cybersecurity Framework


class DataCategory(Enum):
    """Categories of data for compliance"""
    PERSONAL_IDENTIFIABLE = "pii"
    FINANCIAL = "financial"
    HEALTH = "health"
    BIOMETRIC = "biometric"
    LOCATION = "location"
    BEHAVIORAL = "behavioral"
    MARKETING = "marketing"
    TECHNICAL = "technical"


class ConsentType(Enum):
    """Types of consent"""
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    OPT_IN = "opt_in"
    OPT_OUT = "opt_out"
    LEGITIMATE_INTEREST = "legitimate_interest"


@dataclass
class ComplianceRule:
    """Individual compliance rule"""
    id: str
    framework: ComplianceFramework
    title: str
    description: str
    data_categories: List[DataCategory]
    required_actions: List[str]
    max_retention_days: Optional[int] = None
    consent_required: bool = False
    consent_type: Optional[ConsentType] = None
    cross_border_restrictions: bool = False
    encryption_required: bool = False
    audit_required: bool = False
    right_to_deletion: bool = False
    right_to_portability: bool = False
    breach_notification_hours: Optional[int] = None


@dataclass
class DataProcessingRecord:
    """Record of data processing activities"""
    id: str
    timestamp: datetime
    data_subject_id: str
    data_categories: List[DataCategory]
    processing_purpose: str
    legal_basis: str
    consent_obtained: bool
    consent_type: Optional[ConsentType]
    data_source: str
    data_destination: str
    retention_period: Optional[int]
    encryption_used: bool
    cross_border_transfer: bool
    compliance_frameworks: List[ComplianceFramework]


class InternationalizationManager:
    """Manages internationalization and localization"""
    
    def __init__(self):
        self.supported_languages = {
            "en": {"name": "English", "locale": "en_US", "currency": "USD", "date_format": "%Y-%m-%d"},
            "es": {"name": "Español", "locale": "es_ES", "currency": "EUR", "date_format": "%d/%m/%Y"},
            "fr": {"name": "Français", "locale": "fr_FR", "currency": "EUR", "date_format": "%d/%m/%Y"},
            "de": {"name": "Deutsch", "locale": "de_DE", "currency": "EUR", "date_format": "%d.%m.%Y"},
            "ja": {"name": "日本語", "locale": "ja_JP", "currency": "JPY", "date_format": "%Y年%m月%d日"},
            "zh": {"name": "中文", "locale": "zh_CN", "currency": "CNY", "date_format": "%Y-%m-%d"},
            "pt": {"name": "Português", "locale": "pt_BR", "currency": "BRL", "date_format": "%d/%m/%Y"},
            "ru": {"name": "Русский", "locale": "ru_RU", "currency": "RUB", "date_format": "%d.%m.%Y"},
            "ar": {"name": "العربية", "locale": "ar_SA", "currency": "SAR", "date_format": "%d/%m/%Y"},
            "hi": {"name": "हिन्दी", "locale": "hi_IN", "currency": "INR", "date_format": "%d/%m/%Y"}
        }
        
        self.translations: Dict[str, Dict[str, str]] = {}
        self.current_language = "en"
        self._load_translations()
    
    def _load_translations(self):
        """Load translation strings"""
        # Base translations for common terms
        base_translations = {
            "error": {
                "en": "Error",
                "es": "Error",
                "fr": "Erreur",
                "de": "Fehler",
                "ja": "エラー",
                "zh": "错误",
                "pt": "Erro",
                "ru": "Ошибка",
                "ar": "خطأ",
                "hi": "त्रुटि"
            },
            "success": {
                "en": "Success",
                "es": "Éxito",
                "fr": "Succès",
                "de": "Erfolg",
                "ja": "成功",
                "zh": "成功",
                "pt": "Sucesso",
                "ru": "Успех",
                "ar": "نجح",
                "hi": "सफलता"
            },
            "processing": {
                "en": "Processing",
                "es": "Procesando",
                "fr": "Traitement",
                "de": "Verarbeitung",
                "ja": "処理中",
                "zh": "处理中",
                "pt": "Processando",
                "ru": "Обработка",
                "ar": "جاري المعالجة",
                "hi": "प्रसंस्करण"
            },
            "data_privacy": {
                "en": "Data Privacy",
                "es": "Privacidad de Datos",
                "fr": "Confidentialité des Données",
                "de": "Datenschutz",
                "ja": "データプライバシー",
                "zh": "数据隐私",
                "pt": "Privacidade de Dados",
                "ru": "Конфиденциальность Данных",
                "ar": "خصوصية البيانات",
                "hi": "डेटा गोपनीयता"
            },
            "consent_required": {
                "en": "Consent Required",
                "es": "Consentimiento Requerido",
                "fr": "Consentement Requis",
                "de": "Einverständnis Erforderlich",
                "ja": "同意が必要",
                "zh": "需要同意",
                "pt": "Consentimento Necessário",
                "ru": "Требуется Согласие",
                "ar": "الموافقة مطلوبة",
                "hi": "सहमति आवश्यक"
            }
        }
        
        # Reorganize translations by language
        for key, translations in base_translations.items():
            for lang, translation in translations.items():
                if lang not in self.translations:
                    self.translations[lang] = {}
                self.translations[lang][key] = translation
    
    def set_language(self, language_code: str):
        """Set current language"""
        if language_code in self.supported_languages:
            self.current_language = language_code
            logger.info(f"🌍 Language set to: {self.supported_languages[language_code]['name']}")
        else:
            logger.warning(f"⚠️ Unsupported language: {language_code}")
    
    def translate(self, key: str, language: Optional[str] = None) -> str:
        """Translate a key to the specified or current language"""
        target_language = language or self.current_language
        
        if target_language in self.translations and key in self.translations[target_language]:
            return self.translations[target_language][key]
        
        # Fallback to English
        if "en" in self.translations and key in self.translations["en"]:
            return self.translations["en"][key]
        
        # Fallback to key itself
        return key
    
    def format_currency(self, amount: float, language: Optional[str] = None) -> str:
        """Format currency according to locale"""
        target_language = language or self.current_language
        locale_info = self.supported_languages.get(target_language, self.supported_languages["en"])
        currency = locale_info["currency"]
        
        # Simple currency formatting
        currency_symbols = {
            "USD": "$", "EUR": "€", "JPY": "¥", "CNY": "¥", 
            "BRL": "R$", "RUB": "₽", "SAR": "﷼", "INR": "₹"
        }
        
        symbol = currency_symbols.get(currency, currency)
        
        if target_language in ["ja", "zh"]:
            return f"{symbol}{amount:,.0f}"
        else:
            return f"{symbol}{amount:,.2f}"
    
    def format_date(self, date: datetime, language: Optional[str] = None) -> str:
        """Format date according to locale"""
        target_language = language or self.current_language
        locale_info = self.supported_languages.get(target_language, self.supported_languages["en"])
        date_format = locale_info["date_format"]
        
        return date.strftime(date_format)
    
    def get_language_info(self, language: Optional[str] = None) -> Dict[str, str]:
        """Get language information"""
        target_language = language or self.current_language
        return self.supported_languages.get(target_language, self.supported_languages["en"])


class ComplianceEngine:
    """Engine for managing compliance rules and validation"""
    
    def __init__(self):
        self.rules: Dict[str, ComplianceRule] = {}
        self.processing_records: deque = deque(maxlen=10000)
        self.compliance_cache: Dict[str, Any] = {}
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default compliance rules"""
        
        # GDPR Rules
        self.add_rule(ComplianceRule(
            id="gdpr_consent",
            framework=ComplianceFramework.GDPR,
            title="GDPR Consent Requirement",
            description="Explicit consent required for personal data processing",
            data_categories=[DataCategory.PERSONAL_IDENTIFIABLE],
            required_actions=["obtain_consent", "record_consent", "allow_withdrawal"],
            consent_required=True,
            consent_type=ConsentType.EXPLICIT,
            right_to_deletion=True,
            right_to_portability=True
        ))
        
        self.add_rule(ComplianceRule(
            id="gdpr_retention",
            framework=ComplianceFramework.GDPR,
            title="GDPR Data Retention Limits",
            description="Personal data must not be kept longer than necessary",
            data_categories=[DataCategory.PERSONAL_IDENTIFIABLE],
            required_actions=["define_retention_period", "automated_deletion"],
            max_retention_days=365,  # 1 year default
            audit_required=True
        ))
        
        self.add_rule(ComplianceRule(
            id="gdpr_breach_notification",
            framework=ComplianceFramework.GDPR,
            title="GDPR Breach Notification",
            description="Data breaches must be reported within 72 hours",
            data_categories=[DataCategory.PERSONAL_IDENTIFIABLE],
            required_actions=["detect_breach", "assess_risk", "notify_authority", "notify_subjects"],
            breach_notification_hours=72,
            audit_required=True
        ))
        
        # CCPA Rules
        self.add_rule(ComplianceRule(
            id="ccpa_disclosure",
            framework=ComplianceFramework.CCPA,
            title="CCPA Right to Know",
            description="Consumers have right to know what personal information is collected",
            data_categories=[DataCategory.PERSONAL_IDENTIFIABLE],
            required_actions=["disclose_categories", "disclose_sources", "disclose_purposes"],
            right_to_deletion=True,
            audit_required=True
        ))
        
        # PCI DSS Rules
        self.add_rule(ComplianceRule(
            id="pci_dss_encryption",
            framework=ComplianceFramework.PCI_DSS,
            title="PCI DSS Encryption Requirement",
            description="All financial data must be encrypted",
            data_categories=[DataCategory.FINANCIAL],
            required_actions=["encrypt_at_rest", "encrypt_in_transit", "key_management"],
            encryption_required=True,
            audit_required=True
        ))
        
        # SOX Rules
        self.add_rule(ComplianceRule(
            id="sox_financial_controls",
            framework=ComplianceFramework.SOX,
            title="SOX Financial Controls",
            description="Internal controls over financial reporting",
            data_categories=[DataCategory.FINANCIAL],
            required_actions=["implement_controls", "test_controls", "document_controls"],
            audit_required=True,
            max_retention_days=2555  # 7 years
        ))
    
    def add_rule(self, rule: ComplianceRule):
        """Add a compliance rule"""
        self.rules[rule.id] = rule
        logger.info(f"📋 Compliance rule added: {rule.title} ({rule.framework.value})")
    
    def get_applicable_rules(self, data_categories: List[DataCategory], frameworks: List[ComplianceFramework] = None) -> List[ComplianceRule]:
        """Get applicable compliance rules for given data categories"""
        applicable_rules = []
        
        for rule in self.rules.values():
            # Check if rule applies to any of the data categories
            if any(cat in rule.data_categories for cat in data_categories):
                # Check framework filter if provided
                if frameworks is None or rule.framework in frameworks:
                    applicable_rules.append(rule)
        
        return applicable_rules
    
    def validate_compliance(self, processing_record: DataProcessingRecord) -> Dict[str, Any]:
        """Validate compliance for a data processing record"""
        applicable_rules = self.get_applicable_rules(
            processing_record.data_categories,
            processing_record.compliance_frameworks
        )
        
        validation_results = {
            "compliant": True,
            "violations": [],
            "warnings": [],
            "applicable_rules": len(applicable_rules),
            "details": {}
        }
        
        for rule in applicable_rules:
            rule_validation = self._validate_rule(rule, processing_record)
            validation_results["details"][rule.id] = rule_validation
            
            if not rule_validation["compliant"]:
                validation_results["compliant"] = False
                validation_results["violations"].extend(rule_validation["violations"])
            
            validation_results["warnings"].extend(rule_validation.get("warnings", []))
        
        return validation_results
    
    def _validate_rule(self, rule: ComplianceRule, record: DataProcessingRecord) -> Dict[str, Any]:
        """Validate a specific rule against a processing record"""
        violations = []
        warnings = []
        
        # Check consent requirements
        if rule.consent_required and not record.consent_obtained:
            violations.append(f"Consent required for {rule.framework.value} but not obtained")
        
        # Check consent type
        if rule.consent_type and record.consent_type != rule.consent_type:
            violations.append(f"Incorrect consent type: required {rule.consent_type.value}, got {record.consent_type.value if record.consent_type else 'none'}")
        
        # Check retention period
        if rule.max_retention_days and record.retention_period:
            if record.retention_period > rule.max_retention_days:
                violations.append(f"Retention period exceeds maximum: {record.retention_period} > {rule.max_retention_days} days")
        
        # Check encryption requirements
        if rule.encryption_required and not record.encryption_used:
            violations.append(f"Encryption required for {rule.framework.value} but not used")
        
        # Check cross-border transfer restrictions
        if rule.cross_border_restrictions and record.cross_border_transfer:
            warnings.append(f"Cross-border transfer detected - ensure adequate protections for {rule.framework.value}")
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "warnings": warnings,
            "rule_id": rule.id,
            "framework": rule.framework.value
        }
    
    def record_data_processing(self, record: DataProcessingRecord):
        """Record a data processing activity"""
        self.processing_records.append(record)
        
        # Validate compliance
        validation = self.validate_compliance(record)
        
        if not validation["compliant"]:
            logger.warning(f"⚠️ Compliance violations detected for {record.id}: {validation['violations']}")
        
        logger.info(f"📊 Data processing recorded: {record.id} ({len(record.data_categories)} categories)")
        
        return validation
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get compliance summary"""
        recent_records = [r for r in self.processing_records if (datetime.now(timezone.utc) - r.timestamp).days <= 30]
        
        framework_counts = defaultdict(int)
        category_counts = defaultdict(int)
        violation_counts = defaultdict(int)
        
        for record in recent_records:
            for framework in record.compliance_frameworks:
                framework_counts[framework.value] += 1
            
            for category in record.data_categories:
                category_counts[category.value] += 1
            
            validation = self.validate_compliance(record)
            if not validation["compliant"]:
                violation_counts["violations"] += len(validation["violations"])
                violation_counts["records_with_violations"] += 1
        
        compliance_rate = (
            (len(recent_records) - violation_counts["records_with_violations"]) / len(recent_records) * 100
            if recent_records else 100.0
        )
        
        return {
            "total_rules": len(self.rules),
            "recent_processing_records": len(recent_records),
            "compliance_rate": compliance_rate,
            "framework_distribution": dict(framework_counts),
            "category_distribution": dict(category_counts),
            "violation_summary": dict(violation_counts),
            "supported_frameworks": [f.value for f in ComplianceFramework]
        }


class GlobalComplianceSystem:
    """
    Comprehensive Global Compliance System
    Combines internationalization and compliance management
    """
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.i18n_manager = InternationalizationManager()
        self.compliance_engine = ComplianceEngine()
        self.audit_log: deque = deque(maxlen=10000)
        self.data_inventory: Dict[str, Any] = {}
        
    def configure_for_region(self, region: str, language: str = None):
        """Configure system for specific region"""
        region_configs = {
            "EU": {
                "language": "en",
                "frameworks": [ComplianceFramework.GDPR, ComplianceFramework.ISO27001],
                "data_residency": "EU",
                "encryption_required": True
            },
            "US": {
                "language": "en", 
                "frameworks": [ComplianceFramework.CCPA, ComplianceFramework.SOX, ComplianceFramework.PCI_DSS],
                "data_residency": "US",
                "encryption_required": True
            },
            "APAC": {
                "language": language or "en",
                "frameworks": [ComplianceFramework.PDPA, ComplianceFramework.ISO27001],
                "data_residency": "APAC",
                "encryption_required": True
            },
            "LATAM": {
                "language": "pt",
                "frameworks": [ComplianceFramework.LGPD, ComplianceFramework.ISO27001],
                "data_residency": "LATAM",
                "encryption_required": True
            }
        }
        
        config = region_configs.get(region, region_configs["US"])
        
        # Set language
        self.i18n_manager.set_language(config["language"])
        
        # Log configuration
        self.audit_log.append({
            "timestamp": datetime.now(timezone.utc),
            "action": "region_configuration",
            "region": region,
            "language": config["language"],
            "frameworks": [f.value for f in config["frameworks"]],
            "data_residency": config["data_residency"]
        })
        
        logger.info(f"🌍 Configured for region: {region}")
        logger.info(f"   Language: {config['language']}")
        logger.info(f"   Frameworks: {[f.value for f in config['frameworks']]}")
        logger.info(f"   Data Residency: {config['data_residency']}")
        
        return config
    
    def process_data_with_compliance(
        self,
        data_subject_id: str,
        data_categories: List[DataCategory],
        processing_purpose: str,
        legal_basis: str,
        consent_obtained: bool = False,
        consent_type: Optional[ConsentType] = None,
        frameworks: List[ComplianceFramework] = None
    ) -> Dict[str, Any]:
        """Process data with full compliance checking"""
        
        # Create processing record
        record = DataProcessingRecord(
            id=f"proc_{int(time.time())}_{hash(data_subject_id) % 10000}",
            timestamp=datetime.now(timezone.utc),
            data_subject_id=data_subject_id,
            data_categories=data_categories,
            processing_purpose=processing_purpose,
            legal_basis=legal_basis,
            consent_obtained=consent_obtained,
            consent_type=consent_type,
            data_source="application",
            data_destination="internal_system",
            retention_period=365,  # Default 1 year
            encryption_used=True,   # Assume encryption is used
            cross_border_transfer=False,
            compliance_frameworks=frameworks or [ComplianceFramework.GDPR, ComplianceFramework.CCPA]
        )
        
        # Record and validate
        validation = self.compliance_engine.record_data_processing(record)
        
        # Add to audit log
        self.audit_log.append({
            "timestamp": datetime.now(timezone.utc),
            "action": "data_processing",
            "record_id": record.id,
            "data_subject_id": data_subject_id,
            "compliant": validation["compliant"],
            "violations": validation["violations"],
            "frameworks": [f.value for f in record.compliance_frameworks]
        })
        
        return {
            "record_id": record.id,
            "processing_allowed": validation["compliant"],
            "compliance_validation": validation,
            "localized_message": self.i18n_manager.translate(
                "success" if validation["compliant"] else "error"
            )
        }
    
    def handle_data_subject_request(
        self,
        request_type: str,  # "access", "deletion", "portability", "rectification"
        data_subject_id: str,
        frameworks: List[ComplianceFramework] = None
    ) -> Dict[str, Any]:
        """Handle data subject rights requests"""
        
        # Find applicable rules
        applicable_rules = []
        if frameworks:
            for framework in frameworks:
                framework_rules = [r for r in self.compliance_engine.rules.values() if r.framework == framework]
                applicable_rules.extend(framework_rules)
        
        # Check if request type is supported
        supported_rights = {
            "access": ["gdpr_consent", "ccpa_disclosure"],
            "deletion": any(r.right_to_deletion for r in applicable_rules),
            "portability": any(r.right_to_portability for r in applicable_rules),
            "rectification": True  # Generally supported
        }
        
        request_supported = supported_rights.get(request_type, False)
        
        # Process the request
        if request_supported:
            # Find related processing records
            related_records = [
                r for r in self.compliance_engine.processing_records
                if r.data_subject_id == data_subject_id
            ]
            
            response = {
                "request_id": f"req_{int(time.time())}_{hash(data_subject_id) % 10000}",
                "request_type": request_type,
                "data_subject_id": data_subject_id,
                "status": "processed",
                "related_records": len(related_records),
                "processing_time": datetime.now(timezone.utc),
                "localized_status": self.i18n_manager.translate("success")
            }
            
            # Type-specific processing
            if request_type == "access":
                response["data_categories"] = list(set(
                    cat.value for record in related_records for cat in record.data_categories
                ))
            elif request_type == "deletion":
                response["records_deleted"] = len(related_records)
                # In real implementation, would delete the records
            elif request_type == "portability":
                response["export_format"] = "JSON"
                response["export_size"] = f"{len(related_records)} records"
            
        else:
            response = {
                "request_id": f"req_{int(time.time())}_{hash(data_subject_id) % 10000}",
                "request_type": request_type,
                "data_subject_id": data_subject_id,
                "status": "not_supported",
                "reason": f"Request type '{request_type}' not supported under applicable frameworks",
                "localized_status": self.i18n_manager.translate("error")
            }
        
        # Add to audit log
        self.audit_log.append({
            "timestamp": datetime.now(timezone.utc),
            "action": "data_subject_request",
            "request_type": request_type,
            "data_subject_id": data_subject_id,
            "status": response["status"]
        })
        
        return response
    
    def generate_privacy_notice(self, language: str = None) -> Dict[str, Any]:
        """Generate localized privacy notice"""
        target_language = language or self.i18n_manager.current_language
        
        # Get applicable frameworks and rules
        frameworks = [ComplianceFramework.GDPR, ComplianceFramework.CCPA, ComplianceFramework.PDPA]
        applicable_rules = []
        for framework in frameworks:
            framework_rules = [r for r in self.compliance_engine.rules.values() if r.framework == framework]
            applicable_rules.extend(framework_rules)
        
        # Generate notice content
        notice = {
            "language": target_language,
            "last_updated": self.i18n_manager.format_date(datetime.now()),
            "sections": {
                "data_collection": {
                    "title": self.i18n_manager.translate("data_privacy", target_language),
                    "content": "We collect personal information to provide our financial analysis services."
                },
                "legal_basis": {
                    "title": "Legal Basis for Processing",
                    "content": "Processing is based on legitimate business interests and user consent."
                },
                "your_rights": {
                    "title": "Your Rights",
                    "rights": []
                },
                "contact": {
                    "title": "Contact Information",
                    "email": "privacy@finchat.com",
                    "address": "Privacy Office, FinChat Inc."
                }
            },
            "applicable_frameworks": [f.value for f in frameworks]
        }
        
        # Add rights based on applicable rules
        rights = []
        if any(r.right_to_deletion for r in applicable_rules):
            rights.append("Right to deletion")
        if any(r.right_to_portability for r in applicable_rules):
            rights.append("Right to data portability")
        if any(r.consent_required for r in applicable_rules):
            rights.append("Right to withdraw consent")
        
        notice["sections"]["your_rights"]["rights"] = rights
        
        return notice
    
    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive compliance dashboard"""
        compliance_summary = self.compliance_engine.get_compliance_summary()
        
        # Recent audit events
        recent_audit = [
            event for event in self.audit_log
            if (datetime.now(timezone.utc) - event["timestamp"]).days <= 7
        ]
        
        # Language and region info
        current_lang_info = self.i18n_manager.get_language_info()
        
        return {
            "project_name": self.project_name,
            "current_language": {
                "code": self.i18n_manager.current_language,
                "name": current_lang_info["name"],
                "locale": current_lang_info["locale"]
            },
            "supported_languages": len(self.i18n_manager.supported_languages),
            "compliance_summary": compliance_summary,
            "recent_audit_events": len(recent_audit),
            "audit_event_types": list(set(event["action"] for event in recent_audit)),
            "data_inventory_size": len(self.data_inventory),
            "localized_status": self.i18n_manager.translate("success")
        }
    
    def export_compliance_report(self, filename: Optional[str] = None) -> str:
        """Export comprehensive compliance report"""
        if not filename:
            timestamp = int(time.time())
            filename = f"compliance_report_{timestamp}.json"
        
        report = {
            "project_name": self.project_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "language": self.i18n_manager.current_language,
            "compliance_summary": self.compliance_engine.get_compliance_summary(),
            "audit_log": [
                {**event, "timestamp": event["timestamp"].isoformat()}
                for event in list(self.audit_log)
            ],
            "privacy_notices": {
                lang: self.generate_privacy_notice(lang)
                for lang in ["en", "es", "fr", "de"]
            },
            "supported_frameworks": [f.value for f in ComplianceFramework],
            "supported_languages": list(self.i18n_manager.supported_languages.keys())
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"📊 Compliance report exported: {filename}")
        return filename


# Factory function
def create_global_compliance_system(project_name: str) -> GlobalComplianceSystem:
    """Create global compliance system"""
    return GlobalComplianceSystem(project_name)


# Example usage
async def demonstrate_global_compliance():
    """Demonstrate global compliance system"""
    system = create_global_compliance_system("FinChat-SEC-QA")
    
    # Configure for different regions
    eu_config = system.configure_for_region("EU")
    
    # Process some data with compliance
    result = system.process_data_with_compliance(
        data_subject_id="user_123",
        data_categories=[DataCategory.PERSONAL_IDENTIFIABLE, DataCategory.FINANCIAL],
        processing_purpose="Financial analysis and reporting",
        legal_basis="Legitimate business interest",
        consent_obtained=True,
        consent_type=ConsentType.EXPLICIT,
        frameworks=[ComplianceFramework.GDPR, ComplianceFramework.PCI_DSS]
    )
    
    # Handle data subject request
    deletion_request = system.handle_data_subject_request(
        request_type="deletion",
        data_subject_id="user_123",
        frameworks=[ComplianceFramework.GDPR]
    )
    
    # Generate privacy notice
    privacy_notice = system.generate_privacy_notice("en")
    
    # Get dashboard
    dashboard = system.get_compliance_dashboard()
    
    # Export report
    report_file = system.export_compliance_report()
    
    logger.info(f"🌍 Global compliance demonstration completed")
    logger.info(f"   Data processing: {result['processing_allowed']}")
    logger.info(f"   Deletion request: {deletion_request['status']}")
    logger.info(f"   Compliance rate: {dashboard['compliance_summary']['compliance_rate']:.1f}%")
    logger.info(f"   Report saved: {report_file}")
    
    return dashboard, report_file


if __name__ == "__main__":
    # Example usage
    async def main():
        dashboard, report_file = await demonstrate_global_compliance()
        print(f"🌍 Global compliance system completed")
        print(f"📊 Compliance dashboard: {dashboard}")
    
    asyncio.run(main())