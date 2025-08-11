"""
Global-First Implementation for Multi-Region, Multi-Language Financial Analysis.

This module provides comprehensive internationalization, localization, and 
compliance features for global deployment of the financial analysis platform.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import re

from .config import get_config
from .logging_utils import configure_logging

logger = logging.getLogger(__name__)


class SupportedRegion(Enum):
    """Supported global regions for deployment."""
    NORTH_AMERICA = "na"
    EUROPE = "eu"
    ASIA_PACIFIC = "apac"
    LATIN_AMERICA = "latam"
    MIDDLE_EAST_AFRICA = "mea"


class SupportedLanguage(Enum):
    """Supported languages with ISO 639-1 codes."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    KOREAN = "ko"
    DUTCH = "nl"


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"  # EU General Data Protection Regulation
    CCPA = "ccpa"  # California Consumer Privacy Act
    PDPA = "pdpa"  # Singapore Personal Data Protection Act
    PIPEDA = "pipeda"  # Canada Personal Information Protection
    LGPD = "lgpd"  # Brazil Lei Geral de Proteção de Dados
    SOX = "sox"  # Sarbanes-Oxley Act
    MiFID_II = "mifid2"  # Markets in Financial Instruments Directive


@dataclass
class RegionConfig:
    """Configuration for a specific region."""
    region: SupportedRegion
    primary_language: SupportedLanguage
    secondary_languages: List[SupportedLanguage]
    compliance_frameworks: List[ComplianceFramework]
    data_residency_required: bool
    timezone: str
    currency_codes: List[str]
    regulatory_endpoints: Dict[str, str] = field(default_factory=dict)


@dataclass
class LocalizationResource:
    """Localization resource for UI elements and messages."""
    key: str
    language: SupportedLanguage
    text: str
    context: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.now)


class GlobalFirstImplementation:
    """
    Global-first implementation providing internationalization, localization,
    and compliance features for worldwide deployment.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".cache" / "finchat_sec_qa" / "global"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.region_configs = self._initialize_region_configs()
        self.localization_resources: Dict[str, Dict[SupportedLanguage, str]] = {}
        self.compliance_handlers: Dict[ComplianceFramework, Any] = {}
        
        self._load_localizations()
        self._initialize_compliance_handlers()
        configure_logging()

    def _initialize_region_configs(self) -> Dict[SupportedRegion, RegionConfig]:
        """Initialize configuration for all supported regions."""
        return {
            SupportedRegion.NORTH_AMERICA: RegionConfig(
                region=SupportedRegion.NORTH_AMERICA,
                primary_language=SupportedLanguage.ENGLISH,
                secondary_languages=[SupportedLanguage.SPANISH, SupportedLanguage.FRENCH],
                compliance_frameworks=[ComplianceFramework.CCPA, ComplianceFramework.SOX, ComplianceFramework.PIPEDA],
                data_residency_required=False,
                timezone="America/New_York",
                currency_codes=["USD", "CAD", "MXN"],
                regulatory_endpoints={
                    "sec": "https://www.sec.gov/edgar/",
                    "csa": "https://www.securities-administrators.ca/",
                    "cnbv": "https://www.cnbv.gob.mx/"
                }
            ),
            
            SupportedRegion.EUROPE: RegionConfig(
                region=SupportedRegion.EUROPE,
                primary_language=SupportedLanguage.ENGLISH,
                secondary_languages=[SupportedLanguage.GERMAN, SupportedLanguage.FRENCH, SupportedLanguage.SPANISH, SupportedLanguage.ITALIAN, SupportedLanguage.DUTCH],
                compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.MiFID_II],
                data_residency_required=True,
                timezone="Europe/London",
                currency_codes=["EUR", "GBP", "CHF", "SEK", "NOK", "DKK"],
                regulatory_endpoints={
                    "esma": "https://www.esma.europa.eu/",
                    "fca": "https://www.fca.org.uk/",
                    "bafin": "https://www.bafin.de/"
                }
            ),
            
            SupportedRegion.ASIA_PACIFIC: RegionConfig(
                region=SupportedRegion.ASIA_PACIFIC,
                primary_language=SupportedLanguage.ENGLISH,
                secondary_languages=[SupportedLanguage.JAPANESE, SupportedLanguage.CHINESE_SIMPLIFIED, SupportedLanguage.KOREAN],
                compliance_frameworks=[ComplianceFramework.PDPA],
                data_residency_required=True,
                timezone="Asia/Singapore",
                currency_codes=["USD", "JPY", "CNY", "KRW", "SGD", "AUD"],
                regulatory_endpoints={
                    "jfsa": "https://www.fsa.go.jp/",
                    "csrc": "http://www.csrc.gov.cn/",
                    "mas": "https://www.mas.gov.sg/"
                }
            ),
            
            SupportedRegion.LATIN_AMERICA: RegionConfig(
                region=SupportedRegion.LATIN_AMERICA,
                primary_language=SupportedLanguage.SPANISH,
                secondary_languages=[SupportedLanguage.PORTUGUESE, SupportedLanguage.ENGLISH],
                compliance_frameworks=[ComplianceFramework.LGPD],
                data_residency_required=True,
                timezone="America/Sao_Paulo",
                currency_codes=["BRL", "MXN", "ARS", "COP", "CLP", "PEN"],
                regulatory_endpoints={
                    "cvm": "https://www.cvm.gov.br/",
                    "cnv": "https://www.cnv.gov.ar/",
                    "smv": "https://www.smv.gob.pe/"
                }
            ),
            
            SupportedRegion.MIDDLE_EAST_AFRICA: RegionConfig(
                region=SupportedRegion.MIDDLE_EAST_AFRICA,
                primary_language=SupportedLanguage.ENGLISH,
                secondary_languages=[SupportedLanguage.FRENCH],
                compliance_frameworks=[],  # Region-specific frameworks would be added
                data_residency_required=True,
                timezone="Africa/Johannesburg",
                currency_codes=["AED", "SAR", "ZAR", "EGP"],
                regulatory_endpoints={
                    "dfsa": "https://www.dfsa.ae/",
                    "fsb": "https://www.fsb.co.za/",
                    "fra": "https://www.fra.gov.eg/"
                }
            )
        }

    def _load_localizations(self) -> None:
        """Load localization resources from files."""
        try:
            localization_file = self.storage_path / "localizations.json"
            if localization_file.exists():
                with open(localization_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key, translations in data.items():
                        self.localization_resources[key] = {
                            SupportedLanguage(lang): text 
                            for lang, text in translations.items()
                        }
            else:
                self._create_default_localizations()
                
        except Exception as e:
            logger.error(f"Error loading localizations: {e}")
            self._create_default_localizations()

    def _create_default_localizations(self) -> None:
        """Create default localization resources."""
        default_texts = {
            "app_title": {
                SupportedLanguage.ENGLISH: "Financial Analysis Platform",
                SupportedLanguage.SPANISH: "Plataforma de Análisis Financiero",
                SupportedLanguage.FRENCH: "Plateforme d'Analyse Financière",
                SupportedLanguage.GERMAN: "Finanzanalyse-Plattform",
                SupportedLanguage.JAPANESE: "財務分析プラットフォーム",
                SupportedLanguage.CHINESE_SIMPLIFIED: "财务分析平台",
                SupportedLanguage.PORTUGUESE: "Plataforma de Análise Financeira",
                SupportedLanguage.ITALIAN: "Piattaforma di Analisi Finanziaria",
                SupportedLanguage.KOREAN: "금융 분석 플랫폼",
                SupportedLanguage.DUTCH: "Financiële Analyse Platform"
            },
            
            "query_placeholder": {
                SupportedLanguage.ENGLISH: "Enter your financial question...",
                SupportedLanguage.SPANISH: "Ingresa tu pregunta financiera...",
                SupportedLanguage.FRENCH: "Saisissez votre question financière...",
                SupportedLanguage.GERMAN: "Geben Sie Ihre Finanzfrage ein...",
                SupportedLanguage.JAPANESE: "財務に関する質問を入力してください...",
                SupportedLanguage.CHINESE_SIMPLIFIED: "输入您的财务问题...",
                SupportedLanguage.PORTUGUESE: "Digite sua pergunta financeira...",
                SupportedLanguage.ITALIAN: "Inserisci la tua domanda finanziaria...",
                SupportedLanguage.KOREAN: "재무 관련 질문을 입력하세요...",
                SupportedLanguage.DUTCH: "Voer uw financiële vraag in..."
            },
            
            "risk_analysis": {
                SupportedLanguage.ENGLISH: "Risk Analysis",
                SupportedLanguage.SPANISH: "Análisis de Riesgos",
                SupportedLanguage.FRENCH: "Analyse des Risques",
                SupportedLanguage.GERMAN: "Risikoanalyse",
                SupportedLanguage.JAPANESE: "リスク分析",
                SupportedLanguage.CHINESE_SIMPLIFIED: "风险分析",
                SupportedLanguage.PORTUGUESE: "Análise de Risco",
                SupportedLanguage.ITALIAN: "Analisi del Rischio",
                SupportedLanguage.KOREAN: "위험 분석",
                SupportedLanguage.DUTCH: "Risicoanalyse"
            },
            
            "data_processing_consent": {
                SupportedLanguage.ENGLISH: "I consent to the processing of my data for financial analysis purposes",
                SupportedLanguage.SPANISH: "Consiento el procesamiento de mis datos para fines de análisis financiero",
                SupportedLanguage.FRENCH: "Je consens au traitement de mes données à des fins d'analyse financière",
                SupportedLanguage.GERMAN: "Ich stimme der Verarbeitung meiner Daten für Finanzanalysezwecke zu",
                SupportedLanguage.JAPANESE: "財務分析目的でのデータ処理に同意します",
                SupportedLanguage.CHINESE_SIMPLIFIED: "我同意为财务分析目的处理我的数据",
                SupportedLanguage.PORTUGUESE: "Consinto com o processamento dos meus dados para fins de análise financeira",
                SupportedLanguage.ITALIAN: "Acconsento al trattamento dei miei dati per scopi di analisi finanziaria",
                SupportedLanguage.KOREAN: "재무 분석 목적으로 내 데이터 처리에 동의합니다",
                SupportedLanguage.DUTCH: "Ik stem in met de verwerking van mijn gegevens voor financiële analysedoeleinden"
            },
            
            "error_general": {
                SupportedLanguage.ENGLISH: "An error occurred. Please try again.",
                SupportedLanguage.SPANISH: "Ocurrió un error. Por favor, inténtalo de nuevo.",
                SupportedLanguage.FRENCH: "Une erreur s'est produite. Veuillez réessayer.",
                SupportedLanguage.GERMAN: "Es ist ein Fehler aufgetreten. Bitte versuchen Sie es erneut.",
                SupportedLanguage.JAPANESE: "エラーが発生しました。もう一度お試しください。",
                SupportedLanguage.CHINESE_SIMPLIFIED: "发生错误。请重试。",
                SupportedLanguage.PORTUGUESE: "Ocorreu um erro. Tente novamente.",
                SupportedLanguage.ITALIAN: "Si è verificato un errore. Riprova.",
                SupportedLanguage.KOREAN: "오류가 발생했습니다. 다시 시도해 주세요.",
                SupportedLanguage.DUTCH: "Er is een fout opgetreden. Probeer het opnieuw."
            }
        }
        
        self.localization_resources.update(default_texts)
        self._save_localizations()

    def _initialize_compliance_handlers(self) -> None:
        """Initialize compliance framework handlers."""
        self.compliance_handlers = {
            ComplianceFramework.GDPR: GDPRComplianceHandler(),
            ComplianceFramework.CCPA: CCPAComplianceHandler(),
            ComplianceFramework.PDPA: PDPAComplianceHandler(),
            ComplianceFramework.PIPEDA: PIPEDAComplianceHandler(),
            ComplianceFramework.LGPD: LGPDComplianceHandler(),
            ComplianceFramework.SOX: SOXComplianceHandler(),
            ComplianceFramework.MiFID_II: MiFIDIIComplianceHandler()
        }

    def get_localized_text(self, key: str, language: SupportedLanguage, 
                          default: Optional[str] = None) -> str:
        """Get localized text for a specific key and language."""
        if key in self.localization_resources:
            if language in self.localization_resources[key]:
                return self.localization_resources[key][language]
            # Fallback to English if available
            elif SupportedLanguage.ENGLISH in self.localization_resources[key]:
                return self.localization_resources[key][SupportedLanguage.ENGLISH]
        
        return default or key

    def get_region_config(self, region: SupportedRegion) -> RegionConfig:
        """Get configuration for a specific region."""
        return self.region_configs[region]

    def detect_user_region(self, ip_address: str, accept_language: str) -> SupportedRegion:
        """Detect user region based on IP address and browser language."""
        # Simplified region detection - in practice, use GeoIP service
        
        # Check accept-language header for region hints
        if any(lang in accept_language.lower() for lang in ['es', 'pt']):
            if 'br' in accept_language.lower() or 'pt-br' in accept_language.lower():
                return SupportedRegion.LATIN_AMERICA
        
        if any(lang in accept_language.lower() for lang in ['de', 'fr', 'it', 'nl']):
            return SupportedRegion.EUROPE
        
        if any(lang in accept_language.lower() for lang in ['ja', 'zh', 'ko']):
            return SupportedRegion.ASIA_PACIFIC
        
        # Default to North America
        return SupportedRegion.NORTH_AMERICA

    def detect_user_language(self, accept_language: str) -> SupportedLanguage:
        """Detect user language from Accept-Language header."""
        # Parse Accept-Language header
        languages = []
        for lang_range in accept_language.split(','):
            parts = lang_range.strip().split(';')
            lang_code = parts[0].strip().lower()
            
            # Extract primary language code
            primary_lang = lang_code.split('-')[0]
            
            # Map to supported languages
            lang_mapping = {
                'en': SupportedLanguage.ENGLISH,
                'es': SupportedLanguage.SPANISH,
                'fr': SupportedLanguage.FRENCH,
                'de': SupportedLanguage.GERMAN,
                'ja': SupportedLanguage.JAPANESE,
                'zh': SupportedLanguage.CHINESE_SIMPLIFIED,
                'pt': SupportedLanguage.PORTUGUESE,
                'it': SupportedLanguage.ITALIAN,
                'ko': SupportedLanguage.KOREAN,
                'nl': SupportedLanguage.DUTCH
            }
            
            if primary_lang in lang_mapping:
                languages.append(lang_mapping[primary_lang])
        
        return languages[0] if languages else SupportedLanguage.ENGLISH

    def validate_compliance(self, region: SupportedRegion, 
                          data_processing_type: str,
                          user_consent: Dict[str, Any]) -> Dict[str, Any]:
        """Validate compliance requirements for data processing."""
        region_config = self.region_configs[region]
        compliance_results = {
            'compliant': True,
            'warnings': [],
            'requirements': []
        }
        
        for framework in region_config.compliance_frameworks:
            if framework in self.compliance_handlers:
                handler = self.compliance_handlers[framework]
                result = handler.validate_compliance(data_processing_type, user_consent)
                
                if not result['compliant']:
                    compliance_results['compliant'] = False
                
                compliance_results['warnings'].extend(result.get('warnings', []))
                compliance_results['requirements'].extend(result.get('requirements', []))
        
        return compliance_results

    def format_currency(self, amount: float, region: SupportedRegion, 
                       currency_code: Optional[str] = None) -> str:
        """Format currency amount according to regional standards."""
        region_config = self.region_configs[region]
        
        if not currency_code:
            currency_code = region_config.currency_codes[0]  # Use primary currency
        
        # Simplified currency formatting - in practice, use locale-specific formatting
        currency_symbols = {
            'USD': '$', 'EUR': '€', 'GBP': '£', 'JPY': '¥', 'CNY': '¥',
            'KRW': '₩', 'SGD': 'S$', 'CAD': 'C$', 'AUD': 'A$',
            'BRL': 'R$', 'MXN': '$', 'CHF': 'CHF', 'SEK': 'kr'
        }
        
        symbol = currency_symbols.get(currency_code, currency_code)
        
        # Format based on region
        if region == SupportedRegion.EUROPE:
            return f"{amount:,.2f} {symbol}"
        else:
            return f"{symbol}{amount:,.2f}"

    def format_date(self, date: datetime, region: SupportedRegion, 
                   language: SupportedLanguage) -> str:
        """Format date according to regional and language preferences."""
        # Convert to region timezone
        region_config = self.region_configs[region]
        # In practice, would use proper timezone conversion
        
        # Format based on region
        if region in [SupportedRegion.EUROPE, SupportedRegion.MIDDLE_EAST_AFRICA]:
            return date.strftime("%d/%m/%Y")
        elif region == SupportedRegion.ASIA_PACIFIC:
            if language == SupportedLanguage.JAPANESE:
                return date.strftime("%Y年%m月%d日")
            elif language == SupportedLanguage.CHINESE_SIMPLIFIED:
                return date.strftime("%Y年%m月%d日")
            elif language == SupportedLanguage.KOREAN:
                return date.strftime("%Y년 %m월 %d일")
            else:
                return date.strftime("%d/%m/%Y")
        else:  # North America, Latin America
            return date.strftime("%m/%d/%Y")

    def get_regulatory_endpoints(self, region: SupportedRegion) -> Dict[str, str]:
        """Get regulatory endpoints for a specific region."""
        region_config = self.region_configs[region]
        return region_config.regulatory_endpoints

    def is_data_residency_required(self, region: SupportedRegion) -> bool:
        """Check if data residency is required for a region."""
        region_config = self.region_configs[region]
        return region_config.data_residency_required

    def get_supported_languages(self, region: SupportedRegion) -> List[SupportedLanguage]:
        """Get supported languages for a specific region."""
        region_config = self.region_configs[region]
        return [region_config.primary_language] + region_config.secondary_languages

    def _save_localizations(self) -> None:
        """Save localization resources to file."""
        try:
            localization_file = self.storage_path / "localizations.json"
            data = {
                key: {lang.value: text for lang, text in translations.items()}
                for key, translations in self.localization_resources.items()
            }
            
            with open(localization_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving localizations: {e}")


# Compliance Handler Classes
class BaseComplianceHandler:
    """Base class for compliance framework handlers."""
    
    def validate_compliance(self, data_processing_type: str, 
                          user_consent: Dict[str, Any]) -> Dict[str, Any]:
        """Validate compliance for data processing."""
        return {'compliant': True, 'warnings': [], 'requirements': []}


class GDPRComplianceHandler(BaseComplianceHandler):
    """GDPR compliance handler for European Union."""
    
    def validate_compliance(self, data_processing_type: str, 
                          user_consent: Dict[str, Any]) -> Dict[str, Any]:
        result = {'compliant': True, 'warnings': [], 'requirements': []}
        
        # Check for explicit consent
        if not user_consent.get('explicit_consent', False):
            result['compliant'] = False
            result['requirements'].append('Explicit user consent required for data processing')
        
        # Check for legitimate interest basis
        if data_processing_type in ['analytics', 'profiling']:
            if not user_consent.get('legitimate_interest_disclosed', False):
                result['warnings'].append('Legitimate interest basis should be disclosed')
        
        # Data subject rights
        required_rights = ['access', 'rectification', 'erasure', 'portability']
        for right in required_rights:
            if not user_consent.get(f'{right}_right_disclosed', False):
                result['warnings'].append(f'Data subject right to {right} should be disclosed')
        
        return result


class CCPAComplianceHandler(BaseComplianceHandler):
    """CCPA compliance handler for California."""
    
    def validate_compliance(self, data_processing_type: str, 
                          user_consent: Dict[str, Any]) -> Dict[str, Any]:
        result = {'compliant': True, 'warnings': [], 'requirements': []}
        
        # Right to know
        if not user_consent.get('data_collection_disclosed', False):
            result['requirements'].append('Data collection practices must be disclosed')
        
        # Right to opt-out
        if not user_consent.get('opt_out_available', False):
            result['compliant'] = False
            result['requirements'].append('Opt-out mechanism must be provided')
        
        return result


class PDPAComplianceHandler(BaseComplianceHandler):
    """PDPA compliance handler for Singapore."""
    
    def validate_compliance(self, data_processing_type: str, 
                          user_consent: Dict[str, Any]) -> Dict[str, Any]:
        result = {'compliant': True, 'warnings': [], 'requirements': []}
        
        # Purpose limitation
        if not user_consent.get('purpose_specified', False):
            result['compliant'] = False
            result['requirements'].append('Purpose of data collection must be specified')
        
        return result


class PIPEDAComplianceHandler(BaseComplianceHandler):
    """PIPEDA compliance handler for Canada."""
    
    def validate_compliance(self, data_processing_type: str, 
                          user_consent: Dict[str, Any]) -> Dict[str, Any]:
        result = {'compliant': True, 'warnings': [], 'requirements': []}
        
        # Meaningful consent
        if not user_consent.get('meaningful_consent', False):
            result['compliant'] = False
            result['requirements'].append('Meaningful consent required')
        
        return result


class LGPDComplianceHandler(BaseComplianceHandler):
    """LGPD compliance handler for Brazil."""
    
    def validate_compliance(self, data_processing_type: str, 
                          user_consent: Dict[str, Any]) -> Dict[str, Any]:
        result = {'compliant': True, 'warnings': [], 'requirements': []}
        
        # Legal basis
        legal_bases = ['consent', 'legitimate_interest', 'legal_obligation']
        if not any(user_consent.get(f'{basis}_basis', False) for basis in legal_bases):
            result['compliant'] = False
            result['requirements'].append('Legal basis for processing must be established')
        
        return result


class SOXComplianceHandler(BaseComplianceHandler):
    """SOX compliance handler for financial reporting."""
    
    def validate_compliance(self, data_processing_type: str, 
                          user_consent: Dict[str, Any]) -> Dict[str, Any]:
        result = {'compliant': True, 'warnings': [], 'requirements': []}
        
        if data_processing_type == 'financial_reporting':
            if not user_consent.get('audit_trail_enabled', False):
                result['warnings'].append('Audit trail should be enabled for financial data')
        
        return result


class MiFIDIIComplianceHandler(BaseComplianceHandler):
    """MiFID II compliance handler for investment services."""
    
    def validate_compliance(self, data_processing_type: str, 
                          user_consent: Dict[str, Any]) -> Dict[str, Any]:
        result = {'compliant': True, 'warnings': [], 'requirements': []}
        
        if data_processing_type == 'investment_advice':
            if not user_consent.get('suitability_assessment', False):
                result['warnings'].append('Suitability assessment recommended')
        
        return result


# Global instance
_global_global_implementation: Optional[GlobalFirstImplementation] = None


def get_global_implementation() -> GlobalFirstImplementation:
    """Get the global implementation instance."""
    global _global_global_implementation
    if _global_global_implementation is None:
        _global_global_implementation = GlobalFirstImplementation()
    return _global_global_implementation