"""
Internationalization (i18n) support for FinChat-SEC-QA with global-first design.
Supports multiple languages, localized financial terminology, and regional compliance.
"""
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any, List
from enum import Enum

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported languages with their ISO 639-1 codes."""
    ENGLISH = "en"
    SPANISH = "es" 
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-cn"
    CHINESE_TRADITIONAL = "zh-tw"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    DUTCH = "nl"
    RUSSIAN = "ru"
    KOREAN = "ko"


class SupportedRegion(Enum):
    """Supported regions with their regulatory frameworks."""
    US = "us"          # United States (SEC, FINRA)
    EU = "eu"          # European Union (ESMA, MiFID)
    UK = "uk"          # United Kingdom (FCA)
    CANADA = "ca"      # Canada (CSA, IIROC)  
    JAPAN = "jp"       # Japan (JFSA)
    SINGAPORE = "sg"   # Singapore (MAS)
    AUSTRALIA = "au"   # Australia (ASIC)
    HONG_KONG = "hk"   # Hong Kong (SFC)
    SWITZERLAND = "ch" # Switzerland (FINMA)
    GERMANY = "de"     # Germany (BaFin)


@dataclass
class LocalizationContext:
    """Context for localization including language, region, and user preferences."""
    language: SupportedLanguage
    region: SupportedRegion
    currency: str = "USD"
    date_format: str = "%Y-%m-%d"
    number_format: str = "en_US"
    timezone: str = "UTC"
    financial_calendar: str = "us"  # fiscal year conventions
    
    @property
    def locale_id(self) -> str:
        """Get full locale identifier."""
        return f"{self.language.value}_{self.region.value.upper()}"


class TranslationManager:
    """Manages translations and localized content for multiple languages."""
    
    def __init__(self):
        self.translations: Dict[str, Dict[str, str]] = {}
        self.financial_terms: Dict[str, Dict[str, str]] = {}
        self.regulatory_terms: Dict[str, Dict[str, str]] = {}
        self.default_language = SupportedLanguage.ENGLISH
        self._load_translations()
    
    def _load_translations(self):
        """Load translation dictionaries from files or define inline."""
        # Core UI translations
        self.translations = {
            SupportedLanguage.ENGLISH.value: {
                "welcome": "Welcome to FinChat SEC QA",
                "query_placeholder": "Ask a question about financial filings...",
                "search_button": "Search",
                "loading": "Processing your query...",
                "error": "An error occurred",
                "no_results": "No results found",
                "citation": "Citation",
                "source": "Source",
                "risk_analysis": "Risk Analysis",
                "company": "Company",
                "filing_type": "Filing Type",
                "date_filed": "Date Filed",
                "processing_time": "Processing Time",
                "cache_status": "Cache Status",
                "api_response": "API Response",
                "health_check": "Health Check",
                "system_status": "System Status",
                "performance_metrics": "Performance Metrics",
                "security_notice": "This tool provides information from public SEC filings. Not financial advice.",
                "privacy_compliance": "Data processing complies with applicable privacy regulations."
            },
            
            SupportedLanguage.SPANISH.value: {
                "welcome": "Bienvenido a FinChat SEC QA",
                "query_placeholder": "Haga una pregunta sobre documentos financieros...",
                "search_button": "Buscar",
                "loading": "Procesando su consulta...",
                "error": "Ocurrió un error",
                "no_results": "No se encontraron resultados",
                "citation": "Cita",
                "source": "Fuente",
                "risk_analysis": "Análisis de Riesgo",
                "company": "Empresa",
                "filing_type": "Tipo de Documento",
                "date_filed": "Fecha de Presentación",
                "processing_time": "Tiempo de Procesamiento",
                "cache_status": "Estado del Cache",
                "api_response": "Respuesta de API",
                "health_check": "Verificación de Salud",
                "system_status": "Estado del Sistema",
                "performance_metrics": "Métricas de Rendimiento",
                "security_notice": "Esta herramienta proporciona información de documentos públicos de la SEC. No es asesoría financiera.",
                "privacy_compliance": "El procesamiento de datos cumple con las regulaciones de privacidad aplicables."
            },
            
            SupportedLanguage.FRENCH.value: {
                "welcome": "Bienvenue sur FinChat SEC QA",
                "query_placeholder": "Posez une question sur les dépôts financiers...",
                "search_button": "Rechercher",
                "loading": "Traitement de votre requête...",
                "error": "Une erreur s'est produite",
                "no_results": "Aucun résultat trouvé",
                "citation": "Citation",
                "source": "Source",
                "risk_analysis": "Analyse des Risques",
                "company": "Entreprise",
                "filing_type": "Type de Dépôt",
                "date_filed": "Date de Dépôt",
                "processing_time": "Temps de Traitement",
                "cache_status": "Statut du Cache",
                "api_response": "Réponse API",
                "health_check": "Vérification de Santé",
                "system_status": "Statut du Système",
                "performance_metrics": "Métriques de Performance",
                "security_notice": "Cet outil fournit des informations des dépôts publics de la SEC. Pas de conseils financiers.",
                "privacy_compliance": "Le traitement des données respecte les réglementations de confidentialité applicables."
            },
            
            SupportedLanguage.GERMAN.value: {
                "welcome": "Willkommen bei FinChat SEC QA",
                "query_placeholder": "Stellen Sie eine Frage zu Finanzberichten...",
                "search_button": "Suchen",
                "loading": "Ihre Anfrage wird bearbeitet...",
                "error": "Ein Fehler ist aufgetreten",
                "no_results": "Keine Ergebnisse gefunden",
                "citation": "Zitat",
                "source": "Quelle",
                "risk_analysis": "Risikoanalyse",
                "company": "Unternehmen",
                "filing_type": "Dokumenttyp",
                "date_filed": "Einreichungsdatum",
                "processing_time": "Bearbeitungszeit",
                "cache_status": "Cache-Status",
                "api_response": "API-Antwort",
                "health_check": "Gesundheitsprüfung",
                "system_status": "Systemstatus",
                "performance_metrics": "Leistungsmetriken",
                "security_notice": "Dieses Tool stellt Informationen aus öffentlichen SEC-Unterlagen zur Verfügung. Keine Finanzberatung.",
                "privacy_compliance": "Die Datenverarbeitung entspricht den geltenden Datenschutzbestimmungen."
            },
            
            SupportedLanguage.JAPANESE.value: {
                "welcome": "FinChat SEC QAへようこそ",
                "query_placeholder": "財務書類について質問してください...",
                "search_button": "検索",
                "loading": "クエリを処理しています...",
                "error": "エラーが発生しました",
                "no_results": "結果が見つかりません",
                "citation": "引用",
                "source": "ソース",
                "risk_analysis": "リスク分析",
                "company": "会社",
                "filing_type": "提出書類の種類",
                "date_filed": "提出日",
                "processing_time": "処理時間",
                "cache_status": "キャッシュ状況",
                "api_response": "API応答",
                "health_check": "ヘルスチェック",
                "system_status": "システム状況",
                "performance_metrics": "パフォーマンス指標",
                "security_notice": "このツールは公開されているSEC書類から情報を提供します。投資助言ではありません。",
                "privacy_compliance": "データ処理は適用されるプライバシー規制に準拠しています。"
            },
            
            SupportedLanguage.CHINESE_SIMPLIFIED.value: {
                "welcome": "欢迎使用FinChat SEC QA",
                "query_placeholder": "请询问有关财务文件的问题...",
                "search_button": "搜索",
                "loading": "正在处理您的查询...",
                "error": "发生错误",
                "no_results": "未找到结果",
                "citation": "引用",
                "source": "来源",
                "risk_analysis": "风险分析",
                "company": "公司",
                "filing_type": "文件类型",
                "date_filed": "提交日期",
                "processing_time": "处理时间",
                "cache_status": "缓存状态",
                "api_response": "API响应",
                "health_check": "健康检查",
                "system_status": "系统状态",
                "performance_metrics": "性能指标",
                "security_notice": "此工具提供来自公开SEC文件的信息。非投资建议。",
                "privacy_compliance": "数据处理符合适用的隐私法规。"
            }
        }
        
        # Financial terminology translations
        self.financial_terms = {
            SupportedLanguage.ENGLISH.value: {
                "revenue": "Revenue",
                "net_income": "Net Income",
                "assets": "Assets",
                "liabilities": "Liabilities",
                "equity": "Equity",
                "cash_flow": "Cash Flow",
                "debt_to_equity": "Debt-to-Equity Ratio",
                "roe": "Return on Equity",
                "roa": "Return on Assets",
                "gross_margin": "Gross Margin",
                "operating_margin": "Operating Margin",
                "ebitda": "EBITDA",
                "eps": "Earnings Per Share",
                "pe_ratio": "Price-to-Earnings Ratio",
                "market_cap": "Market Capitalization",
                "dividend_yield": "Dividend Yield",
                "risk_factors": "Risk Factors",
                "liquidity": "Liquidity",
                "working_capital": "Working Capital",
                "capex": "Capital Expenditure"
            },
            
            SupportedLanguage.SPANISH.value: {
                "revenue": "Ingresos",
                "net_income": "Beneficio Neto",
                "assets": "Activos",
                "liabilities": "Pasivos",
                "equity": "Patrimonio",
                "cash_flow": "Flujo de Caja",
                "debt_to_equity": "Ratio Deuda-Patrimonio",
                "roe": "Rentabilidad sobre Patrimonio",
                "roa": "Rentabilidad sobre Activos",
                "gross_margin": "Margen Bruto",
                "operating_margin": "Margen Operativo",
                "ebitda": "EBITDA",
                "eps": "Beneficio por Acción",
                "pe_ratio": "Ratio Precio-Beneficio",
                "market_cap": "Capitalización de Mercado",
                "dividend_yield": "Rendimiento por Dividendo",
                "risk_factors": "Factores de Riesgo",
                "liquidity": "Liquidez",
                "working_capital": "Capital de Trabajo",
                "capex": "Inversión de Capital"
            },
            
            SupportedLanguage.GERMAN.value: {
                "revenue": "Umsatz",
                "net_income": "Nettogewinn",
                "assets": "Vermögen",
                "liabilities": "Verbindlichkeiten",
                "equity": "Eigenkapital",
                "cash_flow": "Cashflow",
                "debt_to_equity": "Verschuldungsgrad",
                "roe": "Eigenkapitalrentabilität",
                "roa": "Gesamtkapitalrentabilität",
                "gross_margin": "Rohgewinnmarge",
                "operating_margin": "Betriebsmarge",
                "ebitda": "EBITDA",
                "eps": "Gewinn je Aktie",
                "pe_ratio": "Kurs-Gewinn-Verhältnis",
                "market_cap": "Marktkapitalisierung",
                "dividend_yield": "Dividendenrendite",
                "risk_factors": "Risikofaktoren",
                "liquidity": "Liquidität",
                "working_capital": "Betriebskapital",
                "capex": "Investitionsausgaben"
            }
        }
        
        # Regulatory compliance terms
        self.regulatory_terms = {
            SupportedLanguage.ENGLISH.value: {
                "gdpr_compliance": "GDPR Compliance",
                "data_privacy": "Data Privacy",
                "consent": "Consent",
                "data_subject_rights": "Data Subject Rights",
                "data_controller": "Data Controller",
                "data_processor": "Data Processor",
                "privacy_policy": "Privacy Policy",
                "cookie_policy": "Cookie Policy",
                "data_retention": "Data Retention",
                "data_portability": "Data Portability",
                "right_to_erasure": "Right to Erasure",
                "sec_compliance": "SEC Compliance",
                "financial_disclosure": "Financial Disclosure",
                "insider_trading": "Insider Trading",
                "market_abuse": "Market Abuse",
                "kyc": "Know Your Customer",
                "aml": "Anti-Money Laundering",
                "fiduciary_duty": "Fiduciary Duty"
            },
            
            SupportedLanguage.SPANISH.value: {
                "gdpr_compliance": "Cumplimiento RGPD",
                "data_privacy": "Privacidad de Datos",
                "consent": "Consentimiento",
                "data_subject_rights": "Derechos del Titular de Datos",
                "data_controller": "Responsable del Tratamiento",
                "data_processor": "Encargado del Tratamiento",
                "privacy_policy": "Política de Privacidad",
                "cookie_policy": "Política de Cookies",
                "data_retention": "Retención de Datos",
                "data_portability": "Portabilidad de Datos",
                "right_to_erasure": "Derecho al Olvido",
                "sec_compliance": "Cumplimiento SEC",
                "financial_disclosure": "Divulgación Financiera",
                "insider_trading": "Tráfico de Información Privilegiada",
                "market_abuse": "Abuso de Mercado",
                "kyc": "Conoce a tu Cliente",
                "aml": "Antilavado de Dinero",
                "fiduciary_duty": "Deber Fiduciario"
            }
        }
    
    def get_translation(self, key: str, language: SupportedLanguage = None, 
                       category: str = "ui") -> str:
        """Get translation for a given key and language."""
        if language is None:
            language = self.default_language
        
        # Select appropriate translation dictionary
        if category == "financial":
            translations = self.financial_terms
        elif category == "regulatory":
            translations = self.regulatory_terms
        else:
            translations = self.translations
        
        lang_code = language.value
        
        # Try to get translation
        if lang_code in translations and key in translations[lang_code]:
            return translations[lang_code][key]
        
        # Fallback to English
        english_code = SupportedLanguage.ENGLISH.value
        if english_code in translations and key in translations[english_code]:
            logger.debug(f"Fallback to English for key '{key}' in category '{category}'")
            return translations[english_code][key]
        
        # Return key if no translation found
        logger.warning(f"No translation found for key '{key}' in category '{category}'")
        return key
    
    def format_number(self, number: float, context: LocalizationContext) -> str:
        """Format number according to locale conventions."""
        if context.number_format == "en_US":
            return f"{number:,.2f}"
        elif context.number_format == "de_DE":
            return f"{number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        elif context.number_format == "fr_FR":
            return f"{number:,.2f}".replace(",", " ")
        else:
            return f"{number:.2f}"
    
    def format_currency(self, amount: float, context: LocalizationContext) -> str:
        """Format currency according to locale and regional preferences."""
        formatted_number = self.format_number(amount, context)
        
        currency_symbols = {
            "USD": "$",
            "EUR": "€",
            "GBP": "£",
            "JPY": "¥",
            "CNY": "¥",
            "CHF": "CHF",
            "CAD": "C$",
            "AUD": "A$",
            "SGD": "S$",
            "HKD": "HK$"
        }
        
        symbol = currency_symbols.get(context.currency, context.currency)
        
        # Currency placement varies by locale
        if context.region in [SupportedRegion.US, SupportedRegion.CANADA]:
            return f"{symbol}{formatted_number}"
        elif context.region == SupportedRegion.EU:
            return f"{formatted_number} {symbol}"
        else:
            return f"{symbol} {formatted_number}"
    
    def format_date(self, date_obj, context: LocalizationContext) -> str:
        """Format date according to locale conventions."""
        format_patterns = {
            "us": "%m/%d/%Y",
            "eu": "%d/%m/%Y", 
            "iso": "%Y-%m-%d",
            "jp": "%Y年%m月%d日"
        }
        
        if context.region == SupportedRegion.US:
            pattern = format_patterns["us"]
        elif context.region in [SupportedRegion.EU, SupportedRegion.UK]:
            pattern = format_patterns["eu"]
        elif context.region == SupportedRegion.JAPAN:
            pattern = format_patterns["jp"]
        else:
            pattern = format_patterns["iso"]
        
        return date_obj.strftime(pattern)


class ComplianceManager:
    """Manages regional compliance requirements and data protection."""
    
    def __init__(self):
        self.compliance_frameworks = self._initialize_compliance_frameworks()
        self.data_protection_rules = self._initialize_data_protection()
        
    def _initialize_compliance_frameworks(self) -> Dict[SupportedRegion, Dict[str, Any]]:
        """Initialize compliance frameworks by region."""
        return {
            SupportedRegion.EU: {
                "gdpr_required": True,
                "cookie_consent": True,
                "data_retention_max_days": 2555,  # 7 years
                "cross_border_transfer_rules": True,
                "privacy_impact_assessment": True,
                "data_protection_officer_required": False,  # Depends on processing
                "lawful_basis": ["legitimate_interest", "consent"],
                "special_categories": False,  # No sensitive personal data
                "automated_decision_making": False
            },
            
            SupportedRegion.US: {
                "ccpa_applicable": True,
                "coppa_applicable": False,  # No children's data
                "sox_compliance": True,  # Financial data
                "data_retention_max_days": 2555,
                "state_privacy_laws": ["CCPA", "CPRA"],
                "federal_regulations": ["SOX", "SEC"],
                "data_minimization": True,
                "opt_out_rights": True
            },
            
            SupportedRegion.UK: {
                "uk_gdpr_required": True,
                "ico_registration": True,
                "data_retention_max_days": 2555,
                "international_transfers": True,
                "adequacy_decisions": ["EU", "EEA"],
                "privacy_notices": True,
                "subject_access_requests": True
            },
            
            SupportedRegion.CANADA: {
                "pipeda_compliance": True,
                "provincial_laws": ["PIPA-BC", "PIPA-AB"],
                "data_retention_max_days": 2555,
                "privacy_impact_assessment": True,
                "breach_notification": True,
                "consent_requirements": True
            },
            
            SupportedRegion.SINGAPORE: {
                "pdpa_compliance": True,
                "data_protection_provisions": True,
                "consent_obligations": True,
                "notification_obligations": True,
                "data_retention_max_days": 2555,
                "international_transfers": True
            },
            
            SupportedRegion.JAPAN: {
                "appi_compliance": True,
                "personal_information_protection": True,
                "data_retention_max_days": 2555,
                "cross_border_restrictions": True,
                "consent_requirements": True,
                "breach_notification": True
            }
        }
    
    def _initialize_data_protection(self) -> Dict[str, Any]:
        """Initialize data protection rules."""
        return {
            "encryption_at_rest": True,
            "encryption_in_transit": True,
            "access_logging": True,
            "data_anonymization": True,
            "pseudonymization": True,
            "minimal_data_collection": True,
            "purpose_limitation": True,
            "storage_limitation": True,
            "accuracy_maintenance": True,
            "security_measures": [
                "access_controls",
                "audit_logging", 
                "data_encryption",
                "network_security",
                "incident_response"
            ]
        }
    
    def get_compliance_requirements(self, region: SupportedRegion) -> Dict[str, Any]:
        """Get compliance requirements for a specific region."""
        return self.compliance_frameworks.get(region, {})
    
    def validate_data_processing(self, context: LocalizationContext, 
                                data_types: List[str]) -> Dict[str, Any]:
        """Validate data processing against regional compliance."""
        requirements = self.get_compliance_requirements(context.region)
        
        validation_result = {
            "compliant": True,
            "requirements": [],
            "warnings": [],
            "actions_required": []
        }
        
        # Check GDPR requirements
        if requirements.get("gdpr_required") or requirements.get("uk_gdpr_required"):
            validation_result["requirements"].extend([
                "Privacy notice must be provided",
                "Lawful basis for processing must be established",
                "Data subject rights must be implemented",
                "Data retention policy must be enforced"
            ])
            
            if "personal_data" in data_types:
                validation_result["actions_required"].append(
                    "Implement data subject rights (access, rectification, erasure, portability)"
                )
        
        # Check CCPA requirements
        if requirements.get("ccpa_applicable"):
            validation_result["requirements"].extend([
                "Consumer privacy rights must be implemented",
                "Opt-out mechanism must be provided",
                "Privacy policy must be updated"
            ])
        
        # Check data retention
        max_retention = requirements.get("data_retention_max_days", 2555)
        validation_result["requirements"].append(
            f"Data retention must not exceed {max_retention} days"
        )
        
        return validation_result
    
    def generate_privacy_notice(self, context: LocalizationContext) -> Dict[str, str]:
        """Generate privacy notice based on regional requirements."""
        translator = TranslationManager()
        
        # Base privacy notice content
        privacy_content = {
            "title": translator.get_translation("privacy_policy", context.language, "regulatory"),
            "data_controller": "Terragon Labs (FinChat SEC QA)",
            "purpose": "Processing public financial filings for analytical insights",
            "lawful_basis": "Legitimate interest in financial analysis",
            "data_types": "Query text, usage analytics (anonymized)",
            "retention_period": "Maximum 7 years as per financial regulations",
            "your_rights": "Access, rectification, erasure, portability, objection",
            "contact": "privacy@terragonlabs.com",
            "last_updated": "2025-08-09"
        }
        
        # Add region-specific content
        requirements = self.get_compliance_requirements(context.region)
        
        if requirements.get("gdpr_required"):
            privacy_content["gdpr_notice"] = (
                "This processing is based on Article 6(1)(f) GDPR - legitimate interest. "
                "You have the right to object to this processing."
            )
            privacy_content["dpo_contact"] = "dpo@terragonlabs.com"
            privacy_content["supervisory_authority"] = "Contact your local data protection authority"
        
        if requirements.get("ccpa_applicable"):
            privacy_content["ccpa_notice"] = (
                "California residents have additional rights under CCPA including "
                "right to know, delete, and opt-out of sale of personal information."
            )
            privacy_content["opt_out_link"] = "https://terragonlabs.com/ccpa-opt-out"
        
        return privacy_content


class GlobalizationService:
    """Main service for handling globalization features."""
    
    def __init__(self):
        self.translator = TranslationManager()
        self.compliance_manager = ComplianceManager()
        self.default_context = LocalizationContext(
            language=SupportedLanguage.ENGLISH,
            region=SupportedRegion.US
        )
    
    def detect_locale_from_request(self, headers: Dict[str, str], 
                                 ip_address: Optional[str] = None) -> LocalizationContext:
        """Detect user locale from request headers and IP."""
        # Parse Accept-Language header
        accept_language = headers.get('Accept-Language', 'en-US,en;q=0.9')
        preferred_languages = self._parse_accept_language(accept_language)
        
        # Determine language
        detected_language = SupportedLanguage.ENGLISH
        for lang_code, _ in preferred_languages:
            try:
                detected_language = SupportedLanguage(lang_code[:2])
                break
            except ValueError:
                continue
        
        # Determine region (simplified - in practice would use GeoIP)
        detected_region = SupportedRegion.US
        if ip_address:
            detected_region = self._detect_region_from_ip(ip_address)
        
        # Create localization context
        return LocalizationContext(
            language=detected_language,
            region=detected_region,
            currency=self._get_regional_currency(detected_region),
            timezone=self._get_regional_timezone(detected_region)
        )
    
    def _parse_accept_language(self, accept_language: str) -> List[Tuple[str, float]]:
        """Parse Accept-Language header."""
        languages = []
        
        for lang_item in accept_language.split(','):
            parts = lang_item.strip().split(';')
            lang_code = parts[0].strip()
            
            # Parse quality value
            quality = 1.0
            if len(parts) > 1 and parts[1].strip().startswith('q='):
                try:
                    quality = float(parts[1].strip()[2:])
                except ValueError:
                    quality = 1.0
            
            languages.append((lang_code, quality))
        
        # Sort by quality value (descending)
        languages.sort(key=lambda x: x[1], reverse=True)
        return languages
    
    def _detect_region_from_ip(self, ip_address: str) -> SupportedRegion:
        """Detect region from IP address (simplified implementation)."""
        # In a real implementation, this would use a GeoIP service
        # For now, return default region
        return SupportedRegion.US
    
    def _get_regional_currency(self, region: SupportedRegion) -> str:
        """Get default currency for region."""
        currency_map = {
            SupportedRegion.US: "USD",
            SupportedRegion.EU: "EUR",
            SupportedRegion.UK: "GBP",
            SupportedRegion.CANADA: "CAD",
            SupportedRegion.JAPAN: "JPY",
            SupportedRegion.SINGAPORE: "SGD",
            SupportedRegion.AUSTRALIA: "AUD",
            SupportedRegion.HONG_KONG: "HKD",
            SupportedRegion.SWITZERLAND: "CHF",
            SupportedRegion.GERMANY: "EUR"
        }
        return currency_map.get(region, "USD")
    
    def _get_regional_timezone(self, region: SupportedRegion) -> str:
        """Get default timezone for region."""
        timezone_map = {
            SupportedRegion.US: "America/New_York",
            SupportedRegion.EU: "Europe/Brussels", 
            SupportedRegion.UK: "Europe/London",
            SupportedRegion.CANADA: "America/Toronto",
            SupportedRegion.JAPAN: "Asia/Tokyo",
            SupportedRegion.SINGAPORE: "Asia/Singapore",
            SupportedRegion.AUSTRALIA: "Australia/Sydney",
            SupportedRegion.HONG_KONG: "Asia/Hong_Kong",
            SupportedRegion.SWITZERLAND: "Europe/Zurich",
            SupportedRegion.GERMANY: "Europe/Berlin"
        }
        return timezone_map.get(region, "UTC")
    
    def localize_response(self, data: Dict[str, Any], 
                         context: LocalizationContext) -> Dict[str, Any]:
        """Localize API response data."""
        localized_data = data.copy()
        
        # Localize UI strings
        if "status" in localized_data:
            localized_data["status_text"] = self.translator.get_translation(
                localized_data["status"], context.language
            )
        
        # Localize financial numbers
        if "financial_data" in localized_data:
            financial_data = localized_data["financial_data"]
            for key, value in financial_data.items():
                if isinstance(value, (int, float)):
                    if key.endswith("_amount") or key.endswith("_value"):
                        financial_data[f"{key}_formatted"] = self.translator.format_currency(
                            value, context
                        )
                    else:
                        financial_data[f"{key}_formatted"] = self.translator.format_number(
                            value, context
                        )
        
        # Add localization metadata
        localized_data["localization"] = {
            "language": context.language.value,
            "region": context.region.value,
            "currency": context.currency,
            "locale_id": context.locale_id
        }
        
        # Add compliance notices
        compliance_validation = self.compliance_manager.validate_data_processing(
            context, ["query_text", "usage_analytics"]
        )
        
        localized_data["compliance"] = {
            "privacy_notice_required": True,
            "data_protection_region": context.region.value,
            "requirements": compliance_validation["requirements"][:3]  # First 3 requirements
        }
        
        return localized_data
    
    def get_supported_locales(self) -> List[Dict[str, str]]:
        """Get list of supported locales."""
        locales = []
        
        for language in SupportedLanguage:
            for region in SupportedRegion:
                locales.append({
                    "locale_id": f"{language.value}_{region.value.upper()}",
                    "language": language.value,
                    "language_name": language.name.replace("_", " ").title(),
                    "region": region.value,
                    "region_name": region.name.replace("_", " ").title(),
                    "currency": self._get_regional_currency(region)
                })
        
        return locales


# Global service instance
_globalization_service = GlobalizationService()


def get_globalization_service() -> GlobalizationService:
    """Get the global globalization service instance."""
    return _globalization_service


def localize_for_request(data: Dict[str, Any], request_headers: Dict[str, str],
                        ip_address: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to localize data for a request."""
    service = get_globalization_service()
    context = service.detect_locale_from_request(request_headers, ip_address)
    return service.localize_response(data, context)


def get_privacy_notice(language: str = "en", region: str = "us") -> Dict[str, str]:
    """Get privacy notice for specific language and region."""
    try:
        lang = SupportedLanguage(language)
        reg = SupportedRegion(region)
        context = LocalizationContext(language=lang, region=reg)
        
        service = get_globalization_service()
        return service.compliance_manager.generate_privacy_notice(context)
    except ValueError:
        # Fallback to defaults
        context = LocalizationContext(
            language=SupportedLanguage.ENGLISH,
            region=SupportedRegion.US
        )
        service = get_globalization_service()
        return service.compliance_manager.generate_privacy_notice(context)