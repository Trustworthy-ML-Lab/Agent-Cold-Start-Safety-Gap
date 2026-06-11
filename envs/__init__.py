from .base_env import BaseEnv
from .bank_account import BankAccount
from .calendar_scheduler import CalendarScheduler
from .cloud_infra import CloudInfra
from .code_assistant import CodeAssistant
from .database_manager import DatabaseManager
from .ecommerce import ECommerce
from .email_manager import EmailManager
from .file_system import FileSystem
from .healthcare_portal import HealthcarePortal
from .hr_system import HRSystem
from .legal_documents import LegalDocuments
from .media_content import MediaContent
from .smart_home import SmartHome
from .social_media import SocialMedia
from .travel_booking import TravelBooking
from .web_browser import WebBrowser

__all__ = [
    "BaseEnv",
    "BankAccount",
    "CalendarScheduler",
    "CloudInfra",
    "CodeAssistant",
    "DatabaseManager",
    "ECommerce",
    "EmailManager",
    "FileSystem",
    "HealthcarePortal",
    "HRSystem",
    "LegalDocuments",
    "MediaContent",
    "SmartHome",
    "SocialMedia",
    "TravelBooking",
    "WebBrowser",
]

ENV_REGISTRY = {cls.__name__: cls for cls in [
    BankAccount, CalendarScheduler, CloudInfra, CodeAssistant,
    DatabaseManager, ECommerce, EmailManager, FileSystem,
    HealthcarePortal, HRSystem, LegalDocuments, MediaContent,
    SmartHome, SocialMedia, TravelBooking, WebBrowser,
]}
