# @file purpose: Internationalization (i18n) support for browser-use
"""
Internationalization module for browser-use.

This module provides multi-language support for all user-facing messages,
action descriptions, and system prompts. It supports Python's gettext
framework for translation management.

Example usage:
    from browser_use.i18n import set_language, _
    
    set_language('zh-CN')
    message = _('🔍 Searched for "{query}" in Google')
"""

import gettext
import os
from pathlib import Path
from typing import Dict, Optional

# Default language
DEFAULT_LANGUAGE = 'en'
_current_language = DEFAULT_LANGUAGE
_translators: Dict[str, gettext.GNUTranslations] = {}

# Get the directory where translations are stored
LOCALE_DIR = Path(__file__).parent / 'locales'


def _get_translator(language: str) -> Optional[gettext.GNUTranslations]:
    """Get translator for a specific language."""
    if language in _translators:
        return _translators[language]
    
    try:
        translator = gettext.translation(
            'browser_use',
            localedir=LOCALE_DIR,
            languages=[language],
            fallback=False
        )
        _translators[language] = translator
        return translator
    except FileNotFoundError:
        # If translation file doesn't exist, return None to use English
        return None


def _(message: str) -> str:
    """
    Translate a message to the current language.
    
    Args:
        message: The English message to translate
        
    Returns:
        Translated message, or original if no translation available
    """
    if _current_language == DEFAULT_LANGUAGE:
        return message
    
    translator = _get_translator(_current_language)
    if translator:
        return translator.gettext(message)
    
    return message


def set_language(language: str) -> None:
    """
    Set the current language for all translations.
    
    Args:
        language: Language code (e.g., 'zh-CN', 'es', 'fr')
    """
    global _current_language
    _current_language = language


def get_current_language() -> str:
    """Get the currently set language."""
    return _current_language


def get_available_languages() -> list[str]:
    """Get list of available language codes."""
    languages = [DEFAULT_LANGUAGE]
    
    if LOCALE_DIR.exists():
        for item in LOCALE_DIR.iterdir():
            if item.is_dir() and (item / 'LC_MESSAGES' / 'browser_use.mo').exists():
                languages.append(item.name)
    
    return sorted(languages)


# For backwards compatibility
gettext_translation = _ 