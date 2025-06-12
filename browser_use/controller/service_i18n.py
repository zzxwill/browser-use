# @file purpose: Enhanced Controller with internationalization support
"""
Enhanced Controller service with internationalization (i18n) support.

This demonstrates how to integrate the browser-use i18n system into the controller
to provide translated messages for all actions and responses.

Example usage:
    from browser_use.i18n import set_language
    from browser_use.controller.service_i18n import ControllerI18n
    
    set_language('zh-CN')
    controller = ControllerI18n()
    # All messages will now be in Chinese
"""

import asyncio
import logging
from typing import Generic, TypeVar

from pydantic import BaseModel

from browser_use.agent.views import ActionResult
from browser_use.browser import BrowserSession
from browser_use.controller.registry.service import Registry
from browser_use.controller.views import (
    SearchGoogleAction,
    GoToUrlAction,
    NoParamsAction,
    ClickElementAction,
    InputTextAction,
    ScrollAction,
    SendKeysAction,
    DoneAction,
)
from browser_use.i18n import _

logger = logging.getLogger(__name__)

Context = TypeVar('Context')


class ControllerI18n(Generic[Context]):
    """Enhanced Controller with internationalization support."""
    
    def __init__(
        self,
        exclude_actions: list[str] = [],
        output_model: type[BaseModel] | None = None,
    ):
        self.registry = Registry[Context](exclude_actions)
        
        # Register all actions with i18n support
        self._register_i18n_actions(output_model)
    
    def _register_i18n_actions(self, output_model: type[BaseModel] | None = None):
        """Register all actions with internationalization support."""
        
        # Done action with i18n
        if output_model is not None:
            # Create extended output model
            class ExtendedOutputModel(BaseModel):
                success: bool = True
                data: output_model
                
            @self.registry.action(
                _('Complete task - with return text and if the task is finished (success=True) or not yet completely finished (success=False), because last step is reached'),
                param_model=ExtendedOutputModel,
            )
            async def done(params: ExtendedOutputModel):
                import json
                import enum
                
                output_dict = params.data.model_dump()
                
                # Convert enums to strings
                for key, value in output_dict.items():
                    if isinstance(value, enum.Enum):
                        output_dict[key] = value.value
                        
                return ActionResult(is_done=True, success=params.success, extracted_content=json.dumps(output_dict))
        else:
            @self.registry.action(
                _('Complete task - with return text and if the task is finished (success=True) or not yet completely finished (success=False), because last step is reached'),
                param_model=DoneAction,
            )
            async def done(params: DoneAction):
                return ActionResult(is_done=True, success=params.success, extracted_content=params.text)
        
        # Search Google with i18n
        @self.registry.action(
            _('Search the query in Google, the query should be a search query like humans search in Google, concrete and not vague or super long.'),
            param_model=SearchGoogleAction,
        )
        async def search_google(params: SearchGoogleAction, browser_session: BrowserSession):
            search_url = f'https://www.google.com/search?q={params.query}&udm=14'
            
            page = await browser_session.get_current_page()
            if page.url in ('about:blank', 'https://www.google.com'):
                await page.goto(search_url)
                await page.wait_for_load_state()
            else:
                page = await browser_session.create_new_tab(search_url)
            
            msg = _('🔍  Searched for "{query}" in Google').format(query=params.query)
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)
        
        # Navigate to URL with i18n
        @self.registry.action(_('Navigate to URL in the current tab'), param_model=GoToUrlAction)
        async def go_to_url(params: GoToUrlAction, browser_session: BrowserSession):
            page = await browser_session.get_current_page()
            if page:
                await page.goto(params.url)
                await page.wait_for_load_state()
            else:
                page = await browser_session.create_new_tab(params.url)
            
            msg = _('🔗  Navigated to {url}').format(url=params.url)
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)
        
        # Go back with i18n
        @self.registry.action(_('Go back'), param_model=NoParamsAction)
        async def go_back(params: NoParamsAction, browser_session: BrowserSession):
            await browser_session.go_back()
            msg = _('🔙  Navigated back')
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)
        
        # Wait with i18n
        @self.registry.action(_('Wait for x seconds default 3'))
        async def wait(seconds: int = 3):
            msg = _('🕒  Waiting for {seconds} seconds').format(seconds=seconds)
            logger.info(msg)
            await asyncio.sleep(seconds)
            return ActionResult(extracted_content=msg, include_in_memory=True)
        
        # Click element with i18n
        @self.registry.action(_('Click element by index'), param_model=ClickElementAction)
        async def click_element_by_index(params: ClickElementAction, browser_session: BrowserSession):
            if params.index not in await browser_session.get_selector_map():
                error_msg = _('Element with index {index} does not exist - retry or use alternative actions').format(index=params.index)
                raise Exception(error_msg)
            
            element_node = await browser_session.get_dom_element_by_index(params.index)
            initial_pages = len(browser_session.tabs)
            
            # Check for file upload element
            if await browser_session.find_file_upload_element_by_index(params.index) is not None:
                msg = _('Index {index} - has an element which opens file upload dialog. To upload files please use a specific function to upload files').format(index=params.index)
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            
            try:
                download_path = await browser_session._click_element_node(element_node)
                if download_path:
                    msg = _('💾  Downloaded file to {path}').format(path=download_path)
                else:
                    msg = _('🖱️  Clicked button with index {index}').format(index=params.index)
                
                logger.info(msg)
                
                if len(browser_session.tabs) > initial_pages:
                    new_tab_msg = _('New tab opened - switching to it')
                    msg += f' - {new_tab_msg}'
                    logger.info(new_tab_msg)
                    await browser_session.switch_to_tab(-1)
                    
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                error_msg = _('Element not clickable with index {index} - most likely the page changed').format(index=params.index)
                logger.warning(error_msg)
                return ActionResult(error=str(e))
        
        # Input text with i18n
        @self.registry.action(
            _('Input text into a input interactive element'),
            param_model=InputTextAction,
        )
        async def input_text(params: InputTextAction, browser_session: BrowserSession, has_sensitive_data: bool = False):
            if params.index not in await browser_session.get_selector_map():
                error_msg = _('Element index {index} does not exist - retry or use alternative actions').format(index=params.index)
                raise Exception(error_msg)
            
            element_node = await browser_session.get_dom_element_by_index(params.index)
            await browser_session._input_text_element_node(element_node, params.text)
            
            if not has_sensitive_data:
                msg = _('⌨️  Input {text} into index {index}').format(text=params.text, index=params.index)
            else:
                msg = _('⌨️  Input sensitive data into index {index}').format(index=params.index)
            
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)
        
        # Scroll down with i18n
        @self.registry.action(
            _('Scroll down the page by pixel amount - if none is given, scroll one page'),
            param_model=ScrollAction,
        )
        async def scroll_down(params: ScrollAction, browser_session: BrowserSession):
            page = await browser_session.get_current_page()
            dy = params.amount or await page.evaluate('() => window.innerHeight')
            
            try:
                await browser_session._scroll_container(dy)
            except Exception as e:
                await page.evaluate('(y) => window.scrollBy(0, y)', dy)
                logger.debug('Smart scroll failed; used window.scrollBy fallback', exc_info=e)
            
            if params.amount is not None:
                amount_str = _('{amount} pixels').format(amount=params.amount)
            else:
                amount_str = _('one page')
                
            msg = _('🔍 Scrolled down the page by {amount}').format(amount=amount_str)
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)
        
        # Scroll up with i18n
        @self.registry.action(
            _('Scroll up the page by pixel amount - if none is given, scroll one page'),
            param_model=ScrollAction,
        )
        async def scroll_up(params: ScrollAction, browser_session: BrowserSession):
            page = await browser_session.get_current_page()
            dy = -(params.amount or await page.evaluate('() => window.innerHeight'))
            
            try:
                await browser_session._scroll_container(dy)
            except Exception as e:
                await page.evaluate('(y) => window.scrollBy(0, y)', dy)
                logger.debug('Smart scroll failed; used window.scrollBy fallback', exc_info=e)
            
            if params.amount is not None:
                amount_str = _('{amount} pixels').format(amount=params.amount)
            else:
                amount_str = _('one page')
                
            msg = _('🔍 Scrolled up the page by {amount}').format(amount=amount_str)
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)
        
        # Send keys with i18n
        @self.registry.action(
            _('Send strings of special keys like Escape,Backspace, Insert, PageDown, Delete, Enter, Shortcuts such as `Control+o`, `Control+Shift+T` are supported as well. This gets used in keyboard.press.'),
            param_model=SendKeysAction,
        )
        async def send_keys(params: SendKeysAction, page):
            from playwright.async_api import Page
            
            try:
                await page.keyboard.press(params.keys)
            except Exception as e:
                if 'Unknown key' in str(e):
                    for key in params.keys:
                        try:
                            await page.keyboard.press(key)
                        except Exception as e:
                            logger.debug(f'Error sending key {key}: {str(e)}')
                            raise e
                else:
                    raise e
            
            msg = _('⌨️  Sent keys: {keys}').format(keys=params.keys)
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

    def action(self, description: str, **kwargs):
        """Decorator for registering custom actions with i18n support."""
        # Translate the description
        translated_description = _(description)
        return self.registry.action(translated_description, **kwargs) 