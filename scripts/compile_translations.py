#!/usr/bin/env python3
"""
Translation Compilation Script for browser-use

This script compiles .po translation files into .mo files for use with
the gettext internationalization system.

Usage:
    python scripts/compile_translations.py
    
This will compile all .po files in browser_use/i18n/locales/ into their
corresponding .mo files.
"""

import os
import subprocess
import sys
from pathlib import Path


def compile_translations():
    """Compile all .po files to .mo files."""
    
    # Get the root directory of browser-use
    root_dir = Path(__file__).parent.parent
    locale_dir = root_dir / 'browser_use' / 'i18n' / 'locales'
    
    if not locale_dir.exists():
        print(f"❌ Locale directory not found: {locale_dir}")
        return False
    
    print(f"🔍 Looking for translation files in: {locale_dir}")
    
    compiled_count = 0
    
    # Find all .po files
    for po_file in locale_dir.rglob('*.po'):
        # Get the corresponding .mo file path
        mo_file = po_file.with_suffix('.mo')
        
        print(f"📝 Compiling: {po_file.relative_to(root_dir)} -> {mo_file.relative_to(root_dir)}")
        
        try:
            # Use msgfmt to compile .po to .mo
            result = subprocess.run([
                'msgfmt',
                str(po_file),
                '-o',
                str(mo_file)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"  ✅ Compiled successfully")
                compiled_count += 1
            else:
                print(f"  ❌ Compilation failed: {result.stderr}")
                
        except FileNotFoundError:
            print(f"  ❌ msgfmt not found. Please install gettext tools:")
            print(f"     - Ubuntu/Debian: sudo apt-get install gettext")
            print(f"     - macOS: brew install gettext")
            print(f"     - Windows: https://mlocati.github.io/articles/gettext-iconv-windows.html")
            return False
        except Exception as e:
            print(f"  ❌ Error compiling {po_file}: {e}")
    
    if compiled_count > 0:
        print(f"\n✅ Successfully compiled {compiled_count} translation file(s)")
        return True
    else:
        print(f"\n❌ No translation files were compiled")
        return False


def create_translation_template():
    """Create a .pot template file from the source code."""
    
    root_dir = Path(__file__).parent.parent
    locale_dir = root_dir / 'browser_use' / 'i18n' / 'locales'
    pot_file = locale_dir / 'browser_use.pot'
    
    # Create locale directory if it doesn't exist
    locale_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔍 Creating translation template: {pot_file.relative_to(root_dir)}")
    
    # Find all Python files with translatable strings
    python_files = []
    source_dirs = [
        root_dir / 'browser_use',
        root_dir / 'examples',
    ]
    
    for source_dir in source_dirs:
        if source_dir.exists():
            python_files.extend(source_dir.rglob('*.py'))
    
    try:
        # Use xgettext to extract translatable strings
        cmd = [
            'xgettext',
            '--language=Python',
            '--keyword=_',
            '--keyword=gettext',
            '--output=' + str(pot_file),
            '--from-code=UTF-8',
            '--add-comments=TRANSLATORS',
        ]
        
        # Add all Python files
        cmd.extend(str(f) for f in python_files)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Translation template created successfully")
            return True
        else:
            print(f"❌ Failed to create template: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print(f"❌ xgettext not found. Please install gettext tools.")
        return False
    except Exception as e:
        print(f"❌ Error creating template: {e}")
        return False


def validate_translations():
    """Validate .po files for syntax errors."""
    
    root_dir = Path(__file__).parent.parent
    locale_dir = root_dir / 'browser_use' / 'i18n' / 'locales'
    
    print(f"🔍 Validating translation files...")
    
    valid_count = 0
    
    for po_file in locale_dir.rglob('*.po'):
        print(f"📝 Validating: {po_file.relative_to(root_dir)}")
        
        try:
            result = subprocess.run([
                'msgfmt',
                '--check',
                '--verbose',
                str(po_file)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"  ✅ Valid")
                valid_count += 1
            else:
                print(f"  ❌ Invalid: {result.stderr}")
                
        except FileNotFoundError:
            print(f"  ❌ msgfmt not found")
            return False
        except Exception as e:
            print(f"  ❌ Error validating {po_file}: {e}")
    
    print(f"\n✅ Validated {valid_count} translation file(s)")
    return True


def main():
    """Main function."""
    
    print("🌐 Browser-Use Translation Compiler")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        action = sys.argv[1]
        
        if action == 'template':
            success = create_translation_template()
        elif action == 'validate':
            success = validate_translations()
        elif action == 'compile':
            success = compile_translations()
        else:
            print(f"❌ Unknown action: {action}")
            print("Available actions: template, validate, compile")
            success = False
    else:
        # Default: compile translations
        success = compile_translations()
    
    if success:
        print("\n🎉 Translation operations completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Translation operations failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 