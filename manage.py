#!/usr/bin/env python
import os
import sys
import io

# Force UTF-8 on Windows (avoids cp1252 errors with Russian text)
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'geo_project.settings')
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()
