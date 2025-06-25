import os
import sys

from django.core.wsgi import get_wsgi_application

# Add your project directory to the sys.path
path = '/home/darshil01/mysite'
if path not in sys.path:
    sys.path.insert(0, path)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'csv_analyzer.settings')

application = get_wsgi_application()
