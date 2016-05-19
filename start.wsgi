import sys, os
import site

#add the project dir to the path
sys.path.insert(0, '/var/www/airpollution_modeling/venv_python_modeling/src')

# Add the site-packages of the chosen virtualenv to work with
site.addsitedir('/var/www/airpollution_modeling/venv_python_modeling/local/lib/python2.7/site-packages')

sys.stdout = sys.stderr


# Activate your virtual env
activate_env=os.path.expanduser("/var/www/airpollution_modeling/venv_python_modeling/bin/activate_this.py")
execfile(activate_env, dict(__file__=activate_env))

from webapp import app as application
