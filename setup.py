import sys
import subprocess
import os
#------------ Upgrading PIP --------------------------------------------
directory = os.getcwd()
os.system('"' + directory  + '\venv\Scripts\python.exe' + '"' + '-m pip install --upgrade pip')

#----------------------- Installing Required Packages-------------------
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'meteostat'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pvlib'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scipy'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'frechetdist'])

#------------------- Installing Ng Spice --------------
'''
os.system('pyspice-post-installation --install-ngspice-dll')
os.system('pyspice-post-installation --check-install')
'''
