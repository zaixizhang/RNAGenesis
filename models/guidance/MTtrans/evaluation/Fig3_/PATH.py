import os
import sys

main_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(main_dir)
print('add %s to path'%main_dir)