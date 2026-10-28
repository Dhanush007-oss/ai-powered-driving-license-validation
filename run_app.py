import os
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(APP_DIR)
sys.path.insert(0, APP_DIR)

from app import demo

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, show_error=True)
