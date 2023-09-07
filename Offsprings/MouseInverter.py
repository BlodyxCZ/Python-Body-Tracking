from pynput.mouse import Listener
import win32api
from ctypes import *

windll.user32.BlockInput(True)

def on_move(x, y):
    print('Pointer moved to {0}'.format(
        (x, y)))
    win32api.SetCursorPos((y,x))


with Listener(on_move=on_move) as listener:
    listener.join()
