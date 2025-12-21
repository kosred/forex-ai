import ctypes
import logging
import time
from ctypes import wintypes

logger = logging.getLogger(__name__)

# Windows API Constants
INPUT_MOUSE    = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP       = 0x0002
KEYEVENTF_SCANCODE    = 0x0008

VK_CONTROL = 0x11
VK_E = 0x45

class MOUSEINPUT(ctypes.Structure):
    _fields_ = (("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR))

class KEYBDINPUT(ctypes.Structure):
    _fields_ = (("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR))

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (("uMsg", wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD))

class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = (("mi", MOUSEINPUT),
                    ("ki", KEYBDINPUT),
                    ("hi", HARDWAREINPUT))
    _anonymous_ = ("_input",)
    _fields_ = (("type", wintypes.DWORD),
                ("_input", _INPUT))

LPINPUT = ctypes.POINTER(INPUT)

def _send_input(inputs):
    nInputs = len(inputs)
    pInputs = (INPUT * nInputs)(*inputs)
    cbSize = ctypes.sizeof(INPUT)
    return ctypes.windll.user32.SendInput(nInputs, pInputs, cbSize)

def _input_key(vk, up=False):
    ki = KEYBDINPUT(wVk=vk, wScan=0, dwFlags=KEYEVENTF_KEYUP if up else 0, time=0, dwExtraInfo=0)
    return INPUT(type=INPUT_KEYBOARD, ki=ki)

def send_ctrl_e():
    """Simulate Ctrl+E keypress."""
    inputs = [
        _input_key(VK_CONTROL, up=False),
        _input_key(VK_E, up=False),
        _input_key(VK_E, up=True),
        _input_key(VK_CONTROL, up=True)
    ]
    _send_input(inputs)

def focus_mt5_window():
    """Find and focus the MetaTrader 5 window."""
    def callback(hwnd, extra):
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        buff = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
        title = buff.value
        if "MetaTrader 5" in title:
            logger.info(f"Found MT5 window: {title}")
            # Restore if minimized
            if ctypes.windll.user32.IsIconic(hwnd):
                 ctypes.windll.user32.ShowWindow(hwnd, 9) # SW_RESTORE
            # Bring to front
            ctypes.windll.user32.SetForegroundWindow(hwnd)
            return False # Stop enumeration
        return True

    CMPFUNC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
    ctypes.windll.user32.EnumWindows(CMPFUNC(callback), 0)

def ensure_autotrading_enabled(mt5_module):
    """
    Check if AutoTrading is enabled. If not, focus window and press Ctrl+E.
    Returns True if enabled (eventually), False otherwise.
    """
    if not mt5_module.initialize():
        return False

    info = mt5_module.terminal_info()
    if not info:
        return False

    if info.trade_allowed:
        logger.info("AutoTrading is already ENABLED.")
        return True

    logger.warning("AutoTrading is DISABLED. Attempting to enable via hotkey (Ctrl+E)...")

    focus_mt5_window()
    time.sleep(1.0) # Wait for focus
    send_ctrl_e()
    time.sleep(1.0) # Wait for toggle

    # Verify
    info = mt5_module.terminal_info()
    if info and info.trade_allowed:
        logger.info("Successfully enabled AutoTrading!")
        return True
    else:
        logger.error("Failed to enable AutoTrading via hotkey. Human intervention required.")
        return False
