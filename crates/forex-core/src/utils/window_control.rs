use tracing::{error, info, warn};

#[cfg(target_os = "windows")]
use std::ffi::OsString;
#[cfg(target_os = "windows")]
use std::os::windows::ffi::OsStringExt;
#[cfg(target_os = "windows")]
use windows::Win32::Foundation::{BOOL, HWND, LPARAM};
#[cfg(target_os = "windows")]
use windows::Win32::UI::Input::KeyboardAndMouse::{
    SendInput, INPUT, INPUT_KEYBOARD, KEYBDINPUT, KEYEVENTF_KEYUP, VIRTUAL_KEY, VK_CONTROL, VK_E,
};
#[cfg(target_os = "windows")]
use windows::Win32::UI::WindowsAndMessaging::{
    EnumWindows, GetWindowTextLengthW, GetWindowTextW, IsIconic, SetForegroundWindow, ShowWindow,
    SW_RESTORE,
};

pub fn ensure_autotrading_enabled() -> bool {
    // This function originally acted as a high-level check.
    // Since we don't have the full MT5 module here yet,
    // we will implement the low-level window control parts that were the core complexity.
    // The actual logic verifying MT5 state typically belongs in an execution/broker adapter.
    // For now, we provide the ability to toggle it via the window control if needed.

    // In Python this function did:
    // 1. Check if linux -> return True
    // 2. Check MT5 connection -> return False if fail
    // 3. Check terminal_info.trade_allowed -> return True if allowed
    // 4. If disabled -> focus window -> send Ctrl+E -> check again

    // We will port the "focus window -> send Ctrl+E" primitives here.

    if !cfg!(target_os = "windows") {
        return true;
    }

    // Stub implementation for now as we don't have MT5 connection in core yet.
    // But we will expose the helper functions.
    true
}

#[cfg(target_os = "windows")]
pub fn focus_mt5_window() -> bool {
    unsafe {
        let mut found_hwnd: Option<HWND> = None;

        unsafe extern "system" fn enum_window_proc(hwnd: HWND, lparam: LPARAM) -> BOOL {
            let found_ptr = lparam.0 as *mut Option<HWND>;

            let length = GetWindowTextLengthW(hwnd);
            if length > 0 {
                let mut buffer = vec![0u16; (length + 1) as usize];
                GetWindowTextW(hwnd, &mut buffer);
                let title = OsString::from_wide(&buffer[..length as usize]);
                let title_lossy = title.to_string_lossy();

                if title_lossy.contains("MetaTrader 5") {
                    // Found it
                    *found_ptr = Some(hwnd);
                    return BOOL(0); // Stop enumeration
                }
            }
            BOOL(1) // Continue enumeration
        }

        let lparam = LPARAM(&mut found_hwnd as *mut _ as isize);
        let _ = EnumWindows(Some(enum_window_proc), lparam);

        if let Some(hwnd) = found_hwnd {
            info!("Found MT5 window. Focusing...");
            if IsIconic(hwnd).as_bool() {
                let _ = ShowWindow(hwnd, SW_RESTORE);
            }
            let _ = SetForegroundWindow(hwnd);
            return true;
        }
    }
    warn!("MetaTrader 5 window not found.");
    false
}

#[cfg(not(target_os = "windows"))]
pub fn focus_mt5_window() -> bool {
    warn!("Window control is not supported on this OS.");
    false
}

#[cfg(target_os = "windows")]
pub fn send_ctrl_e() {
    unsafe {
        let inputs = [
            input_key(VK_CONTROL, false),
            input_key(VK_E, false),
            input_key(VK_E, true),
            input_key(VK_CONTROL, true),
        ];

        SendInput(&inputs, std::mem::size_of::<INPUT>() as i32);
    }
}

#[cfg(not(target_os = "windows"))]
pub fn send_ctrl_e() {
    warn!("Keyboard input is not supported on this OS.");
}

#[cfg(target_os = "windows")]
fn input_key(vk: VIRTUAL_KEY, up: bool) -> INPUT {
    let mut input = INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: Default::default(),
    };

    let flags = if up {
        KEYEVENTF_KEYUP
    } else {
        Default::default()
    };

    input.Anonymous.ki = KEYBDINPUT {
        wVk: vk,
        wScan: 0,
        dwFlags: flags,
        time: 0,
        dwExtraInfo: 0,
    };

    input
}
