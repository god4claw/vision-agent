//go:build windows

package executor

import (
	"fmt"
	"strconv"
	"strings"
	"syscall"
	"time"
)

var (
	user32          = syscall.NewLazyDLL("user32.dll")
	procSetCursor   = user32.NewProc("SetCursorPos")
	procMouseEvent  = user32.NewProc("mouse_event")
	procKeybdEvent  = user32.NewProc("keybd_event")
	procGetAsyncKey = user32.NewProc("GetAsyncKeyState")
)

const (
	mouseLeftDown = 0x0002
	mouseLeftUp   = 0x0004
	keyEventUp    = 0x0002
	vkEscape      = 0x1B
)

// sendInput performs real mouse/keyboard input via user32.dll.
func sendInput(a Action) error {
	switch a.Type {
	case ActionNone:
		return nil
	case ActionMove:
		procSetCursor.Call(uintptr(a.X), uintptr(a.Y))
		return nil
	case ActionClick:
		procSetCursor.Call(uintptr(a.X), uintptr(a.Y))
		time.Sleep(15 * time.Millisecond)
		procMouseEvent.Call(mouseLeftDown, 0, 0, 0, 0)
		time.Sleep(15 * time.Millisecond)
		procMouseEvent.Call(mouseLeftUp, 0, 0, 0, 0)
		return nil
	case ActionDrag:
		procSetCursor.Call(uintptr(a.X), uintptr(a.Y))
		time.Sleep(15 * time.Millisecond)
		procMouseEvent.Call(mouseLeftDown, 0, 0, 0, 0)
		time.Sleep(40 * time.Millisecond)
		procSetCursor.Call(uintptr(a.X2), uintptr(a.Y2))
		time.Sleep(40 * time.Millisecond)
		procMouseEvent.Call(mouseLeftUp, 0, 0, 0, 0)
		return nil
	case ActionKey:
		return sendKey(a.Key)
	default:
		return fmt.Errorf("unsupported action %q", a.Type)
	}
}

// sendKey presses a key or modifier combination, e.g. "Ctrl+Shift+P".
func sendKey(s string) error {
	var mods []uintptr
	var main uintptr
	haveMain := false
	for _, part := range strings.Split(strings.TrimSpace(s), "+") {
		p := strings.TrimSpace(part)
		if m, ok := modVK(p); ok {
			mods = append(mods, m)
			continue
		}
		vk, ok := vkFor(p)
		if !ok {
			return fmt.Errorf("unmappable key token %q", p)
		}
		main = vk
		haveMain = true
	}
	if !haveMain {
		return fmt.Errorf("no main key in %q", s)
	}
	for _, m := range mods {
		procKeybdEvent.Call(m, 0, 0, 0)
	}
	procKeybdEvent.Call(main, 0, 0, 0)
	procKeybdEvent.Call(main, 0, keyEventUp, 0)
	for i := len(mods) - 1; i >= 0; i-- {
		procKeybdEvent.Call(mods[i], 0, keyEventUp, 0)
	}
	return nil
}

func modVK(tok string) (uintptr, bool) {
	switch strings.ToLower(tok) {
	case "ctrl", "control":
		return 0x11, true
	case "alt", "option":
		return 0x12, true
	case "shift":
		return 0x10, true
	case "win", "super", "cmd", "meta":
		return 0x5B, true
	}
	return 0, false
}

// vkFor maps a (lowercased) key token to a Windows virtual-key code. Only the
// whitelisted, layout-independent keys are mapped; anything else is rejected so
// real input is never sent for an unmappable token.
func vkFor(tok string) (uintptr, bool) {
	t := strings.ToLower(tok)
	if len(t) == 1 {
		c := t[0]
		switch {
		case c >= 'a' && c <= 'z':
			return uintptr(0x41 + (c - 'a')), true
		case c >= '0' && c <= '9':
			return uintptr(0x30 + (c - '0')), true
		}
	}
	switch t {
	case "enter", "return":
		return 0x0D, true
	case "tab":
		return 0x09, true
	case "esc", "escape":
		return 0x1B, true
	case "space":
		return 0x20, true
	case "backspace":
		return 0x08, true
	case "delete", "del":
		return 0x2E, true
	case "insert", "ins":
		return 0x2D, true
	case "up":
		return 0x26, true
	case "down":
		return 0x28, true
	case "left":
		return 0x25, true
	case "right":
		return 0x27, true
	case "home":
		return 0x24, true
	case "end":
		return 0x23, true
	case "pageup", "pgup":
		return 0x21, true
	case "pagedown", "pgdn":
		return 0x22, true
	case "capslock":
		return 0x14, true
	}
	if len(t) >= 2 && t[0] == 'f' {
		if n, err := strconv.Atoi(t[1:]); err == nil && n >= 1 && n <= 24 {
			return uintptr(0x70 + n - 1), true
		}
	}
	return 0, false
}

// AbortRequested is the kill-switch: true while ESC is physically held down.
func AbortRequested() bool {
	r, _, _ := procGetAsyncKey.Call(uintptr(vkEscape))
	return r&0x8000 != 0
}
