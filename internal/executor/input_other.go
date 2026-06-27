//go:build !windows

package executor

import "errors"

// sendInput is unsupported off Windows; real input is a no-op that errors so
// the agent's watchdog notices rather than silently believing it acted.
func sendInput(a Action) error {
	return errors.New("real input is only supported on windows")
}

// AbortRequested has no global-hotkey backend off Windows.
func AbortRequested() bool { return false }
