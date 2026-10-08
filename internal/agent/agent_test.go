// agent_test.go — unit tests for the backend AI subsystem.
package agent

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func testAgent(t *testing.T) *Agent {
	t.Helper()
	dir := t.TempDir()
	a, err := New(dir, Hooks{})
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func TestStoreQuotaReset(t *testing.T) {
	a := testAgent(t)
	a.store.SetQuotaLimit("qwen/qwen3.8-max", 2)
	a.store.NoteUsage("qwen/qwen3.8-max")
	a.store.NoteUsage("qwen/qwen3.8-max")
	if a.store.QuotaAllows("qwen/qwen3.8-max") {
		t.Fatal("quota should be exhausted after 2/2")
	}
	// simulate day rollover
	a.store.mu.Lock()
	a.store.st.Quotas["qwen/qwen3.8-max"].Day = "2000-01-01"
	a.store.mu.Unlock()
	if !a.store.QuotaAllows("qwen/qwen3.8-max") {
		t.Fatal("quota should reset on UTC day change")
	}
	snap := a.store.QuotaSnapshot()
	if snap["qwen/qwen3.8-max"]["used"] != 0 {
		t.Fatalf("expected used=0 after reset, got %v", snap["qwen/qwen3.8-max"])
	}
}

func TestAllowCandidate(t *testing.T) {
	a := testAgent(t)
	// default: everything allowed
	if !a.AllowCandidate("qwen", "qwen3.8-max") {
		t.Fatal("default should allow")
	}
	a.store.SetModelEnabled("qwen/qwen3.8-max", false)
	if a.AllowCandidate("qwen", "qwen3.8-max") {
		t.Fatal("disabled model should be filtered")
	}
	a.store.SetModelEnabled("qwen/qwen3.8-max", true)
	a.store.SetQuotaLimit("qwen/qwen3.8-max", 1)
	a.NoteUsage("qwen", "qwen3.8-max")
	if a.AllowCandidate("qwen", "qwen3.8-max") {
		t.Fatal("quota-exhausted model should be filtered")
	}
	// provider-level disable
	a.store.SetProvider(&ProviderState{Name: "custom:x", Enabled: false})
	if a.AllowCandidate("custom:x", "m") {
		t.Fatal("disabled provider should be filtered")
	}
}

func TestNoteUsageAndHistory(t *testing.T) {
	a := testAgent(t)
	a.NoteUsage("qwen", "qwen3.8-max")
	snap := a.store.QuotaSnapshot()
	if snap["qwen/qwen3.8-max"]["used"] != 1 {
		t.Fatalf("expected used=1, got %v", snap["qwen/qwen3.8-max"])
	}
	for i := 0; i < maxHistory+10; i++ {
		a.store.AddHistory("user", "hi", nil)
	}
	if len(a.store.History()) != maxHistory {
		t.Fatalf("history should be capped at %d", maxHistory)
	}
}

func TestParseToolCall(t *testing.T) {
	tool, args, ok := parseToolCall(`{"tool":"probe_model","args":{"model":"qwen/qwen3.8-max"}}`)
	if !ok || tool != "probe_model" || args["model"] != "qwen/qwen3.8-max" {
		t.Fatalf("plain JSON parse failed: %v %v %v", tool, args, ok)
	}
	tool, _, ok = parseToolCall("```json\n{\"tool\":\"list_models\",\"args\":{}}\n```")
	if !ok || tool != "list_models" {
		t.Fatal("fenced JSON parse failed")
	}
	if _, _, ok := parseToolCall("Hello, how are you?"); ok {
		t.Fatal("plain text should not parse as tool call")
	}
	if _, _, ok := parseToolCall(`{"foo":"bar"}`); ok {
		t.Fatal("non-tool JSON should not parse")
	}
}

func TestProxyPool(t *testing.T) {
	p := NewProxyPool(nil)
	if p.Next() != "" {
		t.Fatal("empty pool should return empty")
	}
	if !p.Add("http://127.0.0.1:8080") || !p.Add("http://127.0.0.1:8081") {
		t.Fatal("adds should succeed")
	}
	if p.Add("http://127.0.0.1:8080") {
		t.Fatal("duplicate add should fail")
	}
	if p.Add("not-a-url") {
		t.Fatal("invalid URL should fail")
	}
	// round-robin alternates
	a1, a2, a3 := p.Next(), p.Next(), p.Next()
	if a1 == a2 || a1 != a3 {
		t.Fatalf("expected rotation, got %s %s %s", a1, a2, a3)
	}
}

func TestProxyFor(t *testing.T) {
	a := testAgent(t)
	a.store.SetProvider(&ProviderState{Name: "custom:kilo", Enabled: true, UseProxy: true})
	a.pool.Add("http://127.0.0.1:9999")
	if got := a.ProxyFor("custom:kilo"); got != "http://127.0.0.1:9999" {
		t.Fatalf("expected proxy, got %q", got)
	}
	a.store.SetProvider(&ProviderState{Name: "qwen", Enabled: true})
	if got := a.ProxyFor("qwen"); got != "" {
		t.Fatalf("non-proxy provider should return empty, got %q", got)
	}
}

func TestOrderedChain(t *testing.T) {
	chain := OrderedChain([]ComboEntry{
		{Model: "b/m2", Weight: 1},
		{Model: "a/m1", Weight: 5},
		{Model: "c/m3", Weight: 3},
	})
	want := []string{"a/m1", "c/m3", "b/m2"}
	for i := range want {
		if chain[i] != want[i] {
			t.Fatalf("got %v, want %v", chain, want)
		}
	}
}

func TestStorePersistence(t *testing.T) {
	dir := t.TempDir()
	a, err := New(dir, Hooks{})
	if err != nil {
		t.Fatal(err)
	}
	a.store.SetProvider(&ProviderState{Name: "custom:t", Enabled: true, Source: "manual"})
	a.store.SetCombo("s", []ComboEntry{{Model: "custom:t/m", Weight: 2}})
	a.store.SetQuotaLimit("custom:t/m", 42)

	// reload from disk
	b, err := New(dir, Hooks{})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := b.store.GetProvider("custom:t"); !ok {
		t.Fatal("provider not persisted")
	}
	if len(b.store.ListCombos()["s"]) != 1 {
		t.Fatal("combo not persisted")
	}
	if b.store.QuotaSnapshot()["custom:t/m"]["limit"] != 42 {
		t.Fatal("quota not persisted")
	}
	// state file must exist under data dir
	if _, err := os.Stat(filepath.Join(dir, "agent_state.json")); err != nil {
		t.Fatal("state file missing")
	}
}

func TestChatWithoutCredentials(t *testing.T) {
	a := testAgent(t)
	if a.cf != nil {
		t.Skip("CF credentials set in env; skipping unconfigured-path test")
	}
	_, _, err := a.Chat("hello")
	if err == nil || !contains(err.Error(), "CF_ACCOUNT_ID") {
		t.Fatalf("expected config error, got %v", err)
	}
}

func TestDiscoverKeylessShapes(t *testing.T) {
	// structural check only (no network): defaults are well-formed
	seen := map[string]bool{}
	for _, ep := range defaultKeylessEndpoints {
		if ep.Name == "" || ep.ProbeURL == "" || ep.Method == "" {
			t.Fatalf("malformed endpoint entry: %+v", ep)
		}
		if seen[ep.Name] {
			t.Fatalf("duplicate endpoint name %s", ep.Name)
		}
		seen[ep.Name] = true
	}
	if len(defaultKeylessEndpoints) < 3 {
		t.Fatal("need at least 3 researched endpoints")
	}
	_ = time.Second // keep time imported if unused elsewhere
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (func() bool {
		for i := 0; i+len(sub) <= len(s); i++ {
			if s[i:i+len(sub)] == sub {
				return true
			}
		}
		return false
	})()
}
