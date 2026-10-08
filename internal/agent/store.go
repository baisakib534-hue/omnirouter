// store.go — JSON-file persisted state for the backend AI.
//
// State lives at <dataDir>/agent_state.json (data/ is gitignored).
// API keys are NEVER persisted: providers keep only the NAME of the env var
// holding the key (api_key_env); auto-discovered keyless providers may carry
// a non-secret placeholder key such as "unused".

package agent

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// ProviderState is one upstream provider managed by the agent. The Name is
// the core provider id ("qwen", "glm", "custom:llm7", …).
type ProviderState struct {
	Name      string   `json:"name"`
	BaseURL   string   `json:"base_url"`
	APIKeyEnv string   `json:"api_key_env,omitempty"`
	Models    []string `json:"models"`
	Enabled   bool     `json:"enabled"`
	Source    string   `json:"source"` // "manual" | "auto-discovered"
	UseProxy  bool     `json:"use_proxy,omitempty"`
}

// ComboEntry is one weighted member of a named combo.
type ComboEntry struct {
	Model  string `json:"model"`
	Weight int    `json:"weight"`
}

// ModelHealth tracks per-model ("provider/model") liveness.
type ModelHealth struct {
	Enabled   bool   `json:"enabled"`
	LastCheck int64  `json:"last_check,omitempty"` // unix
	LastError string `json:"last_error,omitempty"`
}

// ModelQuota tracks the daily request quota of one model key.
type ModelQuota struct {
	DailyLimit int    `json:"daily_limit"` // 0 = unlimited
	Used       int    `json:"used"`
	Day        string `json:"day"` // YYYY-MM-DD (UTC); counters reset on day change
}

// Exchange is one persisted chat turn.
type Exchange struct {
	Role    string   `json:"role"` // "user" | "assistant" | "system"
	Content string   `json:"content"`
	Time    int64    `json:"time"`
	Actions []string `json:"actions,omitempty"`
}

type agentState struct {
	Providers map[string]*ProviderState `json:"providers"`
	Combos    map[string][]ComboEntry   `json:"combos"`
	Health    map[string]*ModelHealth   `json:"health"`
	Quotas    map[string]*ModelQuota    `json:"quotas"`
	Proxies   []string                  `json:"proxies"`
	History   []Exchange                `json:"history"`
}

// maxHistory bounds the persisted chat history.
const maxHistory = 50

// Store is a mutex-guarded, eagerly-flushed JSON store.
type Store struct {
	mu   sync.Mutex
	path string
	st   agentState
}

// NewStore loads (or creates) the state file under dir.
func NewStore(dir string) (*Store, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	s := &Store{path: filepath.Join(dir, "agent_state.json")}
	s.st = agentState{
		Providers: map[string]*ProviderState{},
		Combos:    map[string][]ComboEntry{},
		Health:    map[string]*ModelHealth{},
		Quotas:    map[string]*ModelQuota{},
	}
	if raw, err := os.ReadFile(s.path); err == nil && len(raw) > 0 {
		_ = json.Unmarshal(raw, &s.st) // corrupt file → start fresh
		if s.st.Providers == nil {
			s.st.Providers = map[string]*ProviderState{}
		}
		if s.st.Combos == nil {
			s.st.Combos = map[string][]ComboEntry{}
		}
		if s.st.Health == nil {
			s.st.Health = map[string]*ModelHealth{}
		}
		if s.st.Quotas == nil {
			s.st.Quotas = map[string]*ModelQuota{}
		}
	}
	return s, nil
}

func (s *Store) saveLocked() {
	raw, err := json.MarshalIndent(s.st, "", "  ")
	if err != nil {
		return
	}
	tmp := s.path + ".tmp"
	if os.WriteFile(tmp, raw, 0o600) != nil {
		return
	}
	_ = os.Rename(tmp, s.path)
}

func todayUTC() string { return time.Now().UTC().Format("2006-01-02") }

// quotaForLocked returns the quota record, lazily resetting the counter at
// UTC midnight.
func (s *Store) quotaForLocked(key string) *ModelQuota {
	q := s.st.Quotas[key]
	if q == nil {
		q = &ModelQuota{Day: todayUTC()}
		s.st.Quotas[key] = q
	}
	if q.Day != todayUTC() {
		q.Day = todayUTC()
		q.Used = 0
	}
	return q
}

// --- providers ---

func (s *Store) SetProvider(p *ProviderState) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.st.Providers[p.Name] = p
	s.saveLocked()
}

func (s *Store) GetProvider(name string) (*ProviderState, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	p, ok := s.st.Providers[name]
	return p, ok
}

func (s *Store) ListProviders() []*ProviderState {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]*ProviderState, 0, len(s.st.Providers))
	for _, p := range s.st.Providers {
		cp := *p
		out = append(out, &cp)
	}
	return out
}

func (s *Store) SetProviderEnabled(name string, enabled bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if p, ok := s.st.Providers[name]; ok {
		p.Enabled = enabled
		s.saveLocked()
	}
}

// --- combos ---

func (s *Store) SetCombo(name string, entries []ComboEntry) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.st.Combos[name] = entries
	s.saveLocked()
}

func (s *Store) DeleteCombo(name string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.st.Combos[name]; !ok {
		return false
	}
	delete(s.st.Combos, name)
	s.saveLocked()
	return true
}

func (s *Store) ListCombos() map[string][]ComboEntry {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make(map[string][]ComboEntry, len(s.st.Combos))
	for k, v := range s.st.Combos {
		out[k] = append([]ComboEntry(nil), v...)
	}
	return out
}

// --- model health ---

func (s *Store) HealthFor(key string) *ModelHealth {
	s.mu.Lock()
	defer s.mu.Unlock()
	h := s.st.Health[key]
	if h == nil {
		h = &ModelHealth{Enabled: true}
		s.st.Health[key] = h
	}
	return h
}

func (s *Store) SetModelEnabled(key string, enabled bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	h := s.st.Health[key]
	if h == nil {
		h = &ModelHealth{}
		s.st.Health[key] = h
	}
	h.Enabled = enabled
	s.saveLocked()
}

func (s *Store) RecordModelCheck(key string, ok bool, errMsg string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	h := s.st.Health[key]
	if h == nil {
		h = &ModelHealth{Enabled: true}
		s.st.Health[key] = h
	}
	h.LastCheck = time.Now().Unix()
	h.LastError = errMsg
	if !ok {
		h.Enabled = false
	}
	s.saveLocked()
}

// --- quotas ---

func (s *Store) SetQuotaLimit(key string, limit int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	q := s.quotaForLocked(key)
	q.DailyLimit = limit
	s.saveLocked()
}

// NoteUsage increments the daily counter for a model key (UTC-day aware).
func (s *Store) NoteUsage(key string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.quotaForLocked(key).Used++
	s.saveLocked()
}

// QuotaAllows reports whether a model key may serve another request.
func (s *Store) QuotaAllows(key string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	q := s.quotaForLocked(key)
	return q.DailyLimit == 0 || q.Used < q.DailyLimit
}

// QuotaSnapshot returns per-model {limit, used, remaining, enabled} for
// models that have a configured limit, recorded usage, or health state.
func (s *Store) QuotaSnapshot() map[string]map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := map[string]map[string]any{}
	seen := map[string]bool{}
	collect := func(key string) {
		if seen[key] {
			return
		}
		seen[key] = true
		q := s.quotaForLocked(key)
		if q.DailyLimit == 0 && q.Used == 0 {
			if _, ok := s.st.Health[key]; !ok {
				return
			}
		}
		remaining := -1 // -1 = unlimited
		if q.DailyLimit > 0 {
			remaining = q.DailyLimit - q.Used
			if remaining < 0 {
				remaining = 0
			}
		}
		enabled := true
		if h, ok := s.st.Health[key]; ok {
			enabled = h.Enabled
		}
		out[key] = map[string]any{
			"limit":     q.DailyLimit,
			"used":      q.Used,
			"remaining": remaining,
			"enabled":   enabled,
			"day":       q.Day,
		}
	}
	for k := range s.st.Quotas {
		collect(k)
	}
	for k := range s.st.Health {
		collect(k)
	}
	return out
}

// QuotaSummary renders a compact plain-text quota/health summary for the
// agent's system prompt.
func (s *Store) QuotaSummary() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.st.Quotas) == 0 && len(s.st.Health) == 0 {
		return "(no quotas configured yet — all models unlimited until you set_quota)"
	}
	var b strings.Builder
	for key := range s.st.Quotas {
		qq := s.quotaForLocked(key)
		enabled := true
		if h, ok := s.st.Health[key]; ok {
			enabled = h.Enabled
		}
		lim := "unlimited"
		if qq.DailyLimit > 0 {
			lim = itoa(qq.DailyLimit)
		}
		b.WriteString("- " + key + ": used " + itoa(qq.Used) + " / limit " + lim)
		if !enabled {
			b.WriteString(" [DISABLED]")
		}
		b.WriteString("\n")
	}
	for key, h := range s.st.Health {
		if _, ok := s.st.Quotas[key]; ok {
			continue
		}
		if !h.Enabled {
			b.WriteString("- " + key + ": [DISABLED]")
			if h.LastError != "" {
				b.WriteString(" last error: " + truncate(h.LastError, 120))
			}
			b.WriteString("\n")
		}
	}
	return strings.TrimRight(b.String(), "\n")
}

// --- proxies ---

func (s *Store) ListProxies() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.st.Proxies...)
}

func (s *Store) AddProxy(url string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, u := range s.st.Proxies {
		if u == url {
			return false
		}
	}
	s.st.Proxies = append(s.st.Proxies, url)
	s.saveLocked()
	return true
}

// --- history ---

// AddHistory appends an exchange, keeping the last maxHistory entries.
func (s *Store) AddHistory(role, content string, actions []string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.st.History = append(s.st.History, Exchange{
		Role: role, Content: content, Time: time.Now().Unix(), Actions: actions,
	})
	if len(s.st.History) > maxHistory {
		s.st.History = s.st.History[len(s.st.History)-maxHistory:]
	}
	s.saveLocked()
}

// History returns a copy of the persisted exchanges (oldest first).
func (s *Store) History() []Exchange {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]Exchange(nil), s.st.History...)
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}
