// agent_wire.go — wires the backend AI (internal/agent) into the router.
//
// The agent package must not import core (import cycle), so core owns this
// glue: it builds the agent.Hooks from the live registry/forwarder/store,
// exposes request-path hook vars consumed by proxy.go, mounts the agent's
// HTTP surface, and starts its scheduler.

package core

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/Godde3s/omnirouter/internal/agent"
)

// Request-path hooks consumed by proxy.go. Nil until the agent mounts, in
// which case the request path behaves exactly as before.
var agentFilterChain func(model string, chain []*Provider) []*Provider
var agentNoteUsage func(providerID, upstreamModel string)
var agentProxyFor func(providerID string) string

// proxyClientFor returns a cached *http.Client routing through proxyURL
// (used for auto-discovered keyless providers when a proxy pool exists).
var proxyClientCache = struct {
	sync.Mutex
	m map[string]*http.Client
}{m: map[string]*http.Client{}}

func proxyClientFor(proxyURL string) *http.Client {
	proxyClientCache.Lock()
	defer proxyClientCache.Unlock()
	if c, ok := proxyClientCache.m[proxyURL]; ok {
		return c
	}
	var tr http.RoundTripper = http.DefaultTransport
	if pu, err := url.Parse(proxyURL); err == nil {
		tr = &http.Transport{Proxy: http.ProxyURL(pu)}
	}
	c := &http.Client{Transport: tr} // Timeout 0, like the forwarder (streams)
	proxyClientCache.m[proxyURL] = c
	return c
}

// setRegistryProviderEnabled flips a provider's enabled flag by core id
// ("qwen") or custom name ("llm7" / "custom:llm7").
func setRegistryProviderEnabled(reg *Registry, id string, enabled bool) {
	want := strings.ToLower(strings.TrimPrefix(strings.TrimSpace(id), "custom:"))
	reg.mu.Lock()
	defer reg.mu.Unlock()
	for _, p := range reg.providers {
		pid := strings.ToLower(p.ID)
		if pid == strings.ToLower(strings.TrimSpace(id)) || strings.TrimPrefix(pid, "custom:") == want {
			p.Enabled = enabled
		}
	}
}

// agentProbe performs one minimal chat completion through the local router
// (used by the agent's probe_model tool and the health scheduler).
// X-Agent-Probe bypasses the agent's own quota filter so disabled models
// can still be probed.
func agentProbe(port int, apiKey, model string) (int64, string) {
	payload, _ := json.Marshal(map[string]any{
		"model":      model,
		"messages":   []map[string]string{{"role": "user", "content": "ping"}},
		"max_tokens": 8,
		"stream":     false,
	})
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "POST",
		fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", port),
		bytes.NewReader(payload))
	if err != nil {
		return 0, err.Error()
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Agent-Probe", "1")
	t0 := time.Now()
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return 0, err.Error()
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	ms := time.Since(t0).Milliseconds()
	if resp.StatusCode != 200 {
		return ms, fmt.Sprintf("HTTP %d: %s", resp.StatusCode, shorten(string(body), 200))
	}
	var chk struct {
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &chk) == nil && chk.Error != nil {
		return ms, "upstream error: "+chk.Error.Message
	}
	return ms, ""
}

func shorten(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

// mountAgent creates the backend AI, installs the request-path hooks,
// re-seeds persisted combos, mounts /agent + /api/agent/* (admin-gated)
// and starts the background scheduler.
func (rt *Router) mountAgent(mux *http.ServeMux, seedKey string) {
	hooks := agent.Hooks{
		Probe: func(model string) (int64, string) {
			return agentProbe(rt.cfg.Port, seedKey, model)
		},
		SetCombo: func(name string, chain []string) {
			rt.registry.SetCombo(name, chain)
		},
		DeleteCombo: func(name string) bool {
			return rt.registry.DeleteCombo(name)
		},
		AddProvider: func(def agent.ProviderDef) error {
			if !strings.HasPrefix(def.ID, "custom:") {
				return fmt.Errorf("refusing to overwrite built-in provider %s", def.ID)
			}
			key := def.APIKey
			if key == "" && def.APIKeyEnv != "" {
				key = strings.TrimSpace(os.Getenv(def.APIKeyEnv))
			}
			cp := CustomProvider{
				Name:    strings.TrimPrefix(def.ID, "custom:"),
				BaseURL: def.BaseURL,
				APIKey:  key,
				Models:  def.Models,
				Enabled: def.Enabled,
			}
			req, _ := http.NewRequest("GET", "/", nil)
			return upsertProvider(req, rt.store, rt.registry, cp)
		},
		SetProviderEnabled: func(id string, enabled bool) {
			setRegistryProviderEnabled(rt.registry, id, enabled)
		},
		ListCoreProviders: func() []agent.CoreProviderInfo {
			out := make([]agent.CoreProviderInfo, 0)
			for _, p := range rt.registry.Providers() {
				out = append(out, agent.CoreProviderInfo{
					ID:      p.ID,
					Label:   p.Label,
					Enabled: p.Enabled,
					Models:  append([]string(nil), p.Models...),
				})
			}
			return out
		},
	}

	ag, err := agent.New(rt.cfg.DataDir, hooks)
	if err != nil {
		log.Printf("[agent] disabled: %v", err)
		return
	}

	// Re-register persisted custom providers (e.g. auto-discovered keyless
	// ones) into the core registry. Core forgets them on restart; without
	// this they stay invisible until the next discovery run, and the health
	// check would probe models the router cannot resolve.
	for _, p := range ag.Providers() {
		if !strings.HasPrefix(p.Name, "custom:") {
			continue
		}
		def := agent.ProviderDef{
			ID: p.Name, BaseURL: p.BaseURL, APIKeyEnv: p.APIKeyEnv,
			Models: p.Models, Source: p.Source, UseProxy: p.UseProxy,
			Enabled: p.Enabled,
		}
		if err := hooks.AddProvider(def); err != nil {
			log.Printf("[agent] re-register provider %s: %v", p.Name, err)
			continue
		}
		if !p.Enabled {
			// upsertProvider hot-registers as enabled; mirror the
			// persisted disabled state in core too.
			hooks.SetProviderEnabled(p.Name, false)
		}
	}

	// Request-path hooks (read by proxy.go).
	agentFilterChain = func(_ string, chain []*Provider) []*Provider {
		out := make([]*Provider, 0, len(chain))
		for _, p := range chain {
			up := ""
			if len(p.Models) == 1 {
				up = p.Models[0]
			}
			if ag.AllowCandidate(p.ID, up) {
				out = append(out, p)
			}
		}
		return out
	}
	agentNoteUsage = func(providerID, upstreamModel string) {
		ag.NoteUsage(providerID, upstreamModel)
	}
	agentProxyFor = func(providerID string) string {
		return ag.ProxyFor(providerID)
	}

	// Re-seed persisted combos (agent store is the source of truth for
	// weighted combos; weights decide the chain order).
	for name, entries := range ag.Combos() {
		rt.registry.SetCombo(name, agent.OrderedChain(entries))
	}

	ag.Mount(mux, func(next http.HandlerFunc) http.HandlerFunc {
		return adminAPI(next, rt.cfg.AdminPassword)
	})
	ag.StartScheduler(rt.stop)
	log.Printf("[agent] backend AI mounted: /agent UI + /api/agent/* (admin-gated)")
}
