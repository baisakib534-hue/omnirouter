// scheduler.go — background operator jobs.
//
//   - Model health check every AGENT_CHECK_INTERVAL (default 24h): probes
//     every enabled model through the router; failures disable the model,
//     and a provider whose every model fails is disabled too. A short
//     summary is appended to the chat history so the user sees it.
//   - Daily keyless discovery: probes the researched free-endpoint list and
//     auto-adds newly working OpenAI-routable providers (enabled, marked
//     source "auto-discovered", traffic via the proxy pool when configured).

package agent

import (
	"fmt"
	"log"
	"os"
	"sort"
	"strings"
	"time"
)

// StartScheduler launches the background jobs; they stop when stop closes.
func (a *Agent) StartScheduler(stop <-chan struct{}) {
	checkInt := parseDurationEnv("AGENT_CHECK_INTERVAL", 24*time.Hour)
	go a.checkLoop(stop, checkInt)
	go a.discoveryLoop(stop, 24*time.Hour)
	log.Printf("[agent] scheduler started (health check every %s, discovery every 24h)", checkInt)
}

func parseDurationEnv(key string, def time.Duration) time.Duration {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d > 0 {
			return d
		}
		log.Printf("[agent] invalid %s=%q, using default %s", key, v, def)
	}
	return def
}

func (a *Agent) checkLoop(stop <-chan struct{}, interval time.Duration) {
	// first run shortly after boot (non-blocking), then on the interval
	select {
	case <-time.After(60 * time.Second):
		a.checkAllModels()
	case <-stop:
		return
	}
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-t.C:
			a.checkAllModels()
		case <-stop:
			return
		}
	}
}

func (a *Agent) discoveryLoop(stop <-chan struct{}, interval time.Duration) {
	select {
	case <-time.After(90 * time.Second):
		a.runDiscovery()
	case <-stop:
		return
	}
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-t.C:
			a.runDiscovery()
		case <-stop:
			return
		}
	}
}

// probeSet collects every model key worth health-checking.
// probeTarget is one model to health-check: the "provider/model" key used
// for probing and health/quota bookkeeping, plus the owning provider id
// (kept separately because upstream model names may themselves contain "/").
type probeTarget struct {
	key      string
	provider string
}

func (a *Agent) probeSet() []probeTarget {
	seen := map[string]bool{}
	var out []probeTarget
	add := func(provider, key string) {
		if !seen[key] {
			seen[key] = true
			out = append(out, probeTarget{key: key, provider: provider})
		}
	}
	for _, p := range a.store.ListProviders() {
		if !p.Enabled {
			continue
		}
		for _, m := range p.Models {
			add(p.Name, p.Name+"/"+m)
		}
	}
	if a.hooks.ListCoreProviders != nil {
		for _, p := range a.hooks.ListCoreProviders() {
			if !p.Enabled {
				continue
			}
			for _, m := range p.Models {
				add(p.ID, p.ID+"/"+m)
			}
		}
	}
	for key, h := range a.store.st.Health {
		if h.Enabled {
			add(a.providerOfKey(key), key)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].key < out[j].key })
	if len(out) > 80 {
		out = out[:80] // keep the daily job bounded
	}
	return out
}

// providerOfKey recovers the provider id from a "provider/model" key by
// longest-prefix match against known providers, since upstream model names
// may themselves contain "/".
func (a *Agent) providerOfKey(key string) string {
	best := ""
	consider := func(id string) {
		if id != "" && (key == id || strings.HasPrefix(key, id+"/")) && len(id) > len(best) {
			best = id
		}
	}
	for _, p := range a.store.ListProviders() {
		consider(p.Name)
	}
	if a.hooks.ListCoreProviders != nil {
		for _, cp := range a.hooks.ListCoreProviders() {
			consider(cp.ID)
		}
	}
	if best != "" {
		return best
	}
	if i := strings.Index(key, "/"); i > 0 {
		return key[:i]
	}
	return key
}

func (a *Agent) checkAllModels() {
	if a.hooks.Probe == nil {
		return
	}
	models := a.probeSet()
	if len(models) == 0 {
		return
	}
	log.Printf("[agent] health check: probing %d models", len(models))
	failed := map[string][]string{} // provider -> failed model keys
	disabled := 0
	for _, t := range models {
		ms, errMsg := a.hooks.Probe(t.key)
		if errMsg != "" {
			a.store.RecordModelCheck(t.key, false, errMsg)
			disabled++
			failed[t.provider] = append(failed[t.provider], t.key)
			log.Printf("[agent] model %s FAILED (%dms): %s — disabled", t.key, ms, truncate(errMsg, 120))
		} else {
			a.store.RecordModelCheck(t.key, true, "")
		}
	}
	// disable providers whose every probed model failed
	for prov, bad := range failed {
		if a.providerFullyFailed(prov, bad) {
			a.store.SetProviderEnabled(prov, false)
			if a.hooks.SetProviderEnabled != nil {
				a.hooks.SetProviderEnabled(prov, false)
			}
			log.Printf("[agent] provider %s disabled: all %d probed models failed", prov, len(bad))
		}
	}
	summary := fmt.Sprintf("Health check: probed %d models, %d failed and were disabled.", len(models), disabled)
	if len(failed) > 0 {
		var ps []string
		for p := range failed {
			ps = append(ps, p)
		}
		summary += " Failing providers: " + strings.Join(ps, ", ") + "."
	}
	a.store.AddHistory("system", summary, []string{"health-check"})
	log.Printf("[agent] %s", summary)
}

// providerFullyFailed reports whether every probed model of prov failed.
// Providers with no successful model left standing get disabled.
func (a *Agent) providerFullyFailed(prov string, bad []string) bool {
	// count enabled models the provider currently has
	total := 0
	if p, ok := a.store.GetProvider(prov); ok && p.Enabled {
		total = len(p.Models)
	} else if a.hooks.ListCoreProviders != nil {
		for _, cp := range a.hooks.ListCoreProviders() {
			if cp.ID == prov && cp.Enabled {
				total = len(cp.Models)
				break
			}
		}
	}
	return total > 0 && len(bad) >= total
}

// runDiscovery probes the keyless list and auto-adds working routable
// providers that are not already known.
func (a *Agent) runDiscovery() {
	results := a.DiscoverKeyless()
	added := 0
	for i, r := range results {
		ep := defaultKeylessEndpoints[i]
		status := "down"
		if r.OK {
			status = "up"
		}
		log.Printf("[agent] discovery: %s %s (%dms)", ep.Name, status, r.LatencyMs)
		if !r.OK || !ep.Routable {
			continue
		}
		id := normalizeProviderID(ep.Name)
		if _, ok := a.store.GetProvider(id); ok {
			continue // already managed
		}
		def := ProviderDef{
			ID: id, BaseURL: ep.BaseURL, APIKey: ep.APIKey,
			Models: ep.Models, Source: "auto-discovered",
			UseProxy: true, Enabled: true,
		}
		if a.hooks.AddProvider == nil {
			continue
		}
		if err := a.hooks.AddProvider(def); err != nil {
			log.Printf("[agent] discovery: failed to add %s: %v", id, err)
			continue
		}
		a.store.SetProvider(&ProviderState{
			Name: id, BaseURL: ep.BaseURL, Models: ep.Models,
			Enabled: true, Source: "auto-discovered", UseProxy: true,
		})
		a.applyDefaultQuota(id, ep.Models)
		added++
		log.Printf("[agent] discovery: auto-added keyless provider %s", id)
	}
	summary := fmt.Sprintf("Keyless discovery: %d endpoint(s) responding, %d new provider(s) auto-added.", countUp(results), added)
	a.store.AddHistory("system", summary, []string{"discover-keyless"})
}

func countUp(results []DiscoverResult) int {
	n := 0
	for _, r := range results {
		if r.OK {
			n++
		}
	}
	return n
}
