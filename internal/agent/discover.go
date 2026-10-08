// discover.go — researched list of free keyless LLM endpoints.
//
// Every entry below was verified against public documentation on 2026-10-08:
//   - Pollinations.ai classic text API (pollinations/pollinations APIDOCS.md,
//     antseed "llm-api-without-account" guide): plain GET, no key, no account.
//   - LLM7.io (same guide + live /v1/models check): OpenAI-compatible,
//     documented anonymous key "Bearer unused", ~30 RPM. Catalog is dynamic;
//     default probe model DeepSeek-V4-Flash-0731 (turbo tier, free).
//   - Kilo Gateway (kilo.ai/docs/gateway + live probe): keyless
//     OpenAI-compatible at https://api.kilo.ai/api/gateway (+ /v1/chat/completions),
//     200 req/hr per IP on :free models. NOTE: Kilo's free routes may log prompts
//     for training — never send confidential data through auto-discovered providers.
//
// These are best-effort free tiers and can change or die at any time; the
// prober below is the source of truth — only endpoints that actually respond
// are reported (and only OpenAI-routable ones are auto-added as providers).

package agent

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"time"
)

// KeylessEndpoint describes one candidate free endpoint.
type KeylessEndpoint struct {
	Name     string   // short id, e.g. "llm7"
	Label    string   // human label
	BaseURL  string   // OpenAI-compatible base for core registration ("" if not routable)
	ProbeURL string   // exact URL probed
	Method   string   // "GET" | "POST"
	Auth     string   // Authorization header value ("" = none)
	Model    string   // model id used in POST probes
	Models   []string // models registered when the probe succeeds
	APIKey   string   // key value stored on the auto-added provider ("" = none)
	Routable bool     // core can route OpenAI traffic to BaseURL
	Note     string
}

var defaultKeylessEndpoints = []KeylessEndpoint{
	{
		Name:     "pollinations",
		Label:    "Pollinations.ai (classic text API)",
		ProbeURL: "https://text.pollinations.ai/hi",
		Method:   "GET",
		Models:   []string{"openai"},
		Routable: false, // plain-text GET API, not OpenAI-compatible
		Note:     "keyless GET endpoint; informational only, cannot be routed as an OpenAI provider",
	},
	{
		Name:     "pollinations-openai",
		Label:    "Pollinations.ai (OpenAI-compatible)",
		BaseURL:  "https://text.pollinations.ai",
		ProbeURL: "https://text.pollinations.ai/v1/chat/completions",
		Method:   "POST",
		Model:    "openai",
		Models:   []string{"openai"},
		Routable: true,
		Note:     "auto-added only if it answers without a key (router appends /v1/chat/completions to BaseURL)",
	},
	{
		Name:     "llm7",
		Label:    "LLM7.io (anonymous tier)",
		BaseURL:  "https://api.llm7.io",
		ProbeURL: "https://api.llm7.io/v1/chat/completions",
		Method:   "POST",
		Auth:     "Bearer unused", // documented anonymous key
		Model:    "DeepSeek-V4-Flash-0731",
		Models:   []string{"DeepSeek-V4-Flash-0731"},
		APIKey:   "unused",
		Routable: true,
		Note:     "llm7 catalog is dynamic — models come and go; the prober is the source of truth",
	},
	{
		Name:     "kilo",
		Label:    "Kilo Gateway (:free models)",
		BaseURL:  "https://api.kilo.ai/api/gateway",
		ProbeURL: "https://api.kilo.ai/api/gateway/v1/chat/completions",
		Method:   "POST",
		Model:    "kilo-auto/free",
		Models:   []string{"kilo-auto/free"},
		Routable: true,
		Note:     "free routes may log prompts for training — not for confidential data",
	},
}

// DiscoverResult is the outcome of probing one endpoint.
type DiscoverResult struct {
	Name      string
	Label     string
	OK        bool
	LatencyMs int64
	Err       string
	Routable  bool
}

// DiscoverKeyless probes every default endpoint and reports which respond.
// Probes go through the proxy pool when proxies are configured.
func (a *Agent) DiscoverKeyless() []DiscoverResult {
	client := a.pool.Client()
	out := make([]DiscoverResult, 0, len(defaultKeylessEndpoints))
	for _, ep := range defaultKeylessEndpoints {
		res := DiscoverResult{Name: ep.Name, Label: ep.Label, Routable: ep.Routable}
		start := time.Now()
		ok, errMsg := probeKeyless(client, ep)
		res.LatencyMs = time.Since(start).Milliseconds()
		res.OK = ok
		res.Err = errMsg
		out = append(out, res)
	}
	return out
}

func probeKeyless(client *http.Client, ep KeylessEndpoint) (bool, string) {
	var req *http.Request
	var err error
	if ep.Method == "GET" {
		req, err = http.NewRequest("GET", ep.ProbeURL, nil)
	} else {
		payload, _ := json.Marshal(map[string]any{
			"model":      ep.Model,
			"messages":   []Msg{{Role: "user", Content: "ping"}},
			"max_tokens": 8,
			"stream":     false,
		})
		req, err = http.NewRequest("POST", ep.ProbeURL, bytes.NewReader(payload))
		req.Header.Set("Content-Type", "application/json")
	}
	if err != nil {
		return false, err.Error()
	}
	if ep.Auth != "" {
		req.Header.Set("Authorization", ep.Auth)
	}
	req.Header.Set("User-Agent", "omnirouter-agent/1.0")

	resp, err := client.Do(req)
	if err != nil {
		return false, "request failed: " + truncate(err.Error(), 160)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	if resp.StatusCode == 429 {
		return false, "HTTP 429 (rate limited — endpoint alive but throttled)"
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return false, "HTTP " + itoa(resp.StatusCode) + ": " + truncate(strings.TrimSpace(string(body)), 160)
	}
	if ep.Method == "GET" {
		if len(bytes.TrimSpace(body)) == 0 {
			return false, "empty response body"
		}
		return true, ""
	}
	if !bytes.Contains(body, []byte(`"choices"`)) {
		return false, "unexpected response shape: " + truncate(strings.TrimSpace(string(body)), 160)
	}
	return true, ""
}
