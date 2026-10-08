// tools.go — the backend AI's operator tools.
//
// Each tool is a Go func plus a compact schema shown to the LLM. The agent
// loop parses {"tool":"name","args":{...}} from the model's reply and runs
// the matching tool. Model keys are always "provider/model"
// (e.g. "qwen/qwen3.8-max", "custom:llm7/DeepSeek-V4-Flash-0731").

package agent

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

// ToolDef is one callable operator tool.
type ToolDef struct {
	Name string
	Desc string
	Args string // compact argument docs for the system prompt
	Run  func(args map[string]any) (string, error)
}

// --- arg helpers ---

func strArg(args map[string]any, key string) string {
	v, _ := args[key].(string)
	return strings.TrimSpace(v)
}

func intArg(args map[string]any, key string) int {
	switch v := args[key].(type) {
	case float64:
		return int(v)
	case int:
		return v
	case json.Number:
		n, _ := v.Int64()
		return int(n)
	}
	return 0
}

func boolArg(args map[string]any, key string) (bool, bool) {
	v, ok := args[key].(bool)
	return v, ok
}

func strSliceArg(args map[string]any, key string) []string {
	var out []string
	switch v := args[key].(type) {
	case []any:
		for _, e := range v {
			if s, ok := e.(string); ok && strings.TrimSpace(s) != "" {
				out = append(out, strings.TrimSpace(s))
			}
		}
	case []string:
		out = v
	}
	return out
}

// normalizeProviderID maps a user-given name to a core provider id:
// built-in bridges keep their id, everything else becomes "custom:<slug>".
func normalizeProviderID(name string) string {
	name = strings.ToLower(strings.TrimSpace(name))
	switch name {
	case "qwen", "glm", "ds", "gemini", "oc", "freebuff":
		return name
	}
	name = strings.TrimPrefix(name, "custom:")
	var b strings.Builder
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			b.WriteRune(r)
		case r == '-' || r == '_' || r == '.':
			b.WriteRune(r)
		}
	}
	slug := b.String()
	if slug == "" {
		slug = "provider"
	}
	return "custom:" + slug
}

func okJSON(v any) (string, error) {
	raw, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return "", err
	}
	return string(raw), nil
}

// tools returns the full operator toolset.
func (a *Agent) tools() []ToolDef {
	return []ToolDef{
		{
			Name: "list_providers",
			Desc: "List providers managed by the agent (id, base_url, enabled, models, source).",
			Args: "{}",
			Run: func(args map[string]any) (string, error) {
				return okJSON(a.store.ListProviders())
			},
		},
		{
			Name: "add_provider",
			Desc: "Add an OpenAI-compatible provider and register it with the router.",
			Args: `{"name":"short id (e.g. llm7)","base_url":"https://...","api_key_env":"ENV_VAR_NAME or empty for keyless","models":["model-id", ...]}`,
			Run: func(args map[string]any) (string, error) {
				name, baseURL := strArg(args, "name"), strArg(args, "base_url")
				keyEnv, models := strArg(args, "api_key_env"), strSliceArg(args, "models")
				if name == "" || baseURL == "" {
					return "", fmt.Errorf("name and base_url are required")
				}
				if len(models) == 0 {
					return "", fmt.Errorf("models must list at least one model id")
				}
				id := normalizeProviderID(name)
				key := ""
				if keyEnv != "" {
					key = strings.TrimSpace(os.Getenv(keyEnv))
					if key == "" {
						return "", fmt.Errorf("env var %s is empty or not set", keyEnv)
					}
				}
				def := ProviderDef{
					ID: id, BaseURL: strings.TrimRight(baseURL, "/"),
					APIKeyEnv: keyEnv, APIKey: key,
					Models: models, Source: "manual", Enabled: true,
				}
				if a.hooks.AddProvider == nil {
					return "", fmt.Errorf("router hook unavailable")
				}
				if err := a.hooks.AddProvider(def); err != nil {
					return "", err
				}
				a.store.SetProvider(&ProviderState{
					Name: id, BaseURL: def.BaseURL, APIKeyEnv: keyEnv,
					Models: models, Enabled: true, Source: "manual",
				})
				a.applyDefaultQuota(id, models)
				return fmt.Sprintf("provider %s added with %d model(s) and registered with the router", id, len(models)), nil
			},
		},
		{
			Name: "update_provider",
			Desc: "Update a provider's base_url, api_key_env and/or model list (partial update).",
			Args: `{"name":"provider id","base_url":"optional","api_key_env":"optional","models":["optional replacement list"]}`,
			Run: func(args map[string]any) (string, error) {
				id := normalizeProviderID(strArg(args, "name"))
				p, ok := a.store.GetProvider(id)
				if !ok {
					return "", fmt.Errorf("unknown provider %s (see list_providers)", id)
				}
				if v := strArg(args, "base_url"); v != "" {
					p.BaseURL = strings.TrimRight(v, "/")
				}
				if v := strArg(args, "api_key_env"); v != "" {
					p.APIKeyEnv = v
				}
				if m := strSliceArg(args, "models"); len(m) > 0 {
					p.Models = m
				}
				key := ""
				if p.APIKeyEnv != "" {
					key = strings.TrimSpace(os.Getenv(p.APIKeyEnv))
					if key == "" {
						return "", fmt.Errorf("env var %s is empty or not set", p.APIKeyEnv)
					}
				}
				if a.hooks.AddProvider == nil {
					return "", fmt.Errorf("router hook unavailable")
				}
				if err := a.hooks.AddProvider(ProviderDef{
					ID: id, BaseURL: p.BaseURL, APIKeyEnv: p.APIKeyEnv,
					APIKey: key, Models: p.Models, Source: p.Source,
					UseProxy: p.UseProxy, Enabled: p.Enabled,
				}); err != nil {
					return "", err
				}
				a.store.SetProvider(p)
				return fmt.Sprintf("provider %s updated", id), nil
			},
		},
		{
			Name: "enable_provider",
			Desc: "Enable a provider so the router may use it again.",
			Args: `{"name":"provider id"}`,
			Run: func(args map[string]any) (string, error) {
				return a.setProviderEnabled(strArg(args, "name"), true)
			},
		},
		{
			Name: "disable_provider",
			Desc: "Disable a provider: the router will skip it for all models.",
			Args: `{"name":"provider id"}`,
			Run: func(args map[string]any) (string, error) {
				return a.setProviderEnabled(strArg(args, "name"), false)
			},
		},
		{
			Name: "list_models",
			Desc: "List every model the router knows, as provider/model keys, with agent health/quota state.",
			Args: "{}",
			Run: func(args map[string]any) (string, error) {
				if a.hooks.ListCoreProviders == nil {
					return "", fmt.Errorf("router hook unavailable")
				}
				type row struct {
					Model   string `json:"model"`
					Enabled bool   `json:"enabled"`
					Quota   string `json:"quota"`
				}
				var out []row
				for _, p := range a.hooks.ListCoreProviders() {
					for _, m := range p.Models {
						key := p.ID + "/" + m
						enabled := p.Enabled && a.modelEnabled(key)
						quota := "unlimited"
						if q := a.store.QuotaSnapshot()[key]; q != nil {
							if lim, _ := q["limit"].(int); lim > 0 {
								quota = fmt.Sprintf("%v/%v used", q["used"], lim)
							}
						}
						out = append(out, row{Model: key, Enabled: enabled, Quota: quota})
					}
				}
				sort.Slice(out, func(i, j int) bool { return out[i].Model < out[j].Model })
				return okJSON(out)
			},
		},
		{
			Name: "enable_model",
			Desc: "Re-enable one model (provider/model).",
			Args: `{"model":"provider/model"}`,
			Run: func(args map[string]any) (string, error) {
				m := strArg(args, "model")
				if m == "" {
					return "", fmt.Errorf("model is required")
				}
				a.store.SetModelEnabled(m, true)
				return fmt.Sprintf("model %s enabled", m), nil
			},
		},
		{
			Name: "disable_model",
			Desc: "Disable one model: the router will skip it (failover still tries the rest).",
			Args: `{"model":"provider/model"}`,
			Run: func(args map[string]any) (string, error) {
				m := strArg(args, "model")
				if m == "" {
					return "", fmt.Errorf("model is required")
				}
				a.store.SetModelEnabled(m, false)
				return fmt.Sprintf("model %s disabled", m), nil
			},
		},
		{
			Name: "create_combo",
			Desc: "Create/update a named combo: ordered failover chain of provider/model entries with weights (higher weight = tried first).",
			Args: `{"name":"my-stack","entries":[{"model":"provider/model","weight":3},{"model":"provider/model","weight":1}]}`,
			Run: func(args map[string]any) (string, error) {
				name := strings.ToLower(strings.TrimSpace(strArg(args, "name")))
				if name == "" {
					return "", fmt.Errorf("name is required")
				}
				raw, _ := args["entries"].([]any)
				if len(raw) == 0 {
					return "", fmt.Errorf("entries must be a non-empty list")
				}
				var entries []ComboEntry
				for _, e := range raw {
					m, _ := e.(map[string]any)
					model := strings.TrimSpace(strArg(m, "model"))
					if model == "" || !strings.Contains(model, "/") {
						return "", fmt.Errorf("each entry needs a \"provider/model\" id")
					}
					w := intArg(m, "weight")
					if w <= 0 {
						w = 1
					}
					entries = append(entries, ComboEntry{Model: model, Weight: w})
				}
				// order by weight desc (stable), then persist + register.
				ordered := append([]ComboEntry(nil), entries...)
				sort.SliceStable(ordered, func(i, j int) bool { return ordered[i].Weight > ordered[j].Weight })
				chain := make([]string, 0, len(ordered))
				for _, e := range ordered {
					chain = append(chain, e.Model)
				}
				a.store.SetCombo(name, entries)
				if a.hooks.SetCombo != nil {
					a.hooks.SetCombo(name, chain)
				}
				return fmt.Sprintf("combo %s saved (%d entries, order: %s)", name, len(chain), strings.Join(chain, " → ")), nil
			},
		},
		{
			Name: "list_combos",
			Desc: "List all named combos with their weighted entries.",
			Args: "{}",
			Run: func(args map[string]any) (string, error) {
				return okJSON(a.store.ListCombos())
			},
		},
		{
			Name: "delete_combo",
			Desc: "Delete a named combo.",
			Args: `{"name":"combo-name"}`,
			Run: func(args map[string]any) (string, error) {
				name := strings.ToLower(strings.TrimSpace(strArg(args, "name")))
				if !a.store.DeleteCombo(name) {
					return "", fmt.Errorf("combo %s does not exist", name)
				}
				if a.hooks.DeleteCombo != nil {
					a.hooks.DeleteCombo(name)
				}
				return fmt.Sprintf("combo %s deleted", name), nil
			},
		},
		{
			Name: "quota_status",
			Desc: "Show per-model daily quota usage: limit, used, remaining, enabled.",
			Args: "{}",
			Run: func(args map[string]any) (string, error) {
				return okJSON(a.store.QuotaSnapshot())
			},
		},
		{
			Name: "set_quota",
			Desc: "Set a model's daily request limit (0 = unlimited). Counters reset at UTC midnight.",
			Args: `{"model":"provider/model","daily_limit":1000}`,
			Run: func(args map[string]any) (string, error) {
				m := strArg(args, "model")
				if m == "" {
					return "", fmt.Errorf("model is required")
				}
				lim := intArg(args, "daily_limit")
				if lim < 0 {
					return "", fmt.Errorf("daily_limit must be >= 0")
				}
				a.store.SetQuotaLimit(m, lim)
				if lim == 0 {
					return fmt.Sprintf("quota for %s set to unlimited", m), nil
				}
				return fmt.Sprintf("daily quota for %s set to %d", m, lim), nil
			},
		},
		{
			Name: "probe_model",
			Desc: "Send one minimal chat completion through the router to a model and report ok/error + latency.",
			Args: `{"model":"provider/model"}`,
			Run: func(args map[string]any) (string, error) {
				m := strArg(args, "model")
				if m == "" {
					return "", fmt.Errorf("model is required")
				}
				if a.hooks.Probe == nil {
					return "", fmt.Errorf("router hook unavailable")
				}
				ms, errMsg := a.hooks.Probe(m)
				if errMsg != "" {
					return fmt.Sprintf("probe %s FAILED in %dms: %s", m, ms, errMsg), nil
				}
				return fmt.Sprintf("probe %s OK in %dms", m, ms), nil
			},
		},
		{
			Name: "proxy_add",
			Desc: "Add an http(s) proxy URL to the rotating pool used for keyless-provider traffic.",
			Args: `{"url":"http://user:pass@host:port"}`,
			Run: func(args map[string]any) (string, error) {
				u := strArg(args, "url")
				if !a.pool.Add(u) {
					return "", fmt.Errorf("invalid or duplicate proxy URL")
				}
				a.store.AddProxy(u)
				return fmt.Sprintf("proxy added (%d in pool)", len(a.pool.List())), nil
			},
		},
		{
			Name: "proxy_list",
			Desc: "List the proxy pool URLs.",
			Args: "{}",
			Run: func(args map[string]any) (string, error) {
				return okJSON(a.pool.List())
			},
		},
		{
			Name: "health_check",
			Desc: "Run the full model health check now (probes every enabled model, disables failures).",
			Args: "{}",
			Run: func(args map[string]any) (string, error) {
				go a.checkAllModels()
				return "health check started in the background; results will appear in the chat history and quota view", nil
			},
		},
		{
			Name: "discover_keyless",
			Desc: "Probe the researched list of free keyless LLM endpoints and report which respond (does not change anything).",
			Args: "{}",
			Run: func(args map[string]any) (string, error) {
				res := a.DiscoverKeyless()
				var b strings.Builder
				for _, r := range res {
					status := "DOWN"
					if r.OK {
						status = "UP"
					}
					fmt.Fprintf(&b, "- %s (%s): %s in %dms", r.Name, r.Label, status, r.LatencyMs)
					if r.Err != "" {
						b.WriteString(" — " + r.Err)
					}
					b.WriteString("\n")
				}
				return strings.TrimRight(b.String(), "\n"), nil
			},
		},
	}
}

func (a *Agent) setProviderEnabled(name string, enabled bool) (string, error) {
	id := normalizeProviderID(name)
	p, ok := a.store.GetProvider(id)
	if !ok {
		// mirror a core-known provider so the record exists
		p = &ProviderState{Name: id, Enabled: true, Source: "manual"}
		if a.hooks.ListCoreProviders != nil {
			for _, cp := range a.hooks.ListCoreProviders() {
				if cp.ID == id {
					p.BaseURL, p.Models = "", cp.Models
					break
				}
			}
		}
	}
	p.Enabled = enabled
	a.store.SetProvider(p)
	if a.hooks.SetProviderEnabled != nil {
		a.hooks.SetProviderEnabled(id, enabled)
	}
	state := "enabled"
	if !enabled {
		state = "disabled"
	}
	return fmt.Sprintf("provider %s %s", id, state), nil
}

func (a *Agent) modelEnabled(key string) bool {
	return a.store.HealthFor(key).Enabled
}

// applyDefaultQuota seeds AGENT_DEFAULT_DAILY_QUOTA on freshly added models.
func (a *Agent) applyDefaultQuota(providerID string, models []string) {
	def := 0
	if v := strings.TrimSpace(os.Getenv("AGENT_DEFAULT_DAILY_QUOTA")); v != "" {
		var n int
		if _, err := fmt.Sscanf(v, "%d", &n); err == nil && n > 0 {
			def = n
		}
	}
	if def == 0 {
		return
	}
	for _, m := range models {
		a.store.SetQuotaLimit(providerID+"/"+m, def)
	}
}
