// agent.go — the backend AI operator: tool-calling loop over Cloudflare
// Workers AI, plus the HTTP API (/api/agent/*) and the /agent chat UI.
//
// The LLM does not get native function-calling; instead the system prompt
// instructs it to emit {"tool":"name","args":{...}} JSON, which the loop
// parses, executes, and feeds back (max 8 iterations per user message).

package agent

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// ProviderDef describes a provider being registered with the router core.
type ProviderDef struct {
	ID        string
	BaseURL   string
	APIKeyEnv string
	APIKey    string // resolved key value ("" = none); never persisted by the agent
	Models    []string
	Source    string // "manual" | "auto-discovered"
	UseProxy  bool
	Enabled   bool
}

// CoreProviderInfo is a read-only view of a core registry provider.
type CoreProviderInfo struct {
	ID      string
	Label   string
	Enabled bool
	Models  []string
}

// Hooks are implemented by the router core (which owns the registry,
// forwarder and config) and injected here to avoid an import cycle.
type Hooks struct {
	// Probe runs one minimal chat completion through the router.
	Probe func(model string) (latencyMs int64, errMsg string)
	// SetCombo/DeleteCombo manage named failover chains in the registry.
	SetCombo    func(name string, chain []string)
	DeleteCombo func(name string) bool
	// AddProvider registers (or replaces) a custom provider in the registry.
	AddProvider func(def ProviderDef) error
	// SetProviderEnabled flips a provider's enabled flag in the registry.
	SetProviderEnabled func(id string, enabled bool)
	// ListCoreProviders snapshots the registry's providers.
	ListCoreProviders func() []CoreProviderInfo
}

// Action records one tool call made while answering.
type Action struct {
	Tool   string `json:"tool"`
	Result string `json:"result"`
}

// Agent is the backend AI operator.
type Agent struct {
	store   *Store
	hooks   Hooks
	cf      *cfClient // nil when CF_ACCOUNT_ID/CF_API_TOKEN are missing
	pool    *ProxyPool
	toolMap map[string]ToolDef
}

// New loads state from dir and wires the hooks. The agent works (probes,
// scheduler) even without Cloudflare credentials; only chat needs them.
func New(dataDir string, hooks Hooks) (*Agent, error) {
	st, err := NewStore(dataDir)
	if err != nil {
		return nil, err
	}
	a := &Agent{store: st, hooks: hooks}
	a.pool = NewProxyPool(st.ListProxies())
	a.cf, _ = newCFClient() // nil when unconfigured; chat reports it
	for _, t := range a.tools() {
		if a.toolMap == nil {
			a.toolMap = map[string]ToolDef{}
		}
		a.toolMap[t.Name] = t
	}
	return a, nil
}

// Combos returns the persisted weighted combos (for boot re-seeding).
func (a *Agent) Combos() map[string][]ComboEntry { return a.store.ListCombos() }

// Providers returns the persisted provider records (custom providers the
// agent manages, including auto-discovered keyless ones) for boot
// re-registration into the core registry.
func (a *Agent) Providers() []*ProviderState { return a.store.ListProviders() }

// OrderedChain sorts combo entries by weight (desc) into a plain chain.
func OrderedChain(entries []ComboEntry) []string {
	cp := append([]ComboEntry(nil), entries...)
	sortByWeight(cp)
	chain := make([]string, 0, len(cp))
	for _, e := range cp {
		chain = append(chain, e.Model)
	}
	return chain
}

func sortByWeight(entries []ComboEntry) {
	for i := 1; i < len(entries); i++ {
		for j := i; j > 0 && entries[j].Weight > entries[j-1].Weight; j-- {
			entries[j], entries[j-1] = entries[j-1], entries[j]
		}
	}
}

// --- request-path integration (called by the router core) ---

// AllowCandidate reports whether a provider/upstream-model candidate may
// serve traffic: enabled provider, enabled model, quota remaining.
func (a *Agent) AllowCandidate(providerID, upstreamModel string) bool {
	key := providerID + "/" + upstreamModel
	if p, ok := a.store.GetProvider(providerID); ok && !p.Enabled {
		return false
	}
	if !a.store.HealthFor(key).Enabled {
		return false
	}
	return a.store.QuotaAllows(key)
}

// NoteUsage increments the daily request counter for a served model.
func (a *Agent) NoteUsage(providerID, upstreamModel string) {
	a.store.NoteUsage(providerID + "/" + upstreamModel)
}

// ProxyFor returns the next proxy URL for a provider flagged use_proxy,
// or "" when no proxy applies.
func (a *Agent) ProxyFor(providerID string) string {
	if p, ok := a.store.GetProvider(providerID); ok && p.UseProxy {
		return a.pool.Next()
	}
	return ""
}

// --- tool-calling chat ---

const maxIterations = 8

func (a *Agent) systemPrompt() string {
	var b strings.Builder
	b.WriteString(`You are the Backend AI, the DevOps operator of this OmniRouter instance — an OpenAI-compatible LLM gateway (single Go binary). You manage providers, combos, quotas and model health. You MAY enable/disable providers and models, create combos, set quotas, probe models, manage the proxy pool, and discover free keyless providers.

Rules:
- Always reply in the same language the user writes in.
- Explain each action briefly in plain language after acting; keep it short.
- Be careful: disabling things affects live traffic. Probe a model before disabling it when you can.
- When asked to route, optimize or pick models, consult the live quota/health summary below and prefer models with remaining quota — route directly to them.
- Never reveal API keys, tokens or secrets. Provider records keep only env var NAMES, never values.

Live quota/health summary (limits reset at UTC midnight):
`)
	b.WriteString(a.store.QuotaSummary())
	b.WriteString(`

Tools — to call a tool, reply with ONLY this JSON object and nothing else:
{"tool":"<name>","args":{...}}

Available tools:
`)
	for _, t := range a.tools() {
		b.WriteString("- " + t.Name + " " + t.Args + " — " + t.Desc + "\n")
	}
	b.WriteString(`- When you are finished, reply in plain text (no JSON). One tool call per turn; I will feed each result back.`)
	return b.String()
}

func (a *Agent) buildMessages() []Msg {
	msgs := []Msg{{Role: "system", Content: a.systemPrompt()}}
	hist := a.store.History()
	if len(hist) > 20 {
		hist = hist[len(hist)-20:]
	}
	for _, e := range hist {
		role := e.Role
		if role != "user" && role != "assistant" && role != "system" {
			role = "user"
		}
		msgs = append(msgs, Msg{Role: role, Content: e.Content})
	}
	return msgs
}

// Chat runs one user message through the tool-calling loop.
func (a *Agent) Chat(userMsg string) (string, []Action, error) {
	if a.cf == nil {
		return "", nil, fmt.Errorf("backend AI not configured: set CF_ACCOUNT_ID and CF_API_TOKEN env vars, then restart")
	}
	userMsg = strings.TrimSpace(userMsg)
	if userMsg == "" {
		return "", nil, fmt.Errorf("empty message")
	}
	a.store.AddHistory("user", userMsg, nil)

	var actions []Action
	msgs := a.buildMessages()
	var reply string
	for i := 0; i < maxIterations; i++ {
		text, err := a.cf.Chat("", msgs)
		if err != nil {
			return "", actions, err
		}
		reply = text
		toolName, args, ok := parseToolCall(text)
		if !ok {
			break // plain-text final answer
		}
		result := a.runTool(toolName, args)
		actions = append(actions, Action{Tool: toolName, Result: truncate(result, 600)})
		msgs = append(msgs,
			Msg{Role: "assistant", Content: text},
			Msg{Role: "user", Content: "Tool result (" + toolName + "):\n" + result},
		)
	}
	names := make([]string, 0, len(actions))
	for _, ac := range actions {
		names = append(names, ac.Tool)
	}
	a.store.AddHistory("assistant", reply, names)
	return reply, actions, nil
}

func (a *Agent) runTool(name string, args map[string]any) string {
	t, ok := a.toolMap[name]
	if !ok {
		return "error: unknown tool " + name
	}
	out, err := t.Run(args)
	if err != nil {
		return "error: " + err.Error()
	}
	return out
}

// parseToolCall extracts {"tool":...,"args":{...}} from model output,
// tolerating ```json fences and surrounding prose.
func parseToolCall(s string) (string, map[string]any, bool) {
	t := strings.TrimSpace(s)
	start := strings.Index(t, "{")
	end := strings.LastIndex(t, "}")
	if start < 0 || end <= start {
		return "", nil, false
	}
	candidate := t[start : end+1]
	if !strings.Contains(candidate, `"tool"`) {
		return "", nil, false
	}
	var tc struct {
		Tool string         `json:"tool"`
		Args map[string]any `json:"args"`
	}
	if json.Unmarshal([]byte(candidate), &tc) != nil || tc.Tool == "" {
		return "", nil, false
	}
	if tc.Args == nil {
		tc.Args = map[string]any{}
	}
	return tc.Tool, tc.Args, true
}

// --- HTTP API ---

// Mount registers /api/agent/* and the /agent chat UI. auth wraps handlers
// with the router's admin authentication.
func (a *Agent) Mount(mux *http.ServeMux, auth func(http.HandlerFunc) http.HandlerFunc) {
	mux.HandleFunc("/api/agent/chat", auth(a.handleChat))
	mux.HandleFunc("/api/agent/quota", auth(a.handleQuota))
	mux.HandleFunc("/agent", auth(a.handleUI))
}

func writeAgentJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func (a *Agent) handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		writeAgentJSON(w, 405, map[string]string{"error": "method not allowed"})
		return
	}
	var body struct {
		Message string `json:"message"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeAgentJSON(w, 400, map[string]string{"error": "invalid json"})
		return
	}
	reply, actions, err := a.Chat(body.Message)
	if err != nil {
		writeAgentJSON(w, 500, map[string]string{"error": err.Error()})
		return
	}
	writeAgentJSON(w, 200, map[string]any{"reply": reply, "actions": actions})
}

func (a *Agent) handleQuota(w http.ResponseWriter, r *http.Request) {
	writeAgentJSON(w, 200, map[string]any{"models": a.store.QuotaSnapshot()})
}

func (a *Agent) handleUI(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write([]byte(agentUIHTML))
}

// agentUIHTML is the minimal single-page chat UI (no framework).
const agentUIHTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Backend AI — OmniRouter operator</title>
<style>
body{background:#0d1117;color:#e6edf3;font-family:system-ui,sans-serif;margin:0;display:flex;flex-direction:column;height:100vh}
header{padding:12px 16px;border-bottom:1px solid #30363d;display:flex;justify-content:space-between;align-items:center}
header h1{font-size:16px;margin:0}
#quota{font-size:12px;color:#8b949e}
#log{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:10px}
.msg{max-width:80%;padding:10px 14px;border-radius:10px;white-space:pre-wrap;word-break:break-word}
.user{align-self:flex-end;background:#1f6feb}
.ai{align-self:flex-start;background:#161b22;border:1px solid #30363d}
.sys{align-self:center;color:#8b949e;font-size:12px}
.act{font-size:11px;color:#8b949e;margin-top:6px}
form{display:flex;gap:8px;padding:12px;border-top:1px solid #30363d}
input{flex:1;background:#0d1117;border:1px solid #30363d;color:#e6edf3;border-radius:8px;padding:10px}
button{background:#1f6feb;color:#fff;border:0;border-radius:8px;padding:10px 18px;cursor:pointer}
button:disabled{opacity:.5}
</style>
</head>
<body>
<header><h1>🤖 Backend AI — router operator</h1><span id="quota"></span></header>
<div id="log"></div>
<form id="f"><input id="in" autocomplete="off" placeholder="Ask the operator… (e.g. probe qwen/qwen3.8-max, disable slow models)"><button>Send</button></form>
<script>
var log=document.getElementById('log'), form=document.getElementById('f'), input=document.getElementById('in');
function add(cls, text, actions){
  var d=document.createElement('div'); d.className='msg '+cls; d.textContent=text;
  if(actions&&actions.length){ var a=document.createElement('div'); a.className='act';
    a.textContent='actions: '+actions.map(function(x){return x.tool}).join(', '); d.appendChild(a); }
  log.appendChild(d); log.scrollTop=log.scrollHeight;
}
function sys(t){ var d=document.createElement('div'); d.className='sys'; d.textContent=t; log.appendChild(d); }
form.addEventListener('submit', function(e){
  e.preventDefault();
  var m=input.value.trim(); if(!m) return;
  input.value=''; add('user', m);
  var btn=form.querySelector('button'); btn.disabled=true;
  fetch('/api/agent/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:m})})
    .then(function(r){return r.json().then(function(j){return {ok:r.ok,j:j}})})
    .then(function(x){
      if(!x.ok){ sys('error: '+(x.j.error||x.j)); }
      else { add('ai', x.j.reply||'(no reply)', x.j.actions); }
      btn.disabled=false; refreshQuota();
    })
    .catch(function(err){ sys('request failed: '+err); btn.disabled=false; });
});
function refreshQuota(){
  fetch('/api/agent/quota').then(function(r){return r.json()}).then(function(j){
    var models=j.models||{}, n=0, dis=0;
    for(var k in models){ n++; if(!models[k].enabled) dis++; }
    document.getElementById('quota').textContent=n+' models tracked'+(dis?' ('+dis+' disabled)':'');
  }).catch(function(){});
}
sys('Backend AI ready. It manages providers, combos, quotas and health.');
refreshQuota();
</script>
</body>
</html>`
