// proxy.go — rotating proxy pool for the backend AI.
//
// PROXY_POOL (comma-separated http(s):// URLs) seeds the pool; the
// proxy_add/proxy_list tools manage the persisted list in the store.
// The pool backs a round-robin http.RoundTripper used for keyless-provider
// probes, and core can route auto-discovered providers' traffic through it
// via Agent.ProxyFor.

package agent

import (
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"
)

// ProxyPool is a thread-safe round-robin list of proxy URLs.
type ProxyPool struct {
	mu   sync.Mutex
	urls []string
	idx  int
}

// NewProxyPool builds the pool from PROXY_POOL plus the persisted list.
func NewProxyPool(stored []string) *ProxyPool {
	p := &ProxyPool{}
	seen := map[string]bool{}
	add := func(u string) {
		u = strings.TrimSpace(u)
		if u == "" || seen[u] {
			return
		}
		seen[u] = true
		p.urls = append(p.urls, u)
	}
	for _, u := range strings.Split(os.Getenv("PROXY_POOL"), ",") {
		add(u)
	}
	for _, u := range stored {
		add(u)
	}
	return p
}

// Next returns the next proxy URL in rotation, or "" when the pool is empty.
func (p *ProxyPool) Next() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.urls) == 0 {
		return ""
	}
	u := p.urls[p.idx%len(p.urls)]
	p.idx++
	return u
}

// List returns a copy of the configured proxy URLs.
func (p *ProxyPool) List() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]string(nil), p.urls...)
}

// Add appends a proxy URL; false when already present or invalid.
func (p *ProxyPool) Add(raw string) bool {
	raw = strings.TrimSpace(raw)
	u, err := url.Parse(raw)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return false
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, e := range p.urls {
		if e == raw {
			return false
		}
	}
	p.urls = append(p.urls, raw)
	return true
}

// rotatingTransport is an http.RoundTripper that spreads requests across
// the proxy pool (direct connection when the pool is empty).
type rotatingTransport struct {
	pool   *ProxyPool
	direct http.RoundTripper
	mu     sync.Mutex
	cache  map[string]*http.Transport
}

func newRotatingTransport(pool *ProxyPool) *rotatingTransport {
	return &rotatingTransport{
		pool:   pool,
		direct: http.DefaultTransport,
		cache:  map[string]*http.Transport{},
	}
}

func (t *rotatingTransport) forProxy(proxyURL string) *http.Transport {
	t.mu.Lock()
	defer t.mu.Unlock()
	if tr, ok := t.cache[proxyURL]; ok {
		return tr
	}
	pu, _ := url.Parse(proxyURL)
	tr := &http.Transport{
		Proxy:                 http.ProxyURL(pu),
		MaxIdleConnsPerHost:   4,
		IdleConnTimeout:       60 * time.Second,
		TLSHandshakeTimeout:   15 * time.Second,
		ResponseHeaderTimeout: 60 * time.Second,
	}
	t.cache[proxyURL] = tr
	return tr
}

func (t *rotatingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if pu := t.pool.Next(); pu != "" {
		return t.forProxy(pu).RoundTrip(req)
	}
	return t.direct.RoundTrip(req)
}

// Client returns an http.Client whose requests rotate through the pool.
func (p *ProxyPool) Client() *http.Client {
	return &http.Client{
		Transport: newRotatingTransport(p),
		Timeout:   45 * time.Second,
	}
}
