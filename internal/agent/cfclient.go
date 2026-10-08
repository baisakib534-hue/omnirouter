// cfclient.go — minimal Cloudflare Workers AI client.
//
// Uses the OpenAI-compatible endpoint:
//
//	POST https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1/chat/completions
//	Authorization: Bearer {CF_API_TOKEN}
//
// Credentials come ONLY from the environment (CF_ACCOUNT_ID, CF_API_TOKEN);
// nothing secret is ever hardcoded or persisted.

package agent

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// defaultCFModel is the default backend-AI model: generous daily free quota
// on Workers AI. Overridable with CF_AI_MODEL.
const defaultCFModel = "@cf/meta/llama-3.1-8b-instruct"

// Msg is one chat message in OpenAI format.
type Msg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type cfClient struct {
	accountID string
	token     string
	model     string
	http      *http.Client
}

func newCFClient() (*cfClient, error) {
	accountID := strings.TrimSpace(os.Getenv("CF_ACCOUNT_ID"))
	token := strings.TrimSpace(os.Getenv("CF_API_TOKEN"))
	if accountID == "" || token == "" {
		return nil, fmt.Errorf("backend AI not configured: set CF_ACCOUNT_ID and CF_API_TOKEN env vars")
	}
	model := strings.TrimSpace(os.Getenv("CF_AI_MODEL"))
	if model == "" {
		model = defaultCFModel
	}
	return &cfClient{
		accountID: accountID,
		token:     token,
		model:     model,
		http:      &http.Client{Timeout: 90 * time.Second},
	}, nil
}

// Chat sends messages to Workers AI and returns the assistant's reply text.
// An empty model selects the configured default.
func (c *cfClient) Chat(model string, messages []Msg) (string, error) {
	if model == "" {
		model = c.model
	}
	payload, _ := json.Marshal(map[string]any{
		"model":      model,
		"messages":   messages,
		"stream":     false,
		"max_tokens": 2048,
	})
	url := fmt.Sprintf("https://api.cloudflare.com/client/v4/accounts/%s/ai/v1/chat/completions", c.accountID)
	req, err := http.NewRequest("POST", url, bytes.NewReader(payload))
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+c.token)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return "", fmt.Errorf("cloudflare ai request failed: %w", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 2<<20))

	if resp.StatusCode == 401 || resp.StatusCode == 403 {
		return "", fmt.Errorf("cloudflare ai auth failed (HTTP %d): check CF_ACCOUNT_ID / CF_API_TOKEN", resp.StatusCode)
	}
	if resp.StatusCode == 429 {
		return "", fmt.Errorf("cloudflare ai rate limited (HTTP 429): daily quota may be exhausted")
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("cloudflare ai HTTP %d: %s", resp.StatusCode, truncate(string(body), 300))
	}

	// OpenAI-compatible shape first …
	var openai struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &openai) == nil {
		if openai.Error != nil {
			return "", fmt.Errorf("cloudflare ai error: %s", openai.Error.Message)
		}
		if len(openai.Choices) > 0 {
			return strings.TrimSpace(openai.Choices[0].Message.Content), nil
		}
	}
	// … then the native Workers AI shape.
	var native struct {
		Success bool `json:"success"`
		Result  struct {
			Response string `json:"response"`
		} `json:"result"`
		Errors []struct {
			Message string `json:"message"`
		} `json:"errors"`
	}
	if json.Unmarshal(body, &native) == nil && native.Result.Response != "" {
		return strings.TrimSpace(native.Result.Response), nil
	}
	if len(native.Errors) > 0 {
		return "", fmt.Errorf("cloudflare ai error: %s", native.Errors[0].Message)
	}
	return "", fmt.Errorf("cloudflare ai: unrecognized response shape: %s", truncate(string(body), 300))
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
