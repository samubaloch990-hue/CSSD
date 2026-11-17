# AI Knowledge Assistant

This Streamlit app uses a free reasoning model through OpenRouter for detailed answers and retrieves relevant passages locally from uploaded PDF, TXT, CSV, and Markdown files.

## Configure the API key

Create an [OpenRouter API key](https://openrouter.ai/keys). In Streamlit Community Cloud, open **Manage app → Settings → Secrets** and add:

```toml
OPENROUTER_API_KEY = "your-api-key"
```

Never commit the key or paste it into chat. The selected model is currently listed as free, but provider limits and availability can change. Questions and relevant document passages are sent to OpenRouter; avoid submitting sensitive documents.

Uploads and conversation history are kept in each active app session and are not written to shared disk.
