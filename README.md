# AI Knowledge Assistant

This Streamlit app uses OpenAI's reasoning model for detailed answers and retrieves relevant passages from uploaded PDF, TXT, CSV, and Markdown files.

## Configure the API key

In Streamlit Community Cloud, open **Manage app → Settings → Secrets** and add:

```toml
OPENAI_API_KEY = "your-api-key"
```

Never commit the key or paste it into chat. API usage may incur charges to the app owner's OpenAI account. Because this app is public, visitors can use that account's quota. Uploaded text and questions are sent to OpenAI; avoid submitting sensitive documents.

Uploads and conversation history are kept in each active app session and are not written to shared disk.
