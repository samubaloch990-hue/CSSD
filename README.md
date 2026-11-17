# AI Knowledge Assistant

This Streamlit app uses OpenRouter for answers and retrieves relevant passages locally from uploaded PDF, TXT, CSV, and Markdown files. Plain greetings are answered immediately; short, uncomplicated questions use a faster response setting, while complex and document-based questions retain deeper reasoning.

## Configure the API key

Create an [OpenRouter API key](https://openrouter.ai/keys). In Streamlit Community Cloud, open **Manage app → Settings → Secrets** and add:

```toml
OPENROUTER_API_KEY = "your-api-key"
```

Never commit the key or paste it into chat. The selected model is currently listed as free, but provider limits and availability can change. Questions and relevant document passages are sent to OpenRouter; avoid submitting sensitive documents.

Uploads and conversation history are kept in each active app session and are not written to shared disk.

## Shared correction memory

The app can remember corrected facts for all users without saving full chat transcripts. Each topic has one saved correction; a new correction for the same topic replaces the previous one. The PIN authorizes a change but does not verify that the fact itself is true.

1. Create a Supabase project and run this in its SQL Editor:

   ```sql
   create table if not exists public.assistant_memory (
       topic_key text primary key,
       topic_label text not null,
       correction text not null,
       updated_at timestamptz not null default now()
   );

   alter table public.assistant_memory enable row level security;
   revoke all on table public.assistant_memory from anon, authenticated;
   grant all on table public.assistant_memory to service_role;
   ```

2. In Streamlit Community Cloud, open **Manage app → Settings → Secrets** and add the Supabase project URL, private service-role key, and a private passphrase of at least 6 characters (shorter passphrases are easier to guess):

   ```toml
   SUPABASE_URL = "https://your-project.supabase.co"
   SUPABASE_SERVICE_ROLE_KEY = "your-private-service-role-key"
   MEMORY_EDIT_PIN = "choose-a-private-passphrase-of-6-or-more-characters"
   ```

   Keep these values private. Do not commit them to GitHub or paste them into chat.

3. In the app sidebar, use **Correct shared knowledge** to submit a topic and its corrected fact. A wrong PIN makes no database change. Only trusted people should know the PIN because approved corrections are shared with all app users.
# AI Knowledge Assistant

This Streamlit app uses OpenRouter for answers and retrieves relevant passages locally from uploaded PDF, TXT, CSV, and Markdown files. Plain greetings are answered immediately; short, uncomplicated questions use a faster response setting, while complex and document-based questions retain deeper reasoning.

## Configure the API key

Create an [OpenRouter API key](https://openrouter.ai/keys). In Streamlit Community Cloud, open **Manage app → Settings → Secrets** and add:

```toml
OPENROUTER_API_KEY = "your-api-key"
```

Never commit the key or paste it into chat. The selected model is currently listed as free, but provider limits and availability can change. Questions and relevant document passages are sent to OpenRouter; avoid submitting sensitive documents.

Uploads and conversation history are kept in each active app session and are not written to shared disk.

## Shared correction memory

The app can remember corrected facts for all users without saving full chat transcripts. Each topic has one saved correction; a new correction for the same topic replaces the previous one. The PIN authorizes a change but does not verify that the fact itself is true.

1. Create a Supabase project and run this in its SQL Editor:

   ```sql
   create table if not exists public.assistant_memory (
       topic_key text primary key,
       topic_label text not null,
       correction text not null,
       updated_at timestamptz not null default now()
   );

   alter table public.assistant_memory enable row level security;
   revoke all on table public.assistant_memory from anon, authenticated;
   grant all on table public.assistant_memory to service_role;
   ```

2. In Streamlit Community Cloud, open **Manage app → Settings → Secrets** and add the Supabase project URL, private service-role key, and a private passphrase of at least 12 characters:

   ```toml
   SUPABASE_URL = "https://your-project.supabase.co"
   SUPABASE_SERVICE_ROLE_KEY = "your-private-service-role-key"
   MEMORY_EDIT_PIN = "choose-a-private-passphrase-of-12-or-more-characters"
   ```

   Keep these values private. Do not commit them to GitHub or paste them into chat.

3. In the app sidebar, use **Correct shared knowledge** to submit a topic and its corrected fact. A wrong PIN makes no database change. Only trusted people should know the PIN because approved corrections are shared with all app users.
