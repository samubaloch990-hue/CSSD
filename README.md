# CSSD AI Assistant

CSSD AI Assistant uses OpenRouter for answers and retrieves relevant passages from shared documents. It supports PDF, TXT, CSV, and Markdown files. Plain greetings are answered immediately; short, uncomplicated questions use a faster response setting, while complex questions retain deeper reasoning.

## Configure the API key

Create an [OpenRouter API key](https://openrouter.ai/keys). In Streamlit Community Cloud, open **Manage app → Settings → Secrets** and add:

```toml
OPENROUTER_API_KEY = "your-api-key"
```

Never commit the key or paste it into chat. The selected model is currently listed as free, but provider limits and availability can change. Questions and relevant document passages are sent to OpenRouter; avoid submitting sensitive documents.

Conversation history is kept only for the active app session. Saved shared documents are stored as extracted text in Supabase and are available to everyone using the app. Relevant passages may be sent to OpenRouter when answering questions, so do not save sensitive documents.

## Shared document library

Use the single uploader in **Manage shared documents** in the app sidebar to save reference documents for future conversations. People with the configured PIN can add, replace, or remove documents; everyone can ask questions about saved documents. Uploading a file with the same name replaces its saved version. The app stores extracted text, not the original file. Scanned English PDFs are read with the free Tesseract OCR engine on the app server; large files may take a few minutes. Each upload can be up to 100 MB. The library supports up to 10 documents and has no app-imposed extracted-text character limit; hosting and database capacity still apply.

Streamlit Community Cloud installs Tesseract from `packages.txt`. For local use, install Tesseract and its English language data separately before uploading scanned PDFs.

## Shared correction memory

The app can remember corrected facts for all users without saving full chat transcripts. Each topic has one saved correction; a new correction for the same topic replaces the previous one. The PIN authorizes a change but does not verify that the fact itself is true.

Users can optionally rate answers and send a short suggestion from **Give feedback** in the sidebar. They can choose which answer to rate. Only the rating and suggestion are stored; the question and answer are not. Feedback is visible only after the owner enters the PIN and is never added to the assistant's knowledge automatically. Review a suggestion, then use **Correct shared knowledge** if it should become a shared correction.

1. Create a Supabase project and run this in its SQL Editor. This creates the shared document, correction memory, and feedback tables:

   ```sql
   create table if not exists public.assistant_memory (
       topic_key text primary key,
       topic_label text not null,
       correction text not null,
       updated_at timestamptz not null default now()
   );

   create table if not exists public.assistant_documents (
       document_id text primary key,
       file_name text not null unique,
       content text not null,
       updated_at timestamptz not null default now()
   );

   create table if not exists public.assistant_feedback (
       feedback_id uuid primary key default gen_random_uuid(),
       rating smallint not null check (rating in (-1, 1)),
       suggestion text not null default ''
           check (char_length(suggestion) <= 1000),
       status text not null default 'pending'
           check (status in ('pending', 'reviewed')),
       created_at timestamptz not null default now()
   );

   alter table public.assistant_memory enable row level security;
   alter table public.assistant_documents enable row level security;
   alter table public.assistant_feedback enable row level security;

   revoke all on table public.assistant_memory, public.assistant_documents,
       public.assistant_feedback
       from anon, authenticated;
   grant all on table public.assistant_memory, public.assistant_documents,
       public.assistant_feedback
       to service_role;
   ```

2. In Streamlit Community Cloud, open **Manage app → Settings → Secrets** and add the Supabase project URL, private service-role key, and a private passphrase of at least 6 characters (shorter passphrases are easier to guess):

   ```toml
   SUPABASE_URL = "https://your-project.supabase.co"
   SUPABASE_SERVICE_ROLE_KEY = "your-private-service-role-key"
   MEMORY_EDIT_PIN = "choose-a-private-passphrase-of-6-or-more-characters"
   ```

   Keep these values private. Do not commit them to GitHub or paste them into chat.

3. In the app sidebar, use **Manage shared documents** to add documents once. Use **Correct shared knowledge** to submit a topic and its corrected fact. A wrong PIN makes no database change. Use **Review user feedback** to review suggestions; marking feedback reviewed does not automatically change assistant knowledge. Only trusted people should know the PIN because saved documents and approved corrections are shared with all app users.
