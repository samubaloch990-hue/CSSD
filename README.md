# CSSD AI Assistant

CSSD AI Assistant uses OpenRouter for answers and retrieves relevant passages from shared documents. It supports PDF, TXT, CSV, and Markdown files. Choose English or Arabic in the sidebar for assistant replies; Arabic replies use Modern Standard Arabic and right-to-left chat text. The **Assistant** and **Notifications** tabs separate chat from setup alerts and recent confirmations; notification history can be cleared there. Plain greetings are answered immediately; short, uncomplicated questions use a faster response setting, while complex questions retain deeper reasoning.

## Configure the API key

Create an [OpenRouter API key](https://openrouter.ai/keys). In Streamlit Community Cloud, open **Manage app → Settings → Secrets** and add:

```toml
OPENROUTER_API_KEY = "your-api-key"
```

Never commit the key or paste it into chat. The selected model is currently listed as free, but provider limits and availability can change. Questions, relevant document passages, and aggregate feedback themes may be sent to OpenRouter; raw feedback notes and full chats are not.

Conversation history is kept only for the active app session. Answers are concise and simple by default; users can ask for more explanation. Document sources are grouped at the end instead of repeated after individual sentences. The assistant is also instructed to interpret typos, slang, fragments, and speech-to-text mistakes, and to ask for clarification when the intent is unclear. Shared feedback can influence future answers for everyone, but this is prompt-level adaptation, not training the underlying model; no full chats are saved for learning. Saved shared documents are stored as extracted text in Supabase and are available to everyone using the app. Relevant passages may be sent to OpenRouter when answering questions, so do not save sensitive documents.

## Shared document library

Use the single uploader in **Manage shared documents** in the app sidebar to save reference documents for future conversations. People with the configured PIN can add, replace, or remove documents; everyone can ask questions about saved documents. Uploading a file with the same name replaces its saved version. The app stores extracted text, not the original file. Scanned English PDFs are read with the free Tesseract OCR engine on the app server; large files may take a few minutes. Each upload can be up to 100 MB. The library supports up to 10 documents and has no app-imposed extracted-text character limit; hosting and database capacity still apply.

Streamlit Community Cloud installs Tesseract from `packages.txt`. For local use, install Tesseract and its English language data separately before uploading scanned PDFs.

## Shared correction memory

The app can remember corrected facts for all users without saving full chat transcripts. Each topic has one saved correction; a new correction for the same topic replaces the previous one. The PIN authorizes a change but does not verify that the fact itself is true.

Users can optionally rate answers and send a short suggestion from **Give feedback** in the sidebar. They can choose which answer to rate. Only the rating and suggestion are stored; the question, answer, and user identity are not. From the latest 100 feedback submissions, the app starts applying recurring style themes (such as simpler wording or examples) after at least 10 submissions and when a theme appears at least 3 times. It sends only aggregate ratings and detected theme labels to OpenRouter, not raw suggestions or chat transcripts. The app has no accounts, so it cannot verify that feedback came from 10 different people. Suggestions do not automatically change factual knowledge; the owner can review them and use **Correct shared knowledge** for approved factual changes.

1. Create a Supabase project and run this in its SQL Editor. This creates the shared document, correction memory, feedback, and PIN rate-limit tables/functions:

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

   create table if not exists public.assistant_admin_pin_rate_limit (
       id smallint primary key default 1 check (id = 1),
       attempt_count integer not null default 0
           check (attempt_count between 0 and 5),
       window_started_at timestamptz not null default now(),
       locked_until timestamptz
   );

   create or replace function public.consume_admin_pin_attempt()
   returns table (allowed boolean, retry_after timestamptz)
   language plpgsql
   security definer
   set search_path = public, pg_temp
   as $$
   declare
       v_attempt public.assistant_admin_pin_rate_limit%rowtype;
       v_now timestamptz;
       v_next_attempt integer;
   begin
       insert into public.assistant_admin_pin_rate_limit (id)
       values (1)
       on conflict (id) do nothing;

       select * into v_attempt
       from public.assistant_admin_pin_rate_limit
       where id = 1
       for update;

       v_now := clock_timestamp();
       if v_attempt.locked_until is not null then
           if v_attempt.locked_until > v_now then
               return query select false, v_attempt.locked_until;
               return;
           end if;
           update public.assistant_admin_pin_rate_limit
           set attempt_count = 0, window_started_at = v_now, locked_until = null
           where id = 1;
           v_attempt.attempt_count := 0;
           v_attempt.window_started_at := v_now;
           v_attempt.locked_until := null;
       end if;

       if v_attempt.window_started_at <= v_now - interval '1 hour' then
           update public.assistant_admin_pin_rate_limit
           set attempt_count = 0, window_started_at = v_now
           where id = 1;
           v_attempt.attempt_count := 0;
       end if;

       if v_attempt.attempt_count >= 5 then
           update public.assistant_admin_pin_rate_limit
           set locked_until = v_now + interval '1 hour'
           where id = 1;
           return query select false, v_now + interval '1 hour';
           return;
       end if;

       v_next_attempt := v_attempt.attempt_count + 1;
       update public.assistant_admin_pin_rate_limit
       set attempt_count = v_next_attempt,
           locked_until = case
               when v_next_attempt >= 5 then v_now + interval '1 hour'
               else null
           end
       where id = 1;
       return query select true, null::timestamptz;
   end;
   $$;

   create or replace function public.reset_admin_pin_attempts()
   returns void
   language plpgsql
   security definer
   set search_path = public, pg_temp
   as $$
   begin
       update public.assistant_admin_pin_rate_limit
       set attempt_count = 0,
           window_started_at = clock_timestamp(),
           locked_until = null
       where id = 1;
   end;
   $$;

   alter table public.assistant_memory enable row level security;
   alter table public.assistant_documents enable row level security;
   alter table public.assistant_feedback enable row level security;
   alter table public.assistant_admin_pin_rate_limit enable row level security;

   revoke all on table public.assistant_memory, public.assistant_documents,
       public.assistant_feedback, public.assistant_admin_pin_rate_limit
       from anon, authenticated;
   grant all on table public.assistant_memory, public.assistant_documents,
       public.assistant_feedback, public.assistant_admin_pin_rate_limit
       to service_role;

   revoke all on function public.consume_admin_pin_attempt()
       from public, anon, authenticated;
   revoke all on function public.reset_admin_pin_attempts()
       from public, anon, authenticated;
   grant execute on function public.consume_admin_pin_attempt() to service_role;
   grant execute on function public.reset_admin_pin_attempts() to service_role;
   ```

   PIN verification allows five failed attempts per hour across the whole app. Reaching the limit locks shared-edit actions for one hour for everyone; a correct PIN resets the counter.

2. In Streamlit Community Cloud, open **Manage app → Settings → Secrets** and add the Supabase project URL, private service-role key, and a six-digit numeric PIN:

   ```toml
   SUPABASE_URL = "https://your-project.supabase.co"
   SUPABASE_SERVICE_ROLE_KEY = "your-private-service-role-key"
   MEMORY_EDIT_PIN = "replace-with-six-digits"
   ```

   Keep these values private. Do not commit them to GitHub or paste them into chat.

3. In the app sidebar, use **Manage shared documents** to add documents once. Use **Correct shared knowledge** to submit a topic and its corrected fact. An incorrect PIN makes no database change. Use **Review user feedback** to review suggestions; marking feedback reviewed does not automatically change assistant knowledge. Only trusted people should know the PIN because saved documents and approved corrections are shared with all app users.
