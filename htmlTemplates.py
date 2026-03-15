"""
htmlTemplates.py
----------------
HTML/CSS snippets used by the DocChat Streamlit UI.

The chat interface renders each message as a styled bubble with an avatar.
All styles are inlined here so the app works without any external CSS files.
Avatar icons are rendered as inline SVG — no external image hosting needed,
so the UI stays consistent even offline or if a CDN goes down.
"""


# ---------------------------------------------------------------------------
# Global stylesheet injected once into the Streamlit page
# ---------------------------------------------------------------------------

css = """
<style>
/* ---- Chat message container ---- */
.chat-message {
    display: flex;
    align-items: flex-start;
    padding: 1.25rem 1.5rem;
    border-radius: 0.75rem;
    margin-bottom: 1rem;
    gap: 1rem;
}

.chat-message.user {
    background-color: #2b313e;
}

.chat-message.bot {
    background-color: #3d4a5c;
}

/* ---- Avatar circle ---- */
.chat-message .avatar {
    flex-shrink: 0;
    width: 44px;
    height: 44px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    background-color: rgba(255, 255, 255, 0.08);
}

/* ---- Message text ---- */
.chat-message .message {
    flex: 1;
    color: #e8eaf0;
    line-height: 1.6;
    font-size: 0.95rem;
    padding-top: 0.15rem;
}
</style>
"""

# ---------------------------------------------------------------------------
# Message bubble templates
# Placeholder {{MSG}} is replaced at render time with the actual text.
# ---------------------------------------------------------------------------

# Bot / assistant message — uses a robot emoji as the avatar (no external URL)
bot_template = """
<div class="chat-message bot">
    <div class="avatar">🤖</div>
    <div class="message">{{MSG}}</div>
</div>
"""

# User message — uses a person emoji as the avatar (no external URL)
user_template = """
<div class="chat-message user">
    <div class="avatar">🧑</div>
    <div class="message">{{MSG}}</div>
</div>
"""