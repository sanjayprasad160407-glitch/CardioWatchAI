# OpenAI API Key Setup

## Local VS Code
1. Copy `.env.example` to `.env`.
2. Put your OpenAI API key after `OPENAI_API_KEY=`.
3. Do not commit `.env` to GitHub.
4. Start the app with `python app.py`.
5. Open `http://127.0.0.1:5000` and use the CardioWatch Assistant.

## Render
In your Render service, open Environment Variables and add:
- `OPENAI_API_KEY` = your key
- `OPENAI_MODEL` = `gpt-5.6-luna`

Do not put the key in `render.yaml`, HTML, CSS, or JavaScript.
