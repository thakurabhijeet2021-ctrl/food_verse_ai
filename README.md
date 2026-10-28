# Food Verse AI — Multimodal Food Intelligence Agent

Skill Wallet Capstone Project 3. A modular Streamlit application with five pillars:
1. Food recognition and nutrition audit
2. Freshness and spoilage diagnostic
3. Kitchen hygiene and safety review
4. Smart recipe assistant
5. Personalized meal planner

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## AI providers

- **Demo mode** works without credentials and is clearly labeled as illustrative output.
- **Gemini**: choose Gemini in the sidebar and enter a Google AI API key. The app uses `gemini-2.5-flash`.
- **Groq**: choose Groq and enter a Groq API key. Vision uses `meta-llama/llama-4-scout-17b-16e-instruct`; recipe generation uses `llama-3.3-70b-versatile`.

For deployment, prefer Streamlit Community Cloud **Secrets** over entering keys in a shared session. Never commit keys to source control. The sidebar key is session-only and is not persisted by this app.

## Streamlit Community Cloud

1. Upload `app.py`, `requirements.txt`, and this README to a GitHub repository.
2. Create a new app in Streamlit Community Cloud and select `app.py`.
3. Deploy. Demo mode needs no secrets.
4. Optional: add secrets and update the provider code to read them from `st.secrets`, or enter a key in the sidebar.

## Important limitations

- Nutrition estimates from images are uncertain because ingredients, cooking oil, recipes and serving weights may be hidden. Use a weighed recipe and trusted nutrition database for higher accuracy.
- Images cannot confirm pathogens, toxins, temperature history, or allergen cross-contact. Do not use the freshness module as a definitive safety test.
- Kitchen score is a visual screening aid, not a regulatory inspection or verified compliance certification.
- Meal plans are templates. The Mifflin–St Jeor equation estimates BMR and activity multipliers estimate TDEE; actual needs differ. Diabetic-friendly and keto menus are not clinical treatment plans. Confirm allergies and consult a qualified clinician or dietitian for medical needs.
- The video flow previews a video but does not perform frame-by-frame inference; upload a representative frame for analysis.
